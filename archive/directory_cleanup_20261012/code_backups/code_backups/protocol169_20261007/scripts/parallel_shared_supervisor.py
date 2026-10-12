#!/usr/bin/env python3
"""Run two backbone generation lanes plus shared Qwen judging and aggregation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_main_experiment import build_plan
from scripts.main_supervisor import complete_results
from utils.artifacts import hash_object, write_json
from utils.model_manifest import verify_manifest


def process_argv(pid):
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except FileNotFoundError:
        return None
    return [s.decode() for s in raw.split(b"\0") if s] or None


def weight_memory_mib(path):
    root = Path(path)
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = root / name
        if index.exists():
            size = json.loads(index.read_text()).get("metadata", {}).get("total_size")
            if size:
                return int(size / 1024**2) + 8192
    return int(sum(p.stat().st_size for p in root.glob("*.safetensors")) / 1024**2) + 8192


def available_checkpoint(cell):
    path = Path(cell["checkpoint"])
    if not (path / "config.json").exists():
        return False
    if cell["method"] in {"zero_shot", "ma2p"}:
        return True
    contract = path / "training_contract.json"
    return contract.exists() and any(path.glob("*.safetensors"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    project = Path(cfg["project_root"])
    output = Path(cfg["outputs_root"])
    control = output / "supervisor"
    logs = control / "logs" / "parallel"
    logs.mkdir(parents=True, exist_ok=True)
    lock = (control / "parallel.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = build_plan(cfg, args.config, project / cfg["plan_path"])
    write_json(project / cfg["plan_path"], plan)
    status_path = control / "parallel_status.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {
        "config_hash": hash_object(cfg), "jobs": {}, "started_at": time.time()}
    if state["config_hash"] != hash_object(cfg):
        raise ValueError("Parallel configuration changed; choose a new result directory")
    children = {}
    cells = sorted(plan["cells"], key=lambda c: (
        {"qwen25": 0, "llama31": 1, "gemma2": 2}[c["backbone"]],
        {"zero_shot": 0, "sft": 1, "ma2p": 2, "grpo_only": 3, "sft_grpo": 4, "ours": 5}[c["method"]]))
    jobs = {j["name"]: j for j in plan["jobs"]}
    expected = cfg["data"]["expected_eval_items"]
    verified_manifests = set()
    local_opener = build_opener(ProxyHandler({}))

    def generation_name(cell):
        return f"{cell['backbone']}-{cell['method']}-r{cell['repeat_id']}"

    def done(cell, phase):
        path = Path(cell["generation_dir"] if phase == "generate"
                    else cell["judge_dirs"]["qwen3_14b"])
        return complete_results(path, expected)

    def launch(name, argv, phase, cell, gpu=None):
        path = logs / (name + ".log")
        with path.open("a") as log:
            log.write("\nSTART " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            log.flush()
            child = subprocess.Popen(argv, cwd=project, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "PYTHONUNBUFFERED": "1"})
        children[name] = child
        state["jobs"][name] = {"status": "running", "pid": child.pid, "argv": argv,
            "phase": phase, "backbone": cell["backbone"], "method": cell["method"],
            "gpu": gpu, "log": str(path), "started_at": time.time()}
        write_json(status_path, state)
        print("STARTED", name, child.pid, "world_model_gpu", gpu, flush=True)

    while True:
        with local_opener.open(cfg["shared_local_endpoint"] + "/health", timeout=10) as response:
            health = json.load(response)
        if health.get("status") != "ready" or health.get("precision") != "bf16":
            raise RuntimeError("Shared BF16 inference service is not ready")
        active_generation, active_judges = {}, []
        for cell in cells:
            genname = generation_name(cell)
            for phase, name in (("generate", genname), ("judge", "qwen3_14b-" + genname)):
                prior = state["jobs"].get(name, {})
                if prior.get("status") == "completed":
                    continue
                if prior.get("status") == "running":
                    child = children.get(name)
                    return_code = child.poll() if child else None
                    actual = process_argv(prior["pid"])
                    if actual:
                        if actual != prior["argv"]:
                            raise RuntimeError("Recorded PID identity changed: " + name)
                        if phase == "generate":
                            active_generation[prior["gpu"]] = prior
                        else:
                            active_judges.append(prior)
                        continue
                    if done(cell, phase):
                        prior.update(status="completed", finished_at=time.time(), return_code=return_code)
                    elif child is None:
                        prior.update(status="pending", interrupted_at=time.time())
                    else:
                        prior.update(status="failed", finished_at=time.time(), return_code=return_code)
                        print("FAILED", name, "see", prior["log"], flush=True)
                elif not prior and done(cell, phase):
                    state["jobs"][name] = {"status": "completed", "verified_at": time.time()}
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits"], text=True)
        free = {int(i): int(m) for i, m in (line.split(",") for line in raw.splitlines())}
        for cell in cells:
            name = generation_name(cell)
            if (state["jobs"].get(name, {}).get("status") in {"running", "completed", "failed"}
                    or not available_checkpoint(cell)
                    or any(j["backbone"] == cell["backbone"] for j in active_generation.values())):
                continue
            preferences = {"qwen25": [0], "llama31": [1], "gemma2": [0, 1]}[cell["backbone"]]
            gpu = next((g for g in preferences if g not in active_generation
                        and free[g] >= weight_memory_mib(cell["checkpoint"])), None)
            if gpu is None:
                continue
            models = {role: spec["path"] for role, spec in cfg["roles"].items() if "path" in spec}
            models["world_model"] = str((project / cell["checkpoint"]).resolve())
            manifest = control / "cell_manifests" / (name + ".json")
            if str(manifest) not in verified_manifests:
                if not manifest.exists():
                    command = [cfg["python"], "make_model_manifest.py", "--output", str(manifest)]
                    for role, path in models.items():
                        command += ["--model", role + "=" + path]
                    subprocess.run(command, cwd=project, check=True)
                verify_manifest(manifest, models.values())
                verified_manifests.add(str(manifest))
            argv = jobs[name]["argv"] + ["--model-manifest", str(manifest),
                                         "--role-device", f"world_model=cuda:{gpu}"]
            launch(name, argv, "generate", cell, gpu)
            active_generation[gpu] = state["jobs"][name]
        if not active_judges:
            for cell in cells:
                genname = generation_name(cell)
                name = "qwen3_14b-" + genname
                if (state["jobs"].get(genname, {}).get("status") == "completed"
                        and state["jobs"].get(name, {}).get("status") not in {"completed", "failed"}):
                    launch(name, jobs[name]["argv"], "judge", cell)
                    active_judges.append(state["jobs"][name])
                    break
        completed = sum(j.get("status") == "completed" for j in state["jobs"].values())
        failures = [name for name, j in state["jobs"].items() if j.get("status") == "failed"]
        state.update(status="running" if active_generation or active_judges else "waiting_for_checkpoints",
                     last_checked_at=time.time(), failed_jobs=failures,
                     expected_dialogues=plan["counts"]["planned_episodes"], completed_jobs=completed)
        write_json(status_path, state)
        if completed == 2 * len(cells):
            for phase in ("freeze", "aggregate"):
                command = next(j["argv"] for j in plan["jobs"] if j["phase"] == phase)
                subprocess.run(command, cwd=project, check=True)
            table = output / "tables/main_tables.json"
            if json.loads(table.read_text())["status"] != "complete":
                raise RuntimeError("Final metric table is incomplete")
            state.update(status="completed", finished_at=time.time())
            write_json(status_path, state)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
