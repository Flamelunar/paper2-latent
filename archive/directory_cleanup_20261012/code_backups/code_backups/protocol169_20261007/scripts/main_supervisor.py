#!/usr/bin/env python3
"""Continue the frozen Qwen-only main experiment after training completes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_main_experiment import build_plan, scoped_repeat_config
from evaluation.modern_pipeline import normalize_item
from utils.artifacts import hash_object, write_json
from utils.artifacts import last_records
from utils.evaluation_io import read_json_or_jsonl
from utils.model_manifest import verify_manifest


def gpu_snapshot() -> list[dict[str, int]]:
    command = ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
               "--format=csv,noheader,nounits"]
    output = subprocess.run(command, check=True, text=True, capture_output=True).stdout
    result = []
    for line in output.splitlines():
        index, free, utilization = (int(field.strip()) for field in line.split(","))
        result.append({"index": index, "free_mib": free, "utilization_percent": utilization})
    return result


GENERATION_MIN_FREE_MIB = 70 * 1024
THREE_CARD_PAIR_MIN_FREE_MIB = 48 * 1024
THREE_CARD_JUDGE_MIN_FREE_MIB = 38 * 1024
THREE_CARD_WORLD_MIN_FREE_MIB = 28 * 1024
# The fixed Persuader and Persuadee use about 36 GiB; the largest World Model
# adds about 18 GiB. Reserve 60 GiB before loading all three in BF16.
COLOCATED_PAIR_WORLD_MIN_FREE_MIB = 60 * 1024


def generation_placement(snapshot: list[dict[str, int]], method: str,
                         world_min_free_mib: int = THREE_CARD_WORLD_MIN_FREE_MIB) -> tuple[list[int], list[str]] | None:
    """Choose a reproducible placement, preferring GPU World Model inference.

    The fixed Persuader and Persuadee share one physical card, Qwen3-14B
    scoring shares a second card, and the World Model uses a third. If a
    valid two-card or three-card layout is unavailable, wait for capacity.
    """
    by_free = sorted(snapshot, key=lambda gpu: (gpu["index"] != 4, -gpu["free_mib"], gpu["index"]))
    # Prefer a two-card layout when the preferred card has enough room for the
    # fixed Persuader/Persuadee pair plus the World Model. This is faster on
    # shared machines where another job is saturating the usual World Model
    # card. CUDA_VISIBLE_DEVICES keeps cuda:0 mapped to the co-located card
    # and cuda:1 mapped to the judge card.
    for pair in by_free:
        if pair["free_mib"] < COLOCATED_PAIR_WORLD_MIN_FREE_MIB:
            continue
        for judge in by_free:
            if judge["index"] == pair["index"] or judge["free_mib"] < THREE_CARD_JUDGE_MIN_FREE_MIB:
                continue
            cards = [pair["index"], judge["index"]]
            flags = ["--role-device", "persuader=cuda:0", "--role-device", "persuadee=cuda:0",
                     "--role-device", "world_model=cuda:0", "--role-device", "progress_scorer=cuda:1",
                     "--role-device", "online_judge=cuda:1"]
            return cards, flags
    for pair in by_free:
        if pair["free_mib"] < THREE_CARD_PAIR_MIN_FREE_MIB:
            continue
        for judge in by_free:
            if judge["index"] == pair["index"] or judge["free_mib"] < THREE_CARD_JUDGE_MIN_FREE_MIB:
                continue
            for world in by_free:
                if world["index"] in {pair["index"], judge["index"]} or world["free_mib"] < world_min_free_mib:
                    continue
                cards = [pair["index"], world["index"], judge["index"]]
                flags = ["--role-device", "persuader=cuda:0", "--role-device", "persuadee=cuda:0",
                         "--role-device", "world_model=cuda:1", "--role-device", "progress_scorer=cuda:2",
                         "--role-device", "online_judge=cuda:2"]
                return cards, flags
    # Keep every role on BF16-capable accelerator placements. A single-card
    # fallback can collide the fixed models and both Qwen3 evaluation roles.
    return None


def judge_placement(snapshot: list[dict[str, int]]) -> list[int] | None:
    candidates = [gpu for gpu in snapshot if gpu["free_mib"] >= 38 * 1024
                  and gpu["utilization_percent"] <= 100]
    candidates.sort(key=lambda gpu: (gpu["index"] != 4, -gpu["free_mib"], gpu["index"]))
    return [candidates[0]["index"]] if candidates else None


def generation_placement_valid(snapshot: list[dict[str, int]], cards: list[int], method: str,
                               world_min_free_mib: int = THREE_CARD_WORLD_MIN_FREE_MIB) -> bool:
    """Recheck the selected physical cards immediately before model startup."""
    by_index = {gpu["index"]: gpu for gpu in snapshot}
    if not cards or len(set(cards)) != len(cards) or any(card not in by_index for card in cards):
        return False
    if len(cards) == 2:
        pair, judge = (by_index[cards[0]], by_index[cards[1]])
        return (pair["free_mib"] >= COLOCATED_PAIR_WORLD_MIN_FREE_MIB
                and judge["free_mib"] >= THREE_CARD_JUDGE_MIN_FREE_MIB)
    if len(cards) == 3:
        pair, world, judge = (by_index[cards[0]], by_index[cards[1]], by_index[cards[2]])
        return (pair["free_mib"] >= THREE_CARD_PAIR_MIN_FREE_MIB
                and world["free_mib"] >= world_min_free_mib
                and judge["free_mib"] >= THREE_CARD_JUDGE_MIN_FREE_MIB)
    if len(cards) == 1:
        return False
    return False


def judge_placement_valid(snapshot: list[dict[str, int]], cards: list[int]) -> bool:
    if len(cards) != 1:
        return False
    gpu = next((item for item in snapshot if item["index"] == cards[0]), None)
    return gpu is not None and gpu["free_mib"] >= 38 * 1024 and gpu["utilization_percent"] <= 100


def complete_results(directory: Path, expected: int) -> bool:
    summary, records = directory / "summary.json", directory / "results.jsonl"
    if not summary.is_file() or not records.is_file():
        return False
    data = json.loads(summary.read_text(encoding="utf-8"))
    if not (all(data.get(key) == expected for key in ("expected", "recorded", "known"))
            and data.get("failed_or_missing") == 0):
        return False
    rows = last_records(read_json_or_jsonl(records))
    return len(rows) == expected and all(row.get("status") == "completed"
                                         and type(row.get("metrics", {}).get("strict_success")) is bool for row in rows)


def replace_option(argv: list[str], option: str, value: str) -> list[str]:
    result = list(argv)
    result[result.index(option) + 1] = value
    return result


def process_argv(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return None
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def early_ready_generation_jobs(jobs: list[dict[str, Any]], sft_ready: Any) -> list[dict[str, Any]]:
    ready = []
    for job in jobs:
        backbone, suffix = job["name"].split("-", 1)
        method = suffix.rsplit("-r", 1)[0]
        if method in {"zero_shot", "ma2p"} or (method == "sft" and sft_ready(backbone)):
            ready.append(job)
    return ready


def validation_smoke_source(cfg: dict[str, Any], project: Path, output: Path) -> Path:
    source = project / cfg["data"]["revision_dir"] / "grpo_validation.jsonl"
    first = read_json_or_jsonl(source)[0]["state"]
    evaluation_ids = {normalize_item(row, i)[0]["id"]
                      for i, row in enumerate(read_json_or_jsonl(project / cfg["data"]["eval"]))}
    if first["id"] in evaluation_ids:
        raise ValueError("Validation smoke scenario overlaps the final evaluation set")
    destination = output / "supervisor" / "validation_smoke_1.json"
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != [first]:
            raise ValueError("Frozen validation smoke source changed")
    else:
        write_json(destination, [first])
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/main_experiment.json")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--repeat-ids", default="0",
                        help="Comma-separated frozen rollout IDs to evaluate (default: 0)")
    parser.add_argument("--plan-path", type=Path, default=None,
                        help="Optional path for the evaluation-only plan")
    parser.add_argument("--ready-baselines", action="store_true",
                        help="Evaluate base Zero-shot/MA2P and verified SFT cells while GRPO training continues")
    args = parser.parse_args()
    if args.poll_seconds < 5:
        raise ValueError("poll-seconds must be at least 5")
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if set(cfg["judges"]) != {"qwen3_14b"} or cfg["judges"]["qwen3_14b"]["provider"] != "local":
        raise ValueError("This supervisor is restricted to the frozen local Qwen3-14B judge")
    project = Path(cfg["project_root"])
    output = Path(cfg["outputs_root"])
    control = output / "supervisor"
    logs = control / "logs" / "main"
    logs.mkdir(parents=True, exist_ok=True)
    lock = (control / "main.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    config_hash = hash_object(cfg)
    status_path = control / "main_status.json"
    if status_path.exists():
        state = json.loads(status_path.read_text(encoding="utf-8"))
        if state.get("config_hash") != config_hash:
            raise ValueError("Main experiment configuration changed; choose another result directory")
    else:
        state = {"config_hash": config_hash, "status": "waiting_for_training",
                 "started_at": time.time(), "jobs": {}}
        write_json(status_path, state)

    try:
        repeat_ids = [int(part.strip()) for part in args.repeat_ids.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--repeat-ids must be comma-separated integers") from exc
    plan_cfg = scoped_repeat_config(cfg, repeat_ids)
    repeat_tag = "_".join(str(value) for value in repeat_ids)
    plan_path = args.plan_path or Path(f"artifacts/main_strategy_plan_eval_repeat_{repeat_tag}.json")
    if not plan_path.is_absolute():
        plan_path = project / plan_path
    plan_cfg["plan_path"] = str(plan_path)
    plan = build_plan(plan_cfg, project / "configs" / args.config.name, plan_path)
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        if old.get("config_hash") != plan["config_hash"]:
            raise ValueError("Existing main plan has a different evaluation configuration")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(plan_path, plan)
    state.update(evaluation_repeat_ids=repeat_ids,
                 expected_generated_dialogues=plan["counts"]["planned_episodes"],
                 planned_generation_jobs=plan["counts"]["generation_jobs"])
    write_json(status_path, state)

    train_status_path = control / "train_status.json"
    if not args.ready_baselines:
        while True:
            training = json.loads(train_status_path.read_text(encoding="utf-8")) if train_status_path.exists() else {}
            if training.get("config_hash") != config_hash:
                raise ValueError("Training supervisor has a different configuration")
            if training.get("status") == "completed":
                expected_training = len([j for j in plan["jobs"] if j["phase"] == "train"])
                if len([j for j in training["jobs"].values() if j.get("status") == "completed"]) != expected_training:
                    raise ValueError(f"Training is marked complete but not all {expected_training} jobs finished")
                break
            if training.get("status") == "failed":
                state.update(status="blocked_by_training", last_checked_at=time.time())
                write_json(status_path, state)
                raise RuntimeError("Training failed; main experiment will not start")
            state.update(status="waiting_for_training", last_checked_at=time.time(),
                         training_status=training.get("status"))
            write_json(status_path, state)
            time.sleep(args.poll_seconds)

    model_paths = [entry.split("=", 1)[1] for entry in next(j for j in plan["jobs"] if j["phase"] == "freeze")["argv"]
                   if "=" in entry and not entry.startswith("--")]
    manifest = project / cfg["model_manifest"]

    def manifest_complete() -> bool:
        if not manifest.is_file():
            return False
        verify_manifest(manifest, model_paths)
        return True

    def run_job(name: str, argv: list[str], done: Any, *, phase: str,
                method: str | None = None, needs_gpu: bool = True,
                yield_when_training_finishes: bool = False) -> bool:
        prior = state["jobs"].get(name, {})
        if done():
            state["jobs"][name] = {**prior, "status": "completed", "verified_at": time.time()}
            write_json(status_path, state)
            return True
        if prior.get("status") == "running":
            pid = prior.get("pid")
            expected_argv = prior.get("argv")
            if not isinstance(pid, int) or not isinstance(expected_argv, list):
                raise RuntimeError(f"Cannot safely reattach to prior {phase} job: {name}")
            actual_argv = process_argv(pid)
            if actual_argv is None:
                state["jobs"][name].update(status="interrupted", interrupted_pid=pid,
                                           interrupted_at=time.time())
                state.update(status="recovering", waiting_job=None, last_checked_at=time.time())
                write_json(status_path, state)
                prior = {}
            elif actual_argv != expected_argv:
                raise RuntimeError(f"PID {pid} no longer matches prior {phase} job: {name}")
            else:
                state.update(status="reattaching", waiting_job=name, last_checked_at=time.time())
                write_json(status_path, state)
                while True:
                    actual_argv = process_argv(pid)
                    if actual_argv is None:
                        break
                    if actual_argv != expected_argv:
                        raise RuntimeError(f"PID {pid} was reused while awaiting prior {phase} job: {name}")
                    time.sleep(min(args.poll_seconds, 30))
                if not done():
                    state["jobs"][name]["status"] = "interrupted"
                    state.update(status="recovering", waiting_job=None, last_checked_at=time.time())
                    write_json(status_path, state)
                    prior = {}
                else:
                    state["jobs"][name].update(status="completed", verified_at=time.time())
                    state.update(status="running", waiting_job=None)
                    write_json(status_path, state)
                    return True
        if prior.get("status") == "failed":
            raise RuntimeError(f"Inspect prior {phase} job before resuming: {name}")
        cards: list[int] = []
        flags: list[str] = []
        if needs_gpu:
            while True:
                if yield_when_training_finishes:
                    training = json.loads(train_status_path.read_text(encoding="utf-8")) if train_status_path.exists() else {}
                    if training.get("status") in {"completed", "failed"}:
                        state.update(status="early_ready_yielded", waiting_job=None,
                                     training_status=training.get("status"), last_checked_at=time.time())
                        write_json(status_path, state)
                        return False
                snapshot = gpu_snapshot()
                backbone = None
                if phase == "generate" and "--backbone" in argv:
                    backbone = argv[argv.index("--backbone") + 1]
                world_min_free_mib = THREE_CARD_WORLD_MIN_FREE_MIB
                placement = (generation_placement(snapshot, method, world_min_free_mib)
                             if phase == "generate" else judge_placement(snapshot))
                if placement:
                    if phase == "generate":
                        candidate_cards, candidate_flags = placement
                        # A competing process can resume between the first
                        # nvidia-smi query and Popen. Confirm the same cards
                        # after a short settling interval before loading any
                        # model, otherwise a race can turn into an OOM.
                        time.sleep(min(5, max(1, args.poll_seconds // 12)))
                        confirmed = gpu_snapshot()
                        if not generation_placement_valid(confirmed, candidate_cards, method, world_min_free_mib):
                            state.update(status="waiting_for_gpu", waiting_job=name,
                                         last_gpu_snapshot=confirmed, last_checked_at=time.time())
                            write_json(status_path, state)
                            continue
                        cards, flags = candidate_cards, candidate_flags
                    else:
                        time.sleep(min(5, max(1, args.poll_seconds // 12)))
                        if not judge_placement_valid(gpu_snapshot(), placement):
                            continue
                        cards = placement
                    break
                state.update(status="waiting_for_gpu", waiting_job=name,
                             last_gpu_snapshot=snapshot, last_checked_at=time.time())
                write_json(status_path, state)
                time.sleep(args.poll_seconds)
        environment = dict(os.environ)
        if cards:
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(card) for card in cards)
        environment["PYTHONUNBUFFERED"] = "1"
        command = argv + flags
        log = logs / f"{name}.log"
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} GPU={cards} ===\n")
            handle.flush()
            child = subprocess.Popen(command, cwd=project, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            state["jobs"][name] = {"status": "running", "phase": phase, "pid": child.pid,
                                   "gpu_indices": cards, "argv": command, "log": str(log),
                                   "started_at": time.time()}
            state.update(status="running", waiting_job=None)
            write_json(status_path, state)
            return_code = child.wait()
        state["jobs"][name].update(return_code=return_code, finished_at=time.time())
        if return_code != 0 or not done():
            state["jobs"][name]["status"] = "failed"
            state["status"] = "failed"
            write_json(status_path, state)
            raise RuntimeError(f"Main experiment job failed: {name}; see {log}")
        state["jobs"][name]["status"] = "completed"
        write_json(status_path, state)
        return True

    if args.ready_baselines:
        train_manifest = json.loads((project / cfg["data"]["revision_dir"] / "manifest.json").read_text(encoding="utf-8"))
        expected_train_hash = train_manifest["files"]["structured_strategy_train.jsonl"]["sha256"]

        def sft_ready(backbone: str) -> bool:
            merged = project / cfg["runs_root"] / backbone / "structured_strategy_merged"
            contract_path = merged / "training_contract.json"
            if not (contract_path.is_file() and (merged / "config.json").is_file()
                    and (merged / "tokenizer_config.json").is_file()
                    and any(merged.glob("*.safetensors"))):
                return False
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            return (contract.get("task") == "structured_strategy"
                    and contract.get("train_file_sha256") == expected_train_hash)

        base_early_models = {
            role: spec["path"] for role, spec in cfg["roles"].items()
            if role in {"persuader", "persuadee", "online_judge", "progress_scorer"}
        }
        base_early_models.update({f"{backbone}_base": path for backbone, path in cfg["backbones"].items()})
        generate_jobs = [job for job in plan["jobs"] if job["phase"] == "generate"]
        judge_jobs = {job["name"]: job for job in plan["jobs"] if job["phase"] == "judge"}

        while True:
            training = json.loads(train_status_path.read_text(encoding="utf-8")) if train_status_path.exists() else {}
            if training.get("config_hash") != config_hash:
                raise ValueError("Training supervisor has a different configuration")
            if training.get("status") == "failed":
                state.update(status="early_ready_yielded", waiting_job=None,
                             training_status="failed", last_checked_at=time.time())
                write_json(status_path, state)
                return
            if training.get("status") == "completed":
                state.update(status="early_ready_yielded", waiting_job=None,
                             training_status="completed", last_checked_at=time.time())
                write_json(status_path, state)
                return

            early_models = dict(base_early_models)
            for backbone in cfg["backbones"]:
                if sft_ready(backbone):
                    early_models[f"{backbone}_sft"] = str(
                        project / cfg["runs_root"] / backbone / "structured_strategy_merged")
            manifest_key = hash_object(early_models)[:12]
            early_manifest = control / "early_model_manifests" / f"{manifest_key}.json"
            if not early_manifest.is_file():
                state.update(status="freezing_early_ready_models", waiting_job=None,
                             last_checked_at=time.time())
                write_json(status_path, state)
                early_manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest_cmd = [cfg["python"], "make_model_manifest.py", "--output", str(early_manifest)]
                for role, path in early_models.items():
                    manifest_cmd += ["--model", f"{role}={path}"]
                with (logs / "early_model_manifest.log").open("a", encoding="utf-8") as handle:
                    result = subprocess.run(manifest_cmd, cwd=project, stdout=handle,
                                            stderr=subprocess.STDOUT, check=False)
                if result.returncode != 0:
                    state.update(status="failed", early_manifest_return_code=result.returncode,
                                 last_checked_at=time.time())
                    write_json(status_path, state)
                    raise RuntimeError(f"Early model manifest failed; see {logs / 'early_model_manifest.log'}")
                verify_manifest(early_manifest, early_models.values())
            else:
                verify_manifest(early_manifest, early_models.values())
            state["early_ready_model_manifest"] = str(early_manifest)

            selected = early_ready_generation_jobs(generate_jobs, sft_ready)
            for job in selected:
                name = job["name"]
                target = Path(job["argv"][job["argv"].index("--output-dir") + 1])
                method = name.rsplit("-r", 1)[0].split("-", 1)[1]
                expected_items = cfg["data"]["expected_eval_items"]
                command = list(job["argv"]) + ["--model-manifest", str(early_manifest)]
                if not run_job(name, command,
                               lambda target=target, expected_items=expected_items: complete_results(target, expected_items),
                               phase="generate", method=method, yield_when_training_finishes=True):
                    return
                judge_name = "qwen3_14b-" + name
                judge_job = judge_jobs[judge_name]
                judge_target = Path(judge_job["argv"][judge_job["argv"].index("--output-dir") + 1])
                if not run_job(judge_name, judge_job["argv"],
                               lambda target=judge_target, expected_items=expected_items: complete_results(target, expected_items),
                               phase="judge", yield_when_training_finishes=True):
                    return

            if all(sft_ready(backbone) for backbone in cfg["backbones"]):
                early_names = {job["name"] for job in generate_jobs
                               if job["name"].split("-", 1)[1].rsplit("-r", 1)[0] in {"zero_shot", "ma2p", "sft"}}
                if all(job_name in state["jobs"] and state["jobs"][job_name].get("status") == "completed"
                       for job_name in early_names):
                    state.update(status="early_ready_completed", waiting_job=None,
                                 last_checked_at=time.time(), early_ready_methods=sorted(early_names))
                    write_json(status_path, state)
                    return

            state.update(status="early_ready_running", waiting_job=None,
                         training_status=training.get("status"), last_checked_at=time.time())
            write_json(status_path, state)
            time.sleep(args.poll_seconds)
        
    freeze_job = next(j for j in plan["jobs"] if j["phase"] == "freeze")

    freeze_job = next(j for j in plan["jobs"] if j["phase"] == "freeze")
    run_job("freeze-model-identities", freeze_job["argv"], manifest_complete,
            phase="freeze", needs_gpu=False)
    smoke_source = validation_smoke_source(cfg, project, output)
    smoke_repeat = repeat_ids[0]
    smoke_names = {f"qwen25-{method}-r{smoke_repeat}"
                   for method in ("zero_shot", "sft", "ma2p", "grpo_only", "sft_grpo", "ours")}
    smoke_names.update({f"{backbone}-ours-r{smoke_repeat}" for backbone in ("llama31", "gemma2")})
    generate_jobs = [job for job in plan["jobs"] if job["phase"] == "generate"]
    for job in generate_jobs:
        if job["name"] not in smoke_names:
            continue
        name = "smoke-" + job["name"]
        target = output / "supervisor" / "smoke" / job["name"]
        command = replace_option(job["argv"], "--input", str(smoke_source))
        command = replace_option(command, "--output-dir", str(target)) + ["--expected-items", "1", "--max-rounds", "1"]
        method = job["name"].rsplit("-r", 1)[0].split("-", 1)[1]
        run_job(name, command, lambda target=target: complete_results(target, 1),
                phase="generate", method=method)

    for job in generate_jobs:
        name = job["name"]
        target = Path(job["argv"][job["argv"].index("--output-dir") + 1])
        method = name.rsplit("-r", 1)[0].split("-", 1)[1]
        expected_items = cfg["data"]["expected_eval_items"]
        run_job(
            name, job["argv"],
            lambda target=target, expected_items=expected_items: complete_results(target, expected_items),
            phase="generate", method=method)
        judge_job = next(j for j in plan["jobs"] if j["phase"] == "judge" and j["name"] == "qwen3_14b-" + name)
        judge_target = Path(judge_job["argv"][judge_job["argv"].index("--output-dir") + 1])
        run_job(
            judge_job["name"], judge_job["argv"],
            lambda target=judge_target, expected_items=expected_items: complete_results(target, expected_items),
            phase="judge")

    aggregate_job = next(j for j in plan["jobs"] if j["phase"] == "aggregate")
    table_dir = output / "tables"
    run_job("paired-main-table", aggregate_job["argv"],
            lambda: (table_dir / "main_tables.json").is_file()
            and json.loads((table_dir / "main_tables.json").read_text(encoding="utf-8")).get("status") == "complete"
            and (table_dir / "main_success_table.tex").is_file(),
            phase="aggregate", needs_gpu=False)
    state.update(status="completed", finished_at=time.time())
    write_json(status_path, state)


if __name__ == "__main__":
    main()
