"""Run the repaired MA2P generation and both judges exclusively on GPU 4."""
import copy
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/app/hzj/liujianjian/ljj-2-paper")
CODE = ROOT / "my_eclp_final/seven_methods_192"
SOURCE = ROOT / "exp_code_ljj/artifacts/seven_methods_192_20261006/generators/ma2p_repaired"
WORK = ROOT / "exp_code_ljj/artifacts/ma2p_repaired_gpu4_192_20261007"
TABLE = ROOT / "exp_code_ljj/results/main_experiment_table.md"
METHOD = "ma2p_repaired"
JUDGES = ("selene8b", "qwen3_14b")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def set_status(stage, state, **extra):
    current = read_json(WORK / "task_status.json") if (WORK / "task_status.json").exists() else {}
    current.update(stage=stage, state=state, updated_unix=time.time(), **extra)
    write_json(WORK / "task_status.json", current)


def prepare_contract():
    import sys

    sys.path.insert(0, str(ROOT / "my_eclp_final"))
    from experiment_io import object_hash

    WORK.mkdir(parents=True, exist_ok=True)
    source_contract_path = SOURCE / "generation_run.json"
    contract = copy.deepcopy(read_json(source_contract_path))
    contract["physical_gpu"] = 4
    contract_path = WORK / "generation_run.json"
    if contract_path.exists():
        if read_json(contract_path) != contract:
            raise ValueError("Existing GPU4 MA2P contract differs from the frozen source settings")
    else:
        write_json(contract_path, contract)
    write_json(WORK / "run_provenance.json", {
        "task": "repaired MA2P, fresh fixed-192 generation and dual-Judge evaluation",
        "physical_gpu": 4,
        "method": METHOD,
        "source_contract": str(source_contract_path),
        "source_contract_sha256": digest(source_contract_path),
        "new_contract_sha256": digest(contract_path),
        "new_run_hash": object_hash(contract),
        "generation_code": str(CODE / "generate_ma2p_recovery.py"),
        "generation_core": str(CODE / "generate_ma2p_core.py"),
        "judge_order": list(JUDGES),
    })


def run_stage(stage, command, env, log_path):
    status = read_json(WORK / "task_status.json") if (WORK / "task_status.json").exists() else {}
    if status.get("stage") == stage and status.get("state") == "complete":
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    set_status(stage, "running", command=command, log=str(log_path), physical_gpu=4)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        set_status(stage, "running", command=command, log=str(log_path), physical_gpu=4,
                   pid=process.pid, started_unix=time.time())
        result = process.wait()
    if result:
        set_status(stage, "failed", exit_code=result, pid=process.pid)
        raise RuntimeError(f"{stage} exited with status {result}; see {log_path}")
    set_status(stage, "complete", exit_code=0, pid=process.pid, finished_unix=time.time())


def wait_for_gpu_admission():
    admission_log = WORK / "logs/gpu4_admission.log"
    admission_log.parent.mkdir(parents=True, exist_ok=True)
    threshold_mib = 48 * 1024
    stable_samples = 0
    while stable_samples < 3:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        free_mib = next(int(line.split(",")[1]) for line in result.stdout.splitlines()
                        if int(line.split(",")[0]) == 4)
        stable_samples = stable_samples + 1 if free_mib >= threshold_mib else 0
        set_status("admission", "waiting", physical_gpu=4, free_mib=free_mib,
                   required_stable_mib=threshold_mib, stable_samples=stable_samples,
                   stable_samples_required=3)
        with admission_log.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"time": time.time(), "gpu": 4, "free_mib": free_mib,
                                  "threshold_mib": threshold_mib,
                                  "stable_samples": stable_samples}) + "\n")
        if stable_samples < 3:
            time.sleep(15)


def run_generation_with_retry(env):
    command = [sys.executable, "-u", str(CODE / "generate_ma2p_recovery.py"),
               "--repo-root", str(ROOT), "--output", str(WORK), "--physical-gpu", "4",
               "--method", METHOD, "--limit", "192"]
    log_path = WORK / "logs/generation_gpu4.log"
    for attempt in range(1, 4):
        wait_for_gpu_admission()
        try:
            run_stage("generation", command, env, log_path)
            return
        except RuntimeError:
            with log_path.open("rb") as log:
                log.seek(max(0, log.seek(0, os.SEEK_END) - 65536))
                recent = log.read().decode("utf-8", errors="replace").lower()
            if "out of memory" not in recent or attempt == 3:
                raise
            set_status("generation", "retry_wait", physical_gpu=4,
                       failed_attempt=attempt, retry_limit=3, reason="CUDA out of memory")
            time.sleep(30)


def validate_generation():
    import sys

    sys.path.insert(0, str(ROOT / "my_eclp_final"))
    from experiment_io import object_hash

    contract = read_json(WORK / "generation_run.json")
    summary_path = WORK / "generation/ma2p_repaired_summary.json"
    generation_path = WORK / "generation/ma2p_repaired.jsonl"
    summary = read_json(summary_path)
    rows = rows_jsonl(generation_path)
    expected = set(contract["expected_indexes"])
    recovery = read_json(WORK / "generation_recovery_revision.json")
    assert contract["physical_gpu"] == 4
    assert summary["complete"] is True and summary["n"] == 192
    assert summary["generation_sha256"] == digest(generation_path)
    assert summary["run_hash"] == object_hash(contract)
    assert len(rows) == 192 and {row["index"] for row in rows} == expected
    for row in rows:
        assert row["method"] == METHOD and row["initial_history"] == [] and row["rounds"] == 5
        assert len(row["trajectory"]) == 10 and row["run_hash"] == object_hash(contract)
        assert row["runtime_physical_gpu"] == 4
        assert row["recovery_core_sha256"] == recovery["recovery_core_sha256"]
        for turn in row["trajectory"]:
            if turn["role"] == "persuader":
                assert turn["generation_recovery_revision"] == "ma2p_ascii_json_recovery_v1"
                assert turn["generation_recovery_source_sha256"] == recovery["recovery_source_sha256"]
    return contract, generation_path, summary


def validate_evaluation(judge, generation_path):
    summary_path = WORK / "judges" / judge / "evaluation" / METHOD / "summary.json"
    launch_path = WORK / "judges" / judge / "evaluation" / METHOD / "execution_launches.jsonl"
    contract_path = WORK / "judges" / judge / "evaluation_run.json"
    summary = read_json(summary_path)
    contract = read_json(contract_path)
    launches = rows_jsonl(launch_path)
    assert summary["complete"] is True and summary["n"] == 192
    assert summary["source_generation_sha256"] == digest(generation_path)
    assert contract["physical_gpu"] == 4
    assert launches and all(row["runtime_physical_gpu"] == 4 for row in launches)
    return summary


def display_metric(summary, key, count_key, digits=2):
    value = summary[key]
    count = summary[count_key]
    if value is None or not count:
        return "Pending"
    return f"{value:.{digits}f} ({count}/192)"


def update_table(summaries):
    if not TABLE.exists():
        raise FileNotFoundError(TABLE)
    if not (WORK / "main_experiment_table_before_gpu4_update.md").exists():
        shutil.copy2(TABLE, WORK / "main_experiment_table_before_gpu4_update.md")
    lines = TABLE.read_text(encoding="utf-8").splitlines()
    sections = {
        "selene8b": "## Selene-1-Mini-Llama-3.1-8B",
        "qwen3_14b": "## Qwen3-14B",
    }
    for judge, heading in sections.items():
        if heading not in lines:
            raise ValueError(f"Table is missing section: {heading}")
        start = lines.index(heading) + 1
        end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
        matches = [i for i in range(start, end) if lines[i].startswith("| MA2P |")]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one MA2P row under {heading}, got {len(matches)}")
        summary = summaries[judge]
        columns = [
            "MA2P",
            f"{summary['n']}/192",
            display_metric(summary, "success_percent", "success_evaluable_n"),
            display_metric(summary, "win1_percent", "win1_evaluable_n"),
            display_metric(summary, "win_full_percent", "win_full_evaluable_n"),
            display_metric(summary, "mean_rounds_fail5", "success_evaluable_n", 3),
            str(summary["format_unresolved"]),
        ]
        lines[matches[0]] = "| " + " | ".join(columns) + " |"
    temporary = TABLE.with_suffix(TABLE.suffix + ".gpu4.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(TABLE)


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise RuntimeError("Controller and every model worker must be pinned to physical GPU 4")
    lock_path = WORK / ".gpu4_controller.lock"
    WORK.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_contract()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="4", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
        generation = WORK / "generation/ma2p_repaired.jsonl"
        generation_summary = WORK / "generation/ma2p_repaired_summary.json"
        if not generation_summary.exists() or not read_json(generation_summary).get("complete"):
            run_generation_with_retry(env)
        contract, generation, gen_summary = validate_generation()
        summaries = {}
        for judge in JUDGES:
            summary_path = WORK / "judges" / judge / "evaluation" / METHOD / "summary.json"
            complete = False
            if summary_path.exists():
                try:
                    summaries[judge] = validate_evaluation(judge, generation)
                    complete = True
                except (AssertionError, KeyError, ValueError, FileNotFoundError):
                    complete = False
            if not complete:
                args = [sys.executable, "-u", str(CODE / "evaluate.py"), "--repo-root", str(ROOT),
                        "--output", str(WORK), "--physical-gpu", "4", "--method", METHOD,
                        "--judge", judge, "--limit", "192"]
                if judge == "selene8b":
                    args += ["--batch-size", "4", "--memory-cap-gib", "22"]
                else:
                    args += ["--batch-size", "2", "--memory-cap-gib", "36"]
                run_stage(f"evaluation_{judge}", args, env, WORK / f"logs/evaluation_{judge}_gpu4.log")
                summaries[judge] = validate_evaluation(judge, generation)
        update_table(summaries)
        audit = {
            "state": "complete",
            "physical_gpu": 4,
            "method": METHOD,
            "generation_n": gen_summary["n"],
            "generation_sha256": digest(generation),
            "generation_run_hash": gen_summary["run_hash"],
            "generation_recovery_revision": read_json(WORK / "generation_recovery_revision.json"),
            "judges": summaries,
            "table": str(TABLE),
            "table_sha256": digest(TABLE),
            "finished_unix": time.time(),
        }
        write_json(WORK / "gpu4_completion_audit.json", audit)
        set_status("complete", "complete", physical_gpu=4,
                   audit=str(WORK / "gpu4_completion_audit.json"))


if __name__ == "__main__":
    main()
