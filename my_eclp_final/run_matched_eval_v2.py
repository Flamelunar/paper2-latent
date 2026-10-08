#!/usr/bin/env python3
"""Run the unified evaluation for six methods and two judges on GPUs 0, 1, and 4."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiment_io import atomic_json, read_jsonl, writer_lock
from evaluate_matched_v2 import METHOD_LABELS
from update_matched_table_v2 import update_table


def free_gib(gpu: int) -> float:
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return float(output.strip()) / 1024


def render(root: Path, work: Path) -> None:
    table = root / "exp_code_ljj/results/main_experiment_table.md"
    update_table(table, work)
    (work / "results.md").write_bytes(table.read_bytes())


def validate_generations(root: Path) -> None:
    folder = root / "exp_code_ljj/artifacts/generation_5rounds"
    for method, filename in (
        ("baseline1_zero_shot", "baseline1_zero_shot.jsonl"),
        ("baseline2_dialogue_sft", "baseline2_dialogue_sft.jsonl"),
        ("baseline3_isr_simdpo", "baseline3_isr_simdpo.jsonl"),
        ("baseline4_ma2p", "baseline4_ma2p.jsonl"),
        ("baseline5_latentratt", "baseline5_latentRATT.json"),
    ):
        rows = read_jsonl(folder / filename)
        if len(rows) != 1000 or {row.get("index") for row in rows} != set(range(1000)):
            raise ValueError(f"{method} must contain the fixed 1,000 indexed test scenarios")
    our_rows = read_jsonl(root / "eclp_runs_gpu4/test_eclp_offline_lpo/generation.jsonl")
    if len(our_rows) != 1000 or {row.get("index") for row in our_rows} != set(range(1000)):
        raise ValueError("Our method must contain the fixed 1,000 indexed test scenarios")


def build_queues(gpus: list[int]) -> dict[int, list[tuple[str, str]]]:
    if len(gpus) < 2 or len(set(gpus)) != len(gpus):
        raise ValueError("At least two distinct GPUs are required for both Judge models")
    memory = {gpu: free_gib(gpu) for gpu in gpus}
    qwen_gpu = 1 if 1 in gpus else max(gpus, key=memory.get)
    selene_gpus = [gpu for gpu in gpus if gpu != qwen_gpu]
    if not selene_gpus:
        selene_gpus = [qwen_gpu]
    queues = {gpu: [] for gpu in gpus}
    methods = list(METHOD_LABELS)
    queues[qwen_gpu] = [("qwen3_14b", method) for method in methods]
    if len(selene_gpus) == 1:
        queues[selene_gpus[0]] = [("selene8b", method) for method in methods]
    else:
        selene_gpus.sort(key=lambda gpu: memory[gpu], reverse=True)
        split = (len(methods) + 1) // 2
        queues[selene_gpus[0]] = [("selene8b", method) for method in methods[:split]]
        queues[selene_gpus[1]] = [("selene8b", method) for method in methods[split:]]
    return queues


def run(args) -> None:
    root = args.repo_root.resolve()
    work = root / "eclp_runs_gpu4/evaluation_v2"
    scripts = Path(__file__).resolve().parent
    method_inputs = {
        "baseline1_zero_shot": root / "exp_code_ljj/artifacts/generation_5rounds/baseline1_zero_shot.jsonl",
        "baseline2_dialogue_sft": root / "exp_code_ljj/artifacts/generation_5rounds/baseline2_dialogue_sft.jsonl",
        "baseline3_isr_simdpo": root / "exp_code_ljj/artifacts/generation_5rounds/baseline3_isr_simdpo.jsonl",
        "baseline4_ma2p": root / "exp_code_ljj/artifacts/generation_5rounds/baseline4_ma2p.jsonl",
        "baseline5_latentratt": root / "exp_code_ljj/artifacts/generation_5rounds/baseline5_latentRATT.json",
        "our_eclp_offline_lpo": root / "eclp_runs_gpu4/test_eclp_offline_lpo",
    }
    work.mkdir(parents=True, exist_ok=True)
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    validate_generations(root)
    queues = build_queues(args.gpus)
    total = sum(map(len, queues.values()))
    completed = 0
    failures: list[dict] = []
    running: dict[int, dict] = {}
    failed_gpus: set[int] = set()
    status_path = work / "status.json"

    with writer_lock(work / ".runner.lock"):
        while completed + len(failures) < total:
            for gpu in args.gpus:
                current = running.get(gpu)
                if current is not None:
                    code = current["process"].poll()
                    if code is not None:
                        current["log"].close()
                        judge, method = current["judge"], current["method"]
                        if code:
                            failure = {"gpu": gpu, "judge": judge, "method": method,
                                       "exit_code": code, "log": str(current["log_path"])}
                            failures.append(failure)
                            failed_gpus.add(gpu)
                            for skipped_judge, skipped_method in queues[gpu]:
                                failures.append({"gpu": gpu, "judge": skipped_judge,
                                                 "method": skipped_method,
                                                 "reason": "not_started_after_worker_failure"})
                            queues[gpu].clear()
                        else:
                            completed += 1
                            print(f"Completed {judge}/{method} on GPU {gpu} ({completed}/{total})", flush=True)
                            render(root, work)
                        del running[gpu]

                if gpu in running or gpu in failed_gpus or not queues[gpu]:
                    continue
                judge, method = queues[gpu][0]
                minimum_free = 34.0 if judge == "qwen3_14b" else 22.0
                reserve = 4.0
                available = free_gib(gpu)
                if available < minimum_free:
                    continue
                cap = min(30.0 if judge == "qwen3_14b" else 22.0, available - reserve)
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
                output = work / judge / method
                log_path = logs / f"gpu{gpu}_{judge}_{method}.log"
                command = [
                    sys.executable, str(scripts / "evaluate_matched_v2.py"),
                    "--repo-root", str(root), "--generation", str(method_inputs[method]),
                    "--golden", str(work / "golden"), "--output", str(output),
                    "--judge", judge, "--method", method, "--physical-gpu", str(gpu),
                    "--batch-size", str(args.batch_size_qwen3 if judge == "qwen3_14b" else args.batch_size_selene),
                    "--memory-cap-gib", f"{cap:.2f}",
                ]
                handle = log_path.open("a", encoding="utf-8")
                try:
                    process = subprocess.Popen(command, cwd=root, env=environment,
                                               stdout=handle, stderr=subprocess.STDOUT)
                except Exception:
                    handle.close()
                    raise
                running[gpu] = {"process": process, "log": handle, "log_path": log_path,
                                "judge": judge, "method": method, "memory_cap_gib": cap,
                                "initial_free_gib": available}
                queues[gpu].pop(0)
                print(f"Started {judge}/{method} on GPU {gpu}; {available:.1f} GiB free, "
                      f"cap {cap:.1f} GiB", flush=True)

            waiting = {str(gpu): {"judge": queues[gpu][0][0], "method": queues[gpu][0][1],
                                  "free_gib": round(free_gib(gpu), 2)}
                       for gpu in args.gpus if gpu not in running and gpu not in failed_gpus and queues[gpu]}
            active = {str(gpu): {"pid": item["process"].pid, "judge": item["judge"],
                                 "method": item["method"], "memory_cap_gib": item["memory_cap_gib"],
                                 "log": str(item["log_path"])} for gpu, item in running.items()}
            atomic_json(status_path, {"state": "running" if completed + len(failures) < total else "finished",
                                      "completed_tasks": completed, "total_tasks": total,
                                      "active": active, "waiting_for_memory": waiting,
                                      "failures": failures, "updated_at": time.strftime("%FT%T%z")})
            if completed + len(failures) < total:
                time.sleep(max(2, min(args.poll_seconds, 60)))

        render(root, work)
        state = "failed" if failures else "complete"
        atomic_json(status_path, {"state": state, "completed_tasks": completed,
                                  "total_tasks": total, "failures": failures,
                                  "updated_at": time.strftime("%FT%T%z")})
    if failures:
        raise SystemExit(f"Some evaluation tasks failed: {json.dumps(failures, ensure_ascii=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 4])
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--batch-size-selene", type=int, default=4)
    parser.add_argument("--batch-size-qwen3", type=int, default=2)
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.batch_size_selene < 1 or args.batch_size_qwen3 < 1:
        parser.error("poll interval and batch sizes must be positive")
    run(args)


if __name__ == "__main__":
    main()
