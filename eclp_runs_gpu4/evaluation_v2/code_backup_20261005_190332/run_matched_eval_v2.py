#!/usr/bin/env python3
"""Run both judges on GPU4 with memory limits and persistent progress."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from experiment_io import atomic_json, read_json, writer_lock
from update_matched_table_v2 import update_table


def render(root: Path, work: Path) -> None:
    table = root / "exp_code_ljj/results/main_experiment_table.md"
    update_table(table, work)
    (work / "results.md").write_bytes(table.read_bytes())


def free_gib() -> float:
    output = subprocess.check_output(["nvidia-smi", "-i", "4", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    return float(output.strip()) / 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pilot", type=int, default=20)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--judges", nargs="+", choices=("selene8b", "qwen3_14b"), default=["selene8b", "qwen3_14b"])
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    work = root / "eclp_runs_gpu4/evaluation_v2"
    scripts = Path(__file__).resolve().parent
    work.mkdir(parents=True, exist_ok=True)
    status = work / "status.json"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="4", PYTHONUNBUFFERED="1")
    with writer_lock(work / ".runner.lock"):
        for judge in args.judges:
            summary_path = work / judge / "our_eclp_offline_lpo" / "summary.json"
            old = read_json(summary_path) if summary_path.exists() else {}
            if old.get("complete") or (not args.full and old.get("evaluated_count", 0) >= args.pilot):
                continue
            needed, cap = (40.0, 32.0) if judge == "qwen3_14b" else (26.0, 22.0)
            while True:
                available = free_gib()
                if available >= needed:
                    cap = min(cap, available - 8.0)
                    break
                atomic_json(status, {"state": "waiting_gpu_memory", "judge": judge, "physical_gpu": 4,
                                     "free_gib": available, "required_free_gib": needed,
                                     "mode": "full" if args.full else "pilot", "updated_at": time.strftime("%FT%T%z")})
                print(f"Waiting for GPU4: {judge} needs {needed:.1f} GiB free; {available:.1f} GiB free", flush=True)
                time.sleep(max(1, min(args.poll_seconds, 60)))
            atomic_json(status, {"state": "running", "judge": judge, "physical_gpu": 4,
                                 "mode": "full" if args.full else "pilot", "updated_at": time.strftime("%FT%T%z")})
            command = [sys.executable, str(scripts / "evaluate_matched_v2.py"), "--repo-root", str(root),
                       "--generation", str(root / "eclp_runs_gpu4/test_eclp_offline_lpo"),
                       "--golden", str(work / "golden"), "--output", str(work / judge / "our_eclp_offline_lpo"),
                       "--judge", judge, "--max-new-scenarios", "0" if args.full else str(max(0, args.pilot - old.get("evaluated_count", 0))),
                       "--memory-cap-gib", str(cap)]
            log_path = work / f"{judge}.log"
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
                atomic_json(status, {"state": "running", "judge": judge, "child_pid": process.pid,
                                     "physical_gpu": 4, "mode": "full" if args.full else "pilot",
                                     "memory_cap_gib": cap, "initial_free_gib": available})
                code = process.wait()
            render(root, work)
            if code:
                atomic_json(status, {"state": "failed", "judge": judge, "exit_code": code, "log": str(log_path)})
                raise SystemExit(code)
        render(root, work)
        atomic_json(status, {"state": "complete" if args.full else "pilot_complete", "judges": args.judges,
                             "physical_gpu": 4, "updated_at": time.strftime("%FT%T%z")})


if __name__ == "__main__":
    main()
