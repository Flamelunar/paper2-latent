#!/usr/bin/env python3
"""Evaluate the existing ECLP 1,000-case run on the fixed 192 cases with Qwen3."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "my_eclp_final"))
from experiment_io import atomic_json, read_json
from matched_eval_protocol_v2 import CONFIG
from update_matched_table_v2 import update_table, verified_complete_summary


def free_gib(gpu: int) -> float:
    value = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return float(value.strip()) / 1024


def evaluators_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "evaluate_.*192_v2[.]py"], text=True, capture_output=True, check=False
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"pgrep failed: {result.stderr.strip()}")
    return any(line.strip().isdigit() for line in result.stdout.splitlines())


def run(args: argparse.Namespace) -> None:
    root = args.repo_root.resolve()
    work = root / "eclp_runs_gpu4/evaluation_v2"
    output = work / "qwen3_14b/our_eclp_offline_lpo"
    source = root / "eclp_runs_gpu4/test_eclp_offline_lpo/generation.jsonl"
    golden = root / "eclp_runs_gpu4/evaluation_v2/golden"
    baseline_status = root / "exp_code_ljj/artifacts/evaluation_from_start192_fullsource_v2/status.json"
    state_path = root / "eclp_runs_gpu4/eclp_qwen3_192_v2/pipeline_status.json"
    log_path = state_path.parent / "evaluation.log"
    scripts = root / "my_eclp_final"
    output.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = scripts / "evaluate_selected192_v2.py"
    base_command = [
        sys.executable, str(evaluator), "--repo-root", str(root),
        "--generation", str(source), "--golden", str(golden), "--output", str(output),
        "--judge", "qwen3_14b", "--method", "our_eclp_offline_lpo", "--physical-gpu", "1",
        "--batch-size", "2", "--scenario-chunk-size", "8", "--memory-cap-gib", "32",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", PYTHONUNBUFFERED="1")

    validation = subprocess.run(
        base_command + ["--validate-only"], cwd=root, env=env, text=True, capture_output=True, check=False
    )
    if validation.returncode:
        raise RuntimeError(f"Qwen3 ECLP 192 validation failed: {validation.stderr.strip()}")

    saved = output / "summary.json"
    if saved.exists():
        summary = verified_complete_summary(output, "qwen3_14b", "our_eclp_offline_lpo")
        if summary is not None and summary["evaluated_count"] == 192:
            report = update_table(root / "exp_code_ljj/results/main_experiment_table.md", work)
            atomic_json(state_path, {"state": "complete", "stage": "table_updated", "summary": summary,
                                     "table_update": report, "revision": CONFIG["revision"],
                                     "finished_at": time.strftime("%FT%T%z")})
            return

    attempt = 0
    while True:
        baseline = read_json(baseline_status) if baseline_status.exists() else {}
        baseline_terminal = baseline.get("state") in {"complete", "failed"}
        available = free_gib(1)
        busy = evaluators_running()
        if baseline_terminal and not busy and available >= args.minimum_free_gib:
            break
        atomic_json(state_path, {
            "state": "waiting", "stage": "gpu_and_baselines", "baseline_state": baseline.get("state", "missing"),
            "baseline_active": baseline.get("active", {}), "gpu": 1,
            "free_gib": available, "minimum_free_gib": args.minimum_free_gib,
            "other_192_evaluator_running": busy, "revision": CONFIG["revision"],
            "updated_at": time.strftime("%FT%T%z"),
        })
        time.sleep(args.poll_seconds)

    while attempt < args.max_attempts:
        attempt += 1
        atomic_json(state_path, {"state": "running", "stage": "qwen3_14b_evaluation", "attempt": attempt,
                                 "gpu": 1, "free_gib": free_gib(1), "revision": CONFIG["revision"],
                                 "command": base_command, "started_at": time.strftime("%FT%T%z")})
        with log_path.open("a", encoding="utf-8") as handle:
            result = subprocess.run(base_command, cwd=root, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode == 0:
            summary = verified_complete_summary(output, "qwen3_14b", "our_eclp_offline_lpo")
            if summary is not None and summary["evaluated_count"] == 192:
                report = update_table(root / "exp_code_ljj/results/main_experiment_table.md", work)
                atomic_json(state_path, {"state": "complete", "stage": "table_updated", "summary": summary,
                                         "table_update": report, "revision": CONFIG["revision"],
                                         "finished_at": time.strftime("%FT%T%z")})
                return
        atomic_json(state_path, {"state": "retrying", "stage": "qwen3_14b_evaluation", "attempt": attempt,
                                 "exit_code": result.returncode, "revision": CONFIG["revision"],
                                 "updated_at": time.strftime("%FT%T%z")})
        time.sleep(args.poll_seconds)

    raise RuntimeError("Qwen3 ECLP evaluation did not produce a verified 192-case result")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--minimum-free-gib", type=float, default=36)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        root = args.repo_root.resolve()
        state_path = root / "eclp_runs_gpu4/eclp_qwen3_192_v2/pipeline_status.json"
        atomic_json(state_path, {"state": "failed", "error": f"{type(exc).__name__}: {exc}",
                                 "updated_at": time.strftime("%FT%T%z")})
        raise


if __name__ == "__main__":
    main()
