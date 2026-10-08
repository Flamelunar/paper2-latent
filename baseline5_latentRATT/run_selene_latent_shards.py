#!/usr/bin/env python3
"""Resume Selene v17-v7 LatentRATT judgments in disjoint A100 shards."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EXP_ROOT = ROOT.parent / "exp_code_ljj"
ARTIFACTS = EXP_ROOT / "artifacts/success_judgments/selene_v17_v7_latentRATT_20260928"
BASE = ARTIFACTS / "latentRATT_persuasion.jsonl"
SHARD_ROOT = ARTIFACTS / "shards_v7_20260928"
GENERATION = EXP_ROOT / "artifacts/generation_5rounds/baseline5_latentRATT.json"
MODEL = "/app/hzj/liujianjian/qwen2.5-7b-latentRATT-all-59868/rl"
JUDGE = "/app/hzj/models/Selene-1-Mini-Llama-3.1-8B"
METHOD = "latentRATT_persuasion"
REVISION = "joint_selene_success_reference_v17_v7_strict_goal_acceptance"
PYTHON = "/app/hzj/anaconda3/envs/ljj/bin/python"
EVALUATOR = ROOT / "evaluate_latent_judges.py"
TABLE_UPDATER = ROOT / "update_latentratt_main_table.py"
WORKERS = 3


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_record(row: dict[str, Any]) -> int:
    index = int(row["index"])
    expected = {
        "trial": 1,
        "method": METHOD,
        "judge_model": JUDGE,
        "model": MODEL,
        "judge_device": "cuda:2",
        "judge_compute_dtype": "bfloat16",
        "judge_quantization": "none",
        "prompt_version": "selene_v17",
        "prompt_revision": REVISION,
        "max_parse_retries": 3,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"Wrong {key} at Selene result index {index}: {row.get(key)!r}")
    return index


def existing_prefix() -> int:
    if not BASE.is_file():
        return 0
    rows = read_jsonl(BASE)
    indexes = [validate_record(row) for row in rows]
    if len(indexes) != len(set(indexes)) or set(indexes) != set(range(len(indexes))):
        raise ValueError("Existing Selene output is not a unique contiguous prefix; refusing to shard")
    if len(indexes) > 1000:
        raise ValueError("Existing Selene output has more than 1,000 rows")
    return len(indexes)


def split_ranges(start: int, workers: int = WORKERS) -> list[tuple[int, int]]:
    remaining = 1000 - start
    quotient, remainder = divmod(remaining, workers)
    ranges = []
    cursor = start
    for worker in range(workers):
        size = quotient + (worker < remainder)
        if size:
            ranges.append((cursor, cursor + size))
            cursor += size
    return ranges


def readable_count(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                if line.strip():
                    json.loads(line)
                    count += 1
            except json.JSONDecodeError:
                continue
    return count


def merge_outputs(ranges: list[tuple[int, int]]) -> None:
    sources = [BASE]
    sources.extend(
        SHARD_ROOT / f"part_{number:02d}_{start:04d}_{end - 1:04d}" / f"{METHOD}.jsonl"
        for number, (start, end) in enumerate(ranges, 1)
    )
    generation = {int(row["index"]): row for row in read_jsonl(GENERATION)}
    if len(generation) != 1000 or set(generation) != set(range(1000)):
        raise ValueError("Generation file must contain exactly indexes 0..999")
    merged: dict[int, dict[str, Any]] = {}
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Missing Selene shard output: {source}")
        for row in read_jsonl(source):
            index = validate_record(row)
            if index in merged:
                raise ValueError(f"Duplicate Selene result for index {index} across shards")
            if row.get("scenario_id") != generation[index].get("scenario_id"):
                raise ValueError(f"Selene result is misaligned with generation index {index}")
            merged[index] = row
    if set(merged) != set(range(1000)):
        missing = sorted(set(range(1000)) - set(merged))
        raise ValueError(f"Merged Selene results are incomplete: {len(missing)} missing, first={missing[:10]}")

    original_mode = BASE.stat().st_mode & 0o777
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=BASE.parent, prefix=".latentRATT_persuasion.",
        suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for index in range(1000):
            handle.write(json.dumps(merged[index], ensure_ascii=False) + "\n")
    os.chmod(temporary, original_mode)
    os.replace(temporary, BASE)
    print(f"Merged 1,000 Selene records into {BASE}", flush=True)


def wait_until_loaded(
    process: subprocess.Popen[bytes], log_path: Path, log_offset: int,
    output_path: Path, expected_rows: int,
    timeout_seconds: int = 1800,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    ready_text = f"Loaded {JUDGE} on cuda:2; BF16, no quantization"
    while time.monotonic() < deadline:
        status = process.poll()
        if status == 0 and readable_count(output_path) == expected_rows:
            return
        if status is not None:
            raise RuntimeError(f"Selene shard exited before loading with status {status}; see {log_path}")
        if log_path.is_file():
            current_log = log_path.read_bytes()[log_offset:].decode("utf-8", errors="replace")
            if ready_text in current_log:
                return
        time.sleep(2)
    raise TimeoutError(f"Selene shard did not load within {timeout_seconds}s; see {log_path}")


def main() -> None:
    SHARD_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = SHARD_ROOT / ".runner.lock"
    with lock_path.open("a+") as run_lock:
        try:
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("A Selene shard runner is already active") from exc

        start = existing_prefix()
        if start == 1000:
            print("Selene v17-v7 output is already complete; updating only its main-table row", flush=True)
            subprocess.run([PYTHON, str(TABLE_UPDATER), "--judges", "selene"], check=True)
            return
        ranges = split_ranges(start)
        print(f"Selene v17-v7 plan: {WORKERS} workers on cuda:2; completed prefix={start}/1000", flush=True)
        print("Shards: " + ", ".join(f"[{a},{b}) ({b-a} rows)" for a, b in ranges), flush=True)

        env = os.environ.copy()
        env["ALLOW_JUDGE_EVAL"] = "1"
        env["DAILYPERSUASION_PROFILE"] = "paper10k"
        processes: list[tuple[subprocess.Popen[bytes], Path, int, int, Any]] = []
        completed = False
        try:
            for number, (range_start, range_end) in enumerate(ranges, 1):
                output_dir = SHARD_ROOT / f"part_{number:02d}_{range_start:04d}_{range_end - 1:04d}"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{METHOD}.jsonl"
                log_path = output_dir / "worker.log"
                log_offset = log_path.stat().st_size if log_path.exists() else 0
                log_handle = log_path.open("a", encoding="utf-8")
                command = [
                    PYTHON, "-u", str(EVALUATOR), "selene-shard",
                    "--generation-dir", str(GENERATION.parent),
                    "--output-dir", str(output_dir),
                    "--full-evaluation", "--methods", METHOD,
                    "--judge-model", JUDGE, "--judge-name", "selene1",
                    "--judge-device", "cuda:2", "--prompt-version", "selene_v17",
                    "--max-tokens", "256", "--max-parse-retries", "3",
                    "--index-start", str(range_start), "--index-end", str(range_end),
                ]
                process = subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
                processes.append((process, output_path, range_start, range_end, log_handle))
                print(
                    f"Started Selene worker {number}: indexes [{range_start},{range_end}), pid={process.pid}",
                    flush=True,
                )
                wait_until_loaded(process, log_path, log_offset, output_path, range_end - range_start)
                print(f"Selene worker {number} loaded; starting the next shard", flush=True)

            last_report = 0.0
            failed: tuple[int, int, int] | None = None
            while True:
                statuses = [process.poll() for process, *_ in processes]
                for (_, _, range_start, range_end, _), status in zip(processes, statuses):
                    if status not in (None, 0):
                        failed = (range_start, range_end, status)
                        break
                if failed or all(status == 0 for status in statuses):
                    break
                now = time.monotonic()
                if now - last_report >= 30:
                    counts = [readable_count(path) for _, path, *_ in processes]
                    done = start + sum(counts)
                    print(f"Selene sharded progress: {done}/1000; shards={counts}", flush=True)
                    last_report = now
                time.sleep(10)
            if failed:
                raise RuntimeError(f"Selene shard [{failed[0]},{failed[1]}) exited with status {failed[2]}")
            completed = True
        finally:
            if not completed:
                for process, *_ in processes:
                    if process.poll() is None:
                        process.terminate()
            for process, _, _, _, handle in processes:
                if process.poll() is None:
                    process.wait()
                handle.close()

        merge_outputs(ranges)
        subprocess.run([PYTHON, str(TABLE_UPDATER), "--judges", "selene"], check=True)


if __name__ == "__main__":
    main()
