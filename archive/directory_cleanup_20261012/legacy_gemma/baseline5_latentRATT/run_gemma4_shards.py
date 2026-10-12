#!/usr/bin/env python3
"""Run disjoint Gemma v11 shards on cuda:2, merge them, then update the table."""
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
ARTIFACTS = EXP_ROOT / "artifacts/success_judgments/gemma4"
BASE = ARTIFACTS / "latentRATT_persuasion.jsonl"
SHARD_ROOT = ARTIFACTS / "shards_v11_20260928"
GENERATION = EXP_ROOT / "artifacts/generation_5rounds/baseline5_latentRATT.json"
MODEL = "/app/hzj/liujianjian/qwen2.5-7b-latentRATT-all-59868/rl"
JUDGE = "/app/hzj/models/gemma-4-E4B-it"
METHOD = "latentRATT_persuasion"
REVISION = "joint_v2_r11"
PYTHON = "/app/hzj/anaconda3/envs/ljj/bin/python"
EVALUATOR = ROOT / "evaluate_latent_judges.py"
TABLE_UPDATER = ROOT / "update_latentratt_main_table.py"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_record(row: dict[str, Any], *, index: int | None = None) -> int:
    row_index = int(row["index"])
    expected = {
        "trial": 1,
        "method": METHOD,
        "judge_model": JUDGE,
        "model": MODEL,
        "judge_device": "cuda:2",
        "judge_compute_dtype": "bfloat16",
        "judge_quantization": "none",
        "prompt_version": "v11",
        "prompt_revision": REVISION,
        "max_parse_retries": 3,
    }
    if index is not None and row_index != index:
        raise ValueError(f"Unexpected index {row_index}; expected {index}")
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"Wrong {key} at Gemma result index {row_index}: {row.get(key)!r}")
    return row_index


def existing_prefix() -> int:
    if not BASE.is_file():
        return 0
    rows = read_jsonl(BASE)
    indexes = [validate_record(row) for row in rows]
    if len(indexes) != len(set(indexes)) or set(indexes) != set(range(len(indexes))):
        raise ValueError("Existing Gemma output is not a unique contiguous prefix; refusing to shard")
    if len(indexes) > 1000:
        raise ValueError("Existing Gemma output has more than 1,000 rows")
    return len(indexes)


def split_ranges(start: int, workers: int = 3) -> list[tuple[int, int]]:
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
            raise FileNotFoundError(f"Missing Gemma shard output: {source}")
        for row in read_jsonl(source):
            index = validate_record(row)
            if index in merged:
                raise ValueError(f"Duplicate Gemma result for index {index} across shards")
            if row.get("scenario_id") != generation[index].get("scenario_id"):
                raise ValueError(f"Gemma result is misaligned with generation index {index}")
            merged[index] = row
    if set(merged) != set(range(1000)):
        missing = sorted(set(range(1000)) - set(merged))
        raise ValueError(f"Merged Gemma results are incomplete: {len(missing)} missing, first={missing[:10]}")

    original_mode = BASE.stat().st_mode & 0o777 if BASE.exists() else 0o644
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=BASE.parent, prefix=".latentRATT_persuasion.",
        suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for index in range(1000):
            handle.write(json.dumps(merged[index], ensure_ascii=False) + "\n")
    os.chmod(temporary, original_mode)
    os.replace(temporary, BASE)
    print(f"Merged 1,000 Gemma records into {BASE}", flush=True)


def main() -> None:
    SHARD_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = SHARD_ROOT / ".runner.lock"
    with lock_path.open("a+") as run_lock:
        try:
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("A Gemma shard runner is already active") from exc

        start = existing_prefix()
        if start == 1000:
            print("Gemma output is already complete; updating only its main-table row", flush=True)
            subprocess.run([PYTHON, str(TABLE_UPDATER), "--judges", "gemma4"], check=True)
            return
        ranges = split_ranges(start)
        print(f"Gemma v11 plan: 3 workers on cuda:2; completed prefix={start}/1000", flush=True)
        print("Shards: " + ", ".join(f"[{a},{b}) ({b-a} rows)" for a, b in ranges), flush=True)

        env = os.environ.copy()
        env["ALLOW_JUDGE_EVAL"] = "1"
        env["DAILYPERSUASION_PROFILE"] = "paper10k"
        processes: list[tuple[subprocess.Popen[bytes], Path, int, int, Any]] = []
        for number, (range_start, range_end) in enumerate(ranges, 1):
            output_dir = SHARD_ROOT / f"part_{number:02d}_{range_start:04d}_{range_end - 1:04d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "worker.log"
            log_handle = log_path.open("a", encoding="utf-8")
            command = [
                PYTHON, "-u", str(EVALUATOR), "gemma-joint",
                "--generation-dir", str(GENERATION.parent),
                "--output-dir", str(output_dir),
                "--full-evaluation", "--methods", METHOD,
                "--judge-model", JUDGE, "--judge-name", "gemma4",
                "--judge-device", "cuda:2", "--prompt-version", "v11",
                "--max-tokens", "384", "--max-parse-retries", "3",
                "--index-start", str(range_start), "--index-end", str(range_end),
            ]
            process = subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            processes.append((process, output_dir / f"{METHOD}.jsonl", range_start, range_end, log_handle))
            print(f"Started Gemma worker {number}: indexes [{range_start},{range_end}), pid={process.pid}", flush=True)

        last_report = 0.0
        failed: tuple[int, int, int] | None = None
        try:
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
                    print(f"Gemma sharded progress: {done}/1000; shards={counts}", flush=True)
                    last_report = now
                time.sleep(10)
        finally:
            if failed:
                for process, *_ in processes:
                    if process.poll() is None:
                        process.terminate()
            for process, _, _, _, handle in processes:
                if process.poll() is None:
                    process.wait()
                handle.close()
        if failed:
            raise RuntimeError(f"Gemma shard [{failed[0]},{failed[1]}) exited with status {failed[2]}")

        merge_outputs(ranges)
        subprocess.run([PYTHON, str(TABLE_UPDATER), "--judges", "gemma4"], check=True)


if __name__ == "__main__":
    main()
