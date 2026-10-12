#!/usr/bin/env python3
"""Canonicalize completed Gemma 4 full-run JSONL files after parallel runs exit."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from repro.constants import ROOT


ARTIFACTS = ROOT / "artifacts"
PRIMARY = ARTIFACTS / "joint_success_win_full/gemma4_v11"
PARALLEL = ARTIFACTS / "joint_success_win_parallel/gemma4_v11_dialogue_sft"
ARCHIVE = ARTIFACTS / "success_judgments/gemma4"
EXPECTED_JUDGE = "/data1/liujianjian/model/gemma-4-E4B-it"
def active_evaluators() -> list[int]:
    pids: set[int] = set()
    for metadata_path in (
        PRIMARY / "run_metadata.json",
        PARALLEL / "run_metadata.json",
        ARCHIVE / "run_metadata.json",
    ):
        if not metadata_path.is_file():
            continue
        pid = int(json.loads(metadata_path.read_text(encoding="utf-8"))["process_pid"])
        command_path = Path(f"/proc/{pid}/cmdline")
        if command_path.exists() and b"evaluate_joint_prompt_probe.py" in command_path.read_bytes():
            pids.add(pid)
    return sorted(pids)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_record(row: dict[str, Any], *, method: str, path: Path) -> None:
    if (row.get("method") != method
            or row.get("judge_model") != EXPECTED_JUDGE
            or row.get("trial") != 1
            or row.get("prompt_version") != "v11"
            or row.get("prompt_revision") != "joint_v2_r11"
            or row.get("judge_compute_dtype") != "bfloat16"
            or row.get("judge_quantization") != "none"
            or row.get("max_parse_retries") != 3
            or row.get("judge_device") not in {"cuda:1", "cuda:2"}):
        raise ValueError(f"Unexpected evaluation metadata in {path} at index {row.get('index')}")


def check_source_alignment(
    rows: list[dict[str, Any]], *, method: str, source_path: Path
) -> dict[int, dict[str, Any]]:
    generated = read_jsonl(source_path)
    generated_by_index = {int(row["index"]): row for row in generated}
    if (len(generated) != 1000 or len(generated_by_index) != 1000
            or set(generated_by_index) != set(range(1000))):
        raise ValueError(f"Generation source must have unique indexes 0..999: {source_path}")
    for row in rows:
        index = int(row["index"])
        if row.get("scenario_id") != generated_by_index[index].get("scenario_id"):
            raise ValueError(f"Scenario mismatch for {method} index {index}")
    return generated_by_index


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def canonicalize() -> dict[str, Any]:
    active = active_evaluators()
    if active:
        raise RuntimeError(f"Evaluation processes are still active; refusing to rewrite outputs: {active}")

    source_dir = ARTIFACTS / "generation_5rounds"
    zero_candidates = (
        PRIMARY / "qwen25_zero_shot.jsonl",
        ARCHIVE / "qwen25_zero_shot.jsonl",
    )
    zero_input = next((path for path in zero_candidates if path.is_file()), None)
    if zero_input is None:
        raise FileNotFoundError("Missing Gemma 4 Zero-shot result JSONL")
    zero_rows = read_jsonl(zero_input)
    zero_by_index = {int(row["index"]): row for row in zero_rows}
    if len(zero_by_index) != 1000 or set(zero_by_index) != set(range(1000)):
        raise ValueError(f"Zero-shot must have exactly 1000 unique indexes: {zero_input}")
    for row in zero_rows:
        validate_record(row, method="qwen25_zero_shot", path=zero_input)
        if row.get("judge_device") != "cuda:1":
            raise ValueError(f"Zero-shot judgments must come from GPU 1: {zero_input}")
    check_source_alignment(
        zero_rows,
        method="qwen25_zero_shot",
        source_path=source_dir / "qwen25_zero_shot.jsonl",
    )
    zero_path = ARCHIVE / "qwen25_zero_shot.jsonl"
    atomic_write_jsonl(zero_path, [zero_by_index[index] for index in range(1000)])

    raw_path = ARCHIVE / "raw/primary_dialogue_sft_before_canonicalization.jsonl"
    primary_candidates = (
        PRIMARY / "qwen25_dialogue_sft.jsonl",
        raw_path,
    )
    primary_input = next((path for path in primary_candidates if path.is_file()), None)
    if primary_input is None:
        raise FileNotFoundError("Missing Gemma 4 primary Dialogue-SFT result JSONL")
    if primary_input != raw_path and not raw_path.exists():
        atomic_write_jsonl(raw_path, read_jsonl(primary_input))

    parallel_candidates = (
        PARALLEL / "qwen25_dialogue_sft.jsonl",
        ARCHIVE / "partial_runs/gpu2_dialogue_sft/qwen25_dialogue_sft.jsonl",
    )
    parallel_input = next((path for path in parallel_candidates if path.is_file()), None)
    sft_sources = [(primary_input, "primary")]
    if parallel_input is not None:
        sft_sources.append((parallel_input, "parallel"))
    grouped: dict[int, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    source_counts: dict[str, dict[str, int]] = {}
    all_sft_rows: list[dict[str, Any]] = []
    for path, source_name in sft_sources:
        rows = read_jsonl(path)
        counts = Counter(int(row["index"]) for row in rows)
        source_counts[source_name] = {"rows": len(rows), "unique_indexes": len(counts)}
        for row in rows:
            validate_record(row, method="qwen25_dialogue_sft", path=path)
            index = int(row["index"])
            if index not in range(1000):
                raise ValueError(f"Out-of-range Dialogue-SFT index {index} in {path}")
            grouped[index].append((row, source_name))
            all_sft_rows.append(row)

        if source_name == "primary":
            primary_gpu1_indexes = {
                int(row["index"]) for row in rows if row.get("judge_device") == "cuda:1"
            }
            if primary_gpu1_indexes != set(range(1000)):
                raise ValueError(
                    "The primary Dialogue-SFT file must contain all 1,000 GPU 1 judgments "
                    f"before canonicalization; found {len(primary_gpu1_indexes)}"
                )

    check_source_alignment(
        all_sft_rows,
        method="qwen25_dialogue_sft",
        source_path=source_dir / "qwen25_dialogue_sft.jsonl",
    )
    missing = sorted(set(range(1000)) - set(grouped))
    if missing:
        raise ValueError(f"Dialogue-SFT results are missing indexes: {missing[:20]}")

    canonical_sft: list[dict[str, Any]] = []
    selected_devices: Counter[str] = Counter()
    duplicate_indexes = 0
    for index in range(1000):
        candidates = grouped[index]
        if len(candidates) > 1:
            duplicate_indexes += 1
            scenario_ids = {row.get("scenario_id") for row, _ in candidates}
            if len(scenario_ids) != 1:
                raise ValueError(f"Conflicting scenarios among duplicate index {index}")
        row, _ = max(
            candidates,
            key=lambda candidate: (
                candidate[0].get("judge_device") == "cuda:1",
                candidate[1] == "primary",
            ),
        )
        canonical_sft.append(row)
        selected_devices[str(row["judge_device"])] += 1

    if selected_devices != Counter({"cuda:1": 1000}):
        raise ValueError(
            "Canonical Dialogue-SFT results must all come from GPU 1; "
            f"selected devices: {dict(selected_devices)}"
        )

    sft_path = ARCHIVE / "qwen25_dialogue_sft.jsonl"
    atomic_write_jsonl(sft_path, canonical_sft)

    report = {
        "judge_model": EXPECTED_JUDGE,
        "prompt_revision": "joint_v2_r11",
        "canonical_dialogue_sft_rows": len(canonical_sft),
        "source_counts": source_counts,
        "duplicate_indexes_across_sources": duplicate_indexes,
        "selected_judge_devices": dict(selected_devices),
        "raw_primary_sft_archive": str(raw_path),
        "canonical_zero_shot_path": str(zero_path),
        "canonical_sft_path": str(sft_path),
    }
    report_path = ARCHIVE / "canonicalization_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(canonicalize(), indent=2, ensure_ascii=False))
