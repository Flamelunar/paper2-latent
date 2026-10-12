#!/usr/bin/env python3
"""Validate and consolidate Gemma 4 artifacts into success_judgments/gemma4."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from repro.constants import ROOT


ARTIFACTS = ROOT / "artifacts"
FULL = ARTIFACTS / "joint_success_win_full/gemma4_v11"
PARALLEL = ARTIFACTS / "joint_success_win_parallel/gemma4_v11_dialogue_sft"
PROBE = ARTIFACTS / "joint_success_win_probe"
QUICKCHECK = ARTIFACTS / "gemma4_quickcheck_20260926"
SHARED_MANIFEST = ARTIFACTS / "success_prompt_v2_diagnostic/sample_manifest.json"
ARCHIVE = ARTIFACTS / "success_judgments/gemma4"
GENERATION = ARTIFACTS / "generation_5rounds/legacy"
JUDGE = "/data1/liujianjian/model/gemma-4-E4B-it"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def move_file(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite archive file: {destination}")
    shutil.move(str(source), str(destination))


def validate_canonical(method: str) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    path = ARCHIVE / f"{method}.jsonl"
    rows = read_jsonl(path)
    by_index = {int(row["index"]): row for row in rows}
    if len(rows) != 1000 or len(by_index) != 1000 or set(by_index) != set(range(1000)):
        raise ValueError(f"{path} must have exactly one row for each index 0..999")
    generated_rows = read_jsonl(GENERATION / f"{method}.jsonl")
    generated = {int(row["index"]): row for row in generated_rows}
    if len(generated_rows) != 1000 or set(generated) != set(range(1000)):
        raise ValueError(f"Generation source is not a unique 1,000-row set: {method}")
    for index, row in by_index.items():
        if (row.get("method") != method or row.get("judge_model") != JUDGE
                or row.get("judge_device") != "cuda:1"
                or row.get("prompt_version") != "v11"
                or row.get("prompt_revision") != "joint_v2_r11"
                or row.get("judge_compute_dtype") != "bfloat16"
                or row.get("judge_quantization") != "none"
                or row.get("scenario_id") != generated[index].get("scenario_id")):
            raise ValueError(f"Invalid Gemma 4 record or generation alignment at {method}:{index}")
    return rows, by_index


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed: list[dict[str, Any]] = []
    unresolved = 0
    retried = 0
    for row in rows:
        if int(row.get("model_calls", 1)) > 1:
            retried += 1
        verdict = row.get("verdict")
        if row.get("parse_error") or not isinstance(verdict, dict):
            unresolved += 1
            continue
        rounds = verdict.get("round_success")
        if not isinstance(rounds, list) or len(rounds) != 5 or not all(isinstance(x, bool) for x in rounds):
            raise ValueError(f"Malformed five-round success verdict at index {row.get('index')}")
        first = next((n for n, success in enumerate(rounds, 1) if success), None)
        if (verdict.get("first_success_round") != first
                or verdict.get("success_within_5_rounds") != (first is not None)):
            raise ValueError(f"Inconsistent success summary at index {row.get('index')}")
        ab, ba = verdict.get("win_ab"), verdict.get("win_ba")
        expected_credit = ((ab == "A") + (ba == "B")) / 2
        if ab not in {"A", "B"} or ba not in {"A", "B"} or verdict.get("win_credit") != expected_credit:
            raise ValueError(f"Malformed Win-Rate verdict at index {row.get('index')}")
        parsed.append(verdict)
    if not parsed:
        raise ValueError("No parsed judgments to summarize")
    successes = sum(bool(item["success_within_5_rounds"]) for item in parsed)
    capped_rounds = [item["first_success_round"] or 5 for item in parsed]
    return {
        "total": len(rows),
        "parsed": len(parsed),
        "unresolved": unresolved,
        "parse_success_percent": len(parsed) / len(rows) * 100,
        "success_count": successes,
        "success_percent_of_parsed": successes / len(parsed) * 100,
        "win_rate_percent_of_parsed": statistics.fmean(float(item["win_credit"]) for item in parsed) * 100,
        "mean_rounds_success_else_5_among_parsed": statistics.fmean(capped_rounds),
        "rows_with_retries": retried,
    }


def update_metadata() -> None:
    path = ARCHIVE / "run_metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["status"] = "completed"
    metadata["status_updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["output_dir"] = "artifacts/success_judgments/gemma4"
    metadata["completed_gpu1_rows"] = {"qwen25_zero_shot": 1000, "qwen25_dialogue_sft": 1000}
    metadata["archive_layout"] = {
        "canonical_results": "artifacts/success_judgments/gemma4/*.jsonl",
        "gpu2_partial": "artifacts/success_judgments/gemma4/partial_runs/gpu2_dialogue_sft/",
        "prompt_diagnostics": "artifacts/success_judgments/gemma4/prompt_diagnostics/",
        "logs": "artifacts/success_judgments/gemma4/logs/full_evaluation/",
        "run_code": "artifacts/success_judgments/gemma4/code/",
    }
    code_files = {
        "evaluate_joint_prompt_probe.py": ROOT / "evaluate_joint_prompt_probe.py",
        "simulation.py": ROOT / "repro/simulation.py",
        "canonicalize_gemma4_joint_full.py": ROOT / "canonicalize_gemma4_joint_full.py",
        "finalize_main_table.py": ROOT / "finalize_main_table.py",
        "resume_gemma4_joint_eval.py": ROOT / "resume_gemma4_joint_eval.py",
        "organize_gemma4_artifacts.py": ROOT / "organize_gemma4_artifacts.py",
    }
    metadata["current_code_sha256"] = {
        name: hashlib.sha256(file.read_bytes()).hexdigest()
        for name, file in code_files.items() if file.is_file()
    }
    metadata["stdout_log"] = "artifacts/success_judgments/gemma4/logs/full_evaluation/" + Path(
        str(metadata.get("stdout_log", "run.log"))
    ).name
    for attempt in metadata.get("process_attempt_history", []):
        if attempt.get("stdout_log"):
            attempt["stdout_log"] = "artifacts/success_judgments/gemma4/logs/full_evaluation/" + Path(
                str(attempt["stdout_log"])
            ).name
    for field in ("resume_code_snapshot", "resume_supervisor_snapshot"):
        if metadata.get(field):
            metadata[field] = "artifacts/success_judgments/gemma4/code/historical_snapshots/" + Path(
                str(metadata[field])
            ).name
    if metadata.get("last_resume_log"):
        metadata["last_resume_log"] = "artifacts/success_judgments/gemma4/logs/full_evaluation/" + Path(
            str(metadata["last_resume_log"])
        ).name
    metadata["completion_source"] = "1,000 unique GPU 1 rows per method; indexes and scenario IDs validated"
    write_json(path, metadata)


def organize() -> dict[str, Any]:
    cron = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if cron.returncode == 0 and "GEMMA4_FULL_EVAL_V11" in cron.stdout:
        raise RuntimeError("Temporary Gemma 4 evaluation schedule is still installed")
    for method in ("qwen25_zero_shot", "qwen25_dialogue_sft"):
        validate_canonical(method)

    target_dirs = (
        ARCHIVE / "raw",
        ARCHIVE / "partial_runs/gpu2_dialogue_sft",
        ARCHIVE / "prompt_diagnostics/revisions",
        ARCHIVE / "prompt_diagnostics/quickcheck",
        ARCHIVE / "logs/full_evaluation",
        ARCHIVE / "code/historical_snapshots",
        ARCHIVE / "code/current",
    )
    for directory in target_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    # Keep the exact pre-canonicalization files, including GPU 2 rows, for audit.
    move_file(FULL / "qwen25_zero_shot.jsonl", ARCHIVE / "raw/primary_zero_shot_original.jsonl")
    move_file(FULL / "qwen25_dialogue_sft.jsonl", ARCHIVE / "raw/primary_dialogue_sft_original_file.jsonl")
    raw_primary = ARCHIVE / "raw/primary_dialogue_sft_before_canonicalization.jsonl"
    raw_original = ARCHIVE / "raw/primary_dialogue_sft_original_file.jsonl"
    if raw_primary.is_file() and raw_original.is_file():
        if hashlib.sha256(raw_primary.read_bytes()).digest() != hashlib.sha256(raw_original.read_bytes()).digest():
            raise ValueError("Original and normalized pre-canonicalization JSONL copies differ")
        raw_original.unlink()
    zero_original = ARCHIVE / "raw/primary_zero_shot_original.jsonl"
    zero_canonical = ARCHIVE / "qwen25_zero_shot.jsonl"
    if zero_original.is_file() and zero_canonical.is_file():
        if hashlib.sha256(zero_original.read_bytes()).digest() != hashlib.sha256(zero_canonical.read_bytes()).digest():
            raise ValueError("Original and canonical Zero-shot JSONL copies differ")
        zero_original.unlink()
    move_file(FULL / "run_metadata.json", ARCHIVE / "run_metadata.json")
    move_file(FULL / "full_eval_complete", ARCHIVE / "full_eval_complete")
    if FULL.is_dir():
        for source in list(FULL.iterdir()):
            if source.is_file():
                if source.suffix in {".py", ".sh"}:
                    move_file(source, ARCHIVE / "code/historical_snapshots" / source.name)
                elif source.suffix == ".log" or source.name == ".dialogue_eval.lock":
                    move_file(source, ARCHIVE / "logs/full_evaluation" / source.name)
                elif source.name == "canonicalization_report.json":
                    move_file(source, ARCHIVE / "canonicalization_report.json")
                else:
                    move_file(source, ARCHIVE / "raw/unclassified_full_run" / source.name)

    if PARALLEL.is_dir():
        for source in list(PARALLEL.iterdir()):
            if source.name == "qwen25_dialogue_sft.jsonl":
                destination = ARCHIVE / "partial_runs/gpu2_dialogue_sft/qwen25_dialogue_sft.jsonl"
            else:
                destination = ARCHIVE / "partial_runs/gpu2_dialogue_sft" / source.name
            move_file(source, destination)

    # Move only Gemma 4 probe outputs; the shared directory also contains other Judges.
    if PROBE.is_dir():
        for source in list(PROBE.glob("gemma4_*")):
            move_file(source, ARCHIVE / "prompt_diagnostics/revisions" / source.name)
    if QUICKCHECK.is_dir():
        for source in list(QUICKCHECK.iterdir()):
            move_file(source, ARCHIVE / "prompt_diagnostics/quickcheck" / source.name)
    if SHARED_MANIFEST.is_file():
        destination = ARCHIVE / "prompt_diagnostics/shared_case_control_manifest.json"
        if not destination.exists():
            shutil.copy2(SHARED_MANIFEST, destination)

    # Store the exact current evaluator and the post-run processing scripts together.
    current_code = {
        "evaluate_joint_prompt_probe.py": ROOT / "evaluate_joint_prompt_probe.py",
        "simulation.py": ROOT / "repro/simulation.py",
        "canonicalize_gemma4_joint_full.py": ROOT / "canonicalize_gemma4_joint_full.py",
        "finalize_main_table.py": ROOT / "finalize_main_table.py",
        "resume_gemma4_joint_eval.py": ROOT / "resume_gemma4_joint_eval.py",
        "organize_gemma4_artifacts.py": ROOT / "organize_gemma4_artifacts.py",
        "run_gemma4_dialogue_cron_batch.sh": ARCHIVE / "code/historical_snapshots/run_gemma4_dialogue_cron_batch.sh",
    }
    for name, source in current_code.items():
        if source.is_file():
            shutil.copy2(source, ARCHIVE / "code/current" / name)

    # Preserve the 41 GPU 2 rows mixed into the original full-run checkpoint as a named subset.
    primary_original_path = ARCHIVE / "raw/primary_dialogue_sft_original_file.jsonl"
    if not primary_original_path.exists():
        primary_original_path = ARCHIVE / "raw/primary_dialogue_sft_before_canonicalization.jsonl"
    raw_primary = read_jsonl(primary_original_path)
    gpu2_primary_rows = [row for row in raw_primary if row.get("judge_device") == "cuda:2"]
    if gpu2_primary_rows:
        with (ARCHIVE / "partial_runs/gpu2_dialogue_sft/primary_file_gpu2_rows.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in gpu2_primary_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Mark the GPU 2 experiment clearly as partial and excluded from the main table.
    (ARCHIVE / "partial_runs/gpu2_dialogue_sft/README.md").write_text(
        "# GPU 2 partial run\n\n"
        "This directory contains the interrupted CUDA:2 Dialogue-SFT judgment run. "
        "Its 615-row parallel JSONL and the 41 CUDA:2 rows mixed into the original full-run "
        "checkpoint are retained for audit only. None of these rows are used in the main table; "
        "the canonical Dialogue-SFT result contains 1,000 CUDA:1 rows.\n",
        encoding="utf-8",
    )

    metadata = json.loads((ARCHIVE / "run_metadata.json").read_text(encoding="utf-8"))
    metadata["output_dir"] = "artifacts/success_judgments/gemma4/partial_runs/gpu2_dialogue_sft"
    metadata["status"] = "partial_not_used_in_main_table"
    metadata["stdout_log"] = "artifacts/success_judgments/gemma4/partial_runs/gpu2_dialogue_sft/run.log"
    write_json(ARCHIVE / "partial_runs/gpu2_dialogue_sft/run_metadata.json", metadata)

    methods = {}
    for method in ("qwen25_zero_shot", "qwen25_dialogue_sft"):
        rows, _ = validate_canonical(method)
        methods[method] = summarize(rows)
    summary = {
        "judge_model": JUDGE,
        "judge_compute_dtype": "bfloat16",
        "judge_quantization": "none",
        "device_for_canonical_results": "cuda:1",
        "prompt_version": "v11",
        "prompt_revision": "joint_v2_r11",
        "max_parse_retries": 3,
        "generation_source": "artifacts/generation_5rounds/legacy",
        "methods": methods,
        "metric_denominators": {
            "success_percent": "parsed records only",
            "win_rate_percent": "parsed records only; two order-swapped decisions",
            "mean_rounds": "parsed records only; no success counts as round 5",
            "unresolved": "unparseable records out of 1,000",
        },
    }
    write_json(ARCHIVE / "summary.json", summary)

    (ARCHIVE / "README.md").write_text(
        "# Gemma 4 evaluation archive\n\n"
        "Judge: `/data1/liujianjian/model/gemma-4-E4B-it`, BF16 without quantization. "
        "The canonical 1,000-row results use GPU 1, prompt v11 (`joint_v2_r11`), and the "
        "shared held-out generations in `../../generation_5rounds/legacy/`.\n\n"
        "## Contents\n\n"
        "- `qwen25_zero_shot.jsonl`, `qwen25_dialogue_sft.jsonl`: canonical full results, one "
        "GPU 1 record for each index 0–999.\n"
        "- `summary.json`: independently recomputed metrics and denominators.\n"
        "- `run_metadata.json`, `canonicalization_report.json`: settings, code hashes, and "
        "source counts used to select the GPU 1 rows.\n"
        "- `raw/`: original pre-canonicalization outputs, including records from other devices.\n"
        "- `partial_runs/gpu2_dialogue_sft/`: interrupted GPU 2 run, explicitly excluded from "
        "the main table.\n"
        "- `prompt_diagnostics/`: small-sample prompt revisions and quickcheck manifests; these "
        "are diagnostic data, not full-test estimates.\n"
        "- `logs/full_evaluation/`: full-run, retry, scheduler, and interruption logs.\n"
        "- `code/current/`: current evaluation and score-processing code; "
        "`code/historical_snapshots/` preserves run-time script snapshots.\n\n"
        "## Metric rules\n\n"
        "Success rate, Win-Rate, and mean rounds use parsed records only. Win-Rate compares all "
        "five Persuader replies with the held-out golden reply in both orders; a split vote gets "
        "half credit. Mean rounds uses the first successful round, with failure capped at round 5. "
        "Unresolved parse failures are reported separately out of 1,000. See `summary.json` and "
        "`../../../results/main_experiment_table.md`.\n",
        encoding="utf-8",
    )

    # Update the explanatory report to point to the organized files and remove stale status text.
    report = ROOT / "results/gemma4_joint_prompt_diagnostic.md"
    if report.is_file():
        text = report.read_text(encoding="utf-8")
        text = text.replace(
            "artifacts/joint_success_win_probe/gemma4_joint_v2_r11.jsonl",
            "artifacts/success_judgments/gemma4/prompt_diagnostics/revisions/gemma4_joint_v2_r11.jsonl",
        )
        text = text.replace(
            "artifacts/success_prompt_v2_diagnostic/sample_manifest.json",
            "artifacts/success_judgments/gemma4/prompt_diagnostics/shared_case_control_manifest.json",
        )
        text = text.replace(
            "artifacts/joint_success_win_probe/",
            "artifacts/success_judgments/gemma4/prompt_diagnostics/revisions/",
        )
        text = text.replace(
            "Full 1,000-case Gemma 4 results have not been produced and are not entered into the main experiment rows.",
            "The full 1,000-case Gemma 4 results are now included in the main experiment rows and "
            "archived under `artifacts/success_judgments/gemma4/`; the selected-sample figures in "
            "this diagnostic remain prompt-development results, not full-test estimates.",
        )
        report.write_text(text, encoding="utf-8")

    # The complete marker now lives beside the canonical results, so the old run folder is empty.
    for directory in (FULL, PARALLEL, QUICKCHECK):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    for directory in (FULL.parent, PARALLEL.parent):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()

    update_metadata()
    return summary


if __name__ == "__main__":
    print(json.dumps(organize(), ensure_ascii=False, indent=2))
