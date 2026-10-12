#!/usr/bin/env python3
"""Render all four baseline rows from complete five-round Judge records."""

from __future__ import annotations

import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

from repro.constants import ROOT


RUNS = ROOT / "artifacts"
TABLE = ROOT / "results/main_experiment_table.md"
SECTION = "## Main experiment: five-round evaluation"
START_MARKER = "<!-- baseline-five-rounds:start -->"
END_MARKER = "<!-- baseline-five-rounds:end -->"
OLD_START_MARKER = "<!-- qwen25-five-rounds:start -->"
OLD_END_MARKER = "<!-- qwen25-five-rounds:end -->"
WIN_PROTOCOLS = {
    "qwen25_all_five_persuader_replies_vs_golden_bf16_v1",
    "paper10k_all_five_persuader_replies_vs_golden_bf16_v1",
}
METHODS = (
    ("baseline1_zero_shot", "Baseline 1: Zero-shot", "qwen25_zero_shot"),
    ("baseline2_dialogue_sft", "Baseline 2: Dialogue-SFT", "qwen25_dialogue_sft"),
    ("baseline3_isr_simdpo", "Baseline 3: ISR-SFT + SimDPO", None),
    ("baseline4_ma2p", "Baseline 4: MA2P", None),
)
JUDGES = (
    (
        "Meta-Llama-3.1-8B-Instruct (BF16)",
        "/data1/liujianjian/model/Meta-Llama-3.1-8B-Instruct",
        "success_judgments/llama31",
        "win_judgments/llama31",
    ),
    (
        "Qwen3.5-9B (BF16)",
        "/data1/liujianjian/model/Qwen3.5-9B",
        "success_judgments/qwen35",
        "win_judgments/qwen35",
    ),
    (
        "Gemma 4 E4B-it (BF16)",
        "/data1/liujianjian/model/gemma-4-E4B-it",
        "success_judgments/gemma4",
        None,
    ),
)


def read_records(path: Path, *, method: str, judge_model: str) -> dict[int, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1000:
        raise ValueError(f"Expected 1000 records in {path}, found {len(rows)}")
    by_index = {int(row["index"]): row for row in rows}
    if len(by_index) != 1000 or set(by_index) != set(range(1000)):
        raise ValueError(f"Missing or duplicate indexes in {path}")
    for index, row in by_index.items():
        if row.get("method") != method or row.get("judge_model") != judge_model:
            raise ValueError(f"Wrong method or Judge in {path} at index {index}")
    return by_index


def verify_generation_alignment(records: dict[int, dict[str, Any]], *, method: str) -> None:
    """Require each Judge record to refer to the matching archived generation row."""
    source = RUNS / "generation_5rounds" / f"{method}.jsonl"
    if not source.is_file() and method in {"qwen25_zero_shot", "qwen25_dialogue_sft"}:
        source = RUNS / "generation_5rounds/legacy" / f"{method}.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"Missing generation source for {method}: {source}")
    generated_rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generated = {int(row["index"]): row for row in generated_rows}
    if len(generated_rows) != 1000 or len(generated) != 1000 or set(generated) != set(range(1000)):
        raise ValueError(f"Generation source must contain exactly indexes 0..999: {source}")
    for index, record in records.items():
        if record.get("scenario_id") != generated[index].get("scenario_id"):
            raise ValueError(
                f"Scenario mismatch for {method} index {index}: "
                f"Judge={record.get('scenario_id')}, generation={generated[index].get('scenario_id')}"
            )


def joint_records_summary(path: Path, *, method: str, judge_model: str) -> tuple[dict[int, dict[str, Any]], dict[str, int | float]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1000:
        raise ValueError(f"Expected 1000 joint Judge records in {path}, found {len(rows)}")
    by_index = {int(row["index"]): row for row in rows}
    if len(by_index) != 1000 or set(by_index) != set(range(1000)):
        raise ValueError(f"Missing or duplicate indexes in {path}")
    verify_generation_alignment(by_index, method=method)

    valid: list[dict[str, Any]] = []
    unresolved = 0
    for index, row in by_index.items():
        if (row.get("method") != method or row.get("judge_model") != judge_model
                or row.get("trial") != 1 or row.get("prompt_version") != "v11"
                or row.get("prompt_revision") != "joint_v2_r11"
                or row.get("judge_compute_dtype") != "bfloat16"
                or row.get("judge_quantization") != "none"
                or row.get("max_parse_retries") != 3
                or row.get("judge_device") != "cuda:1"):
            raise ValueError(f"Wrong method, Judge, prompt, or precision in {path} at index {index}")
        verdict = row.get("verdict")
        if row.get("parse_error") or not isinstance(verdict, dict):
            unresolved += 1
            continue
        rounds = verdict.get("round_success")
        if not isinstance(rounds, list) or len(rounds) != 5 or not all(isinstance(x, bool) for x in rounds):
            raise ValueError(f"Invalid five-round verdict at index {index} in {path}")
        first = next((number for number, success in enumerate(rounds, 1) if success), None)
        if (verdict.get("first_success_round") != first
                or verdict.get("success_within_5_rounds") != (first is not None)):
            raise ValueError(f"Inconsistent success summary at index {index} in {path}")
        ab, ba = verdict.get("win_ab"), verdict.get("win_ba")
        expected_credit = ((ab == "A") + (ba == "B")) / 2
        if ab not in {"A", "B"} or ba not in {"A", "B"} or verdict.get("win_credit") != expected_credit:
            raise ValueError(f"Invalid Win-Rate verdict at index {index} in {path}")
        valid.append(verdict)
    if not valid:
        raise ValueError(f"No parsed joint judgments in {path}")
    rounds_or_cap = [item["first_success_round"] or 5 for item in valid]
    successes = sum(item["success_within_5_rounds"] for item in valid)
    summary: dict[str, int | float] = {
        "valid_count": len(valid),
        "success_count": successes,
        "success_percent": successes / len(valid) * 100,
        "win_rate": statistics.fmean(float(item["win_credit"]) for item in valid) * 100,
        "mean_rounds_or_cap5": statistics.fmean(rounds_or_cap),
        "unresolved_cases": unresolved,
    }
    return by_index, summary


def success_summary(rows: dict[int, dict[str, Any]]) -> dict[str, int | float]:
    rounds: list[int] = []
    successes = 0
    unresolved = 0
    for index, row in rows.items():
        judgments = row.get("strict_success_by_round")
        if not isinstance(judgments, list) or [item.get("round") for item in judgments] != [1, 2, 3, 4, 5]:
            raise ValueError(f"Incomplete five-round success judgments at index {index}")
        first = next((item["round"] for item in judgments if item.get("success")), None)
        if first != row.get("first_success_round") or bool(first) != bool(row.get("strict_success_within_5_rounds")):
            raise ValueError(f"Inconsistent success result at index {index}")
        successes += first is not None
        rounds.append(first if first is not None else 5)
        unresolved += sum(bool(item.get("parse_unresolved")) for item in judgments)
    return {
        "success_count": successes,
        "success_percent": successes / len(rows) * 100,
        "mean_rounds_or_cap5": statistics.fmean(rounds),
        "unresolved_success": unresolved,
    }


def completed_win_rate(
    path: Path, *, method: str, judge_model: str, success_rows: dict[int, dict[str, Any]]
) -> float | None:
    if not path.exists():
        if all(row.get("protocol") in WIN_PROTOCOLS
               and row.get("comparison_unit") == "all_five_persuader_replies_vs_single_golden_reply"
               and "win_credit" in row
               for row in success_rows.values()):
            return 100 * statistics.fmean(float(row["win_credit"]) for row in success_rows.values())
        return None
    rows = read_records(path, method=method, judge_model=judge_model)
    for index, row in rows.items():
        if row.get("protocol") not in WIN_PROTOCOLS:
            raise ValueError(f"Old or incompatible Win-Rate protocol in {path} at index {index}")
        if row.get("scenario_id") != success_rows[index].get("scenario_id"):
            raise ValueError(f"Win-Rate and success scenario mismatch at index {index}")
    return 100 * statistics.fmean(float(row["win_credit"]) for row in rows.values())


def source_method(success_dir: str, slug: str, archived_slug: str | None) -> str | None:
    """Prefer a new baseline result, then an archived result for rows 1 and 2."""
    if (RUNS / success_dir / f"{slug}.jsonl").is_file():
        return slug
    if archived_slug and (RUNS / success_dir / f"{archived_slug}.jsonl").is_file():
        return archived_slug
    return None


def model_label(rows: dict[int, dict[str, Any]]) -> str:
    model = str(next(iter(rows.values())).get("model", "")).strip()
    if not model:
        return "-"
    path = Path(model)
    adapter_config = path / "adapter_config.json"
    if adapter_config.is_file():
        config = json.loads(adapter_config.read_text(encoding="utf-8"))
        model = str(config.get("base_model_name_or_path") or model)
    return Path(model).name


def render_section() -> str:
    lines = [
        START_MARKER,
        SECTION,
        "",
        "Completed rows use the same 1,000 held-out contexts and five-round protocol. The Judge models use BF16. `-` means no complete result is available.",
        "",
        "| Judge | Method | Persuader model | N | Strict acceptance within 5 rounds (%) | Win-Rate (vs golden) (%) | Mean rounds to success (failures = 5) | Success parse unresolved |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    all_scenarios: set[str] | None = None
    for judge_label, judge_model, success_dir, win_dir in JUDGES:
        joint_judge = win_dir is None
        for method_index, (slug, display_name, archived_slug) in enumerate(METHODS):
            method = source_method(success_dir, slug, archived_slug)
            judge_text = judge_label if method_index == 0 else ""
            if method is None:
                lines.append(f"| {judge_text} | {display_name} | - | - | - | - | - | - |")
                continue
            success_path = RUNS / success_dir / f"{method}.jsonl"
            if joint_judge:
                success_rows, summary = joint_records_summary(
                    success_path, method=method, judge_model=judge_model
                )
            else:
                success_rows = read_records(
                    success_path, method=method, judge_model=judge_model
                )
                summary = success_summary(success_rows)
            scenarios = {str(row["scenario_id"]) for row in success_rows.values()}
            if all_scenarios is None:
                all_scenarios = scenarios
            elif scenarios != all_scenarios:
                raise ValueError("The baseline evaluations use different scenario sets")
            win_rate = (float(summary["win_rate"]) if joint_judge else completed_win_rate(
                RUNS / win_dir / f"{method}.jsonl",
                method=method,
                judge_model=judge_model,
                success_rows=success_rows,
            ))
            win_text = f"{win_rate:.2f}" if win_rate is not None else "-"
            parsed_count = f"{summary['valid_count']}/1000" if joint_judge else "1000"
            success_fraction = f"{summary['success_count']}/{summary['valid_count']}" if joint_judge else f"{summary['success_count']}/1000"
            lines.append(
                f"| {judge_text} | {display_name} | {model_label(success_rows)} | {parsed_count} | "
                f"{summary['success_percent']:.2f} ({success_fraction}) | "
                f"{win_text} | {summary['mean_rounds_or_cap5']:.3f} | "
                f"{summary['unresolved_cases'] if joint_judge else summary['unresolved_success']} |"
            )
    lines.extend([
        "",
        "The Meta-Llama and Qwen3.5 groups retain their original five-round v1 success-evaluation protocol; unresolved round judgments count as failure. The original PersuGPT evaluation files remain archived separately.",
        "Mean rounds uses the first successful round; a trajectory without success counts as round 5.",
        "Win-Rate (vs golden) compares all five generated Persuader replies with the single held-out golden reply using two order-swapped Judge decisions; a split vote contributes 0.5.",
    ])
    lines.append("A missing Win-Rate requires new five-round Judge evaluations; previous first-turn Win-Rate scores are not reused.")
    lines.append("Gemma 4 jointly judges five-round success and Win-Rate using prompt revision v11. Its success rate, Win-Rate, and mean rounds use only parsed cases; the N and unresolved columns show parsed and unresolved counts out of 1,000. Up to three additional generations are made after a parse or evidence validation error.")
    lines.extend([END_MARKER, ""])
    return "\n".join(lines)


def replace_section(original: str, section: str) -> str:
    for start_marker, end_marker in ((START_MARKER, END_MARKER), (OLD_START_MARKER, OLD_END_MARKER)):
        if start_marker in original and end_marker in original:
            start = original.index(start_marker)
            end = original.index(end_marker, start) + len(end_marker)
            return original[:start] + section.rstrip() + original[end:]
    if SECTION in original:
        start = original.index(SECTION)
        next_heading = original.find("\n## ", start + len(SECTION))
        end = len(original) if next_heading == -1 else next_heading + 1
        return original[:start].rstrip() + "\n\n" + section + original[end:]
    return original.rstrip() + "\n\n" + section


def main() -> None:
    original = TABLE.read_text(encoding="utf-8") if TABLE.exists() else ""
    updated = replace_section(original, render_section()).strip() + "\n"
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".main_experiment_table.", suffix=".tmp", dir=TABLE.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, TABLE)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(TABLE)


if __name__ == "__main__":
    main()
