#!/usr/bin/env python3
"""Render the Qwen2.5 five-round section from completed Judge records."""

from __future__ import annotations

import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

from repro.constants import ROOT


RUNS = ROOT / "runs/qwen25_dialogue_sft"
TABLE = ROOT / "results/main_experiment_table.md"
SECTION = "## Qwen2.5 five-round evaluation"
START_MARKER = "<!-- qwen25-five-rounds:start -->"
END_MARKER = "<!-- qwen25-five-rounds:end -->"
WIN_PROTOCOL = "qwen25_all_five_persuader_replies_vs_golden_bf16_v1"
METHODS = (
    ("qwen25_zero_shot", "Qwen2.5-7B-Instruct zero-shot"),
    ("qwen25_dialogue_sft", "Qwen2.5-7B-Instruct Dialogue-SFT"),
)
JUDGES = (
    (
        "PersuGPT (BF16)",
        "/data1/liujianjian/PersuGPT",
        "evaluation_persugpt",
        "evaluation_all_5_rounds_persugpt",
    ),
    (
        "Qwen3.5-9B (BF16)",
        "/data1/liujianjian/model/Qwen3.5-9B",
        "evaluation_qwen3_5_9B",
        "evaluation_all_5_rounds_qwen3_5_9B",
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
        return None
    rows = read_records(path, method=method, judge_model=judge_model)
    for index, row in rows.items():
        if row.get("protocol") != WIN_PROTOCOL:
            raise ValueError(f"Old or incompatible Win-Rate protocol in {path} at index {index}")
        if row.get("scenario_id") != success_rows[index].get("scenario_id"):
            raise ValueError(f"Win-Rate and success scenario mismatch at index {index}")
    return 100 * statistics.fmean(float(row["win_credit"]) for row in rows.values())


def render_section() -> str:
    lines = [
        START_MARKER,
        SECTION,
        "",
        "Each row covers 1,000 held-out five-round trajectories. The Judge models use BF16.",
        "",
        "| Judge | Method | N | Strict acceptance within 5 rounds (%) | Win-Rate (vs golden) (%) | Mean rounds to success (failures = 5) | Success parse unresolved |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    all_scenarios: set[str] | None = None
    pending = False
    for judge_label, judge_model, success_dir, win_dir in JUDGES:
        for method_index, (method, display_name) in enumerate(METHODS):
            success_rows = read_records(
                RUNS / success_dir / f"{method}.jsonl", method=method, judge_model=judge_model
            )
            scenarios = {str(row["scenario_id"]) for row in success_rows.values()}
            if all_scenarios is None:
                all_scenarios = scenarios
            elif scenarios != all_scenarios:
                raise ValueError("The four success evaluations use different scenario sets")
            summary = success_summary(success_rows)
            win_rate = completed_win_rate(
                RUNS / win_dir / f"{method}.jsonl",
                method=method,
                judge_model=judge_model,
                success_rows=success_rows,
            )
            pending |= win_rate is None
            win_text = f"{win_rate:.2f}" if win_rate is not None else "pending"
            judge_text = judge_label if method_index == 0 else ""
            lines.append(
                f"| {judge_text} | {display_name} | 1000 | "
                f"{summary['success_percent']:.2f} ({summary['success_count']}/1000) | "
                f"{win_text} | {summary['mean_rounds_or_cap5']:.3f} | "
                f"{summary['unresolved_success']} |"
            )
    lines.extend([
        "",
        "Strict success requires the persuadee to explicitly and unconditionally accept the exact goal in any of five rounds. Unparseable success judgments count as failure.",
        "Mean rounds uses the first successful round; a trajectory without success counts as round 5.",
        "Win-Rate (vs golden) compares all five generated Persuader replies with the single held-out golden reply using two order-swapped Judge decisions; a split vote contributes 0.5.",
    ])
    if pending:
        lines.append("Pending Win-Rate cells require new Judge evaluations; previous first-turn Win-Rate scores are not reused.")
    lines.extend([END_MARKER, ""])
    return "\n".join(lines)


def replace_section(original: str, section: str) -> str:
    if START_MARKER in original and END_MARKER in original:
        start = original.index(START_MARKER)
        end = original.index(END_MARKER, start) + len(END_MARKER)
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
