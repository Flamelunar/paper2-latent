#!/usr/bin/env python3
"""Update only the Baseline 3 / Gemma row after validating full results."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from finalize_main_table import joint_records_summary, model_label
from repro.constants import ROOT


METHOD = "baseline3_isr_simdpo"
JUDGE_MODEL = "/app/hzj/models/gemma-4-E4B-it"
SUCCESS_DIR = "success_judgments/gemma4"
SUCCESS_PATH = ROOT / "artifacts" / SUCCESS_DIR / f"{METHOD}.jsonl"
TABLE = ROOT / "results/main_experiment_table.md"


def main() -> None:
    records, summary = joint_records_summary(
        SUCCESS_PATH,
        method=METHOD,
        judge_model=JUDGE_MODEL,
        success_dir=SUCCESS_DIR,
    )
    output_row = (
        f"|  | Baseline 3: ISR-SFT + SimDPO | {model_label(records)} | "
        f"{summary['valid_count']}/1000 | "
        f"{summary['success_percent']:.2f} "
        f"({summary['success_count']}/{summary['valid_count']}) | "
        f"{summary['win_rate']:.2f} | "
        f"{summary['mean_rounds_or_cap5']:.3f} | "
        f"{summary['unresolved_cases']} |"
    )

    lines = TABLE.read_text(encoding="utf-8").splitlines()
    judge_index = next(
        (i for i, line in enumerate(lines)
         if line.startswith("| Gemma 4 E4B-it (BF16) |")),
        None,
    )
    if judge_index is None:
        raise ValueError("Could not locate the existing Gemma 4 BF16 section")
    row_index = next(
        (i for i in range(judge_index + 1, len(lines))
         if lines[i].startswith("|  | Baseline 3: ISR-SFT + SimDPO |")),
        None,
    )
    if row_index is None:
        raise ValueError("Could not locate the Baseline 3 row under Gemma 4")
    lines[row_index] = output_row
    updated = "\n".join(lines) + "\n"

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".main_experiment_table.baseline3.", suffix=".tmp", dir=TABLE.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, TABLE)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Updated only the Baseline 3 Gemma 4 row: {TABLE}")


if __name__ == "__main__":
    main()
