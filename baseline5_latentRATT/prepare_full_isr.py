#!/usr/bin/env python3
"""Validate and record the complete ISR training corpus for LatentRATT."""
from __future__ import annotations

import json
from pathlib import Path

from persuasion_data import ISR_FILE, TEST_FILE, validate_data
from repro.baseline_data import sha256

EXPECTED_ROWS = 59_868
EXPECTED_TRAIN_SCENARIOS = 10_000
OUT = Path(__file__).resolve().parent / "data/isr_all_59868.manifest.json"


def main() -> None:
    summary = validate_data()
    train_scenarios: set[str] = set()
    train_rows = 0
    with ISR_FILE.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            train_scenarios.add(str(row["metadata"]["scenario_id"]))
            train_rows += 1

    test_scenarios = {
        str(json.loads(line)["scenario_id"])
        for line in TEST_FILE.open(encoding="utf-8")
    }
    overlap = train_scenarios & test_scenarios
    if train_rows != EXPECTED_ROWS:
        raise ValueError(f"Expected all {EXPECTED_ROWS} ISR sessions; found {train_rows}")
    if len(train_scenarios) != EXPECTED_TRAIN_SCENARIOS:
        raise ValueError(f"Expected {EXPECTED_TRAIN_SCENARIOS} train scenarios; found {len(train_scenarios)}")
    if overlap:
        raise ValueError(f"Training/test scenario overlap: {sorted(overlap)[:5]}")
    if summary["sft_sessions"] != train_rows or summary["sft_scenarios"] != len(train_scenarios):
        raise ValueError("Fixed-split validator disagrees with the complete ISR corpus")

    manifest = {
        "source_file": str(ISR_FILE),
        "source_sha256": sha256(ISR_FILE),
        "training_rows": train_rows,
        "training_scenarios": len(train_scenarios),
        "training_mode": "all ISR sessions; no per-scenario sampling",
        "preference_scenarios_used": 0,
        "dpo_used": False,
        "test_file": str(TEST_FILE),
        "test_scenarios": len(test_scenarios),
        "training_test_overlap": len(overlap),
        "fixed_split_summary": summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(OUT.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
