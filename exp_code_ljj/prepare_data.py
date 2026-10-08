#!/usr/bin/env python3
"""Rebuild the exact Dialogue-SFT and test formats from already split source JSON."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from repro.constants import DATA_DIR, SFT_SCENARIOS, TEST_SCENARIOS
from repro.data import (
    build_eval_point,
    build_persuader_rows,
    load_json,
    public_scenario_key,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-source", type=Path, required=True,
                        help="The original pre-split sft.json containing 10,000 scenarios")
    parser.add_argument("--test-source", type=Path, required=True,
                        help="The original pre-split test.json containing 1,000 scenarios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Replace the copied prepared JSONL files")
    args = parser.parse_args()
    train_path = DATA_DIR / "dialogue_sft.jsonl"
    test_path = DATA_DIR / "evaluation_points.jsonl"
    if not args.force and (train_path.exists() or test_path.exists()):
        raise SystemExit("Prepared data already exists; use --force only to rebuild intentionally")
    train = load_json(args.sft_source)
    test = load_json(args.test_source)
    if len(train) != SFT_SCENARIOS or len(test) != TEST_SCENARIOS:
        raise ValueError("Expected 10,000 training and 1,000 test scenarios")
    train_keys = {public_scenario_key(item) for item in train}
    test_keys = {public_scenario_key(item) for item in test}
    if len(train_keys) != len(train) or len(test_keys) != len(test) or train_keys & test_keys:
        raise ValueError("Duplicate or overlapping scenario content in source splits")
    train_count, train_hash = write_jsonl(train_path, build_persuader_rows(train))
    rng = random.Random(args.seed)
    test_count, test_hash = write_jsonl(
        test_path, (build_eval_point(item, rng) for item in test)
    )
    manifest = {
        "seed": args.seed,
        "profile": "paper10k",
        "scenario_split": {"sft": len(train), "test": len(test)},
        "source_files": {"sft": str(args.sft_source), "test": str(args.test_source)},
        "files": {
            "dialogue_sft": {"path": str(train_path), "rows": train_count, "sha256": train_hash},
            "evaluation_points": {"path": str(test_path), "rows": test_count, "sha256": test_hash},
        },
    }
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["files"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
