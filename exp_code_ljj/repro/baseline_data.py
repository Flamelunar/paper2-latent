"""Check the fixed, released paper10k splits before baseline 3 work."""

from __future__ import annotations

import hashlib
import json
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator

from .constants import DATA_DIR


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL {path}:{number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object in {path}:{number}")
                yield row


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_fixed_splits() -> dict[str, Any]:
    """Require exact copied files and aligned dialogue/ISR sessions with no leakage."""
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    names = ("dialogue_sft", "isr_sft", "user_model_sft",
             "preference_seeds", "evaluation_points")
    paths = {name: DATA_DIR / f"{name}.jsonl" for name in names}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = manifest["files"][name]
        actual_hash = sha256(path)
        if actual_hash != expected["sha256"]:
            raise ValueError(f"Changed fixed split {name}: {actual_hash} != {expected['sha256']}")

    training_ids: set[str] = set()
    count = 0
    for count, (dialogue, isr) in enumerate(
        zip_longest(rows(paths["dialogue_sft"]), rows(paths["isr_sft"])), 1
    ):
        if dialogue is None or isr is None:
            raise ValueError("Dialogue-SFT and ISR-SFT session counts differ")
        a, b = dialogue["metadata"], isr["metadata"]
        key = ("scenario_id", "source_id", "session_id", "persuader_turns")
        if tuple(a[k] for k in key) != tuple(b[k] for k in key):
            raise ValueError(f"Dialogue and ISR session mismatch at row {count}")
        if not str(isr["output"]).startswith("Intent to Strategy Reasoning:"):
            raise ValueError(f"Missing ISR target at row {count}")
        training_ids.add(str(a["scenario_id"]))
    if count != manifest["files"]["dialogue_sft"]["rows"]:
        raise ValueError("Unexpected training row count")

    preference = list(rows(paths["preference_seeds"]))
    evaluation = list(rows(paths["evaluation_points"]))
    if len(preference) != manifest["files"]["preference_seeds"]["rows"]:
        raise ValueError("Unexpected preference seed count")
    if len(evaluation) != manifest["files"]["evaluation_points"]["rows"]:
        raise ValueError("Unexpected held-out test count")
    pref_ids = {str(row["scenario_id"]) for row in preference}
    test_ids = {str(row["scenario_id"]) for row in evaluation}
    if len(training_ids) != 10000 or len(pref_ids) != 2000 or len(test_ids) != 1000:
        raise ValueError("Duplicate or incomplete scenario split")
    if training_ids & pref_ids or training_ids & test_ids or pref_ids & test_ids:
        raise ValueError("Training, preference, and test scenarios overlap")
    return {"sft_sessions": count, "sft_scenarios": len(training_ids),
            "preference_scenarios": len(pref_ids), "test_scenarios": len(test_ids),
            "sha256": {name: manifest["files"][name]["sha256"] for name in names}}
