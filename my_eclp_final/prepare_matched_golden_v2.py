#!/usr/bin/env python3
"""Recover true held-out golden continuations without extending/padding them."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from matched_eval_protocol_v2 import validate_available_dialogue


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scenario_id(item: dict) -> str:
    keys = ("tag", "background", "persuadee", "persuader", "goal", "domain")
    text = json.dumps({k: item["scenario"].get(k) for k in keys}, ensure_ascii=False,
                      sort_keys=True, separators=(",", ":"))
    return "dp_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def recover(points: list[dict], items: list[dict]) -> tuple[list[dict], dict]:
    lookup = collections.defaultdict(list)
    for item in items:
        lookup[scenario_id(item)].append(item)
    result = []
    tail_counts = collections.Counter()
    for index, point in enumerate(points):
        matches = []
        start = 2 * (point["target_turn"] - 1)
        for item in lookup[point["scenario_id"]]:
            if item["scenario"] != point["scenario"]:
                continue
            session = item.get("dialog", {}).get(point["session_id"])
            if not isinstance(session, list) or len(session) <= start:
                continue
            turns = [{"role": t["role"], "content": str(t["response"]).strip()} for t in session]
            if turns[:start] == point["history"] and turns[start]["content"] == point["reference"]:
                matches.append(turns[start:])
        if not matches or any(m != matches[0] for m in matches[1:]):
            raise ValueError(f"Missing or ambiguous exact golden session at index {index}")
        tail = matches[0]
        validate_available_dialogue(tail)
        tail_counts[len(tail) // 2] += 1
        result.append({
            "index": index, "scenario_id": point["scenario_id"],
            "source_id": point["source_id"], "session_id": point["session_id"],
            "target_turn": point["target_turn"], "scenario": point["scenario"],
            "initial_history": point["history"], "golden_first": point["reference"],
            "golden_trajectory": tail,
            "golden_remaining_complete_rounds": len(tail) // 2,
            "golden_persuader_turns": (len(tail) + 1) // 2,
            "golden_persuadee_turns": len(tail) // 2,
            "golden_ends_with_persuadee": tail[-1]["role"] == "persuadee",
        })
    length_groups = collections.Counter(str(row["golden_persuader_turns"]) for row in result)
    return result, {"n": len(result), "win1_eligible": len(result),
                    "win_full_eligible": len(result),
                    "golden_persuader_turns_distribution": dict(sorted(length_groups.items())),
                    "golden_ends_after_persuader_count": sum(not r["golden_ends_with_persuadee"] for r in result),
                    "golden_remaining_rounds_distribution": dict(sorted(tail_counts.items())),
                    "matching": "exact_public_scenario_session_history_reference",
                    "padding_or_synthetic_golden": False,
                    "golden_persuadee_source": "original_released_dialogue; differs from current simulated_persuadee"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    points = [json.loads(l) for l in args.points.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows, manifest = recover(points, json.loads(args.source.read_text(encoding="utf-8")))
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "golden.jsonl"
    content = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    if target.exists() and target.read_text(encoding="utf-8") != content:
        raise ValueError("Existing golden recovery differs; use a new output directory")
    target.write_text(content, encoding="utf-8")
    manifest.update(source=str(args.source), source_sha256=digest(args.source),
                    points=str(args.points), points_sha256=digest(args.points),
                    golden_sha256=digest(target))
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
