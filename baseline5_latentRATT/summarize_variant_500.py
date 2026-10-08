#!/usr/bin/env python3
"""Summarize paired first-500 results against the existing v0 judgments."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "exp_code_ljj"
V0_JUDGMENTS = {
    "selene1": EXP / "artifacts/success_judgments/selene_v17_v7_latentRATT_20260928/latentRATT_persuasion.jsonl",
    "gemma4": EXP / "artifacts/success_judgments/gemma4/latentRATT_persuasion.jsonl",
}


def read_rows(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                rows[int(item["index"])] = item
    return rows


def values(rows: dict[int, dict], end: int = 500) -> dict[int, dict[str, float | None]]:
    result = {}
    for index in range(end):
        row = rows.get(index)
        verdict = (row or {}).get("verdict") or {}
        result[index] = {
            "success": verdict.get("success_within_5_rounds"),
            "win": verdict.get("win_credit"),
            "parse_error": (row or {}).get("parse_error"),
        }
    return result


def rate(items: list[float]) -> float | None:
    return sum(items) / len(items) if items else None


def paired_bootstrap(old: dict[int, dict], new: dict[int, dict], metric: str,
                     *, seed: int, iterations: int = 10_000):
    pairs = [
        (float(old[i][metric]), float(new[i][metric]))
        for i in range(500)
        if old[i][metric] is not None and new[i][metric] is not None
    ]
    if not pairs:
        return 0, None, None, None
    deltas = [b - a for a, b in pairs]
    generator = random.Random(seed)
    means = []
    for _ in range(iterations):
        means.append(sum(deltas[generator.randrange(len(deltas))] for _ in deltas) / len(deltas))
    means.sort()
    return len(pairs), sum(deltas) / len(deltas), means[int(0.025 * iterations)], means[int(0.975 * iterations)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-slug", required=True)
    parser.add_argument("--selene-dir", type=Path, required=True)
    parser.add_argument("--gemma-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    new_rows = {
        "selene1": read_rows(args.selene_dir / f"{args.method_slug}.jsonl"),
        "gemma4": read_rows(args.gemma_dir / f"{args.method_slug}.jsonl"),
    }
    lines = [f"# {args.method_slug}: first-500 evaluation", "",
             "Compared with the earlier v0 judgment records on fixed test indices 0-499. Success and Win-Rate use parsed cases; unresolved parses are reported. Paired bootstrap samples test contexts (10,000 iterations). The earlier v0 generations did not use the paired seed; use `latentRATT_first500_paired_seed42.md` for the controlled v0/v1/v2 comparison.", "",
             "| Judge | Version | N success | Success | N win | Win-Rate | Unresolved | Paired success delta (95% CI) | Paired win delta (95% CI) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for judge, label in (("selene1", "Selene V17 v7"), ("gemma4", "Gemma V11")):
        old = values(read_rows(V0_JUDGMENTS[judge]))
        new = values(new_rows[judge])
        for version, current in (("v0", old), (args.method_slug, new)):
            success = [float(row["success"]) for row in current.values() if row["success"] is not None]
            wins = [float(row["win"]) for row in current.values() if row["win"] is not None]
            unresolved = sum(row["success"] is None or row["win"] is None for row in current.values())
            lines.append(
                f"| {label} | {version} | {len(success)} | "
                f"{100 * rate(success):.2f}% | {len(wins)} | {100 * rate(wins):.2f}% | "
                f"{unresolved} | - | - |"
            )
        success_n, success_delta, success_low, success_high = paired_bootstrap(
            old, new, "success", seed=51,
        )
        win_n, win_delta, win_low, win_high = paired_bootstrap(
            old, new, "win", seed=52,
        )
        def format_delta(n, mean, low, high):
            if mean is None:
                return "-"
            return f"{100*mean:+.2f} pp [{100*low:+.2f}, {100*high:+.2f}] (n={n})"
        lines.append(
            f"| {label} | paired delta | - | - | - | - | - | "
            f"{format_delta(success_n, success_delta, success_low, success_high)} | "
            f"{format_delta(win_n, win_delta, win_low, win_high)} |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
