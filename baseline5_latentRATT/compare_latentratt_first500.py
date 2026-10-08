#!/usr/bin/env python3
"""Compare fixed-seed LatentRATT first-500 judgments with paired bootstrap CIs."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp_code_ljj"
JUDGMENTS = EXP / "artifacts/success_judgments"
VERSIONS = {
    "v0 SFT": "latentRATT_v0_paired_seed42",
    "v1 outcome-RL": "latentRATT_v1_outcome",
    "v2 dual-slot": "latentRATT_v2_dualslot",
}
JUDGES = {
    "Selene V17 v7": "selene_v17",
    "Gemma V11": "gemma_v11",
}


def load_judgments(slug: str, judge: str, prompt: str) -> dict[int, dict]:
    path = JUDGMENTS / f"{slug}_{prompt}" / f"{slug}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {judge} judgments: {path}")
    rows: dict[int, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            index = item.get("index")
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 500:
                raise ValueError(f"Invalid test index at {path}:{line_number}: {index!r}")
            if index in rows:
                raise ValueError(f"Duplicate test index {index} in {path}")
            rows[index] = item
    if set(rows) != set(range(500)):
        missing = sorted(set(range(500)) - set(rows))
        raise ValueError(f"Expected all 500 judgments in {path}; missing {missing[:10]}")
    return rows


def metrics(rows: dict[int, dict]) -> dict[int, dict[str, float | None]]:
    result = {}
    for index, row in rows.items():
        verdict = row.get("verdict") or {}
        success = verdict.get("success_within_5_rounds")
        win = verdict.get("win_credit")
        result[index] = {
            "scenario_id": row.get("scenario_id"),
            "success": float(success) if isinstance(success, bool) else None,
            "win": float(win) if isinstance(win, (int, float)) and math.isfinite(win) else None,
        }
    return result


def paired_bootstrap(
    baseline: dict[int, dict],
    candidate: dict[int, dict],
    metric: str,
    *,
    seed: int,
    iterations: int = 10_000,
) -> tuple[int, float | None, float | None, float | None]:
    deltas = [
        float(candidate[index][metric]) - float(baseline[index][metric])
        for index in range(500)
        if baseline[index][metric] is not None and candidate[index][metric] is not None
    ]
    if not deltas:
        return 0, None, None, None
    rng = random.Random(seed)
    means = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(iterations)
    )
    return (
        len(deltas),
        sum(deltas) / len(deltas),
        means[int(0.025 * iterations)],
        means[int(0.975 * iterations)],
    )


def percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}%"


def delta_text(values: tuple[int, float | None, float | None, float | None]) -> str:
    count, mean, low, high = values
    if mean is None:
        return f"- (n={count})"
    return f"{100 * mean:+.2f} pp [{100 * low:+.2f}, {100 * high:+.2f}] (n={count})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=EXP / "results/latentRATT_first500_paired_seed42.md",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    args = parser.parse_args()
    if args.bootstrap_iterations < 1:
        parser.error("--bootstrap-iterations must be positive")

    loaded = {
        (judge, version): metrics(load_judgments(slug, judge, prompt))
        for judge, prompt in JUDGES.items()
        for version, slug in VERSIONS.items()
    }
    for judge in JUDGES:
        reference = loaded[(judge, "v0 SFT")]
        for version in VERSIONS:
            candidate = loaded[(judge, version)]
            if any(reference[i]["scenario_id"] != candidate[i]["scenario_id"] for i in range(500)):
                raise ValueError(f"Scenario indices do not align for {judge}: v0 vs {version}")

    lines = [
        "# LatentRATT first-500 paired comparison",
        "",
        "Fixed evaluation indices 0-499; generation seed 42 is reset by test case and dialogue turn. "
        "Selene uses V17 v7 and Gemma uses V11. Bootstrap samples are test contexts, with 10,000 "
        "iterations and percentile 95% confidence intervals. No results are written to the main table.",
        "",
        "| Judge | Version | Success n/500 | Success | Win n/500 | Win-Rate | Unresolved |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for judge in JUDGES:
        for version in VERSIONS:
            values = loaded[(judge, version)]
            success = [v["success"] for v in values.values() if v["success"] is not None]
            wins = [v["win"] for v in values.values() if v["win"] is not None]
            unresolved = sum(v["success"] is None or v["win"] is None for v in values.values())
            lines.append(
                f"| {judge} | {version} | {len(success)}/500 | "
                f"{percent(sum(success) / len(success) if success else None)} | "
                f"{len(wins)}/500 | {percent(sum(wins) / len(wins) if wins else None)} | "
                f"{unresolved} |"
            )

    lines.extend([
        "",
        "## Paired Differences",
        "",
        "Positive deltas favor the first named candidate; percentage-point differences use cases "
        "with valid judgments from both versions for that metric.",
        "",
        "| Judge | Comparison | Metric | Delta (95% CI) |",
        "| --- | --- | --- | ---: |",
    ])
    comparisons = (("v1 outcome-RL", "v0 SFT"), ("v2 dual-slot", "v0 SFT"),
                   ("v2 dual-slot", "v1 outcome-RL"))
    for judge_index, judge in enumerate(JUDGES):
        for left_index, (candidate_name, baseline_name) in enumerate(comparisons):
            for metric_index, metric in enumerate(("success", "win")):
                result = paired_bootstrap(
                    loaded[(judge, baseline_name)], loaded[(judge, candidate_name)], metric,
                    seed=42 + judge_index * 100 + left_index * 10 + metric_index,
                    iterations=args.bootstrap_iterations,
                )
                label = "Success" if metric == "success" else "Win-Rate"
                lines.append(
                    f"| {judge} | {candidate_name} vs {baseline_name} | {label} | "
                    f"{delta_text(result)} |"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
