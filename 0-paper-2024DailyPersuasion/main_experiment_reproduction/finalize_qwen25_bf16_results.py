#!/usr/bin/env python3
"""Validate and combine BF16 Qwen2.5 five-round evaluation outputs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from repro.constants import ROOT

RUNS = ROOT / "runs/qwen25_dialogue_sft"
WIN_PROTOCOL = "qwen25_all_five_persuader_replies_vs_golden_bf16_v1"
METHODS = {
    "qwen25_zero_shot": "Qwen2.5-7B-Instruct zero-shot",
    "qwen25_dialogue_sft": "Qwen2.5-7B-Instruct Dialogue-SFT",
}
JUDGES = {
    "Qwen3.5-9B": {
        "model": "/data1/liujianjian/model/Qwen3.5-9B",
        "zero_shot": RUNS / "evaluation_reuse_zero_shot/qwen25_zero_shot.jsonl",
        "dialogue_sft": RUNS / "evaluation_final_bf16_qwen35/qwen25_dialogue_sft.jsonl",
    },
    "PersuGPT": {
        "model": "/data1/liujianjian/PersuGPT",
        "zero_shot": RUNS / "evaluation_persugpt/qwen25_zero_shot.jsonl",
        "dialogue_sft": RUNS / "evaluation_final_bf16_persugpt/qwen25_dialogue_sft.jsonl",
    },
}


def load_and_validate(path: Path, *, method: str, judge: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1000:
        raise ValueError(f"{path}: expected 1,000 rows, found {len(rows)}")
    if {int(row["index"]) for row in rows} != set(range(1000)):
        raise ValueError(f"{path}: indices are missing or duplicated")

    for row in rows:
        if row.get("method") != method or row.get("judge_model") != judge:
            raise ValueError(f"{path}: unexpected method or judge at index {row.get('index')}")
        if int(row.get("generated_rounds", -1)) != 5:
            raise ValueError(f"{path}: expected five generated rounds at index {row.get('index')}")
        judgments = row.get("strict_success_by_round")
        if not isinstance(judgments, list) or [item.get("round") for item in judgments] != [1, 2, 3, 4, 5]:
            raise ValueError(f"{path}: incomplete round judgments at index {row.get('index')}")
        first = next((int(item["round"]) for item in judgments if item.get("success")), None)
        if row.get("first_success_round") != first:
            raise ValueError(f"{path}: inconsistent first success round at index {row.get('index')}")
        if bool(row.get("strict_success_within_5_rounds")) != bool(first):
            raise ValueError(f"{path}: inconsistent five-round success at index {row.get('index')}")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["first_success_round"] is not None]
    capped_rounds = [int(row["first_success_round"] or 5) for row in rows]
    protocols = {row["protocol"] for row in rows}
    if len(protocols) != 1:
        raise ValueError("A method mixes incompatible evaluation protocols")
    all_reply_win_rate = (
        100 * statistics.fmean(float(row["win_credit"]) for row in rows)
        if protocols == {WIN_PROTOCOL} else None
    )
    by_round = {
        str(round_number): 100 * statistics.fmean(
            bool(row["strict_success_by_round"][round_number - 1]["success"]) for row in rows
        )
        for round_number in range(1, 6)
    }
    return {
        "n": len(rows),
        "win_rate_percent": all_reply_win_rate,
        "strict_success_within_5_rounds_percent": 100 * len(successful) / len(rows),
        "strict_success_count": len(successful),
        "mean_first_success_round_among_successes": (
            statistics.fmean(int(row["first_success_round"]) for row in successful) if successful else None
        ),
        "mean_rounds_to_success_or_cap5": statistics.fmean(capped_rounds),
        "strict_success_rate_by_round_percent": by_round,
        "unresolved_pairwise_judgments": sum(
            sum(bool(item) for item in row.get("pairwise_unresolved", [])) for row in rows
        ),
        "unresolved_success_judgments": sum(
            bool(item.get("parse_unresolved"))
            for row in rows for item in row["strict_success_by_round"]
        ),
        "protocols": sorted(protocols),
        "judge_batch_sizes": sorted({int(row["judge_batch_size"]) for row in rows}),
    }


def main() -> None:
    generation_config_path = RUNS / "generation_final_bf16/run_config.json"
    generation_config = json.loads(generation_config_path.read_text(encoding="utf-8"))
    if generation_config.get("bf16_autocast") is not True:
        raise ValueError("Final SFT generation must record BF16 autocast")
    if generation_config.get("persuader_compute_dtype") != "bfloat16" or generation_config.get("persuadee_compute_dtype") != "bfloat16":
        raise ValueError("Final SFT generation must use BF16 for both models")
    if int(generation_config.get("executed_contexts", 0)) != 1000 or generation_config.get("rounds") != 5:
        raise ValueError("Final SFT generation must cover 1,000 five-round test contexts")

    summaries: dict[str, dict[str, Any]] = {}
    scenario_sets: list[set[str]] = []
    for judge_name, config in JUDGES.items():
        for method, label in METHODS.items():
            source = config["zero_shot"] if method == "qwen25_zero_shot" else config["dialogue_sft"]
            rows = load_and_validate(source, method=method, judge=config["model"])
            scenario_sets.append({str(row["scenario_id"]) for row in rows})
            summaries[f"{judge_name} | {label}"] = summarize(rows)
    if any(scenarios != scenario_sets[0] for scenarios in scenario_sets[1:]):
        raise ValueError("The four evaluations do not cover the same held-out scenarios")

    lines = [
        "# Qwen2.5-7B Dialogue-SFT: BF16 five-round automatic evaluation",
        "",
        "All rows cover the same 1,000 held-out contexts. The Dialogue-SFT trajectories were generated for five rounds with BF16 autocast; both local Judges use BF16.",
        "",
        "| Judge | Persuader | N | Strict acceptance within 5 rounds (%) | Win-Rate (vs golden) (%) | Mean rounds to success (failures = 5) | Success parse unresolved |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, (key, item) in enumerate(summaries.items()):
        judge, method = key.split(" | ", 1)
        judge_text = judge if index % len(METHODS) == 0 else ""
        win = item["win_rate_percent"]
        win_text = f"{win:.2f}" if win is not None else "pending"
        lines.append(
            f"| {judge_text} | {method} | {item['n']} | "
            f"{item['strict_success_within_5_rounds_percent']:.2f} "
            f"({item['strict_success_count']}/1000) | {win_text} | "
            f"{item['mean_rounds_to_success_or_cap5']:.3f} | "
            f"{item['unresolved_success_judgments']} |"
        )
    lines.extend([
        "",
        "## Protocol notes",
        "",
        "- Win-Rate (vs golden) compares all five generated Persuader replies with the one held-out golden reply; each local Judge scores twice with response order swapped and split votes receive 0.5 credit.",
        "- A pending Win-Rate means that method only has an earlier first-turn comparison. Its old score is not relabeled as an all-reply score.",
        "- Strict success is an additional automatic measure: the persuadee must explicitly and unconditionally accept, commit to, or act toward the exact goal within five rounds.",
        "- Mean rounds counts the first successful round, or 5 when no success occurs.",
        "- The five-round cap is this reproduction's setting; the paper does not specify a maximum dialogue length.",
        "- The paper uses GPT-4+ISR target replies and ChatGPT as Judge. This run uses held-out target replies and local Qwen3.5-9B and PersuGPT Judges, so scores are not directly comparable to the published Table 2 values.",
        "- Human evaluation is omitted.",
    ])

    output = ROOT / "results/qwen25_dialogue_sft_table2_bf16_final.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "generation_config": str(generation_config_path),
        "generation_compute_dtype": "bfloat16",
        "bf16_autocast": True,
        "rounds": 5,
        "test_contexts": 1000,
        "evaluations": summaries,
    }
    output.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
