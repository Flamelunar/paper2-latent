#!/usr/bin/env python3
"""Legacy first-turn Win-Rate and ROUGE-L evaluator for the paper-style comparison."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from repro.profile import add_profile_argument, configure_profile_from_argv

PROFILE = configure_profile_from_argv()
from repro.constants import ROOT, TEST_SCENARIOS
from repro.persuadee_simulation import PersuaderMethod
from repro.simulation import LocalTransformersEndpoint, generation_endpoint, read_jsonl
from repro.trajectory_evaluation import judge_prompt, parse_winner, rouge_l_character

GENERATION_DIR = ROOT / "runs/qwen25_dialogue_sft/generation_5rounds"
REUSE_DIR = ROOT / "runs/qwen25_dialogue_sft/evaluation_qwen3_5_9B"
OUTPUT_DIR = ROOT / "runs/qwen25_dialogue_sft/evaluation_qwen3_5_9B_table2"
JUDGE_MODEL = "/data1/liujianjian/model/Qwen3.5-9B"
PROTOCOL = "qwen25_table2_winrate_rougel_bf16_qwen35_v1"
METHODS = (
    PersuaderMethod(
        slug="qwen25_zero_shot",
        display_name="Qwen2.5-7B-Instruct zero-shot",
        model_path="/data1/liujianjian/model/Qwen2.5-7B-Instruct",
        with_isr=False,
    ),
    PersuaderMethod(
        slug="qwen25_dialogue_sft",
        display_name="Qwen2.5-7B-Instruct Dialogue-SFT",
        model_path="/data1/liujianjian/Persuader/qwen2.5-7b-instruct-dialogue-sft-paper10k",
        with_isr=False,
    ),
)


def load_generation(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    indexes = [int(row.get("index", -1)) for row in rows]
    if len(rows) < expected_rows or set(indexes) != set(range(len(rows))) or len(set(indexes)) != len(indexes):
        raise ValueError(f"Expected consecutive, unique rows with at least {expected_rows} entries in {path}")
    for row in rows:
        trajectory = row.get("trajectory")
        if not isinstance(trajectory, list) or len(trajectory) != 10:
            raise ValueError(f"Expected five dialogue rounds at index {row.get('index')}")
        roles = [turn.get("role") for turn in trajectory]
        if roles != ["persuader", "persuadee"] * 5:
            raise ValueError(f"Invalid role sequence at index {row.get('index')}: {roles}")
        if not str(row.get("reference", "")).strip():
            raise ValueError(f"Missing first-turn reference at index {row.get('index')}")
    return sorted(rows, key=lambda row: int(row["index"]))[:expected_rows]


def resolve_winner(
    raw: str,
    response_a: str,
    response_b: str,
    judge: LocalTransformersEndpoint,
) -> tuple[str | None, list[str]]:
    retries: list[str] = []
    try:
        return parse_winner(raw), retries
    except ValueError:
        clarification = (
            "Your previous selection was:\n"
            f"{raw}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}\n\n"
            "Which response label matches your previous selection? "
            "Reply with exactly one letter: A or B."
        )
    for attempt in range(2):
        retry = judge.complete_batch([{
            "system": "Identify the label of the selected response.",
            "user": clarification,
            "temperature": 0.0,
            "max_tokens": 64,
        }])[0]
        retries.append(retry)
        try:
            return parse_winner(retry), retries
        except ValueError:
            if attempt == 0:
                clarification = (
                    f"Response A:\n{response_a}\n\nResponse B:\n{response_b}\n\n"
                    "Choose the more persuasive response. Return only A or B."
                )
    return None, retries


def score_pairwise_batch(
    rows: list[dict[str, Any]], judge: LocalTransformersEndpoint
) -> list[dict[str, Any]]:
    system = "You are a careful, order-blind DailyPersuasion judge."
    requests: list[dict[str, Any]] = []
    response_pairs: list[tuple[str, str]] = []
    for row in rows:
        candidate = str(row["trajectory"][0]["content"]).strip()
        reference = str(row["reference"]).strip()
        requests.extend([
            {
                "system": system,
                "user": judge_prompt(row, candidate, reference),
                "temperature": 0.0,
                "max_tokens": 128,
            },
            {
                "system": system,
                "user": judge_prompt(row, reference, candidate),
                "temperature": 0.0,
                "max_tokens": 128,
            },
        ])
        response_pairs.extend(((candidate, reference), (reference, candidate)))
    raw_outputs = judge.complete_batch(requests)
    scored: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        candidate = str(row["trajectory"][0]["content"]).strip()
        reference = str(row["reference"]).strip()
        winners: list[str | None] = []
        retry_outputs: list[list[str]] = []
        raw_pair = raw_outputs[2 * row_index : 2 * row_index + 2]
        response_pair = response_pairs[2 * row_index : 2 * row_index + 2]
        for raw, (response_a, response_b) in zip(raw_pair, response_pair):
            winner, retries = resolve_winner(raw, response_a, response_b, judge)
            winners.append(winner)
            retry_outputs.append(retries)
        credits = [
            0.5 if winner is None else float(winner == ("A" if position == 0 else "B"))
            for position, winner in enumerate(winners)
        ]
        scored.append({
            "candidate": candidate,
            "reference": reference,
            "rouge_l": rouge_l_character(candidate, reference),
            "win_first": winners[0] == "A",
            "win_second": winners[1] == "B",
            "win_credit": statistics.fmean(credits),
            "position_consistent": winners[0] is not None and winners[1] is not None and (winners[0] == "A") == (winners[1] == "B"),
            "pairwise_winners": winners,
            "pairwise_judge_outputs": raw_pair,
            "pairwise_retry_outputs": retry_outputs,
            "pairwise_unresolved": [winner is None for winner in winners],
        })
    return scored


def load_rows_by_index(path: Path, judge_model: str) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    result = {}
    for row in rows:
        if row.get("judge_model") != judge_model:
            raise ValueError(f"Unexpected Judge in reusable evaluation row: {row.get('judge_model')}")
        index = int(row["index"])
        if index in result:
            raise ValueError(f"Duplicate reusable index {index} in {path}")
        result[index] = row
    return result


def reused_pairwise(row: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    candidate = str(generation["trajectory"][0]["content"]).strip()
    reference = str(generation["reference"]).strip()
    if row.get("candidate") != candidate or row.get("reference") != reference:
        raise ValueError(f"Reusable pairwise result does not match generation index {row.get('index')}")
    credit = row.get("win_credit")
    if credit is None:
        credit = (int(bool(row["win_first"])) + int(bool(row["win_second"]))) / 2
    return {
        "candidate": candidate,
        "reference": reference,
        "rouge_l": rouge_l_character(candidate, reference),
        "win_first": bool(row["win_first"]),
        "win_second": bool(row["win_second"]),
        "win_credit": float(credit),
        "position_consistent": bool(row.get("position_consistent", row["win_first"] == row["win_second"])),
        "pairwise_winners": row.get("pairwise_winners"),
        "pairwise_judge_outputs": row.get("pairwise_judge_outputs", []),
        "pairwise_retry_outputs": row.get("pairwise_retry_outputs", []),
        "pairwise_unresolved": row.get("pairwise_unresolved", [False, False]),
        "reused_from_judge_batch_size": row.get("judge_batch_size"),
    }


def evaluate_method(
    method: PersuaderMethod,
    rows: list[dict[str, Any]],
    judge: LocalTransformersEndpoint,
    judge_model: str,
    output_dir: Path,
    reusable: dict[int, dict[str, Any]],
    context_batch_size: int,
) -> Path:
    output = output_dir / f"{method.slug}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = load_rows_by_index(output, judge_model)
    if existing:
        if any(row.get("protocol") != PROTOCOL for row in existing.values()):
            raise ValueError(f"Incompatible Table 2 protocol in {output}")
        if set(existing) - {int(row["index"]) for row in rows}:
            raise ValueError(f"Unexpected indexes in {output}")

    pending: list[dict[str, Any]] = []
    for row_number, generation in enumerate(rows, 1):
        index = int(generation["index"])
        if index in existing:
            continue
        old = reusable.get(index)
        if old is None:
            pending.append(generation)
            continue
        scored = reused_pairwise(old, generation)
        result = {
            "protocol": PROTOCOL,
            "index": index,
            "scenario_id": generation["scenario_id"],
            "source_id": generation["source_id"],
            "session_id": generation["session_id"],
            "method": method.slug,
            "model": method.model_path,
            "judge_model": judge_model,
            "judge_compute_dtype": "bfloat16",
            "scoring_batch_size": old.get("judge_batch_size"),
            **scored,
        }
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        existing[index] = result
        if row_number % 10 == 0:
            print(f"{method.slug}: {row_number}/{len(rows)} rows present", flush=True)

    for batch_start in range(0, len(pending), context_batch_size):
        batch = pending[batch_start : batch_start + context_batch_size]
        scored_rows = score_pairwise_batch(batch, judge)
        for generation, scored in zip(batch, scored_rows):
            index = int(generation["index"])
            result = {
                "protocol": PROTOCOL,
                "index": index,
                "scenario_id": generation["scenario_id"],
                "source_id": generation["source_id"],
                "session_id": generation["session_id"],
                "method": method.slug,
                "model": method.model_path,
                "judge_model": judge_model,
                "judge_compute_dtype": "bfloat16",
                "scoring_batch_size": 2 * len(batch),
                **scored,
            }
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            existing[index] = result
        print(
            f"{method.slug}: {len(existing)}/{len(rows)} rows present "
            f"(scored {len(batch)} contexts in one batch)",
            flush=True,
        )

    if set(existing) != {int(row["index"]) for row in rows}:
        raise ValueError(f"Incomplete Table 2 evaluation for {method.slug}")
    return output


def summarize(path: Path) -> dict[str, float | int]:
    rows = read_jsonl(path)
    unresolved = sum(sum(bool(value) for value in row.get("pairwise_unresolved", [])) for row in rows)
    return {
        "count": len(rows),
        "win_rate_percent": 100 * statistics.fmean(float(row["win_credit"]) for row in rows),
        "rouge_l_character": statistics.fmean(float(row["rouge_l"]) for row in rows),
        "pairwise_unresolved_judgments": unresolved,
    }


def write_report(
    output_dir: Path,
    summaries: dict[str, dict[str, float | int]],
    judge_model: str,
    reused_rows: dict[str, int],
    context_batch_size: int,
) -> Path:
    lines = [
        "# Qwen2.5 Dialogue-SFT: Table 2 Automatic Metrics",
        "",
        "| Persuader | Win-Rate (%) | Character ROUGE-L |",
        "| --- | ---: | ---: |",
    ]
    for method in METHODS:
        metric = summaries[method.slug]
        lines.append(
            f"| {method.display_name} | {metric['win_rate_percent']:.2f} | "
            f"{metric['rouge_l_character']:.2f} |"
        )
    lines.extend([
        "",
        "## Evaluation details",
        "",
        f"- Test contexts: {TEST_SCENARIOS}.",
        "- The candidate is the first Persuader reply; the reference is the held-out dataset target reply.",
        f"- Win-Rate uses two order-swapped local Judge decisions per context, scored in BF16 with `{Path(judge_model).name}`; split decisions contribute 0.5.",
        "- ROUGE-L is character-level F1 on the same first reply and target reply.",
        "- Human evaluation is omitted. This run uses the held-out target replies and a local Judge; the published Table 2 uses GPT-4+ISR references and ChatGPT, so the values are not directly comparable.",
        f"- Existing order-swapped judgments were reused when available; new judgments were decoded in BF16 batches of up to {2 * context_batch_size} prompts.",
        "",
        "| Persuader | Reused judgments | Unresolved pairwise decisions |",
        "| --- | ---: | ---: |",
    ])
    for method in METHODS:
        metric = summaries[method.slug]
        lines.append(
            f"| {method.display_name} | {reused_rows[method.slug]} | "
            f"{metric['pairwise_unresolved_judgments']} |"
        )
    report = output_dir / "qwen25_table2_metrics.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_argument(parser)
    parser.add_argument("--generation-dir", type=Path, default=GENERATION_DIR)
    parser.add_argument("--reuse-dir", type=Path, default=REUSE_DIR)
    parser.add_argument("--evaluation-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--judge-device", default="cuda:0,cuda:1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--context-batch-size", type=int, default=8)
    args = parser.parse_args()
    if PROFILE != "paper10k":
        raise SystemExit("Table 2 metrics require the paper10k held-out test split")
    expected = TEST_SCENARIOS if args.limit is None else args.limit
    if expected < 1 or expected > TEST_SCENARIOS:
        raise SystemExit(f"--limit must be between 1 and {TEST_SCENARIOS}")
    if args.context_batch_size < 1 or args.context_batch_size > 8:
        raise SystemExit("--context-batch-size must be between 1 and 8")

    generations = {
        method.slug: load_generation(args.generation_dir / f"{method.slug}.jsonl", expected)
        for method in METHODS
    }
    for method in METHODS:
        if not Path(method.model_path).is_dir():
            raise SystemExit(f"Missing Persuader model: {method.model_path}")

    from evaluate_qwen25_table2 import check_judge_bf16_capacity

    check_judge_bf16_capacity(args.judge_model, args.judge_device, 1)
    endpoint = generation_endpoint(
        base_url=f"local://{args.judge_device}",
        model=args.judge_model,
        api_key="local-not-secret",
        max_retries=1,
    )
    if not isinstance(endpoint, LocalTransformersEndpoint):
        raise TypeError("Table 2 scoring requires a local BF16 Transformers Judge")

    summaries: dict[str, dict[str, float | int]] = {}
    reused_counts: dict[str, int] = {}
    try:
        for method in METHODS:
            old = load_rows_by_index(args.reuse_dir / f"{method.slug}.jsonl", args.judge_model)
            output = evaluate_method(
                method=method,
                rows=generations[method.slug],
                judge=endpoint,
                judge_model=args.judge_model,
                output_dir=args.evaluation_dir,
                reusable=old,
                context_batch_size=args.context_batch_size,
            )
            reused_counts[method.slug] = len(set(old) & {int(row["index"]) for row in generations[method.slug]})
            summaries[method.slug] = summarize(output)
    finally:
        endpoint.close()

    report = write_report(
        args.evaluation_dir, summaries, args.judge_model, reused_counts, args.context_batch_size
    )
    summary = {
        "protocol": PROTOCOL,
        "profile": PROFILE,
        "judge_model": args.judge_model,
        "judge_compute_dtype": "bfloat16",
        "persuader_compute_dtype": "bfloat16",
        "persuadee_compute_dtype": "bfloat16",
        "context_batch_size": args.context_batch_size,
        "test_count": expected,
        "metrics": summaries,
        "reused_rows": reused_counts,
        "report": str(report),
    }
    (args.evaluation_dir / "qwen25_table2_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Table 2 automatic metrics written to {report}", flush=True)


if __name__ == "__main__":
    main()
