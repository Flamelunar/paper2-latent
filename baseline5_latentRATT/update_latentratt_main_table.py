#!/usr/bin/env python3
"""Validate LatentRATT Judge outputs and update only its two main-table rows."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import stat
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EXP_ROOT = ROOT.parent / "exp_code_ljj"
ARTIFACTS = EXP_ROOT / "artifacts"
GENERATION = ARTIFACTS / "generation_5rounds/baseline5_latentRATT.json"
TABLE = EXP_ROOT / "results/main_experiment_table.md"
METHOD = "latentRATT_persuasion"
SELENE = "/app/hzj/models/Selene-1-Mini-Llama-3.1-8B"
GEMMA = "/app/hzj/models/gemma-4-E4B-it"
MODEL_LABEL = "Qwen2.5-7B-Instruct + LatentRATT"
GENERATION_PROTOCOL = "latentRATT_persuasion_paper10k_5rounds_no_judge_v1"
MODEL = "/app/hzj/liujianjian/qwen2.5-7b-latentRATT-all-59868/rl"
SELENE_RESULT = ARTIFACTS / "success_judgments/selene_v17_v7_latentRATT_20260928/baseline5_latentRATT.json"
SELENE_REVISION = "joint_selene_success_reference_v17_v7_strict_goal_acceptance"
PERSUADEE = "/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged"
TEST_POINTS = EXP_ROOT / "data/evaluation_points.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def indexed(path: Path, *, judge: str | None = None) -> dict[int, dict[str, Any]]:
    rows = read_jsonl(path)
    by_index = {int(row["index"]): row for row in rows}
    if len(rows) != 1000 or len(by_index) != 1000 or set(by_index) != set(range(1000)):
        raise ValueError(f"Expected exactly one record for every test index in {path}")
    for index, row in by_index.items():
        if row.get("method") != METHOD:
            raise ValueError(f"Wrong method in {path} at index {index}")
        if judge and row.get("judge_model") != judge:
            raise ValueError(f"Wrong Judge in {path} at index {index}")
    return by_index


def has_complete_indexes(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        rows = read_jsonl(path)
        indexes = {int(row["index"]) for row in rows}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return len(rows) == 1000 and indexes == set(range(1000))


def validate_generation() -> dict[int, dict[str, Any]]:
    rows = indexed(GENERATION)
    points = read_jsonl(TEST_POINTS)
    if len(points) != 1000:
        raise ValueError(f"Expected 1,000 fixed test points, found {len(points)}")
    for index, row in rows.items():
        point = points[index]
        turns = row.get("trajectory")
        if not isinstance(turns, list) or len(turns) != 10:
            raise ValueError(f"Generation index {index} does not contain five complete rounds")
        if [turn.get("role") for turn in turns] != ["persuader", "persuadee"] * 5:
            raise ValueError(f"Invalid dialogue role order at test index {index}")
        if any(not str(turn.get("content", "")).strip() for turn in turns):
            raise ValueError(f"Empty dialogue turn at test index {index}")
        if (row.get("protocol") != GENERATION_PROTOCOL
                or row.get("rounds") != 5 or row.get("max_new_tokens") != 512
                or row.get("temperature") != 0.95 or row.get("top_p") != 0.7
                or row.get("top_k") != 50 or row.get("persuadee_model") != PERSUADEE
                or row.get("persuader_model") != MODEL
                or row.get("scenario_id") != point.get("scenario_id")
                or row.get("source_id") != point.get("source_id")
                or row.get("session_id") != point.get("session_id")
                or row.get("reference") != point.get("reference")
                or row.get("initial_history") != point.get("history")):
            raise ValueError(f"Generation metadata or fixed context mismatch at test index {index}")
    return rows


def selene_metrics(generation: dict[int, dict[str, Any]]) -> dict[str, Any]:
    path = SELENE_RESULT
    records = indexed(path, judge=SELENE)
    valid: list[dict[str, Any]] = []
    unresolved = 0
    for index, row in records.items():
        if row.get("scenario_id") != generation[index].get("scenario_id"):
            raise ValueError(f"Selene result is misaligned at index {index}")
        if (row.get("trial") != 1 or row.get("prompt_version") != "selene_v17"
                or row.get("prompt_revision") != SELENE_REVISION
                or row.get("model") != MODEL or row.get("judge_device") != "cuda:2"
                or row.get("judge_compute_dtype") != "bfloat16"
                or row.get("judge_quantization") != "none"
                or row.get("max_parse_retries") != 3):
            raise ValueError(f"Selene prompt or precision metadata mismatch at index {index}")
        verdict = row.get("verdict")
        if row.get("parse_error") or not isinstance(verdict, dict):
            unresolved += 1
            continue
        rounds = verdict.get("round_success")
        if not isinstance(rounds, list) or len(rounds) != 5 or not all(isinstance(x, bool) for x in rounds):
            raise ValueError(f"Invalid Selene success verdict at index {index}")
        first = next((round_number for round_number, value in enumerate(rounds, 1) if value), None)
        if (verdict.get("first_success_round") != first
                or verdict.get("success_within_5_rounds") != (first is not None)):
            raise ValueError(f"Inconsistent Selene success verdict at index {index}")
        ab, ba = verdict.get("win_ab"), verdict.get("win_ba")
        credit = ((ab == "A") + (ba == "B")) / 2
        if (ab not in {"A", "B"} or ba not in {"A", "B"}
                or verdict.get("win_credit") != credit):
            raise ValueError(f"Invalid Selene Win-Rate verdict at index {index}")
        valid.append(verdict)
    if not valid:
        raise ValueError("No parsed Selene joint judgments are available")
    rounds_or_cap = [item["first_success_round"] or 5 for item in valid]
    successes = sum(bool(item["success_within_5_rounds"]) for item in valid)
    return {
        "n": f"{len(valid)}/1000",
        "success_count": successes,
        "success_percent": successes / len(valid) * 100,
        "win_rate": statistics.fmean(float(row["win_credit"]) for row in valid) * 100,
        "mean_rounds": statistics.fmean(rounds_or_cap),
        "unresolved": unresolved,
    }


def gemma_metrics(generation: dict[int, dict[str, Any]]) -> dict[str, Any]:
    path = ARTIFACTS / f"success_judgments/gemma4/{METHOD}.jsonl"
    records = indexed(path, judge=GEMMA)
    valid: list[dict[str, Any]] = []
    unresolved = 0
    for index, row in records.items():
        if row.get("scenario_id") != generation[index].get("scenario_id"):
            raise ValueError(f"Gemma result is misaligned at index {index}")
        if (row.get("trial") != 1 or row.get("prompt_version") != "v11"
                or row.get("prompt_revision") != "joint_v2_r11"
                or row.get("model") != MODEL
                or row.get("judge_device") != "cuda:2"
                or row.get("judge_compute_dtype") != "bfloat16"
                or row.get("judge_quantization") != "none"
                or row.get("max_parse_retries") != 3):
            raise ValueError(f"Gemma prompt or precision metadata mismatch at index {index}")
        verdict = row.get("verdict")
        if row.get("parse_error") or not isinstance(verdict, dict):
            unresolved += 1
            continue
        rounds = verdict.get("round_success")
        if not isinstance(rounds, list) or len(rounds) != 5 or not all(isinstance(x, bool) for x in rounds):
            raise ValueError(f"Invalid Gemma success verdict at index {index}")
        first = next((round_number for round_number, value in enumerate(rounds, 1) if value), None)
        if (verdict.get("first_success_round") != first
                or verdict.get("success_within_5_rounds") != (first is not None)):
            raise ValueError(f"Inconsistent Gemma success verdict at index {index}")
        ab, ba = verdict.get("win_ab"), verdict.get("win_ba")
        credit = ((ab == "A") + (ba == "B")) / 2
        if (ab not in {"A", "B"} or ba not in {"A", "B"}
                or verdict.get("win_credit") != credit):
            raise ValueError(f"Invalid Gemma Win-Rate verdict at index {index}")
        valid.append(verdict)
    if not valid:
        raise ValueError("No parsed Gemma joint judgments are available")
    rounds_or_cap = [item["first_success_round"] or 5 for item in valid]
    successes = sum(bool(item["success_within_5_rounds"]) for item in valid)
    return {
        "n": f"{len(valid)}/1000",
        "success_count": successes,
        "success_percent": successes / len(valid) * 100,
        "win_rate": statistics.fmean(float(row["win_credit"]) for row in valid) * 100,
        "mean_rounds": statistics.fmean(rounds_or_cap),
        "unresolved": unresolved,
    }


def replace_latent_row(lines: list[str], judge_name: str, metrics: dict[str, Any]) -> int:
    current_judge = ""
    replaced = 0
    for index, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cells = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        if cells[0]:
            current_judge = cells[0]
        if not cells[1].startswith("our LatentRATT"):
            continue
        if judge_name.casefold() not in current_judge.casefold():
            continue
        n_text = str(metrics["n"])
        success_text = f"{metrics['success_percent']:.2f} ({metrics['success_count']}/{1000 if n_text == '1000' else int(n_text.split('/')[0])})"
        replacement = (
            f"| {cells[0]} | our LatentRATT | {MODEL_LABEL} | {n_text} | "
            f"{success_text} | {metrics['win_rate']:.2f} | "
            f"{metrics['mean_rounds']:.3f} | {metrics['unresolved']} |"
        )
        lines[index] = replacement
        replaced += 1
    if replaced != 1:
        raise ValueError(f"Expected one {judge_name} LatentRATT row in the main table, found {replaced}")
    return replaced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument(
        "--judges", nargs="+", choices=("selene", "gemma4"), default=["gemma4"],
        help="Judge rows to update; defaults to Gemma 4 and leaves Selene untouched",
    )
    args = parser.parse_args()
    generation = validate_generation()
    text = args.table.read_text(encoding="utf-8")
    lines = text.splitlines()
    metrics_by_judge: dict[str, dict[str, Any]] = {}
    selene_path = SELENE_RESULT
    gemma_path = ARTIFACTS / f"success_judgments/gemma4/{METHOD}.jsonl"
    selected = set(args.judges)
    if "selene" in selected and has_complete_indexes(selene_path):
        metrics_by_judge["selene"] = selene_metrics(generation)
        replace_latent_row(lines, "Selene-1-Mini-Llama-3.1-8B", metrics_by_judge["selene"])
    elif "selene" in selected:
        raise ValueError("Selene was requested, but its 1,000-index result is incomplete")
    if "gemma4" in selected and has_complete_indexes(gemma_path):
        metrics_by_judge["gemma4"] = gemma_metrics(generation)
        replace_latent_row(lines, "Gemma 4 E4B-it", metrics_by_judge["gemma4"])
    elif "gemma4" in selected:
        raise ValueError("Gemma 4 was requested, but its 1,000-index result is incomplete")
    if not metrics_by_judge:
        raise ValueError("No complete 1,000-index LatentRATT Judge result is available yet")
    updated_text = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=args.table.parent,
                                     prefix=f".{args.table.name}.", suffix=".tmp",
                                     delete=False) as handle:
        handle.write(updated_text)
        temporary = Path(handle.name)
    os.chmod(temporary, stat.S_IMODE(args.table.stat().st_mode))
    os.replace(temporary, args.table)
    print(f"Updated completed LatentRATT Judge rows in {args.table}")
    print(json.dumps(metrics_by_judge, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
