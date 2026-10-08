"""Use the fixed ISR split while keeping private reasoning out of spoken turns."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from direct_answer_prompt import direct_answer_ids

BASELINES = Path(__file__).resolve().parents[1] / "exp_code_ljj"
sys.path.insert(0, str(BASELINES))
from repro.baseline_data import validate_fixed_splits  # noqa: E402
from repro.persuadee_simulation import dialogue_prompt  # noqa: E402
from repro.simulation import extract_utterance  # noqa: E402

ISR_FILE = BASELINES / "data/isr_sft.jsonl"
TEST_FILE = BASELINES / "data/evaluation_points.jsonl"


def parse_isr(text: str) -> tuple[str, str, str]:
    reasoning, marker, rest = text.partition("\nSelected Strategy:")
    strategy, utterance_marker, utterance = rest.partition("\nUtterance:")
    if not marker or not utterance_marker or not reasoning.startswith("Intent to Strategy Reasoning:"):
        raise ValueError("Malformed ISR label")
    if not strategy.strip() or not utterance.strip():
        raise ValueError("Empty ISR strategy or utterance")
    return reasoning.removeprefix("Intent to Strategy Reasoning:").strip(), strategy.strip(), utterance.strip()


def training_point(row: dict) -> tuple[dict, str, dict]:
    history = row.get("history") or []
    first = history[0][0] if history else row["instruction"]
    background = first.split("The persuasion background: ", 1)[1].split("\nThe persuasion goal: ", 1)[0]
    goal = first.split("\nThe persuasion goal: ", 1)[1].split("\n", 1)[0]
    point = {"scenario": {"background": background, "goal": goal}, "history": []}
    for index, (user, assistant) in enumerate(history):
        if index > 0:
            point["history"].append({"role": "persuadee", "content": user})
        _, _, utterance = parse_isr(assistant)
        point["history"].append({"role": "persuader", "content": utterance})
    if history:
        point["history"].append({"role": "persuadee", "content": row["instruction"]})
    reasoning, strategy, target = parse_isr(row["output"])
    return point, target, {"reasoning": reasoning, "strategy": strategy,
                           "scenario_id": row["metadata"]["scenario_id"]}


def format_prompt(point: dict, tokenizer, *, max_prompt_tokens: int) -> list[int]:
    # Reuse the exact public dialogue prompt used by the other baselines.
    user = dialogue_prompt(point, point["history"])
    messages = [{"role": "system", "content": "You are a skilled persuader."},
                {"role": "user", "content": user}]
    ids = direct_answer_ids(tokenizer, messages)
    return ids[-max_prompt_tokens:]


def iter_training_rows(path: Path = ISR_FILE, *, limit: int | None = None):
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            yield training_point(json.loads(line))


def validate_data() -> dict:
    summary = validate_fixed_splits()
    checked = 0
    for point, utterance, private in iter_training_rows():
        if not utterance or not private["reasoning"] or not private["strategy"]:
            raise ValueError("ISR fields incomplete")
        checked += 1
    if checked != summary["sft_sessions"]:
        raise ValueError("ISR count mismatch")
    return summary
