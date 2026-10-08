"""Prepare dialogue rows and held-out contexts from the fixed source split."""

from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable
from .constants import SYSTEM_PERSUADER

def load_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return value

def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    return count, digest.hexdigest()

def public_scenario_key(item: dict[str, Any]) -> str:
    """Return the stable scenario identity used by the prepared split builder."""
    scenario = item["scenario"]
    return json.dumps(
        {
            key: scenario.get(key)
            for key in ("tag", "background", "persuadee", "persuader", "goal", "domain")
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

def scenario_id(item: dict[str, Any]) -> str:
    digest = hashlib.sha256(public_scenario_key(item).encode("utf-8")).hexdigest()
    return f"dp_{digest[:20]}"

def scenario_header(item: dict[str, Any], *, include_strategies: bool) -> str:
    scenario = item["scenario"]
    lines = [
        f"The persuasion background: {scenario['background']}",
        f"The persuasion goal: {scenario['goal']}",
    ]
    if include_strategies:
        lines.extend(
            [
                "Next, you need first to give candidate persuasion strategies, then "
                "analyze, select strategy, and interact.",
                "Candidate persuasion strategies: " + "; ".join(item["strategy"]),
            ]
        )
    else:
        lines.append("Give the next persuader utterance only.")
    return "\n".join(lines)

def valid_session(session: Any) -> bool:
    if not isinstance(session, list) or len(session) < 2:
        return False
    if session[0].get("role") != "persuader":
        return False
    for index, turn in enumerate(session):
        expected = "persuader" if index % 2 == 0 else "persuadee"
        if not isinstance(turn, dict) or turn.get("role") != expected:
            return False
        if not str(turn.get("response", "")).strip():
            return False
    return True

def build_persuader_rows(
    records: list[dict[str, Any]]
) -> Iterable[dict[str, Any]]:
    for item in records:
        for session_id, session in sorted(item["dialog"].items()):
            if not valid_session(session):
                continue
            pairs: list[list[str]] = []
            for turn_index in range(0, len(session), 2):
                persuader = session[turn_index]
                query = (
                    scenario_header(item, include_strategies=False)
                    if turn_index == 0
                    else str(session[turn_index - 1]["response"]).strip()
                )
                answer = str(persuader["response"]).strip()
                pairs.append([query, answer])
            yield {
                "instruction": pairs[-1][0],
                "input": "",
                "output": pairs[-1][1],
                "history": pairs[:-1] or None,
                "system": SYSTEM_PERSUADER,
                "metadata": {
                    "scenario_id": scenario_id(item),
                    "source_id": str(item.get("id", "")),
                    "session_id": session_id,
                    "persuader_turns": len(pairs),
                    "format": "dialogue_only",
                },
            }

def build_eval_point(item: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    sessions = [
        (session_id, session)
        for session_id, session in sorted(item["dialog"].items())
        if valid_session(session)
    ]
    if not sessions:
        raise ValueError(f"No valid sessions for test scenario {item['id']}")
    session_id, session = rng.choice(sessions)
    target_index = rng.choice(list(range(0, len(session), 2)))
    return {
        "scenario_id": scenario_id(item),
        "source_id": str(item.get("id", "")),
        "session_id": session_id,
        "target_turn": target_index // 2 + 1,
        "scenario": item["scenario"],
        "strategies": item["strategy"],
        "history": [
            {"role": turn["role"], "content": str(turn["response"]).strip()}
            for turn in session[:target_index]
        ],
        "reference": str(session[target_index]["response"]).strip(),
    }
