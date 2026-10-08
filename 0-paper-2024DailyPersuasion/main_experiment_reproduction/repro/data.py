from __future__ import annotations

import hashlib
import json
import random
import re
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from tqdm.auto import tqdm

from .constants import (
    PROFILE, SFT_SCENARIOS, PREFERENCE_SCENARIOS, TEST_SCENARIOS,
    DATA_DIR,
    DATA_FILENAMES,
    ORIGINAL_ROOT,
    PREFERENCE_SOURCE,
    SMALL_ORIGINAL_ROOT,
    SFT_SOURCE,
    SYSTEM_PERSUADER,
    TEST_SOURCE,
    data_path,
)

QUALITY_SEED = 2024
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
    "for", "from", "has", "have", "i", "if", "in", "is", "it", "me", "my",
    "of", "on", "or", "our", "that", "the", "their", "them", "there", "this",
    "to", "we", "with", "you", "your",
}
REFUSAL_RE = re.compile(
    r"\b(?:no|never|not interested|won't|will not|can't|cannot|don't want|"
    r"refuse|decline|stick with|not convinced|rather not)\b", re.I
)
COMMITMENT_RE = re.compile(
    r"\b(?:i will|i'll|let's|i agree|i accept|i am convinced|i'm convinced|"
    r"i'm willing|i am willing|give it a try|give this a try|sounds good|"
    r"that makes sense|go ahead|count me in|i'll do|i will try)\b", re.I
)
SOFT_ACCEPT_RE = re.compile(
    r"\b(?:open to|willing to consider|consider it|consider this|give it a shot|"
    r"try it|try this|explore|look into|learn more|sounds promising|could try|"
    r"might try|i could|i may)\b", re.I
)
HESITATION_RE = re.compile(
    r"\b(?:not sure|unsure|hesitant|concern|worried|skeptic|skeptical|doubt|"
    r"busy|expensive|uncertain|maybe|perhaps|think about|need time|however|but)\b", re.I
)


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


def isr_output(turn: dict[str, Any]) -> str:
    return (
        f"Intent to Strategy Reasoning: {str(turn['analysis']).strip()}\n"
        f"Selected Strategy: {str(turn['strategy']).strip()}\n"
        f"Utterance: {str(turn['response']).strip()}"
    )


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


def _content_tokens(text: str) -> set[str]:
    return {
        token for token in TOKEN_RE.findall(str(text).lower())
        if token not in STOPWORDS and len(token) > 2
    }


def _response_level(text: str) -> int:
    """Approximate the persuadee's response state for deterministic ranking."""
    if REFUSAL_RE.search(text) and not COMMITMENT_RE.search(text):
        return -1
    if COMMITMENT_RE.search(text):
        return 3
    if SOFT_ACCEPT_RE.search(text):
        return 2
    if HESITATION_RE.search(text):
        return 1
    return 0


def trajectory_quality(
    item: dict[str, Any], session_id: str, session: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return a reproducible, content-based quality record for one trajectory.

    This is a selection heuristic, not a paper-provided quality annotation.  It
    requires a complete multi-turn dialogue and rewards evidence of initial
    resistance, strategy adaptation, and a later positive persuadee response.
    """
    if not valid_session(session) or session[-1]["role"] != "persuadee":
        return None
    persuader_turns = session[::2]
    if len(persuader_turns) < 5:
        return None
    if any(
        not str(turn.get(field, "")).strip()
        for turn in persuader_turns
        for field in ("analysis", "strategy", "response")
    ):
        return None

    persuadee_replies = [str(turn["response"]).strip() for turn in session[1::2]]
    levels = [_response_level(reply) for reply in persuadee_replies]
    final_level = levels[-1]
    resistance_count = sum(level <= 0 for level in levels[:-1])
    positive_steps = sum(levels[index] > levels[index - 1] for index in range(1, len(levels)))
    progress_delta = levels[-1] - levels[0]
    strategies = [str(turn["strategy"]).strip() for turn in persuader_turns]
    unique_strategies = len({value.lower() for value in strategies})
    overlaps: list[float] = []
    strategy_response_overlaps: list[float] = []
    for index, turn in enumerate(persuader_turns):
        latest_reply = persuadee_replies[index - 1] if index else item["scenario"]["background"]
        analysis_words = _content_tokens(str(turn["analysis"]))
        reply_words = _content_tokens(latest_reply)
        strategy_words = _content_tokens(str(turn["strategy"]))
        utterance_words = _content_tokens(str(turn["response"]))
        overlaps.append(len(analysis_words & reply_words) / max(1, len(analysis_words)))
        strategy_response_overlaps.append(
            len(strategy_words & utterance_words) / max(1, len(strategy_words))
        )
    coherence = sum(overlaps) / len(overlaps)
    strategy_response = sum(strategy_response_overlaps) / len(strategy_response_overlaps)
    adaptation = min(1.0, max(0.0, unique_strategies - 1) / max(1, len(strategies) - 1))
    score = (
        final_level * 100
        + progress_delta * 20
        + positive_steps * 8
        + min(3, resistance_count) * 10
        + adaptation * 15
        + coherence * 10
        + strategy_response * 10
        + min(len(persuader_turns), 8)
    )
    if final_level >= 3 and resistance_count >= 1 and (progress_delta >= 1 or adaptation >= 0.5):
        quality_level = "A"
    elif final_level >= 2 and resistance_count >= 1:
        quality_level = "B"
    elif final_level >= 3:
        quality_level = "C"
    elif final_level >= 1:
        quality_level = "D"
    else:
        quality_level = "E"
    return {
        "session_id": session_id,
        "score": round(score, 4),
        "quality_level": quality_level,
        "final_level": final_level,
        "resistance_count": resistance_count,
        "positive_steps": positive_steps,
        "progress_delta": progress_delta,
        "coherence": round(coherence, 4),
        "strategy_response": round(strategy_response, 4),
        "adaptation": round(adaptation, 4),
        "persuader_turns": len(persuader_turns),
    }


def select_high_quality_trajectories(
    records: list[dict[str, Any]], *, count: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one best complete trajectory for every selected SFT scenario."""
    quality_rank = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0}
    candidates: list[dict[str, Any]] = []
    quality_counts: dict[str, int] = {key: 0 for key in quality_rank}
    for record in tqdm(records, desc="Scoring SFT trajectories", unit="scenario"):
        options = [
            (trajectory_quality(record, session_id, session), session)
            for session_id, session in sorted(record["dialog"].items())
        ]
        options = [(metrics, session) for metrics, session in options if metrics is not None]
        if not options:
            continue
        metrics, session = max(
            options,
            key=lambda pair: (
                quality_rank[pair[0]["quality_level"]],
                pair[0]["score"],
                pair[0]["final_level"],
                pair[0]["persuader_turns"],
                pair[0]["session_id"],
            ),
        )
        quality_counts[metrics["quality_level"]] += 1
        candidates.append({"record": record, "session": session, "metrics": metrics})
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} complete quality-ranked SFT scenarios; need {count}")
    # Shuffle before sorting so exact score ties have a stable but non-source-order tie break.
    random.Random(seed).shuffle(candidates)
    candidates.sort(
        key=lambda item: (
            quality_rank[item["metrics"]["quality_level"]],
            item["metrics"]["score"],
            item["metrics"]["final_level"],
            item["metrics"]["resistance_count"],
            item["metrics"]["persuader_turns"],
        ),
        reverse=True,
    )
    selected = candidates[:count]
    selected_counts = {key: 0 for key in quality_rank}
    for item in selected:
        selected_counts[item["metrics"]["quality_level"]] += 1
    return selected, {
        "selection_heuristic": "complete trajectory with outcome, resistance, adaptation, and ISR-field checks",
        "candidate_scenarios": len(candidates),
        "candidate_quality_counts": {key: value for key, value in quality_counts.items() if value},
        "selected_quality_counts": {key: value for key, value in selected_counts.items() if value},
        "selected_score_min": min(item["metrics"]["score"] for item in selected),
        "selected_score_median": median(item["metrics"]["score"] for item in selected),
        "selected_score_max": max(item["metrics"]["score"] for item in selected),
    }


def _json_file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_pretty_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sample_scenarios(records: list[dict[str, Any]], *, count: int, seed: int) -> list[dict[str, Any]]:
    if len(records) < count:
        raise ValueError(f"Only {len(records)} scenarios available; need {count}")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    return [records[index] for index in indices[:count]]


def _write_small_sft_format(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Provide readable JSON arrays alongside the JSONL files used by runners."""
    sft_root = SMALL_ORIGINAL_ROOT.parent / "sft_format"
    dialogue_rows = list(build_persuader_rows(records, with_isr=False))
    isr_rows = list(build_persuader_rows(records, with_isr=True))
    if len(dialogue_rows) != SFT_SCENARIOS or len(isr_rows) != SFT_SCENARIOS:
        raise ValueError("Small SFT format does not contain exactly one session per selected scenario")
    _write_pretty_json(sft_root / "sft_dialogue.json", dialogue_rows)
    _write_pretty_json(sft_root / "sft_add_isr.json", isr_rows)
    _write_pretty_json(
        sft_root / "dataset_info.json",
        {
            "small5000_dialogue_sft": {
                "file_name": "sft_dialogue.json",
                "formatting": "alpaca",
                "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history", "system": "system"},
            },
            "small5000_isr_sft": {
                "file_name": "sft_add_isr.json",
                "formatting": "alpaca",
                "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history", "system": "system"},
            },
        },
    )
    return {
        "dialogue_rows": len(dialogue_rows),
        "isr_rows": len(isr_rows),
        "files": {
            name: _json_file_digest(sft_root / name)
            for name in ("sft_dialogue.json", "sft_add_isr.json", "dataset_info.json")
        },
    }


def _write_small_profile_readme() -> None:
    root = SMALL_ORIGINAL_ROOT.parent
    (root / "README.md").write_text(
        "# DailyPersuasion small5000 profile\n\n"
        "This isolated reduced-scale profile contains 5,000 selected SFT scenarios, "
        "1,000 preference scenarios, and 500 held-out test scenarios. "
        "`original_format/*.json` keeps the released schema with readable indentation. "
        "Each SFT record retains exactly one highest-ranked complete trajectory; "
        "the ranking is a deterministic heuristic based on complete roles, ISR fields, "
        "resistance, response progress, strategy adaptation, and a positive ending.\n\n"
        "`sft_format/sft_dialogue.json` and `sft_format/sft_add_isr.json` are readable "
        "Alpaca-style views of those same 5,000 sessions. Runner JSONL files are "
        "created under `main_experiment_reproduction/runs/small5000/data/` by:\n\n"
        "`sft_dialogue.jsonl`, `sft_add_isr.jsonl`, `sft_persuadee.jsonl`, "
        "`preference_seed.jsonl`, and `test.jsonl`.\n\n"
        "```bash\n"
        "python prepare_data.py --profile small5000\n"
        "```\n\n"
        "`manifest.json` records selected session IDs, quality evidence, source hashes, "
        "and split-leakage checks.\n",
        encoding="utf-8",
    )


def build_small_profile_sources(*, seed: int = QUALITY_SEED) -> dict[str, Any]:
    """Materialize the selected 5k/1k/500 sources without altering paper files."""
    expected_files = [
        SMALL_ORIGINAL_ROOT / "sft.json",
        SMALL_ORIGINAL_ROOT / "preference.json",
        SMALL_ORIGINAL_ROOT / "test.json",
        SMALL_ORIGINAL_ROOT.parent / "manifest.json",
    ]
    if all(path.is_file() for path in expected_files):
        manifest_path = SMALL_ORIGINAL_ROOT.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sft_root = SMALL_ORIGINAL_ROOT.parent / "sft_format"
        if not all((sft_root / name).is_file() for name in ("sft_dialogue.json", "sft_add_isr.json", "dataset_info.json")):
            manifest["sft_format"] = _write_small_sft_format(load_json(SMALL_ORIGINAL_ROOT / "sft.json"))
            _write_pretty_json(manifest_path, manifest)
        if not (SMALL_ORIGINAL_ROOT.parent / "README.md").is_file():
            _write_small_profile_readme()
        return manifest
    if SMALL_ORIGINAL_ROOT.parent.exists():
        raise RuntimeError(
            f"Incomplete small-profile directory: {SMALL_ORIGINAL_ROOT.parent}. "
            "Remove or repair that exact directory before rebuilding it."
        )

    # Always select from the immutable paper-sized source splits, not a prior run.
    sft_source = ORIGINAL_ROOT / "sft.json"
    preference_source = ORIGINAL_ROOT / "preference.json"
    test_source = ORIGINAL_ROOT / "test.json"
    source_records = [load_json(path) for path in (sft_source, preference_source, test_source)]
    expected_sizes = (10_000, 2_000, 1_000)
    if tuple(map(len, source_records)) != expected_sizes:
        raise ValueError(f"Unexpected paper source sizes: {tuple(map(len, source_records))}")
    source_ids = [set(public_scenario_key(record) for record in records) for records in source_records]
    if tuple(map(len, source_ids)) != expected_sizes or source_ids[0] & source_ids[1] or source_ids[0] & source_ids[2] or source_ids[1] & source_ids[2]:
        raise ValueError("Paper source scenario split is invalid or overlaps")

    selected, quality_summary = select_high_quality_trajectories(
        source_records[0], count=SFT_SCENARIOS, seed=seed
    )
    selected_sft: list[dict[str, Any]] = []
    selection_log: list[dict[str, Any]] = []
    for item in selected:
        record = deepcopy(item["record"])
        session_id = item["metrics"]["session_id"]
        record["dialog"] = {session_id: deepcopy(item["session"])}
        selected_sft.append(record)
        selection_log.append(
            {
                "scenario_id": scenario_id(record),
                "source_id": str(record.get("id", "")),
                **item["metrics"],
            }
        )
    selected_preference = _sample_scenarios(
        source_records[1], count=PREFERENCE_SCENARIOS, seed=seed + 1
    )
    selected_test = _sample_scenarios(source_records[2], count=TEST_SCENARIOS, seed=seed + 2)
    selected_ids = [set(public_scenario_key(record) for record in records) for records in (selected_sft, selected_preference, selected_test)]
    if selected_ids[0] & selected_ids[1] or selected_ids[0] & selected_ids[2] or selected_ids[1] & selected_ids[2]:
        raise ValueError("Small-profile scenario leakage detected")

    root = SMALL_ORIGINAL_ROOT.parent
    _write_pretty_json(root / "original_format" / "sft.json", selected_sft)
    _write_pretty_json(root / "original_format" / "preference.json", selected_preference)
    _write_pretty_json(root / "original_format" / "test.json", selected_test)
    manifest = {
        "profile": "small5000",
        "seed": seed,
        "scenario_counts": {
            "sft": len(selected_sft),
            "preference": len(selected_preference),
            "test": len(selected_test),
        },
        "selection": {
            "sft": "one highest-ranked complete trajectory per scenario",
            "preference": "seeded scenario sample from the paper preference split",
            "test": "seeded scenario sample from the paper test split",
            "quality_summary": quality_summary,
            "sft_selection_log": selection_log,
        },
        "source_files": {
            str(path): {"sha256": _json_file_digest(path), "scenarios": len(records)}
            for path, records in zip((sft_source, preference_source, test_source), source_records)
        },
        "leakage": {"sft_preference": 0, "sft_test": 0, "preference_test": 0},
        "format_note": "All source subset JSON is pretty-printed and retains the released record schema.",
    }
    manifest["sft_format"] = _write_small_sft_format(selected_sft)
    _write_pretty_json(root / "manifest.json", manifest)
    _write_small_profile_readme()
    return manifest


def build_persuader_rows(
    records: list[dict[str, Any]], *, with_isr: bool
) -> Iterable[dict[str, Any]]:
    for item in records:
        for session_id, session in sorted(item["dialog"].items()):
            if not valid_session(session):
                continue
            pairs: list[list[str]] = []
            for turn_index in range(0, len(session), 2):
                persuader = session[turn_index]
                query = (
                    scenario_header(item, include_strategies=with_isr)
                    if turn_index == 0
                    else str(session[turn_index - 1]["response"]).strip()
                )
                answer = isr_output(persuader) if with_isr else str(persuader["response"]).strip()
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
                    "format": "isr" if with_isr else "dialogue_only",
                },
            }


def build_user_model_rows(records: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for item in records:
        scenario = item["scenario"]
        system = (
            f"You are {scenario['persuadee']}, the persuadee in the quoted scenario. "
            "Reply naturally to the persuader and do not leave character."
        )
        context = (
            f"Persuasion background: {scenario['background']}\n"
            f"Persuasion goal: {scenario['goal']}\n"
            f"Persuader: {scenario['persuader']}\n"
            f"Persuadee: {scenario['persuadee']}"
        )
        for session_id, session in sorted(item["dialog"].items()):
            if not valid_session(session):
                continue
            pairs: list[list[str]] = []
            for turn_index in range(1, len(session), 2):
                persuader_text = str(session[turn_index - 1]["response"]).strip()
                query = f"{context}\n\nPersuader: {persuader_text}" if turn_index == 1 else persuader_text
                answer = str(session[turn_index]["response"]).strip()
                pairs.append([query, answer])
            if not pairs:
                continue
            yield {
                "instruction": pairs[-1][0],
                "input": "",
                "output": pairs[-1][1],
                "history": pairs[:-1] or None,
                "system": system,
                "metadata": {
                    "scenario_id": scenario_id(item),
                    "source_id": str(item.get("id", "")),
                    "session_id": session_id,
                    "persuadee_turns": len(pairs),
                },
            }


def select_preference_seed(item: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    sessions = [
        (session_id, session)
        for session_id, session in sorted(item["dialog"].items())
        if valid_session(session)
    ]
    if not sessions:
        raise ValueError(f"No valid sessions for preference scenario {item['id']}")
    session_id, session = rng.choice(sessions)
    persuader_indices = list(range(0, len(session), 2))
    target_index = rng.choice(persuader_indices)
    history = [
        {"role": turn["role"], "content": str(turn["response"]).strip()}
        for turn in session[:target_index]
    ]
    return {
        "scenario_id": scenario_id(item),
        "source_id": str(item.get("id", "")),
        "session_id": session_id,
        "target_turn": target_index // 2 + 1,
        "scenario": item["scenario"],
        "strategies": item["strategy"],
        "history": history,
        "gold_response": str(session[target_index]["response"]).strip(),
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


def dataset_info() -> dict[str, Any]:
    columns = {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "history": "history",
        "system": "system",
    }
    return {
        "dialogue_sft": {"file_name": DATA_FILENAMES["dialogue_sft"], "formatting": "alpaca", "columns": columns},
        "isr_sft": {"file_name": DATA_FILENAMES["isr_sft"], "formatting": "alpaca", "columns": columns},
        "user_model_sft": {"file_name": DATA_FILENAMES["user_model_sft"], "formatting": "alpaca", "columns": columns},
        "simulation_preferences": {
            "file_name": DATA_FILENAMES["simulation_preferences"],
            "formatting": "alpaca",
            "ranking": True,
            "columns": columns,
        },
    }


def prepare_all(*, seed: int = 42, force: bool = False) -> dict[str, Any]:
    if PROFILE == "small5000":
        # This writes only the isolated small-profile directory and never
        # modifies the paper-sized source JSON files.
        build_small_profile_sources()
    required = [SFT_SOURCE, PREFERENCE_SOURCE, TEST_SOURCE]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    outputs = {
        "dialogue_sft": data_path("dialogue_sft"),
        "isr_sft": data_path("isr_sft"),
        "user_model_sft": data_path("user_model_sft"),
        "preference_seeds": data_path("preference_seeds"),
        "evaluation_points": data_path("evaluation_points"),
    }
    if not force and all(path.is_file() for path in outputs.values()):
        return validate_prepared_data(outputs)
    sft = load_json(SFT_SOURCE)
    preference = load_json(PREFERENCE_SOURCE)
    test = load_json(TEST_SOURCE)
    expected_source_sizes = (SFT_SCENARIOS, PREFERENCE_SCENARIOS, TEST_SCENARIOS)
    if (len(sft), len(preference), len(test)) != expected_source_sizes:
        raise ValueError(
            f"Unexpected {PROFILE} split sizes: {(len(sft), len(preference), len(test))}"
        )
    split_ids = [set(public_scenario_key(row) for row in rows) for rows in (sft, preference, test)]
    if tuple(map(len, split_ids)) != expected_source_sizes:
        raise ValueError("Duplicate scenario content exists inside a prepared split")
    if split_ids[0] & split_ids[1] or split_ids[0] & split_ids[2] or split_ids[1] & split_ids[2]:
        raise ValueError("Prepared scenario splits overlap")

    manifest: dict[str, Any] = {
        "seed": seed,
        "profile": PROFILE,
        "scenario_split": {"sft": len(sft), "preference": len(preference), "test": len(test)},
        "source_note": (
            "The paper does not release original split IDs or seed. The paper profile "
            "uses the project's seeded reconstruction; small5000 records its own "
            "quality-selection manifest beside the source subset."
        ),
        "identity_note": (
            "The released id field is not unique. Leakage checks and resumable jobs "
            "therefore use a SHA-256 fingerprint of the public scenario fields."
        ),
        "files": {},
    }
    builders = (
        ("dialogue_sft", build_persuader_rows(sft, with_isr=False)),
        ("isr_sft", build_persuader_rows(sft, with_isr=True)),
        ("user_model_sft", build_user_model_rows(sft)),
    )
    for name, rows in builders:
        count, sha = write_jsonl(outputs[name], tqdm(rows, desc=f"Building {name}", unit="row"))
        manifest["files"][name] = {"path": str(outputs[name]), "rows": count, "sha256": sha}

    pref_rng = random.Random(seed)
    count, sha = write_jsonl(
        outputs["preference_seeds"],
        (select_preference_seed(item, pref_rng) for item in tqdm(preference, desc="Sampling preference histories")),
    )
    manifest["files"]["preference_seeds"] = {
        "path": str(outputs["preference_seeds"]), "rows": count, "sha256": sha
    }
    eval_rng = random.Random(seed)
    count, sha = write_jsonl(
        outputs["evaluation_points"],
        (build_eval_point(item, eval_rng) for item in tqdm(test, desc="Sampling evaluation contexts")),
    )
    manifest["files"]["evaluation_points"] = {
        "path": str(outputs["evaluation_points"]), "rows": count, "sha256": sha
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "dataset_info.json").write_text(
        json.dumps(dataset_info(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validate_prepared_data(outputs)
    return manifest


def validate_prepared_data(outputs: dict[str, Path] | None = None) -> dict[str, Any]:
    outputs = outputs or {
        "dialogue_sft": data_path("dialogue_sft"),
        "isr_sft": data_path("isr_sft"),
        "user_model_sft": data_path("user_model_sft"),
        "preference_seeds": data_path("preference_seeds"),
        "evaluation_points": data_path("evaluation_points"),
    }
    expected_exact = {
        "dialogue_sft": 59_868,
        "isr_sft": 59_868,
        "user_model_sft": 59_868,
        "preference_seeds": 2_000,
        "evaluation_points": 1_000,
    }
    if PROFILE == "small5000":
        expected_exact = {
            "dialogue_sft": SFT_SCENARIOS,
            "isr_sft": SFT_SCENARIOS,
            "user_model_sft": SFT_SCENARIOS,
            "preference_seeds": PREFERENCE_SCENARIOS,
            "evaluation_points": TEST_SCENARIOS,
        }
    summary: dict[str, Any] = {}
    identities: dict[str, set[str]] = {}
    session_keys: dict[str, set[tuple[str, str]]] = {}
    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    for name, path in outputs.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        count = 0
        seen: set[str] = set()
        sessions: set[tuple[str, str]] = set()
        digest = hashlib.sha256()
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL {path}:{line_number}") from exc
                digest.update(line.encode("utf-8"))
                count += 1
                metadata = row.get("metadata", row)
                stable_id = str(metadata["scenario_id"])
                seen.add(stable_id)
                if name not in {"preference_seeds", "evaluation_points"}:
                    key = (stable_id, str(metadata["session_id"]))
                    if key in sessions:
                        raise ValueError(f"Duplicate scenario/session in {path}: {key}")
                    sessions.add(key)
        # Preference and evaluation files contain exactly one row per scenario.
        if name in {"preference_seeds", "evaluation_points"} and len(seen) != count:
            raise ValueError(f"Duplicate scenario_id in {path}")
        if count != expected_exact[name]:
            raise ValueError(f"Expected {expected_exact[name]} rows in {path}, found {count}")
        expected_sha = manifest.get("files", {}).get(name, {}).get("sha256")
        actual_sha = digest.hexdigest()
        if expected_sha is not None and actual_sha != expected_sha:
            raise ValueError(f"SHA-256 mismatch for {path}")
        identities[name] = seen
        session_keys[name] = sessions
        summary[name] = {"path": str(path), "rows": count, "sha256": actual_sha}
    if not (
        session_keys["dialogue_sft"]
        == session_keys["isr_sft"]
        == session_keys["user_model_sft"]
    ):
        raise ValueError("The three SFT datasets must cover identical scenario/session pairs")
    train_ids = identities["dialogue_sft"]
    if len(train_ids) != SFT_SCENARIOS:
        raise ValueError(f"Expected {SFT_SCENARIOS} distinct SFT scenarios, found {len(train_ids)}")
    preference_ids = identities["preference_seeds"]
    evaluation_ids = identities["evaluation_points"]
    if train_ids & preference_ids or train_ids & evaluation_ids or preference_ids & evaluation_ids:
        raise ValueError("Prepared training, preference, and evaluation scenarios overlap")
    return summary
