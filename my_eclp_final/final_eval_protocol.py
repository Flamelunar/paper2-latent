#!/usr/bin/env python3
"""Frozen final Judge protocol matching the Qwen2.5 baseline table.

Qwen3 neutral V11 profile and nonmonotonic reply parser were recovered from
exp_code_ljj/__pycache__/evaluate_joint_prompt_probe.cpython-38.pyc.
Final metrics use the FIRST positive reply even if a later reply retracts it.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from repro.simulation import LocalTransformersEndpoint
PROMPT_FILE = Path(__file__).with_name("final_judge_prompts.json")
PROMPT_CONFIG = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
PROMPT_PROFILES = {'selene_v17': {'judge_name': 'selene1', 'rule_key': 'selene1_v17'}, 'gemma_v11': {'judge_name': 'gemma4', 'rule_key': 'gemma4_v11'}, 'qwen3_v11': {'judge_name': 'qwen3_14b', 'rule_key': 'gemma4_v11', 'revision': 'qwen3_gemma_v11_neutral_json_v2', 'round_evidence': 'omitted', 'output_contract': 'Output requirements:\n- The format instructions do not change the success or Win-Rate criteria above.\n- Return exactly one valid JSON object and nothing else.\n- Include exactly these three keys: round_success, win_ab, and win_ba.\n- Use this structure: {"round_success":[boolean,boolean,boolean,boolean,boolean],"win_ab":"A or B","win_ba":"A or B"}.\n- Use exactly five bare lowercase JSON booleans in R1-R5 order; do not quote them.\n- Each win value must be exactly uppercase A or B.\n- Do not include round_evidence, explanations, markdown, code fences, extra keys, or text outside the JSON object.\n- Do not use trailing commas or null.'}}

PROMPT_REVISIONS = {version: profile.get('revision', PROMPT_CONFIG[profile['rule_key']]['revision']) for version, profile in PROMPT_PROFILES.items()}

def prompt_requires_evidence(prompt_version: str) -> bool:
    profile = PROMPT_PROFILES[prompt_version]
    default = PROMPT_CONFIG[profile["rule_key"]]["round_evidence"]
    return profile.get("round_evidence", default) == "required"

def prompt_output_contract(prompt_version: str) -> str:
    profile = PROMPT_PROFILES[prompt_version]
    return profile.get("output_contract", PROMPT_CONFIG[profile["rule_key"]]["output_contract"])

def joint_prompt(row: dict[str, Any], prompt_version: str, judge_name: str) -> str:
    profile = PROMPT_PROFILES[prompt_version]
    if profile["judge_name"] != judge_name:
        raise ValueError(f"{prompt_version} is configured for {profile['judge_name']}, not {judge_name}")

    scenario = row["scenario"]
    turns = row["trajectory"]
    if len(turns) != 10 or any(
        turns[i]["role"] != ("persuader" if i % 2 == 0 else "persuadee")
        for i in range(10)
    ):
        raise ValueError(f"Expected five complete rounds at index {row.get('index')}")

    dialogue = "\n".join(
        f"{'G' if turn['role'] == 'persuader' else 'R'}{i // 2 + 1} "
        f"{turn['role'].title()}: {turn['content']}"
        for i, turn in enumerate(turns)
    )
    earlier_dialogue = "\n".join(
        f"{str(turn['role']).title()}: {turn['content']}"
        for turn in row.get("initial_history", [])
    ) or "(No earlier dialogue.)"
    success_rule = PROMPT_CONFIG[profile["rule_key"]]["success_rule"]
    if not prompt_requires_evidence(prompt_version):
        evidence_instruction = (
            "\n- If the judgment is true, copy the shortest exact commitment from that same R line "
            "into round_evidence. If false, use an empty string."
        )
        if not success_rule.endswith(evidence_instruction):
            raise ValueError("Expected the V11 evidence-format instruction at the end of success_rule")
        success_rule = success_rule[:-len(evidence_instruction)]
    win_rule = PROMPT_CONFIG["shared_win_rate_rule"]
    output_contract = prompt_output_contract(prompt_version)

    return f"""You are an expert judge of persuasion success and comparative persuasive quality.

You will receive:
1. Persuasion goal: {scenario['goal']}
2. Background: {scenario['background']}
3. Earlier dialogue before the generated five-round conversation:
{earlier_dialogue}
4. Generated five-round conversation:
{dialogue}
5. Held-out golden Persuader reply:
{row['reference']}

Judgment criteria:

Success judgment:
- Evaluate each generated Persuadee reply R1-R5 independently, using the earlier dialogue and generated conversation only up to that round.
- Earlier dialogue provides context but does not itself count as success; only generated replies R1-R5 can establish success.
- Later replies must not change the judgment for an earlier round.
- Use the matching Persuadee reply as the evidence; the Persuader's words may clarify context but cannot prove success.
{success_rule}

Win-Rate judgment:
{win_rule}

{output_contract}"""

def selene_round_prompt(row: dict[str, Any], round_number: int) -> str:
    scenario = row["scenario"]
    turns = row["trajectory"]
    if round_number not in range(1, 6) or len(turns) != 10:
        raise ValueError(f"Expected a round from a complete five-round dialogue at index {row.get('index')}")

    record = [
        f"{str(turn['role']).title()}: {turn['content']}"
        for turn in row.get("initial_history", [])
    ]
    for index, turn in enumerate(turns[: 2 * round_number]):
        label = "Persuader" if index % 2 == 0 else "Persuadee"
        record.append(f"Round {index // 2 + 1} {label}: {turn['content']}")

    config = PROMPT_CONFIG["selene1_v17"]
    return f"""You are an expert in judging whether persuasion has been successful based on the conversation record.

You will receive:
1. Conversation record through generated round {round_number}:
{chr(10).join(record)}
2. Persuasion goal: {scenario['goal']}
3. Background: {scenario.get('background', '')}

Judgment criteria:
{config['success_rule']}

Determine whether the Persuadee has shown success by this point in the conversation. Consider the whole conversation record above, but no later generated rounds.

{config['round_output_contract']}"""

def selene_win_prompt(row: dict[str, Any]) -> str:
    scenario = row["scenario"]
    turns = row["trajectory"]
    if len(turns) != 10:
        raise ValueError(f"Expected five complete rounds at index {row.get('index')}")
    generated = "\n".join(
        f"G{index + 1}: {turns[2 * index]['content']}" for index in range(5)
    )
    return f"""You are an impartial judge comparing the persuasive quality of two candidates.

You will receive:
1. Persuasion goal: {scenario['goal']}
2. Background: {scenario.get('background', '')}
3. Generated Persuader replies as one candidate:
{generated}
4. Held-out golden Persuader reply as the other candidate:
{row['reference']}

Win-Rate judgment criteria:
{PROMPT_CONFIG['shared_win_rate_rule']}

{PROMPT_CONFIG['selene1_v17']['win_output_contract']}"""

def parse_json_object(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    try:
        parsed, end = json.JSONDecoder().raw_decode(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("No parseable JSON object") from exc
    if not isinstance(parsed, dict) or cleaned[end:].strip():
        raise ValueError("Response must contain one JSON object and nothing else")
    return parsed

def parse_selene_success(raw: str) -> bool:
    parsed = parse_json_object(raw)
    if set(parsed) != {"success"} or type(parsed["success"]) is not bool:
        raise ValueError('JSON must contain only "success" with a bare true/false value')
    return parsed["success"]

def parse_selene_win(raw: str) -> dict[str, str]:
    parsed = parse_json_object(raw)
    if set(parsed) != {"win_ab", "win_ba"}:
        raise ValueError("JSON must contain exactly win_ab and win_ba")
    if any(not isinstance(parsed[key], str) or parsed[key] not in {"A", "B"}
           for key in ("win_ab", "win_ba")):
        raise ValueError("win_ab and win_ba must each be exactly A or B")
    return parsed

def judge_with_retries(
    judge: LocalTransformersEndpoint,
    *,
    prompt: str,
    output_contract: str,
    max_tokens: int,
    max_parse_retries: int,
    parser: Any,
) -> tuple[Any | None, list[str], list[str], str | None]:
    outputs: list[str] = []
    errors: list[str] = []
    verdict = None
    for attempt in range(max_parse_retries + 1):
        retry_note = ""
        if attempt:
            retry_note = (
                f"\n\nThe previous response did not match the required format: {errors[-1]}. "
                f"This is correction attempt {attempt} of {max_parse_retries}. "
                "Keep the same judgment criteria and return a complete response using this exact format: "
                f"{output_contract}"
            )
        raw = judge.complete(
            system="You are an impartial evaluator. Follow the requested judgment criteria and output format exactly.",
            user=prompt + retry_note,
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=True,
        )
        outputs.append(raw)
        try:
            verdict = parser(raw)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            return verdict, outputs, errors, None
    return None, outputs, errors, errors[-1] if errors else "Unknown output error"

def evaluate_selene_sample(
    row: dict[str, Any],
    sample: dict[str, Any],
    judge: LocalTransformersEndpoint,
    *,
    judge_model: str,
    judge_device: str,
    max_tokens: int,
    max_parse_retries: int,
) -> dict[str, Any]:
    config = PROMPT_CONFIG["selene1_v17"]
    round_judgments: list[dict[str, Any]] = []
    all_outputs: list[str] = []
    errors_by_attempt: list[dict[str, str]] = []
    success_parse_error: str | None = None
    first_success_round: int | None = None

    for round_number in range(1, 6):
        decision, outputs, errors, error = judge_with_retries(
            judge,
            prompt=selene_round_prompt(row, round_number),
            output_contract=config["round_output_contract"],
            max_tokens=max_tokens,
            max_parse_retries=max_parse_retries,
            parser=parse_selene_success,
        )
        all_outputs.extend(outputs)
        errors_by_attempt.extend(
            {"stage": f"success_round_{round_number}", "error": item} for item in errors
        )
        round_judgments.append({
            "round": round_number,
            "success": decision,
            "raw_outputs": outputs,
            "parse_errors": errors,
            "parse_error": error,
        })
        if error is not None:
            success_parse_error = f"R{round_number}: {error}"
            break
        if decision:
            first_success_round = round_number
            break

    win_decision, win_outputs, win_errors, win_parse_error = judge_with_retries(
        judge,
        prompt=selene_win_prompt(row),
        output_contract=config["win_output_contract"],
        max_tokens=max_tokens,
        max_parse_retries=max_parse_retries,
        parser=parse_selene_win,
    )
    all_outputs.extend(win_outputs)
    errors_by_attempt.extend(
        {"stage": "win_rate", "error": item} for item in win_errors
    )

    success_resolved = (
        (success_parse_error is None and len(round_judgments) == 5)
        or first_success_round is not None
    )
    observed_rounds = [item["success"] for item in round_judgments]
    if first_success_round is not None:
        observed_rounds.extend([True] * (5 - len(observed_rounds)))
    verdict: dict[str, Any] = {
        "goal_action": "",
        "round_success": observed_rounds if success_resolved else None,
        "first_success_round": first_success_round,
        "success_within_5_rounds": (first_success_round is not None) if success_resolved else None,
        "win_ab": win_decision["win_ab"] if win_decision is not None else None,
        "win_ba": win_decision["win_ba"] if win_decision is not None else None,
        "win_credit": (
            ((win_decision["win_ab"] == "A") + (win_decision["win_ba"] == "B")) / 2
            if win_decision is not None else None
        ),
    }
    parse_error = "; ".join(
        message for message in (success_parse_error, win_parse_error) if message
    ) or None
    return {
        **sample,
        "scenario_id": row["scenario_id"],
        "goal": row["scenario"]["goal"],
        "model": row.get("persuader_model", row.get("model", "")),
        "prompt_version": "selene_v17",
        "prompt_revision": PROMPT_CONFIG["selene1_v17"]["revision"],
        "judge_model": judge_model,
        "judge_device": judge_device,
        "judge_compute_dtype": "bfloat16",
        "judge_quantization": "none",
        "model_calls": len(all_outputs),
        "max_parse_retries": max_parse_retries,
        "raw_output": all_outputs[-1] if all_outputs else "",
        "raw_outputs": all_outputs,
        "retry_outputs": (
            [output for item in round_judgments for output in item["raw_outputs"][1:]]
            + win_outputs[1:]
        ),
        "parse_errors_by_attempt": errors_by_attempt,
        "parse_error": parse_error,
        "success_parse_error": success_parse_error,
        "win_parse_error": win_parse_error,
        "success_judgments": round_judgments,
        "win_judgment": {
            "verdict": win_decision,
            "raw_outputs": win_outputs,
            "parse_errors": win_errors,
            "parse_error": win_parse_error,
        },
        "verdict": verdict,
    }

def parse_joint(raw: str, *, include_evidence: bool) -> dict[str, Any]:
    cleaned = raw.strip()
    decoder = json.JSONDecoder()
    try:
        parsed, end = decoder.raw_decode(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("No parseable JSON object") from exc
    if not isinstance(parsed, dict) or cleaned[end:].strip():
        raise ValueError("Response must contain one JSON object and nothing else")

    expected_keys = {"round_success", "win_ab", "win_ba"}
    if include_evidence:
        expected_keys.add("round_evidence")
    if set(parsed) != expected_keys:
        raise ValueError(f"JSON object must contain exactly: {', '.join(sorted(expected_keys))}")

    rounds = parsed["round_success"]
    if not isinstance(rounds, list) or len(rounds) != 5 or any(type(value) is not bool for value in rounds):
        raise ValueError("round_success must contain exactly five JSON true/false values")
    if include_evidence:
        evidence = parsed["round_evidence"]
        if not isinstance(evidence, list) or len(evidence) != 5 or any(not isinstance(item, str) for item in evidence):
            raise ValueError("round_evidence must contain exactly five strings")
        if any((success and not quote.strip()) or (not success and quote.strip())
               for success, quote in zip(rounds, evidence)):
            raise ValueError("True needs a quote and false needs empty evidence")
    else:
        evidence = [""] * 5

    win_ab, win_ba = parsed["win_ab"], parsed["win_ba"]
    if not isinstance(win_ab, str) or not isinstance(win_ba, str) or win_ab not in {"A", "B"} or win_ba not in {"A", "B"}:
        raise ValueError("win_ab and win_ba must each be exactly A or B")

    first = next((number for number, value in enumerate(rounds, 1) if value), None)
    return {
        "goal_action": "",
        "round_evidence": evidence,
        "round_success": rounds,
        "first_success_round": first,
        "success_within_5_rounds": first is not None,
        "win_ab": win_ab,
        "win_ba": win_ba,
        "win_credit": ((win_ab == "A") + (win_ba == "B")) / 2,
    }
