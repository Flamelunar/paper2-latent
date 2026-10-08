#!/usr/bin/env python3
"""Judge five-round success and Win-Rate with the versioned strict prompts."""

from __future__ import annotations

import argparse
import os
from contextlib import ExitStack
import json
import re
from pathlib import Path
from typing import Any

from repro.profile import configure_profile_from_argv

configure_profile_from_argv()
from repro.constants import ROOT
from repro.simulation import LocalTransformersEndpoint, read_jsonl
from evaluate_baselines import ARCHIVED_METHODS, load_generation
from evaluate_success_prompt_probe import read_manifest


OUTPUT_DIR = ROOT / "artifacts/joint_success_win_probe"
PROMPT_FILE = ROOT / "prompts/joint_judge_prompts.json"
PROMPT_CONFIG = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
PROMPT_PROFILES = {
    "selene_v17": {"judge_name": "selene1", "rule_key": "selene1_v17"},
    "gemma_v11": {"judge_name": "gemma4", "rule_key": "gemma4_v11"},
    "qwen3_v11": {
        "judge_name": "qwen3_14b",
        "rule_key": "strict_v18",
        "revision": "joint_strict_v18_json_schema",
        "round_evidence": "omitted",
        "output_contract": """Output requirements:
- The format instructions do not change the success or Win-Rate criteria above.
- Return exactly one valid JSON object and nothing else.
- Include exactly these three keys: round_success, win_ab, and win_ba.
- Use this structure: {\"round_success\":[boolean,boolean,boolean,boolean,boolean],\"win_ab\":\"A or B\",\"win_ba\":\"A or B\"}.
- Use exactly five bare lowercase JSON booleans in R1-R5 order; do not quote them.
- Each win value must be exactly uppercase A or B.
- Do not include round_evidence, explanations, markdown, code fences, extra keys, or text outside the JSON object.
- Do not use trailing commas or null.""",
    },
    "qwen3_v18": {
        "judge_name": "qwen3_14b",
        "rule_key": "strict_v18",
        "revision": "joint_strict_v18_json_schema",
        "round_evidence": "omitted",
        "output_contract": """Output requirements:
- Return exactly one valid JSON object and nothing else.
- Include exactly these three keys: round_success, win_ab, and win_ba.
- Use exactly five bare lowercase JSON booleans in R1-R5 order.
- Each win value must be exactly uppercase A or B.
- Do not include explanations, evidence, markdown, code fences, extra keys, null, or text outside the JSON object.""",
    },
    "selene_v18": {
        "judge_name": "selene1",
        "rule_key": "strict_v18",
        "revision": "joint_strict_v18_json_schema",
        "round_evidence": "omitted",
        "output_contract": """Output requirements:
- Return exactly one valid JSON object and nothing else.
- Include exactly these three keys: round_success, win_ab, and win_ba.
- Use exactly five bare lowercase JSON booleans in R1-R5 order.
- Each win value must be exactly uppercase A or B.
        - Do not include explanations, evidence, markdown, code fences, extra keys, null, or text outside the JSON object.""",
    },
    # v18 is the public protocol name used by the new experiment plan.
    # The strict schema is shared by both supported Judge model families; the
    # model-specific aliases above remain available for older launchers.
    "v18": {
        "judge_name": "any",
        "rule_key": "strict_v18",
        "revision": "joint_strict_v18_json_schema",
        "round_evidence": "omitted",
        "output_contract": """Output requirements:
- Return exactly one valid JSON object and nothing else.
- Include exactly these three keys: round_success, win_ab, and win_ba.
- Use exactly five bare lowercase JSON booleans in R1-R5 order.
- Each win value must be exactly uppercase A or B.
- Do not include explanations, evidence, markdown, code fences, extra keys, null, or text outside the JSON object.""",
    },
}
PROMPT_REVISIONS = {
    version: profile.get("revision", PROMPT_CONFIG[profile["rule_key"]]["revision"])
    for version, profile in PROMPT_PROFILES.items()
}


def prompt_requires_evidence(prompt_version: str) -> bool:
    profile = PROMPT_PROFILES[prompt_version]
    default = PROMPT_CONFIG[profile["rule_key"]]["round_evidence"]
    return profile.get("round_evidence", default) == "required"


def prompt_output_contract(prompt_version: str) -> str:
    profile = PROMPT_PROFILES[prompt_version]
    return profile.get("output_contract", PROMPT_CONFIG[profile["rule_key"]]["output_contract"])


def joint_prompt(row: dict[str, Any], prompt_version: str, judge_name: str) -> str:
    profile = PROMPT_PROFILES[prompt_version]
    if profile["judge_name"] not in ("any", judge_name):
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
        # Older shared prompt configs carried the V11 evidence tail even when
        # evidence was disabled. The strict v18 config already omits it.
        if success_rule.endswith(evidence_instruction):
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


def model_weight_size_bytes(model: str) -> int | None:
    model_path = Path(model).expanduser()
    if not model_path.is_dir():
        return None

    adapter_config = model_path / "adapter_config.json"
    if adapter_config.is_file():
        adapter = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_path = Path(str(adapter["base_model_name_or_path"])).expanduser()
        base_bytes = model_weight_size_bytes(str(base_path))
        adapter_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("pytorch_model*.bin"))
        if base_bytes is None or not adapter_files:
            return None
        return base_bytes + sum(path.stat().st_size for path in adapter_files)

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = model_path / index_name
        if index_path.is_file():
            total_size = json.loads(index_path.read_text(encoding="utf-8")).get("metadata", {}).get("total_size")
            if isinstance(total_size, int) and total_size > 0:
                return total_size

    weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("pytorch_model*.bin"))
    if not weight_files:
        return None
    return sum(path.stat().st_size for path in weight_files)


def check_multi_gpu_headroom(device: str, reserve_gib: float, model: str) -> None:
    devices = [value.strip() for value in device.split(",")]
    if len(devices) < 2:
        return
    if any(not re.fullmatch(r"cuda:\d+", value) for value in devices):
        raise ValueError(f"Expected comma-separated CUDA devices, got: {device}")

    import torch

    minimum_free = reserve_gib + 1.0
    low_memory = []
    free_by_device: dict[str, float] = {}
    for value in devices:
        gpu_index = int(value.split(":", 1)[1])
        free_bytes, _ = torch.cuda.mem_get_info(gpu_index)
        free_gib = free_bytes / 1024**3
        free_by_device[value] = free_gib
        if free_gib < minimum_free:
            low_memory.append(f"{value}: {free_gib:.2f} GiB free")
    if low_memory:
        raise RuntimeError(
            f"Judge loading not started; each selected GPU needs at least "
            f"{minimum_free:.2f} GiB free (reserve {reserve_gib:.2f} GiB plus 1 GiB load margin): "
            + ", ".join(low_memory)
        )
    weight_bytes = model_weight_size_bytes(model)
    if weight_bytes is not None:
        weight_gib = weight_bytes / 1024**3
        weight_budget_gib = sum(free - reserve_gib for free in free_by_device.values())
        required_gib = weight_gib + 1.0
        if weight_budget_gib < required_gib:
            raise RuntimeError(
                f"Judge loading not started; BF16 weights need about {weight_gib:.2f} GiB "
                f"plus a 1 GiB load margin, but the selected GPUs have only "
                f"{weight_budget_gib:.2f} GiB available after reserves."
            )


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

    if any(rounds[index] and not all(rounds[index:]) for index in range(len(rounds))):
        raise ValueError("round_success must be monotonic: once successful, later rounds cannot be false")

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/success_prompt_v2_diagnostic/sample_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--generation-dir", type=Path, default=ROOT / "artifacts/generation_5rounds_first500",
                        help="Directory containing the five-round generations to judge")
    parser.add_argument("--full-evaluation", action="store_true",
                        help="Evaluate every row for the selected five-round generations")
    parser.add_argument("--index-start", type=int, default=0,
                        help="First test index for a full-evaluation shard (inclusive)")
    parser.add_argument("--index-end", type=int, default=None,
                        help="End test index for a full-evaluation shard (exclusive; defaults to the number of rows in the generation files)")
    supported_methods = tuple(method.slug for method in ARCHIVED_METHODS) + (
        "baseline1_zero_shot", "baseline2_dialogue_sft", "baseline3_isr_simdpo",
        "baseline4_ma2p", "latentRATT_persuasion", "baseline5_latentratt",
    )
    parser.add_argument("--methods", nargs="+", choices=supported_methods,
                        help="Methods to evaluate in full mode; defaults to both archived methods")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-name", choices=("selene1", "gemma4", "qwen3_14b"), required=True)
    parser.add_argument("--judge-device", required=True)
    parser.add_argument("--prompt-version", choices=tuple(PROMPT_REVISIONS), default="selene_v17")
    parser.add_argument("--max-evaluations", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-reserve-gib", type=float, default=3.0,
                        help="Per-GPU headroom for multi-GPU loading (default: 3 GiB)")
    parser.add_argument("--max-parse-retries", type=int, default=3,
                        help="Additional model generations after a parsing or evidence error (default: 3)")
    args = parser.parse_args()
    expected_judge = PROMPT_PROFILES[args.prompt_version]["judge_name"]
    if expected_judge != "any" and args.judge_name != expected_judge:
        parser.error(f"{args.prompt_version} requires --judge-name {expected_judge}")
    if os.environ.get("ALLOW_JUDGE_EVAL") != "1":
        print("Judge evaluation is disabled for the active LatentRATT run; set ALLOW_JUDGE_EVAL=1 to resume.", flush=True)
        return
    if args.max_parse_retries < 0:
        parser.error("--max-parse-retries must be nonnegative")
    if not args.full_evaluation and (args.index_start != 0 or args.index_end is not None):
        parser.error("--index-start/--index-end are only valid with --full-evaluation")

    methods = set(supported_methods)
    if args.full_evaluation:
        selected_methods = args.methods or [method.slug for method in ARCHIVED_METHODS]
    else:
        if args.methods:
            parser.error("--methods is only valid with --full-evaluation")
        samples = read_manifest(args.manifest)
    generation = {
        method: {int(row["index"]): row for row in load_generation(
            args.generation_dir / f"{method}.jsonl", expected_rows=None
        )}
        for method in (selected_methods if args.full_evaluation else {sample["method"] for sample in samples})
        if method in methods
    }
    if args.full_evaluation:
        row_counts = {method: len(rows) for method, rows in generation.items()}
        if len(row_counts) != len(selected_methods) or len(set(row_counts.values())) != 1:
            raise ValueError(f"Selected methods must each have a generation file with the same row count; found {row_counts}")
        available_rows = next(iter(row_counts.values()))
        index_end = available_rows if args.index_end is None else args.index_end
        if not 0 <= args.index_start <= index_end <= available_rows:
            parser.error(f"--index-start/--index-end must satisfy 0 <= start <= end <= {available_rows} for this generation directory")
        samples = [
            {"trial": 1, "method": method, "index": index, "stratum": "full_test"}
            for method in selected_methods for index in range(args.index_start, index_end)
        ]
    else:
        for sample in samples:
            method, index = str(sample["method"]), int(sample["index"])
            if method not in generation or index not in generation[method]:
                raise ValueError(f"Generation row missing for method={method}, index={index} in {args.generation_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    revision = PROMPT_REVISIONS[args.prompt_version]
    if args.full_evaluation:
        output_paths = {
            method: args.output_dir / f"{method}.jsonl"
            for method in {sample["method"] for sample in samples}
        }
    else:
        output_paths = {"probe": args.output_dir / f"{args.judge_name}_{revision}.jsonl"}
    sample_keys = {
        (int(sample["trial"]), str(sample["method"]), int(sample["index"]))
        for sample in samples
    }
    existing: dict[tuple[int, str, int], dict[str, Any]] = {}
    for output in output_paths.values():
        if output.exists():
            for item in read_jsonl(output):
                key = (int(item["trial"]), str(item["method"]), int(item["index"]))
                if (key not in sample_keys or item.get("judge_model") != args.judge_model
                        or item.get("prompt_revision") != revision
                        or item.get("judge_compute_dtype") != "bfloat16"
                        or item.get("judge_quantization") != "none"
                        or int(item.get("max_parse_retries", 0)) != args.max_parse_retries):
                    raise ValueError(f"Unexpected key, model mismatch, prompt mismatch, or retry-policy mismatch in {output}: {key}")
                source_row = generation[str(item["method"])].get(int(item["index"]))
                if source_row is None or item.get("scenario_id") != source_row.get("scenario_id"):
                    raise ValueError(f"Scenario mismatch in existing record {output}: {key}")
                # Reuse valid BF16, unquantized judgments when resuming on a new GPU layout;
                # each record retains the device configuration that actually produced it.
                previous = existing.get(key)
                if previous is None or (previous.get("parse_error") and not item.get("parse_error")):
                    existing[key] = item
    pending = [sample for sample in samples
               if (sample["trial"], sample["method"], sample["index"]) not in existing]
    if args.max_evaluations > 0:
        pending = pending[:args.max_evaluations]
    print(f"{args.judge_name} {revision}: {len(existing)} saved, {len(pending)} pending", flush=True)
    if not pending:
        return

    check_multi_gpu_headroom(args.judge_device, args.gpu_memory_reserve_gib, args.judge_model)
    judge = LocalTransformersEndpoint(
        model=args.judge_model,
        device=args.judge_device,
        quantization="none",
        gpu_memory_reserve_gib=args.gpu_memory_reserve_gib,
    )
    device_map = getattr(judge.model, "hf_device_map", {})
    offloaded = [(name, str(device)) for name, device in device_map.items()
                 if str(device).lower() in {"cpu", "disk"}]
    if offloaded:
        judge.close()
        raise RuntimeError(f"Judge has CPU/disk offload: {offloaded[:5]}")
    print(f"Loaded {args.judge_model} on {args.judge_device}; BF16, no quantization", flush=True)
    unresolved_count = sum(bool(item.get("parse_error")) for item in existing.values())
    try:
        with ExitStack() as stack:
            handles = {
                method: stack.enter_context(path.open("a", encoding="utf-8"))
                for method, path in output_paths.items()
            }
            for number, sample in enumerate(pending, 1):
                row = generation[sample["method"]][sample["index"]]
                if args.prompt_version == "selene_v17":
                    result = evaluate_selene_sample(
                        row,
                        sample,
                        judge,
                        judge_model=args.judge_model,
                        judge_device=args.judge_device,
                        max_tokens=args.max_tokens,
                        max_parse_retries=args.max_parse_retries,
                    )
                    unresolved_count += result["parse_error"] is not None
                else:
                    prompt = joint_prompt(row, args.prompt_version, args.judge_name)
                    outputs: list[str] = []
                    parse_errors: list[str] = []
                    verdict = None
                    error = None
                    for attempt in range(args.max_parse_retries + 1):
                        retry_note = ""
                        if attempt:
                            retry_note = (
                                f"\n\nThe previous response failed validation: {parse_errors[-1]}. "
                                f"This is correction attempt {attempt} of {args.max_parse_retries}. "
                                "Return a complete JSON object again and follow this exact output contract: "
                                f"{prompt_output_contract(args.prompt_version)}"
                            )
                            bad_quote = re.fullmatch(r"Round ([1-5]) positive evidence is not a verbatim persuadee quote", parse_errors[-1])
                            if bad_quote and prompt_requires_evidence(args.prompt_version):
                                bad_round = int(bad_quote.group(1))
                                reply = row["trajectory"][2 * (bad_round - 1) + 1]["content"]
                                retry_note += (
                                    f"\nR{bad_round} is exactly: {reply}\n"
                                    f"For round_evidence item {bad_round}, copy only an exact quote from this line."
                                )
                        raw = judge.complete(
                            system="You are an impartial evaluator of persuasion success and comparative persuasive quality.",
                            user=prompt + retry_note, temperature=0.0, max_tokens=args.max_tokens,
                            json_mode=True,
                        )
                        outputs.append(raw)
                        try:
                            include_evidence = prompt_requires_evidence(args.prompt_version)
                            verdict = parse_joint(raw, include_evidence=include_evidence)
                            if include_evidence:
                                for round_index, (is_success, quote) in enumerate(zip(
                                    verdict["round_success"], verdict["round_evidence"]
                                )):
                                    reply = row["trajectory"][2 * round_index + 1]["content"]
                                    if is_success and quote.strip().casefold() not in reply.casefold():
                                        raise ValueError(f"Round {round_index + 1} positive evidence is not a verbatim persuadee quote")
                        except ValueError as exc:
                            verdict = None
                            error = str(exc)
                            parse_errors.append(error)
                        else:
                            error = None
                            break
                    unresolved_count += error is not None
                    result = {
                        **sample,
                        "scenario_id": row["scenario_id"],
                        "goal": row["scenario"]["goal"],
                        "model": row.get("persuader_model", row.get("model", "")),
                        "prompt_version": args.prompt_version,
                        "prompt_revision": revision,
                        "judge_model": args.judge_model,
                        "judge_device": args.judge_device,
                        "judge_compute_dtype": "bfloat16",
                        "judge_quantization": "none",
                        "model_calls": len(outputs),
                        "max_parse_retries": args.max_parse_retries,
                        "raw_output": outputs[-1],
                        "raw_outputs": outputs,
                        "retry_outputs": outputs[1:],
                        "parse_errors_by_attempt": parse_errors,
                        "parse_error": error,
                        "verdict": verdict,
                    }
                handle_key = sample["method"] if args.full_evaluation else "probe"
                handles[handle_key].write(json.dumps(result, ensure_ascii=False) + "\n")
                handles[handle_key].flush()
                if number % 5 == 0 or number == len(pending):
                    print(f"{args.judge_name} {revision}: {number}/{len(pending)} new, {unresolved_count} total parse errors", flush=True)
    finally:
        judge.close()
    for output in output_paths.values():
        print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
