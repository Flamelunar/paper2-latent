"""Shared multi-turn test runner for local persuader baselines.

This is deliberately separate from :mod:`repro.evaluation`: that module
measures one generated utterance against a dataset reference.  Here, each
persuader must continue the same held-out context with the fixed learned Qwen
persuadee, which is the protocol needed before trajectory-level evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from tqdm.auto import tqdm

from .constants import (
    MAX_NEW_TOKENS,
    OUTPUT_DIR,
    PROFILE,
    RESULT_DIR,
    ROOT,
    SYSTEM_PERSUADER,
    TEST_SCENARIOS,
    data_path,
)
from .data import scenario_header
from .profile import add_profile_argument
from .simulation import (
    ChatEndpoint,
    LocalTransformersEndpoint,
    append_row,
    extract_utterance,
    generation_endpoint,
    read_jsonl,
    transcript,
    user_prompt,
)

TEST_PROTOCOL = "dailypersuasion_fixed_qwen_persuadee_multiturn_v2_no_thinking"
DEFAULT_TEST_DATA = (
    ROOT / "data" / "small5000" / "test.jsonl"
    if PROFILE == "small5000"
    else data_path("evaluation_points")
)
DEFAULT_OUTPUT_DIR = (
    RESULT_DIR / "persuadee_simulation"
    if PROFILE == "paper10k"
    else ROOT / "results" / "persuadee_simulation"
)
DEFAULT_PERSUADEE_MODEL = "/data1/liujianjian/Persuadee/Qwen3.5-9B-persuadee/merged"

if PROFILE == "paper10k":
    DIALOGUE_MODEL = str(OUTPUT_DIR / "01_llama31_dialogue_sft")
    ISR_MODEL = str(OUTPUT_DIR / "02_llama31_isr_sft")
    OURS_MODEL = str(OUTPUT_DIR / "03_llama31_isr_sim_dpo")
else:
    DIALOGUE_MODEL = "/data1/liujianjian/Persuader/llama3.1-8B-Instruct-dialogue-merged"
    ISR_MODEL = "/data1/liujianjian/Persuader/llama3.1-8B-Instruct-isr-merged"
    OURS_MODEL = "/data1/liujianjian/Persuader/llama3.1-8B-Instruct-ours-merged"


@dataclass(frozen=True)
class PersuaderMethod:
    """One trained/untrained persuader and its inference prompt convention."""

    slug: str
    display_name: str
    model_path: str
    with_isr: bool


BASE_METHODS = (
    PersuaderMethod(
        slug="01_zero_shot",
        display_name="01 Zero-shot (Llama-3.1-8B-Instruct)",
        model_path="/data1/liujianjian/model/Meta-Llama-3.1-8B-Instruct",
        with_isr=False,
    ),
    PersuaderMethod(
        slug="02_dialogue_sft",
        display_name="02 Dialogue-SFT (Llama-3.1-8B-Instruct)",
        model_path=DIALOGUE_MODEL,
        with_isr=False,
    ),
    PersuaderMethod(
        slug="03_isr_sft",
        display_name="03 ISR-SFT (Llama-3.1-8B-Instruct)",
        model_path=ISR_MODEL,
        with_isr=True,
    ),
)
OURS_METHOD = PersuaderMethod(
    slug="04_ours",
    display_name="04 Ours (Llama-3.1 ISR-SFT + SimDPO)",
    model_path=OURS_MODEL,
    with_isr=True,
)
METHODS = BASE_METHODS + ((OURS_METHOD,) if PROFILE == "paper10k" else ())
METHOD_BY_SLUG = {method.slug: method for method in METHODS}


def dialogue_prompt(point: dict[str, Any], history: list[dict[str, str]]) -> str:
    """Prompt used by the base and dialogue-only SFT persuaders."""
    item = {"scenario": point["scenario"], "strategy": point.get("strategies", [])}
    return f"""{scenario_header(item, include_strategies=False)}

Historical dialogue:
{transcript(history)}

Write only the next persuader utterance. Do not add a speaker label."""


def isr_prompt(point: dict[str, Any], history: list[dict[str, str]]) -> str:
    """Prompt matching the ISR-SFT target format used by route 03."""
    scenario = point["scenario"]
    return f"""The persuasion background: {scenario['background']}
The persuasion goal: {scenario['goal']}
The persuader: {scenario['persuader']}
The persuadee: {scenario['persuadee']}
Candidate persuasion strategies: {'; '.join(point.get('strategies', []))}

Historical dialogue:
{transcript(history)}

Analyze the persuadee's current intent, select one candidate strategy, and
write the next persuader response. Return exactly this three-line format:
Intent to Strategy Reasoning: <analysis>
Selected Strategy: <strategy>
Utterance: <persuader response>"""


def persuader_prompt(
    point: dict[str, Any], method: PersuaderMethod, history: list[dict[str, str]]
) -> str:
    return isr_prompt(point, history) if method.with_isr else dialogue_prompt(point, history)


def clean_utterance(text: str, *, is_isr: bool, speaker_name: str | None = None) -> str:
    """Keep the spoken utterance while retaining the raw model answer separately."""
    utterance = extract_utterance(text) if is_isr else text.strip()
    # Qwen3.5 may emit its private chain-of-thought delimiters despite the
    # role-play instruction. Preserve it in ``raw_output`` but never feed the
    # reasoning text back into the next dialogue turn.
    if "</think>" in utterance.lower():
        utterance = re.split(r"</think>", utterance, maxsplit=1, flags=re.I)[1].strip()
    utterance = re.sub(r"<think>.*?</think>", "", utterance, flags=re.I | re.S).strip()
    # Models occasionally ignore the request and prefix the literal role name.
    prefixes = ["Persuader:", "Persuadee:", "Assistant:"]
    if speaker_name:
        prefixes.append(f"{speaker_name}:")
    for prefix in prefixes:
        if utterance.lower().startswith(prefix.lower()):
            utterance = utterance[len(prefix) :].strip()
    if not utterance:
        raise ValueError("Model returned an empty persuader utterance")
    return utterance


def validate_devices(persuader_device: str, persuadee_device: str) -> None:
    """Require disjoint CUDA devices for the two resident BF16 models."""
    persuader_devices = set(LocalTransformersEndpoint._parse_devices(persuader_device))
    persuadee_devices = set(LocalTransformersEndpoint._parse_devices(persuadee_device))
    if persuader_devices & persuadee_devices:
        raise ValueError(
            "The local persuader and Qwen persuadee use the same device or overlapping devices. "
            "Use two GPUs, for example --persuader-device cuda:0 "
            "--persuadee-device cuda:1."
        )


def load_test_points(path: Path, *, expected_rows: int = TEST_SCENARIOS) -> list[dict[str, Any]]:
    points = read_jsonl(path)
    if len(points) != expected_rows:
        raise ValueError(f"Expected {expected_rows} fixed test contexts in {path}, found {len(points)}")
    required = {"history", "scenario", "scenario_id", "source_id", "session_id", "target_turn"}
    for index, point in enumerate(points):
        missing = required - point.keys()
        if missing:
            raise ValueError(f"Test context {index} is missing required fields: {sorted(missing)}")
        if not isinstance(point["history"], list):
            raise ValueError(f"Test context {index} has a non-list history")
    return points


def simulate_case(
    *,
    point: dict[str, Any],
    method: PersuaderMethod,
    persuader: ChatEndpoint | LocalTransformersEndpoint,
    persuadee: ChatEndpoint | LocalTransformersEndpoint,
    rounds: int,
    temperature: float,
    max_new_tokens: int,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Produce ``rounds`` persuader/persuadee exchanges from one fixed context."""
    if rounds < 1:
        raise ValueError("rounds must be positive")
    history = [dict(turn) for turn in point["history"]]
    initial_history = [dict(turn) for turn in history]
    trajectory: list[dict[str, Any]] = []

    for round_number in range(1, rounds + 1):
        prompt = persuader_prompt(point, method, history)
        raw_persuader = persuader.complete(
            system=SYSTEM_PERSUADER,
            user=prompt,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
        )
        persuader_text = clean_utterance(
            raw_persuader,
            is_isr=method.with_isr,
            speaker_name=str(point["scenario"]["persuader"]),
        )
        persuader_turn = {
            "round": round_number,
            "role": "persuader",
            "content": persuader_text,
            "raw_output": raw_persuader,
            "prompt_type": "isr" if method.with_isr else "dialogue",
        }
        trajectory.append(persuader_turn)
        history.append({"role": "persuader", "content": persuader_text})

        persuadee_system, persuadee_prompt = user_prompt(point, history)
        raw_persuadee = persuadee.complete(
            system=persuadee_system,
            user=persuadee_prompt,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
        )
        persuadee_text = clean_utterance(
            raw_persuadee,
            is_isr=False,
            speaker_name=str(point["scenario"]["persuadee"]),
        )
        persuadee_turn = {
            "round": round_number,
            "role": "persuadee",
            "content": persuadee_text,
            "raw_output": raw_persuadee,
        }
        trajectory.append(persuadee_turn)
        history.append({"role": "persuadee", "content": persuadee_text})

    return {
        "initial_history": initial_history,
        "trajectory": trajectory,
        "final_history": history,
    }


def _complete_many(
    endpoint: ChatEndpoint | LocalTransformersEndpoint,
    requests: list[dict[str, Any]],
) -> list[str]:
    complete_batch = getattr(endpoint, "complete_batch", None)
    if callable(complete_batch):
        responses = complete_batch(requests)
    else:
        responses = [endpoint.complete(**request) for request in requests]
    if len(responses) != len(requests):
        raise ValueError(
            f"Batched endpoint returned {len(responses)} responses for {len(requests)} prompts"
        )
    return responses


def simulate_cases_batch(
    *,
    points: list[dict[str, Any]],
    method: PersuaderMethod,
    persuader: ChatEndpoint | LocalTransformersEndpoint,
    persuadee: ChatEndpoint | LocalTransformersEndpoint,
    rounds: int,
    temperature: float,
    max_new_tokens: int,
    top_p: float | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Batch independent contexts while preserving turn order within each dialogue."""
    if rounds < 1:
        raise ValueError("rounds must be positive")
    histories = [[dict(turn) for turn in point["history"]] for point in points]
    initial_histories = [[dict(turn) for turn in history] for history in histories]
    trajectories: list[list[dict[str, Any]]] = [[] for _ in points]

    for round_number in range(1, rounds + 1):
        persuader_requests = [
            {
                "system": SYSTEM_PERSUADER,
                "user": persuader_prompt(point, method, history),
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "top_p": top_p,
                "top_k": top_k,
            }
            for point, history in zip(points, histories)
        ]
        raw_persuader = _complete_many(persuader, persuader_requests)
        for index, (point, history, raw) in enumerate(zip(points, histories, raw_persuader)):
            text = clean_utterance(
                raw,
                is_isr=method.with_isr,
                speaker_name=str(point["scenario"]["persuader"]),
            )
            trajectories[index].append({
                "round": round_number,
                "role": "persuader",
                "content": text,
                "raw_output": raw,
                "prompt_type": "isr" if method.with_isr else "dialogue",
            })
            history.append({"role": "persuader", "content": text})

        persuadee_requests = []
        for point, history in zip(points, histories):
            system, prompt = user_prompt(point, history)
            persuadee_requests.append({
                "system": system,
                "user": prompt,
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "top_p": top_p,
                "top_k": top_k,
            })
        raw_persuadee = _complete_many(persuadee, persuadee_requests)
        for index, (point, history, raw) in enumerate(zip(points, histories, raw_persuadee)):
            text = clean_utterance(
                raw,
                is_isr=False,
                speaker_name=str(point["scenario"]["persuadee"]),
            )
            trajectories[index].append({
                "round": round_number,
                "role": "persuadee",
                "content": text,
                "raw_output": raw,
            })
            history.append({"role": "persuadee", "content": text})

    return [
        {
            "initial_history": initial_histories[index],
            "trajectory": trajectories[index],
            "final_history": histories[index],
        }
        for index in range(len(points))
    ]


def _existing_indices(
    output: Path, protocol: str, expected_config: dict[str, Any]
) -> set[int]:
    if not output.exists():
        return set()
    rows = read_jsonl(output)
    protocols = {row.get("protocol") for row in rows}
    if protocols != {protocol}:
        raise ValueError(f"Stale or incompatible protocol in {output}; rerun with --force.")
    for key, expected in expected_config.items():
        values = {row.get(key) for row in rows}
        if values != {expected}:
            raise ValueError(
                f"Incompatible existing output in {output}: {key} is {values}, "
                f"expected {expected!r}; rerun with --force."
            )
    return {int(row["index"]) for row in rows}


def generate_method(
    *,
    points: list[dict[str, Any]],
    method: PersuaderMethod,
    persuader: ChatEndpoint | LocalTransformersEndpoint,
    persuadee: ChatEndpoint | LocalTransformersEndpoint,
    output_dir: Path,
    persuadee_model: str,
    rounds: int,
    temperature: float,
    max_new_tokens: int,
    force: bool,
    top_p: float | None = None,
    top_k: int | None = None,
    context_batch_size: int = 1,
    protocol: str = TEST_PROTOCOL,
) -> Path:
    if context_batch_size < 1:
        raise ValueError("context_batch_size must be positive")
    output = output_dir / f"{method.slug}.jsonl"
    if force and output.exists():
        output.unlink()
    expected_config = {
        "method": method.slug,
        "persuader_model": method.model_path,
        "persuadee_model": persuadee_model,
        "rounds": rounds,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "top_k": top_k,
    }
    if context_batch_size > 1:
        expected_config["context_batch_size"] = context_batch_size
    existing = _existing_indices(output, protocol, expected_config)
    pending = [(index, point) for index, point in enumerate(points) if index not in existing]
    progress = tqdm(total=len(pending), desc=f"Simulate {method.slug}", unit="context")
    for start in range(0, len(pending), context_batch_size):
        batch = pending[start : start + context_batch_size]
        batch_points = [point for _, point in batch]
        if context_batch_size == 1:
            simulations = [
                simulate_case(
                    point=batch_points[0],
                    method=method,
                    persuader=persuader,
                    persuadee=persuadee,
                    rounds=rounds,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    top_k=top_k,
                )
            ]
        else:
            simulations = simulate_cases_batch(
                points=batch_points,
                method=method,
                persuader=persuader,
                persuadee=persuadee,
                rounds=rounds,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
            )
        for (index, point), simulation in zip(batch, simulations):
            result = {
                "protocol": protocol,
                "index": index,
                "method": method.slug,
                "method_name": method.display_name,
                "persuader_model": method.model_path,
                "persuadee_model": persuadee_model,
                "scenario_id": point["scenario_id"],
                "source_id": point["source_id"],
                "session_id": point["session_id"],
                "target_turn": point["target_turn"],
                "reference": point.get("reference", ""),
                "scenario": point["scenario"],
                "strategies": point.get("strategies", []),
                "rounds": rounds,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
                **simulation,
            }
            if context_batch_size > 1:
                result["context_batch_size"] = context_batch_size
            append_row(output, result)
            progress.update(1)
    progress.close()
    rows = read_jsonl(output)
    if len(rows) != len(points) or {int(row["index"]) for row in rows} != set(range(len(points))):
        raise ValueError(f"Incomplete or duplicate simulation output: {output}")
    return output


def write_run_config(
    *,
    output_dir: Path,
    test_data: Path,
    methods: Iterable[PersuaderMethod],
    persuadee_model: str,
    rounds: int,
    temperature: float,
    max_new_tokens: int,
    persuader_device: str,
    persuadee_device: str,
    persuader_quantization: str,
    executed_contexts: int,
    limit: int | None,
    top_p: float | None = None,
    top_k: int | None = None,
    context_batch_size: int = 1,
    protocol: str = TEST_PROTOCOL,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_config.json"
    payload = {
        "protocol": protocol,
        "test_data": str(test_data),
        "test_contexts": TEST_SCENARIOS,
        "executed_contexts": executed_contexts,
        "limit": limit,
        "methods": [asdict(method) for method in methods],
        "persuadee_model": persuadee_model,
        "persuader_device": persuader_device,
        "persuadee_device": persuadee_device,
        "persuader_quantization": persuader_quantization,
        "persuader_quantization_scheme": "NF4" if persuader_quantization == "4bit" else None,
        # Both the NF4 and full-precision inference paths use BF16 matrix
        # computation through BF16 autocast; recording it makes the run auditable.
        "persuader_compute_dtype": "bfloat16",
        "persuadee_compute_dtype": "bfloat16",
        "bf16_autocast": True,
        "persuader_double_quantization": persuader_quantization == "4bit",
        "final_output_directory_label": "persuadee_simulation-4-bit" if persuader_quantization == "4bit" else "persuadee_simulation",
        "rounds": rounds,
        "temperature": temperature,
        "do_sample": temperature > 0,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "context_batch_size": context_batch_size,
        "execution": (
            "Methods run sequentially; independent contexts are decoded in batches, while each context preserves round order."
            if context_batch_size > 1
            else "Methods run sequentially; each context runs synchronously to preserve its dialogue history."
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def run_simulation(
    *,
    methods: Iterable[PersuaderMethod],
    test_data: Path,
    output_dir: Path,
    persuadee_model: str,
    persuader_device: str,
    persuadee_device: str,
    persuader_quantization: str,
    rounds: int,
    temperature: float,
    max_new_tokens: int,
    limit: int | None,
    force: bool,
    top_p: float | None = None,
    top_k: int | None = None,
    context_batch_size: int = 1,
    protocol: str = TEST_PROTOCOL,
) -> list[Path]:
    if max_new_tokens < 1 or max_new_tokens > MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if context_batch_size < 1:
        raise ValueError("context_batch_size must be positive")
    validate_devices(persuader_device, persuadee_device)
    selected_methods = tuple(methods)
    if not selected_methods:
        raise ValueError("At least one persuader method is required")
    points = load_test_points(test_data)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive when provided")
        points = points[:limit]
    if not Path(persuadee_model).is_dir():
        raise FileNotFoundError(f"Fixed persuadee model is unavailable: {persuadee_model}")
    for method in selected_methods:
        if not Path(method.model_path).is_dir():
            raise FileNotFoundError(f"Persuader model is unavailable for {method.slug}: {method.model_path}")
    write_run_config(
        output_dir=output_dir,
        test_data=test_data,
        methods=selected_methods,
        persuadee_model=persuadee_model,
        rounds=rounds,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        persuader_device=persuader_device,
        persuadee_device=persuadee_device,
        persuader_quantization=persuader_quantization,
        executed_contexts=len(points),
        limit=limit,
        top_p=top_p,
        top_k=top_k,
        context_batch_size=context_batch_size,
        protocol=protocol,
    )

    local_key = "local-not-secret"
    persuadee = generation_endpoint(
        base_url=f"local://{persuadee_device}",
        model=persuadee_model,
        api_key=local_key,
        max_retries=1,
    )
    outputs: list[Path] = []
    try:
        for method in selected_methods:
            persuader = generation_endpoint(
                base_url=f"local://{persuader_device}",
                model=method.model_path,
                api_key=local_key,
                max_retries=1,
                quantization=persuader_quantization,
            )
            try:
                outputs.append(
                    generate_method(
                        points=points,
                        method=method,
                        persuader=persuader,
                        persuadee=persuadee,
                        output_dir=output_dir,
                        persuadee_model=persuadee_model,
                        rounds=rounds,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        force=force,
                        top_p=top_p,
                        top_k=top_k,
                        context_batch_size=context_batch_size,
                        protocol=protocol,
                    )
                )
            finally:
                persuader.close()
    finally:
        persuadee.close()
    return outputs


def main(argv: list[str] | None = None, *, forced_method: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_argument(parser)
    if forced_method is None:
        parser.add_argument(
            "--method", choices=("all", *METHOD_BY_SLUG), default="all",
            help=f"Run all {len(METHODS)} methods for this profile, or only one named method.",
        )
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--persuadee-model", default=DEFAULT_PERSUADEE_MODEL)
    parser.add_argument("--persuader-device", default="cuda:0")
    parser.add_argument("--persuadee-device", default="cuda:1")
    parser.add_argument(
        "--persuader-quantization", choices=("none", "4bit"), default="none",
        help="Optional NF4 loading for the local Persuader; useful when GPU memory is fragmented.",
    )
    parser.add_argument("--rounds", type=int, default=4, help="Persuader/persuadee exchanges per test context.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None, help=f"Smoke-test only: run the first N of the fixed {TEST_SCENARIOS} contexts.")
    parser.add_argument("--force", action="store_true", help="Replace selected method output instead of resuming it.")
    args = parser.parse_args(argv)
    selection = forced_method or args.method
    methods = METHODS if selection == "all" else (METHOD_BY_SLUG[selection],)
    for output in run_simulation(
        methods=methods,
        test_data=args.test_data,
        output_dir=args.output_dir,
        persuadee_model=args.persuadee_model,
        persuader_device=args.persuader_device,
        persuadee_device=args.persuadee_device,
        persuader_quantization=args.persuader_quantization,
        rounds=args.rounds,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        force=args.force,
        top_p=args.top_p,
        top_k=args.top_k,
    ):
        print(output)
