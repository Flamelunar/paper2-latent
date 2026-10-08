"""Five-round generation with the fixed Qwen3.5 persuadee."""

from __future__ import annotations
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from tqdm.auto import tqdm
from .constants import MAX_NEW_TOKENS, ROOT, SYSTEM_PERSUADER, TEST_SCENARIOS
from .data import scenario_header
from .simulation import ChatEndpoint, LocalTransformersEndpoint, append_row, extract_utterance, generation_endpoint, read_jsonl, transcript, user_prompt
from .public_answer import public_turn, strip_thinking

TEST_PROTOCOL = "qwen25_table2_fixed_qwen35_persuadee_5rounds_bf16_contextbatch2_v1"
DEFAULT_PERSUADEE_MODEL = "/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged"


@dataclass(frozen=True)
class PersuaderMethod:
    """One persuader; with_isr selects utterance extraction, not the prompt."""

    slug: str
    display_name: str
    model_path: str
    with_isr: bool

def dialogue_prompt(point: dict[str, Any], history: list[dict[str, str]]) -> str:
    """Prompt used by the base and dialogue-only SFT persuaders."""
    item = {"scenario": point["scenario"], "strategy": point.get("strategies", [])}
    return f"""{scenario_header(item, include_strategies=False)}

Historical dialogue:
{transcript(history)}

Write only the next persuader utterance. Do not add a speaker label."""

def persuader_prompt(point: dict[str, Any], method: PersuaderMethod, history: list[dict[str, str]]) -> str:
    if method.slug == "baseline1_zero_shot":
        from .zero_shot_contract import zero_shot_prompt
        return zero_shot_prompt(point, history)
    return dialogue_prompt(point, history)


def clean_utterance(text: str, *, is_isr: bool, speaker_name: str | None = None) -> str:
    """Keep the spoken utterance while retaining the raw model answer separately."""
    public_text, had_thinking_marker = strip_thinking(text)
    utterance = extract_utterance(public_text) if is_isr else public_text.strip()
    # Qwen3.5 may emit its private chain-of-thought delimiters despite the
    # role-play instruction. Preserve it in ``raw_output`` but never feed the
    # reasoning text back into the next dialogue turn.
    # DeepSeek can emit an unmatched closing tag because the opening marker
    # was prefetched in the prompt. Other models can stop after an opening
    # tag. Remove complete and incomplete blocks before saving a turn.
    utterance = re.sub(
        r"<think\b[^>]*>.*?</think\b[^>]*>", "", utterance,
        flags=re.I | re.S,
    )
    utterance = re.sub(r"<think\b[^>]*>.*\Z", "", utterance, flags=re.I | re.S)
    utterance = re.sub(r"</think\b[^>]*>", "", utterance, flags=re.I)
    utterance = utterance.strip()
    if had_thinking_marker:
        # If a model copied the transcript, keep the last non-empty labelled
        # assistant/role segment rather than the whole prompt.
        labelled = list(re.finditer(
            r"(?im)^\s*(assistant|persuader|persuadee)\s*:?[ \t]*", utterance
        ))
        segments: list[str] = []
        for index, match in enumerate(labelled):
            end = labelled[index + 1].start() if index + 1 < len(labelled) else len(utterance)
            segment = utterance[match.end():end].strip()
            if segment:
                segments.append(segment)
        if segments:
            utterance = segments[-1]
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


def _needs_direct_rewrite(raw: str, cleaned: str, method: PersuaderMethod) -> bool:
    """Detect DeepSeek outputs that expose planning instead of spoken dialogue."""
    model_hint = method.model_path.lower()
    if not any(marker in model_hint for marker in ("deepseek-r1", "deepseek_r1", "deepseekr1")):
        return False
    # A closing tag may be emitted without its opening tag because the opening
    # marker was prefetched in the prompt. Any thinking marker needs a clean
    # public-answer rewrite for DeepSeek.
    if re.search(r"<think\b|</think\b", raw, re.I):
        return True
    if len(cleaned) < 240:
        return False
    markers = (
        "next persuader utterance:",
        "next persuader response:",
        "i need to figure out",
        "let's break this down",
        "the key is to",
        "first, i should",
        "first, i need to",
        "the goal is to",
    )
    lowered = cleaned.casefold()
    return any(marker in lowered for marker in markers)


def _direct_rewrite(
    endpoint: ChatEndpoint | LocalTransformersEndpoint,
    raw: str,
    point: dict[str, Any],
    speaker_name: str,
    speaker_role: str = "Persuader",
    allow_empty: bool = False,
) -> tuple[str, str]:
    """Ask the same model to expose only the public utterance after meta-planning."""
    rewritten = endpoint.complete(
        system=(
            f"Return only one natural spoken {speaker_role} utterance. Do not explain, "
            "reason, plan, mention drafts, or add a speaker label."
        ),
        user=(
            f"Goal: {point['scenario']['goal']}\n"
            f"You are {speaker_name}, the {speaker_role.lower()}.\n"
            "Rewrite the draft below as the final message to the other speaker. "
            "Keep only the public answer:\n---\n" + raw + "\n---"
        ),
        temperature=0.0,
        max_tokens=512,
        top_p=1.0,
        top_k=0,
    )
    try:
        cleaned = clean_utterance(rewritten, is_isr=False, speaker_name=speaker_name)
    except ValueError:
        if not allow_empty:
            raise
        # The boundary caller must see a rejected private-only attempt too,
        # otherwise its filtering provenance silently omits that attempt.
        cleaned = ''
    return cleaned, rewritten

def validate_devices(persuader_device: str, persuadee_device: str) -> None:
    """Validate CUDA device syntax; co-location is allowed after memory preflight."""
    LocalTransformersEndpoint._parse_devices(persuader_device)
    LocalTransformersEndpoint._parse_devices(persuadee_device)

def final_public_reply(raw, *, point, role, endpoint, method=None):
    is_isr = method.with_isr if method is not None else False
    try:
        preliminary = clean_utterance(raw, is_isr=is_isr,
                                      speaker_name=str(point['scenario'][role]))
    except ValueError:
        preliminary = ''
    force = role == 'persuadee' and bool(re.search(r'</?think\b', raw, re.I))
    if method is not None:
        force = force or _needs_direct_rewrite(raw, preliminary, method)
    return public_turn(raw, seed=point, role=role, endpoint=endpoint,
                       is_isr=is_isr, force_rewrite=force)


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
        public = final_public_reply(raw_persuader, point=point, role='persuader',
                                   endpoint=persuader, method=method)
        persuader_text = public['content']
        direct_answer_raw = persuader_text if public['direct_answer_rewritten'] else None
        persuader_turn = {
            "round": round_number,
            "role": "persuader",
            "content": persuader_text,
            "raw_output": persuader_text,
            "prompt_type": "dialogue",
            "thinking_policy": "final_answer_only",
            "thinking_filtered": public['thinking_filtered'],
            "public_answer_retries": public['public_answer_retries'],
            "direct_answer_rewritten": direct_answer_raw is not None,
            "direct_answer_raw_output": persuader_text if direct_answer_raw is not None else None,
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
        public = final_public_reply(raw_persuadee, point=point, role='persuadee', endpoint=persuadee)
        persuadee_text = public['content']
        persuadee_direct_answer_raw = persuadee_text if public['direct_answer_rewritten'] else None
        persuadee_turn = {
            "round": round_number,
            "role": "persuadee",
            "content": persuadee_text,
            "raw_output": persuadee_text,
            "thinking_policy": "final_answer_only",
            "thinking_filtered": public['thinking_filtered'],
            "public_answer_retries": public['public_answer_retries'],
            "direct_answer_rewritten": persuadee_direct_answer_raw is not None,
            "direct_answer_raw_output": persuadee_text if persuadee_direct_answer_raw is not None else None,
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
            public = final_public_reply(raw, point=point, role='persuader',
                                       endpoint=persuader, method=method)
            text = public['content']
            direct_answer_raw = text if public['direct_answer_rewritten'] else None
            trajectories[index].append({
                "round": round_number,
                "role": "persuader",
                "content": text,
                "raw_output": text,
                "prompt_type": "dialogue",
                "thinking_policy": "final_answer_only",
                "thinking_filtered": public['thinking_filtered'],
                "public_answer_retries": public['public_answer_retries'],
                "direct_answer_rewritten": direct_answer_raw is not None,
                "direct_answer_raw_output": text if direct_answer_raw is not None else None,
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
            public = final_public_reply(raw, point=point, role='persuadee', endpoint=persuadee)
            text = public['content']
            direct_answer_raw = text if public['direct_answer_rewritten'] else None
            trajectories[index].append({
                "round": round_number,
                "role": "persuadee",
                "content": text,
                "raw_output": text,
                "thinking_policy": "final_answer_only",
                "thinking_filtered": public['thinking_filtered'],
                "public_answer_retries": public['public_answer_retries'],
                "direct_answer_rewritten": direct_answer_raw is not None,
                "direct_answer_raw_output": text if direct_answer_raw is not None else None,
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
                # Keep per-trajectory provenance explicit as well as in
                # run_config.json. This makes resumed and independently
                # inspected JSONL rows auditable without relying on sidecars.
                "compute_dtype": "bfloat16",
                "quantization": "none",
                "enable_thinking": False,
                "thinking_policy": "final_answer_only",
                "persuader_device": (
                    ",".join(f"cuda:{device}" for device in getattr(persuader, "devices", []))
                    or None
                ),
                "persuadee_device": (
                    ",".join(f"cuda:{device}" for device in getattr(persuadee, "devices", []))
                    or None
                ),
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
        "persuader_quantization": "none",
        "persuader_quantization_scheme": None,
        # Both the NF4 and full-precision inference paths use BF16 matrix
        # computation through BF16 autocast; recording it makes the run auditable.
        "persuader_compute_dtype": "bfloat16",
        "persuadee_compute_dtype": "bfloat16",
        "bf16_autocast": True,
        "persuader_double_quantization": False,
        "final_output_directory_label": "persuadee_simulation",
        "rounds": rounds,
        "temperature": temperature,
        "do_sample": temperature > 0,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "context_batch_size": context_batch_size,
        "thinking_policy": "final_answer_only",
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
    if persuader_quantization != "none":
        raise ValueError("This reproduction requires full BF16 weights; quantization is disabled")
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
