"""ISR simulation and ranked preference construction for baseline 3.

Adapted from the project's earlier paper reconstruction. Test evaluation imports
neither this module nor its prompts; it uses the common baseline evaluator.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .baseline_data import sha256, validate_fixed_splits
from .constants import DATA_DIR, PREFERENCE_SCENARIOS, SYSTEM_PERSUADER, data_path
from .gpu_capacity import wait_for_bf16_capacity
from .public_answer import public_turn, strip_thinking, model_preference_directory, POLICY
from .simulation import (
    ChatEndpoint, LocalTransformersEndpoint, endpoint_api_key,
    extract_utterance, generation_endpoint, read_jsonl, transcript, user_prompt,
)

WRITE_LOCK = threading.Lock()
SIMULATION_PROTOCOL = "persugpt_simulation_table5_reconstruction_v4_final_answer"


def complete_many(
    endpoint: ChatEndpoint | LocalTransformersEndpoint,
    requests: list[dict[str, Any]],
) -> list[str]:
    """Batch local BF16 requests while preserving the same prompt parameters."""
    if isinstance(endpoint, LocalTransformersEndpoint):
        return endpoint.complete_batch(requests)
    return [endpoint.complete(**request) for request in requests]


def latest_by_source(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["metadata"]["scenario_id"]): row for row in read_jsonl(path)}


def isr_prompt(seed: dict[str, Any], history: list[dict[str, str]] | None = None) -> str:
    scenario = seed["scenario"]
    history = seed["history"] if history is None else history
    return f"""The persuasion background: {scenario['background']}
The persuasion goal: {scenario['goal']}
The persuader: {scenario['persuader']}
The persuadee: {scenario['persuadee']}
Candidate persuasion strategies: {'; '.join(seed['strategies'])}

Historical dialogue:
{transcript(history)}

Analyze the persuadee's current intent, select one candidate strategy, and
write the next persuader response. Return exactly this three-line format:
Intent to Strategy Reasoning: <analysis>
Selected Strategy: <strategy>
Utterance: <persuader response>"""


def simulate_path(
    seed: dict[str, Any],
    candidate_full: str,
    persuader: ChatEndpoint | LocalTransformersEndpoint,
    user_model: ChatEndpoint | LocalTransformersEndpoint,
    simulation_rounds: int,
    top_p: float | None = None,
    top_k: int | None = None,
) -> list[dict[str, str]]:
    path = [public_turn(candidate_full, seed=seed, role="persuader", endpoint=persuader)]
    combined = [dict(turn) for turn in seed["history"]] + path
    # A reward prefix must contain user feedback. Round 1 is the candidate and
    # its reply; later rounds add one persuader follow-up and one user reply.
    for round_index in range(simulation_rounds):
        if round_index > 0:
            response = persuader.complete(
                system=SYSTEM_PERSUADER,
                user=isr_prompt(seed, combined),
                temperature=0.7,
                top_p=top_p,
                top_k=top_k,
            )
            turn = public_turn(response, seed=seed, role="persuader", endpoint=persuader)
            path.append(turn)
            combined.append(turn)

        system, prompt = user_prompt(seed, combined)
        response = user_model.complete(
            system=system,
            user=prompt,
            temperature=0.7,
            top_p=top_p,
            top_k=top_k,
        )
        turn = public_turn(response, seed=seed, role="persuadee", endpoint=user_model)
        path.append(turn)
        combined.append(turn)
    return path


def comparison_prompt(
    seed: dict[str, Any], path_x: list[dict[str, str]], path_y: list[dict[str, str]]
) -> str:
    scenario = seed["scenario"]
    return f"""I will provide a persuasion background, goal, persuader, persuadee,
historical conversation, and continuations in two parallel universes, Uni-X
and Uni-Y. Judge which universe the persuader performs better.

Rules adapted from Table 5 of Persuading across Diverse Domains:
1. Evaluate persuasiveness, user feedback, semantic relevance, emotional
   factors, factual correctness, and overall quality.
2. Summarize the history, then summarize Uni-X and Uni-Y separately.
3. Compare the two universes and decide where the persuader performs better.
4. Do not be affected by universe order; use only conversation content.
5. Choose exactly one universe; do not answer tie, both, or neither.

Background: {scenario['background']}
Goal: {scenario['goal']}
Persuader: {scenario['persuader']}
Persuadee: {scenario['persuadee']}

Historical dialogue:
{transcript(seed['history'])}

Uni-X continuation:
{transcript(path_x)}

Uni-Y continuation:
{transcript(path_y)}

Return only a JSON object:
{{"summary_history":"...","summary_X":"...","summary_Y":"...",
"explain":"objective comparison, ending with the decision",
"result":"Uni-X"}}
The result value must be exactly "Uni-X" or "Uni-Y"."""


def parse_comparison(raw: str) -> dict[str, str]:
    raw, _ = strip_thinking(raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"Judge output is not JSON: {raw[:300]}")
    value = json.loads(raw[start : end + 1])
    if value.get("result") not in {"Uni-X", "Uni-Y"}:
        raise ValueError(f"Invalid comparison result: {value.get('result')}")
    return {key: str(value.get(key, "")) for key in (
        "summary_history", "summary_X", "summary_Y", "explain", "result"
    )}


def compare_prefixes(
    seed: dict[str, Any],
    path_x: list[dict[str, str]],
    path_y: list[dict[str, str]],
    judge: ChatEndpoint | LocalTransformersEndpoint,
    weights: list[float],
) -> tuple[float, float, list[dict[str, Any]]]:
    requests = []
    for round_number, _ in enumerate(weights, 1):
        prefix_utterances = 2 * round_number
        requests.append({
            "system": "You are a careful path-comparison evaluator following the Table 5 criteria.",
            "user": comparison_prompt(
                seed, path_x[:prefix_utterances], path_y[:prefix_utterances]
            ),
            "temperature": 0.0,
            "json_mode": True,
        })
    raw_comparisons = complete_many(judge, requests)
    return score_prefix_comparisons(raw_comparisons, weights)


def score_prefix_comparisons(
    raw_comparisons: list[str], weights: list[float]
) -> tuple[float, float, list[dict[str, Any]]]:
    if len(raw_comparisons) != len(weights):
        raise ValueError("Judge must return one comparison for every reward prefix")
    decisions: list[dict[str, Any]] = []
    score_x = 0.0
    score_y = 0.0
    for round_number, (weight, raw) in enumerate(zip(weights, raw_comparisons), 1):
        parsed = parse_comparison(raw)
        prefix_utterances = 2 * round_number
        x_wins = parsed["result"] == "Uni-X"
        score_x += weight * int(x_wins)
        score_y += weight * int(not x_wins)
        decisions.append(
            {
                "simulation_round": round_number,
                "prefix_utterances": prefix_utterances,
                "weight": weight,
                "judgment": parsed,
            }
        )
    return score_x, score_y, decisions


def simulate_paths(
    seed: dict[str, Any],
    candidates: list[str],
    persuader: ChatEndpoint | LocalTransformersEndpoint,
    user_model: ChatEndpoint | LocalTransformersEndpoint,
    simulation_rounds: int,
) -> list[list[dict[str, str]]]:
    """Simulate both candidate continuations together using local batch decode."""
    return simulate_paths_batch(
        [seed], [candidates], persuader, user_model, simulation_rounds
    )[0]


def simulate_paths_batch(
    seeds: list[dict[str, Any]],
    candidates_by_seed: list[list[str]],
    persuader: ChatEndpoint | LocalTransformersEndpoint,
    user_model: ChatEndpoint | LocalTransformersEndpoint,
    simulation_rounds: int,
) -> list[list[list[dict[str, str]]]]:
    """Simulate several independent scenarios in shared local decode batches."""
    if len(seeds) != len(candidates_by_seed):
        raise ValueError("Every preference seed must have its own candidate pair")

    paths_by_seed: list[list[list[dict[str, str]]]] = []
    entry_seed_indexes: list[int] = []
    paths: list[list[dict[str, str]]] = []
    combined: list[list[dict[str, str]]] = []
    for seed_index, (seed, candidates) in enumerate(zip(seeds, candidates_by_seed)):
        if len(candidates) != 2:
            raise ValueError("Each preference scenario must have exactly two candidates")
        seed_paths = [
            [public_turn(candidate, seed=seed, role="persuader", endpoint=persuader)]
            for candidate in candidates
        ]
        paths_by_seed.append(seed_paths)
        for path in seed_paths:
            entry_seed_indexes.append(seed_index)
            paths.append(path)
            combined.append([dict(turn) for turn in seed["history"]] + path)

    for round_index in range(simulation_rounds):
        if round_index > 0:
            prompts = [
                {
                    "system": SYSTEM_PERSUADER,
                    "user": isr_prompt(seeds[seed_index], history),
                    "temperature": 0.7,
                }
                for seed_index, history in zip(entry_seed_indexes, combined)
            ]
            responses = complete_many(persuader, prompts)
            for seed_index, history, path, response in zip(entry_seed_indexes, combined, paths, responses):
                turn = public_turn(response, seed=seeds[seed_index], role="persuader", endpoint=persuader)
                path.append(turn)
                history.append(turn)

        prompts = []
        for seed_index, history in zip(entry_seed_indexes, combined):
            system, prompt = user_prompt(seeds[seed_index], history)
            prompts.append({"system": system, "user": prompt, "temperature": 0.7})
        responses = complete_many(user_model, prompts)
        for seed_index, history, path, response in zip(entry_seed_indexes, combined, paths, responses):
            turn = public_turn(response, seed=seeds[seed_index], role="persuadee", endpoint=user_model)
            path.append(turn)
            history.append(turn)

    return paths_by_seed


def build_preference_paths(
    seed: dict[str, Any],
    persuader: ChatEndpoint | LocalTransformersEndpoint,
    user_model: ChatEndpoint | LocalTransformersEndpoint,
    simulation_rounds: int,
) -> dict[str, Any]:
    """Generate both candidate paths without loading the comparison Judge."""
    return build_preference_paths_batch(
        [seed], persuader, user_model, simulation_rounds
    )[0]


def build_preference_paths_batch(
    seeds: list[dict[str, Any]],
    persuader: ChatEndpoint | LocalTransformersEndpoint,
    user_model: ChatEndpoint | LocalTransformersEndpoint,
    simulation_rounds: int,
) -> list[dict[str, Any]]:
    """Generate and simulate candidate pairs for a batch of scenarios."""
    if not seeds:
        return []
    candidate_requests = []
    for seed in seeds:
        prompt = isr_prompt(seed)
        candidate_requests.extend([
            {"system": SYSTEM_PERSUADER, "user": prompt, "temperature": 0.9},
            {"system": SYSTEM_PERSUADER, "user": prompt, "temperature": 0.9},
        ])
    generated_candidates = complete_many(persuader, candidate_requests)
    candidates_by_seed = []
    candidate_flags_by_seed = []
    for seed_index, seed in enumerate(seeds):
        pair, flags = [], []
        for raw in generated_candidates[seed_index * 2:seed_index * 2 + 2]:
            candidate, filtered = strip_thinking(raw)
            for _ in range(3):
                if candidate:
                    break
                raw = persuader.complete(system=SYSTEM_PERSUADER,
                                         user=isr_prompt(seed) + "\nReturn the final three-line answer, not thinking tags.",
                                         temperature=.9)
                candidate, found = strip_thinking(raw)
                filtered = filtered or found
            if not candidate:
                raise ValueError(f"Empty public candidate for scenario {seed['scenario_id']}")
            pair.append(candidate)
            flags.append(filtered)
        if len(pair) != 2:
            raise ValueError("Incomplete candidate pair")
        candidates_by_seed.append(pair)
        candidate_flags_by_seed.append(flags)
    for seed, candidates, flags in zip(seeds, candidates_by_seed, candidate_flags_by_seed):
        if extract_utterance(candidates[0]) == extract_utterance(candidates[1]):
            alternative = persuader.complete(
                system=SYSTEM_PERSUADER,
                user=isr_prompt(seed)
                + "\nUse a substantially different valid strategy from the first sample.",
                temperature=1.0,
            )
            candidates[1], found = strip_thinking(alternative)
            flags[1] = flags[1] or found
            if not candidates[1]:
                raise ValueError("Alternative candidate contains no final answer")

    paths_by_seed = simulate_paths_batch(
        seeds, candidates_by_seed, persuader, user_model, simulation_rounds
    )
    rows = []
    for seed, candidates, paths, flags in zip(seeds, candidates_by_seed, paths_by_seed, candidate_flags_by_seed):
        for path, filtered in zip(paths, flags):
            path[0]["thinking_filtered"] = path[0]["thinking_filtered"] or filtered
        rows.append({
            "seed": seed,
            "candidate_x": candidates[0],
            "candidate_y": candidates[1],
            "path_x": paths[0],
            "path_y": paths[1],
            "metadata": {
                "scenario_id": seed["scenario_id"],
                "protocol": SIMULATION_PROTOCOL,
                "thinking_policy": POLICY,
                "thinking_filtered_turns": sum(bool(turn.get("thinking_filtered")) for path in paths for turn in path),
            },
        })
    return rows


def judge_preference_paths(
    paths: dict[str, Any],
    judge: ChatEndpoint | LocalTransformersEndpoint,
    weights: list[float],
) -> dict[str, Any]:
    return judge_preference_paths_batch([paths], judge, weights)[0]


def judge_preference_paths_batch(
    paths_batch: list[dict[str, Any]],
    judge: ChatEndpoint | LocalTransformersEndpoint,
    weights: list[float],
) -> list[dict[str, Any]]:
    """Compare reward prefixes from several scenarios in shared decode batches."""
    if not paths_batch:
        return []
    requests: list[dict[str, Any]] = []
    for paths in paths_batch:
        seed = paths["seed"]
        path_x = paths["path_x"]
        path_y = paths["path_y"]
        for round_number, _ in enumerate(weights, 1):
            prefix_utterances = 2 * round_number
            requests.append({
                "system": "You are a careful path-comparison evaluator following the Table 5 criteria.",
                "user": comparison_prompt(
                    seed, path_x[:prefix_utterances], path_y[:prefix_utterances]
                ),
                "temperature": 0.0,
                "json_mode": True,
            })
    raw_comparisons = complete_many(judge, requests)
    if len(raw_comparisons) != len(paths_batch) * len(weights):
        raise ValueError("Judge returned an incomplete batch of preference comparisons")

    invalid_indices = []
    for index, raw in enumerate(raw_comparisons):
        try:
            parse_comparison(raw)
        except ValueError:
            invalid_indices.append(index)
    if invalid_indices:
        print(
            f"Retrying {len(invalid_indices)} malformed preference judgments "
            "with a 2048-token limit"
        )
        retry_requests = []
        for index in invalid_indices:
            request = dict(requests[index])
            request["max_tokens"] = 2048
            request["user"] += (
                "\n\nThe previous response was incomplete or invalid. Retry the same "
                "comparison and return one complete JSON object with string fields "
                "summary_history, summary_X, summary_Y, explain, and result. The "
                "result must be exactly Uni-X or Uni-Y. Do not use markdown."
            )
            retry_requests.append(request)
        retry_outputs = complete_many(judge, retry_requests)
        if len(retry_outputs) != len(invalid_indices):
            raise ValueError("Judge returned an incomplete retry batch")
        for index, raw in zip(invalid_indices, retry_outputs):
            raw_comparisons[index] = raw
        for raw in raw_comparisons:
            parse_comparison(raw)

    rows = []
    for index, paths in enumerate(paths_batch):
        start = index * len(weights)
        score_x, score_y, decisions = score_prefix_comparisons(
            raw_comparisons[start : start + len(weights)], weights
        )
        rows.append(preference_row_from_scores(paths, score_x, score_y, decisions, weights))
    return rows


def preference_row_from_scores(
    paths: dict[str, Any],
    score_x: float,
    score_y: float,
    decisions: list[dict[str, Any]],
    weights: list[float],
) -> dict[str, Any]:
    seed = paths["seed"]
    candidate_x = paths["candidate_x"]
    candidate_y = paths["candidate_y"]
    path_x = paths["path_x"]
    path_y = paths["path_y"]
    tied = score_x == score_y
    # The paper does not define reward-tie handling; prefer the judgment with
    # the longest available continuation instead of always favoring Uni-X.
    x_chosen = score_x > score_y or (
        tied and decisions[-1]["judgment"]["result"] == "Uni-X"
    )
    return {
        "instruction": isr_prompt(seed),
        "input": "",
        "output": [candidate_x, candidate_y] if x_chosen else [candidate_y, candidate_x],
        "history": [],
        "system": SYSTEM_PERSUADER,
        "metadata": {
            "scenario_id": seed["scenario_id"],
            "source_id": seed["source_id"],
            "session_id": seed["session_id"],
            "target_turn": seed["target_turn"],
            "protocol": SIMULATION_PROTOCOL,
            "thinking_policy": POLICY,
            "thinking_filtered_turns": sum(bool(turn.get("thinking_filtered")) for path in (path_x, path_y) for turn in path),
            "weights": weights,
            "score_x": score_x,
            "score_y": score_y,
            "reward_tied": tied,
            "tie_break": "longest_prefix_judgment" if tied else None,
            "chosen_universe": "X" if x_chosen else "Y",
            "path_x": path_x,
            "path_y": path_y,
            "prefix_decisions": decisions,
        },
    }


def build_one_preference(
    seed: dict[str, Any],
    persuader: ChatEndpoint | LocalTransformersEndpoint,
    user_model: ChatEndpoint | LocalTransformersEndpoint,
    judge: ChatEndpoint | LocalTransformersEndpoint,
    weights: list[float],
) -> dict[str, Any]:
    """Single-record convenience path retained for callers using live models."""
    paths = build_preference_paths(seed, persuader, user_model, len(weights))
    return judge_preference_paths(paths, judge, weights)


def append_row(path: Path, row: dict[str, Any]) -> None:
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def build_preferences(
    *,
    persuader_url: str,
    persuader_model: str,
    user_url: str,
    user_model_name: str,
    user_api_key: str | None,
    judge_url: str,
    judge_model: str,
    simulation_rounds: int,
    workers: int,
    simulation_batch_size: int,
    max_retries: int,
    limit: int | None,
    force: bool,
    preference_directory: Path | None = None,
) -> Path:
    if simulation_rounds < 1:
        raise ValueError("simulation_rounds must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if simulation_batch_size < 1:
        raise ValueError("simulation_batch_size must be positive")
    if max_retries < 1:
        raise ValueError("max_retries must be positive")
    validate_fixed_splits()
    for url, model_name in ((persuader_url, persuader_model),
                            (user_url, user_model_name), (judge_url, judge_model)):
        if url.startswith("local://") and not Path(model_name).expanduser().is_dir():
            raise FileNotFoundError(f"Local simulation model is unavailable: {model_name}")
    def wait_for_local_phase(
        phase_models: list[tuple[str, str]], description: str,
    ) -> None:
        local_models = [
            (url.removeprefix("local://") or "cuda:0", model_name)
            for url, model_name in phase_models if url.startswith("local://")
        ]
        if not local_models:
            return
        import torch
        device_ids = sorted({
            device_id
            for device_spec, _ in local_models
            for device_id in LocalTransformersEndpoint._parse_devices(device_spec)
        })
        for device_id in device_ids:
            with torch.cuda.device(device_id):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(f"cuda:{device_id} does not support BF16")
                if "A100" not in torch.cuda.get_device_name(device_id):
                    raise RuntimeError(
                        f"Simulation is restricted to A100 GPUs; cuda:{device_id} is "
                        f"{torch.cuda.get_device_name(device_id)}"
                    )

        wait_for_bf16_capacity(
            [model_name for _, model_name in local_models],
            device_ids,
            reserve_gib=8,
            description=description,
        )

    seeds = read_jsonl(data_path("preference_seeds"))
    if len(seeds) != PREFERENCE_SCENARIOS:
        raise ValueError(f"Expected {PREFERENCE_SCENARIOS} preference seeds, found {len(seeds)}")
    if limit is not None:
        seeds = seeds[: max(0, limit)]
    preference_directory = preference_directory or model_preference_directory(persuader_model, DATA_DIR)
    preference_directory.mkdir(parents=True, exist_ok=True)
    output = preference_directory / "simulation_preferences.jsonl"
    paths_output = output.with_name(f"{output.stem}_paths{output.suffix}")
    config_path = preference_directory / "simulation_preferences_config.json"
    configuration = {
        "protocol": SIMULATION_PROTOCOL,
        "seeds_sha256": sha256(data_path("preference_seeds")),
        "persuader_url": persuader_url, "persuader_model": persuader_model,
        "user_url": user_url, "user_model": user_model_name,
        "judge_url": judge_url, "judge_model": judge_model,
        "simulation_rounds": simulation_rounds,
        "thinking_policy": POLICY,
    }
    if force:
        for path in (output, paths_output, config_path):
            if path.exists():
                path.unlink()
    if (output.exists() or paths_output.exists()) and not config_path.exists():
        raise ValueError("Existing preferences have no run configuration; use --force to rebuild")
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != configuration:
            raise ValueError("Simulation settings changed; use --force to start a new preference run")
    else:
        config_path.write_text(json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if output.exists():
        protocols = {row.get("metadata", {}).get("protocol") for row in read_jsonl(output)}
        if protocols != {SIMULATION_PROTOCOL}:
            raise ValueError(
                f"Stale simulation protocol in {output}: {protocols}. Rerun with --force."
            )
    if paths_output.exists():
        path_protocols = {
            row.get("metadata", {}).get("protocol") for row in read_jsonl(paths_output)
        }
        if path_protocols != {SIMULATION_PROTOCOL}:
            raise ValueError(
                f"Stale simulation path protocol in {paths_output}: {path_protocols}. "
                "Rerun with --force."
            )
    done = latest_by_source(output)
    if output.exists() and len(done) != len(read_jsonl(output)):
        raise ValueError("Duplicate preference scenario IDs in existing output")
    pending = [seed for seed in seeds if str(seed["scenario_id"]) not in done]
    generated_paths = latest_by_source(paths_output)
    if paths_output.exists() and len(generated_paths) != len(read_jsonl(paths_output)):
        raise ValueError("Duplicate preference scenario IDs in staged simulation paths")
    paths_pending = [
        seed for seed in pending if str(seed["scenario_id"]) not in generated_paths
    ]
    local_key = os.getenv("LOCAL_OPENAI_API_KEY", "local-not-secret")
    local_user = user_url.startswith("local://")
    local_judge = judge_url.startswith("local://")
    user_key = local_key if local_user else endpoint_api_key(user_url, user_api_key)
    judge_key = local_key if local_judge else endpoint_api_key(judge_url)
    if not judge_key:
        raise RuntimeError("The selected judge API endpoint requires a configured API key")
    if not user_key:
        raise RuntimeError("The selected persuadee API endpoint requires a configured API key")
    weights = [float(simulation_rounds - index) for index in range(simulation_rounds)]

    if paths_pending:
        wait_for_local_phase(
            [(persuader_url, persuader_model), (user_url, user_model_name)],
            "Persuader and Persuadee simulation",
        )
        persuader = generation_endpoint(
            base_url=persuader_url,
            model=persuader_model,
            api_key=local_key,
            max_retries=max_retries,
        )
        user_model = generation_endpoint(
            base_url=user_url,
            model=user_model_name,
            api_key=user_key,
            max_retries=max_retries,
        )

        path_batches = [
            paths_pending[index : index + simulation_batch_size]
            for index in range(0, len(paths_pending), simulation_batch_size)
        ]

        def path_task(seed_batch: list[dict[str, Any]]) -> int:
            rows = build_preference_paths_batch(
                seed_batch, persuader, user_model, simulation_rounds
            )
            for row in rows:
                row["metadata"]["path_generation_batch_size"] = len(seed_batch)
                append_row(paths_output, row)
            return len(rows)

        progress = tqdm(total=len(paths_pending), desc="Simulation paths", unit="history")
        try:
            if workers == 1:
                for seed_batch in path_batches:
                    progress.update(path_task(seed_batch))
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(path_task, seed_batch) for seed_batch in path_batches]
                    for future in as_completed(futures):
                        progress.update(future.result())
        finally:
            progress.close()
            persuader.close()
            user_model.close()

    generated_paths = latest_by_source(paths_output)
    if not {str(seed["scenario_id"]) for seed in pending}.issubset(generated_paths):
        raise ValueError("Path simulation did not complete every pending preference scenario")
    paths_to_judge = [generated_paths[str(seed["scenario_id"])] for seed in pending]

    if paths_to_judge:
        wait_for_local_phase([(judge_url, judge_model)], "Preference comparison Judge")
        judge = generation_endpoint(
            base_url=judge_url,
            model=judge_model,
            api_key=judge_key,
            max_retries=max_retries,
        )
        # A single scenario expands to one request per reward prefix. Keep
        # path generation batched, but compare one scenario at a time so the
        # long-context Judge cannot accumulate a large KV-cache batch.
        judge_batch_size = 1
        judge_batches = [
            paths_to_judge[index : index + judge_batch_size]
            for index in range(0, len(paths_to_judge), judge_batch_size)
        ]

        def judge_task(paths_batch: list[dict[str, Any]]) -> int:
            rows = judge_preference_paths_batch(paths_batch, judge, weights)
            for row in rows:
                append_row(output, row)
            return len(rows)

        progress = tqdm(total=len(paths_to_judge), desc="Preference comparisons", unit="history")
        try:
            if workers == 1:
                for paths_batch in judge_batches:
                    progress.update(judge_task(paths_batch))
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(judge_task, paths_batch) for paths_batch in judge_batches]
                    for future in as_completed(futures):
                        progress.update(future.result())
        finally:
            progress.close()
            judge.close()
    rows = latest_by_source(output)
    selected_ids = {str(seed["scenario_id"]) for seed in seeds}
    if not selected_ids.issubset(rows):
        raise ValueError("Simulation did not complete every selected preference scenario")
    if limit is None and set(rows) != selected_ids:
        raise ValueError(f"Expected exactly {PREFERENCE_SCENARIOS} preference scenarios, found {len(rows)}")
    path_batch_sizes: dict[str, int] = {}
    for row in read_jsonl(paths_output):
        batch_size = str(row.get("metadata", {}).get("path_generation_batch_size", 1))
        path_batch_sizes[batch_size] = path_batch_sizes.get(batch_size, 0) + 1
    summary = {
        "protocol": SIMULATION_PROTOCOL,
        "rows": len(rows),
        "simulation_rounds_k": simulation_rounds,
        "judge_comparison_batch_size": 1,
        "path_generation_batch_sizes": path_batch_sizes,
        "path_semantics": (
            "Each reward prefix contains the initial candidate, a persuadee reply, "
            "and for later rounds one additional persuader/persuadee exchange."
        ),
        "weights": weights,
        "persuader_url": persuader_url,
        "persuader_model": persuader_model,
        "user_url": user_url,
        "user_model": user_model_name,
        "judge_url": judge_url,
        "judge_model": judge_model,
        "persuader_compute_dtype": "bfloat16" if persuader_url.startswith("local://") else None,
        "user_compute_dtype": "bfloat16" if local_user else None,
        "judge_compute_dtype": "bfloat16" if local_judge else None,
        "disclosure": (
            "The paper does not publish k or numerical w_j values. This run uses "
            "the recorded configurable reconstruction values."
        ),
    }
    (preference_directory / "simulation_preferences_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output
