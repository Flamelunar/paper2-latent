#!/usr/bin/env python3
"""Blindly rejudge fixed REAL dialogues; never regenerate or read search traces."""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
import os
from pathlib import Path
from typing import Any
from evaluation.matched_protocol169 import goal_prompt as matched_goal_prompt, parse_goal as matched_parse_goal, win_prompt as matched_win_prompt, parse_win, VERSION as MATCHED_VERSION
from evaluation.status_policy import parse_failure, excluded_record
from evaluation.ledger import RequestLedger, StructuralFailure
from evaluation.runtime import LocalHFRunner, OpenAICompatibleRunner
from evaluation.revision_prompts import final_judge_prompt, PROMPT_VERSION
from evaluation.revision_schema import parse_final_judgment, SCHEMA_VERSION, strict_json, single_text
from evaluation.prompts import quality_prompt, outcome_stance_prompt
from evaluation.metrics import extract_json, token_f1, token_cosine
from evaluation.modern_pipeline import summarize
from utils.artifacts import check_lock, hash_object, sha256_file, code_fingerprint, last_records, write_json
from utils.evaluation_io import read_json_or_jsonl, append_jsonl


def quality_parser(raw: str) -> dict[str, int] | None:
    value = strict_json(raw)
    keys = ("persuasiveness", "logical_coherence", "helpfulness")
    if value is None or any(type(value.get(k)) is not int or not 1 <= value[k] <= 10 for k in keys):
        return None
    return {k: value[k] for k in keys}


def stance_parser(raw: str) -> dict[str, Any] | None:
    value = strict_json(raw)
    if value is None or value.get("stance") not in {"success", "partial_accept", "hesitant", "rejected"}:
        return None
    if type(value.get("target_semantic_alignment")) is not int or not 1 <= value["target_semantic_alignment"] <= 10:
        return None
    return value



def compare_with_reference(record: dict[str, Any], item: dict[str, Any], turns: list[dict[str, Any]],
                           ledger: RequestLedger, max_tokens: int, metric: str = "win_full") -> dict[str, Any]:
    generated = turns
    reference = record.get("reference_dialogue", [])
    if not isinstance(reference, list) or not reference or not generated:
        return {"status": "unknown", "reason": "missing_full_reference", "win_credit": None}
    for i, t in enumerate(reference):
        if not isinstance(t, dict) or t.get("role") != ("persuader" if i % 2 == 0 else "persuadee") or not isinstance(t.get("text"), str) or not t["text"].strip():
            raise ValueError("Invalid full reference dialogue")
    decisions: list[dict[str, Any]] = []
    for index, generated_first in enumerate((True, False)):
        try:
            decision = ledger.structured(
                "judge", matched_win_prompt(item, generated, reference, metric, generated_first),
                f"{metric}-comparison-{index}", parse_win, max_tokens=max_tokens)
            decisions.append({"generated_first": generated_first, **decision, "parse_status": "valid"})
        except Exception as exc:
            decisions.append({"generated_first": generated_first, "winner": None, "parse_status": "unknown",
                              "error": f"{type(exc).__name__}: {exc}"})
    if len(decisions) != 2 or any(d.get("winner") not in {"A", "B", "tie"} for d in decisions):
        return {"status": "unknown", "reason": "comparison_parse_failure", "decisions": decisions, "win_credit": None}
    outcomes = ["tie" if d["winner"] == "tie" else
                ("generated" if d["winner"] == ("A" if d["generated_first"] else "B") else "reference")
                for d in decisions]
    order_credits = [{"generated": 1.0, "reference": 0.0, "tie": 0.5}[o] for o in outcomes]
    credit = sum(order_credits) / 2
    return {"status": "completed", "decisions": decisions,
            "generated_more_persuasive": credit == 1.0,
            "reference_more_persuasive": credit == 0.0,
            "order_credits": order_credits, "mapped_outcomes": outcomes,
            "swap_conflict": outcomes[0] != outcomes[1],
            "strict_swap_conflict": set(outcomes) == {"generated", "reference"},
            "win_credit": credit,
            "comparison_rule": "mean of two anonymous orderings; each tie contributes 0.5"}


def replay_one(record: dict[str, Any], runner: Any, judge_id: str, *, max_tokens: int = 256,
               retry_limit: int = 1) -> dict[str, Any]:
    # Explicitly construct only the public target and REAL dialogue view.
    item = {k: record["item"][k] for k in ("id", "background", "goal", "success_criteria", "persuader", "persuadee", "domain")}
    turns = [{"role": t["role"], "text": t["text"]} for t in record["dialogue"]]
    if any(t.get("simulated") for t in record["dialogue"]):
        raise ValueError("A simulated reply appeared in the recorded real dialogue")
    ledger = RequestLedger({"judge": runner}, record["episode_id"] + "/judge/" + judge_id,
                           record["scenario_id"], record["rollout_seed_id"], retry_limit)
    judgments = []
    start = record.get("initial_history_length", 0)
    if type(start) is not int or start < 0 or start % 2 or (len(turns) - start) % 2:
        raise ValueError("Dialogue must contain complete role-alternating turns")
    for index, t in enumerate(turns):
        expected_role = "persuader" if index % 2 == 0 else "persuadee"
        if t["role"] != expected_role or single_text(json.dumps({expected_role + "_utterance": t["text"]}), expected_role + "_utterance") != t["text"]:
            raise ValueError("Recorded dialogue contains wrong-role or multi-turn contamination")
    completed_rounds = (len(turns) - start) // 2
    for turn in range(1, completed_rounds + 1):
        prefix = turns[:start + turn * 2]
        try:
            judgment = ledger.structured("judge", matched_goal_prompt(item, prefix), f"prefix-{turn}",
                lambda raw: matched_parse_goal(raw, prefix, item), max_tokens=max_tokens)
            judgments.append({"round": turn, **judgment, "parse_status": "valid"})
        except Exception as exc:
            judgments.append({"round": turn, "success": None, "parse_status": "unknown",
                              "error": f"{type(exc).__name__}: {exc}"})
    try:
        quality = ledger.structured("judge", quality_prompt(item, turns), "quality", quality_parser, max_tokens=max_tokens)
    except Exception:
        quality = {"persuasiveness": None, "logical_coherence": None, "helpfulness": None}
    try:
        stance = ledger.structured("judge", outcome_stance_prompt(item, turns), "stance", stance_parser, max_tokens=max_tokens)
    except Exception:
        stance = {"stance": None, "target_semantic_alignment": None}
    values = [j["success"] for j in judgments]
    success = True if True in values else (False if values and all(v is False for v in values) else None)
    outcome = next((t["text"] for t in reversed(turns) if t["role"] == "persuadee"), "")
    weights = {"success": 1.0, "partial_accept": .67, "hesitant": .33, "rejected": 0.0}
    weight = weights.get(stance["stance"])
    comparative = compare_with_reference(record, item, turns, ledger, max_tokens, "win_full")
    win1 = compare_with_reference(record, item, turns, ledger, max_tokens, "win1")
    result = {"id": record["id"], "scenario_id": record["scenario_id"], "episode_id": record["episode_id"],
        "method": record["method"], "backbone": record["backbone"], "rollout_seed_id": record["rollout_seed_id"],
        "training_seed": record.get("training_seed"), "judge_id": judge_id, "item": item,
        "dialogue_sha256": hash_object(turns), "status": "completed" if
            (success is not None and all(j["parse_status"] == "valid" for j in judgments)
             and all(v is not None for v in quality.values()) and stance["stance"] is not None
             and comparative["status"] == "completed" and win1["status"] == "completed") else "failed",
        "metrics": {"strict_success": success, "success_judgments": judgments,
            "success_round": next((j["round"] for j in judgments if j["success"] is True), None) if success else None,
            "final_success": values[-1] if values else None,
            "assigned_turn": next((j["round"] for j in judgments if j["success"] is True), record["max_real_turns"]) if success else record["max_real_turns"],
            "quality": quality, "stance": stance, "comparative": comparative, "win_full": comparative, "win1": win1,
            "target_alignment": {"token_proxy_weighted_f1": weight * token_f1(outcome, item["goal"]) if weight is not None else None,
                                 "token_proxy_weighted_cosine": weight * token_cosine(outcome, item["goal"]) if weight is not None else None,
                                 "BERT-F1*": None, "SemScore*": None}},
        "generation_metrics": {"persuader_reply_output_tokens":
            record.get("metrics", {}).get("persuader_reply_output_tokens")},
        "requests": ledger.requests, "costs": ledger.costs(), "generation_costs": record["costs"],
        "generation_protocol_hash": record["protocol_hash"]}
    return excluded_record(result) if parse_failure(result) else result


def run_replay(source: Path, output: Path, runner: Any, judge_id: str, runner_spec: dict[str, Any],
               *, expected_items: int = 500, max_tokens: int = 256, retry_limit: int = 1, allow_partial: bool = False) -> dict[str, Any]:
    records = last_records(read_json_or_jsonl(source))
    if len(records) > expected_items or (not allow_partial and len(records) != expected_items):
        raise ValueError(f"Expected {expected_items} frozen dialogues, found {len(records)}")
    identity = {(r.get("method"), r.get("backbone"), r.get("rollout_seed_id"),
                 r.get("protocol_hash"), r.get("data_manifest_hash")) for r in records}
    if len(identity) != 1 or len({r["scenario_id"] for r in records}) != len(records):
        raise ValueError("Mixed method/scenario/protocol sources in offline replay")
    generation_contract_path = source.parent / "evaluation_config.json"
    if generation_contract_path.is_file():
        generation_contract = json.loads(generation_contract_path.read_text(encoding="utf-8"))
        if (not allow_partial and {r["scenario_id"] for r in records} != set(generation_contract["scenario_ids"])):
            raise ValueError("Replay scenario IDs differ from the frozen generation source")
        if any(r.get("data_manifest_hash") != generation_contract["data_sha256"]
               or r.get("protocol_hash") != generation_contract["protocol_hash"] for r in records):
            raise ValueError("Replay dataset/protocol fingerprint mismatch")
    if expected_items in {169, 249, 500} and any(r.get("initial_history_length") != 0 for r in records):
        raise ValueError("Formal replay requires the same 500 scenarios with empty initial histories")
    contract = {"input_sha256": sha256_file(source), "judge": judge_id, "runner": runner_spec,
        "schema_version": SCHEMA_VERSION, "prompt_version": PROMPT_VERSION,
        "code_sha256": code_fingerprint(Path(__file__).resolve().parent),
        "max_tokens": max_tokens, "retry_limit": retry_limit, "expected_items": expected_items,
        "success_max_tokens": max_tokens, "matched_evaluation_version": MATCHED_VERSION,
        "comparison_schema": "anonymous-two-orderings-v2-tie-aware"}
    check_lock(output / "judge_config.json", contract)
    result_path = output / "results.jsonl"
    previous = last_records(read_json_or_jsonl(result_path)) if result_path.exists() else []
    done = {r["id"] for r in previous if r.get("status") == "completed" or parse_failure(r)}
    for record in records:
        if record["id"] in done:
            continue
        if parse_failure(record):
            skipped = excluded_record({"id": record["id"], "scenario_id": record["scenario_id"],
                "episode_id": record.get("episode_id"), "method": record.get("method"),
                "backbone": record.get("backbone"), "rollout_seed_id": record.get("rollout_seed_id"),
                "training_seed": record.get("training_seed"), "judge_id": judge_id,
                "item": record.get("item", {}), "dialogue_sha256": None, "metrics": {},
                "generation_metrics": record.get("metrics", {}), "generation_costs": record.get("costs", {}),
                "requests": [], "costs": {}, "generation_protocol_hash": record.get("protocol_hash")})
            append_jsonl(result_path, skipped)
            continue
        if record.get("status") != "completed":
            raise ValueError("Generation contains a failed dialogue; repair/resume generation before rejudging")
        judged = replay_one(record, runner, judge_id, max_tokens=max_tokens, retry_limit=retry_limit)
        append_jsonl(result_path, judged)
        if judged["status"] == "failed":
            break
    result = summarize(read_json_or_jsonl(result_path), expected=expected_items, judge=judge_id)
    write_json(output / "summary.json", result)
    if result["failed_or_missing"] and not allow_partial:
        raise RuntimeError("Offline judgments are incomplete; see results.jsonl and resume with the same locked protocol")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--judge-id", required=True)
    p.add_argument("--provider", choices=("local", "openai-compatible"), default="local")
    p.add_argument("--model", default="/app/hzj/models/Qwen3-14B")
    p.add_argument("--base-url", default="")
    p.add_argument("--base-url-env", default="JUDGE_API_BASE_URL", help="Endpoint is resolved at execution, not when emitting a plan")
    p.add_argument("--api-key-env", default="JUDGE_API_KEY")
    p.add_argument("--api-protocol", choices=("chat_completions", "responses"), default="chat_completions")
    p.add_argument("--api-token-parameter", choices=("max_tokens", "max_completion_tokens"), default="max_completion_tokens")
    p.add_argument("--api-send-sampling", action="store_true")
    p.add_argument("--api-send-seed", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--shared-local-endpoint", default="")
    p.add_argument("--precision", default="bf16", choices=("bf16", "fp16", "fp32"))
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--max-output-tokens", type=int, default=256)
    p.add_argument("--retry-limit", type=int, default=1)
    p.add_argument("--expected-items", type=int, default=500)
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    if "AAAI_PARSE_RETRY_LIMIT" in os.environ:
        a.retry_limit = int(os.environ["AAAI_PARSE_RETRY_LIMIT"])
    if a.provider == "local":
        if a.shared_local_endpoint:
            from evaluation.shared_local import SharedLocalRunner
            runner = SharedLocalRunner(a.model, a.shared_local_endpoint, a.precision, a.max_input_tokens)
        else:
            runner = LocalHFRunner(a.model, a.device, a.precision, a.max_input_tokens)
        spec = {"provider": a.provider, "model": a.model, "device": a.device, "precision": a.precision,
                "config_sha256": sha256_file(Path(a.model) / "config.json")}
        if a.shared_local_endpoint:
            spec["shared_local_endpoint"] = a.shared_local_endpoint
            spec["inference_backend"] = "shared_local_hf_v1"
    else:
        base_url = a.base_url or os.environ.get(a.base_url_env, "")
        if not base_url:
            raise SystemExit(f"API base URL is required (--base-url or {a.base_url_env})")
        runner = OpenAICompatibleRunner(a.model, base_url, a.api_key_env, protocol=a.api_protocol,
            token_parameter=a.api_token_parameter, send_sampling=a.api_send_sampling, send_seed=a.api_send_seed)
        spec = {"provider": a.provider, "model": a.model, "base_url": base_url, "base_url_env": a.base_url_env, "api_key_env": a.api_key_env,
                "protocol": a.api_protocol, "token_parameter": a.api_token_parameter,
                "send_sampling": a.api_send_sampling, "send_seed": a.api_send_seed}
    result = run_replay(a.input, a.output_dir, runner, a.judge_id, spec, expected_items=a.expected_items,
                       max_tokens=a.max_output_tokens, retry_limit=a.retry_limit, allow_partial=a.allow_partial)
    import json
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
