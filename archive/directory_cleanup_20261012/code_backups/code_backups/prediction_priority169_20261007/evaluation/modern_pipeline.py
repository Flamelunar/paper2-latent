"""Generate real dialogues for the five-method matrix; judges replay separately."""
from __future__ import annotations
from dataclasses import asdict
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
import time
from typing import Any
from evaluation.ledger import RequestLedger, StructuralFailure
from evaluation.status_policy import parse_failure, excluded_record
from evaluation.matched_protocol169 import goal_prompt as matched_goal_prompt, parse_goal as matched_parse_goal, VERSION as MATCHED_VERSION
FACTUALITY_VERSION = 'disabled-prompt-constraints-only-v1'
from evaluation.ma2p import ma2p_action
from evaluation.revision_config import RevisionConfig, build_runners
from evaluation.revision_prompts import (PROMPT_VERSION, utterance_prompt, strategy_prompt,
    structured_strategy_prompt, strategy_score_prompt, environment_prompt, final_judge_prompt)
from evaluation.revision_schema import (SCHEMA_VERSION, single_text, parse_score, parse_final_judgment,
    parse_eval_structured_strategy as parse_structured_strategy, parse_strategy_output as parse_strategy)
from evaluation.rollout import PlannerConfig, RolloutPlanner, append_pair
from utils.artifacts import sha256_file, code_fingerprint, hash_object, write_json, check_lock, last_records
from utils.evaluation_io import append_jsonl, read_json_or_jsonl

ROOT = Path(__file__).resolve().parents[1]
MAX_CONSECUTIVE_EPISODE_FAILURES = 5


def _reference_persuader_replies(row: dict[str, Any], scenario: dict[str, Any],
                                 audit: dict[str, Any] | None = None) -> list[str]:
    """Extract the held-out Persuader answer for offline comparison only.

    The reference dialogue is deliberately kept out of all generation prompts.
    It is copied into the completed record so the blind judge can compare the
    generated Persuader turns with the held-out answer after generation.
    """
    source = row.get("dialog", scenario.get("dialog", []))
    if not isinstance(source, list):
        raise ValueError("Held-out dialogue must be a list")
    replies: list[str] = []
    for turn_index, turn in enumerate(source):
        role, text = "", ""
        if isinstance(turn, dict):
            role = turn.get("role", "")
            if not isinstance(role, str):
                raise ValueError("Invalid reference role type")
            role = role.strip().lower()
            text = turn.get("text", turn.get("content", ""))
        elif isinstance(turn, str) and ":" in turn:
            role, text = turn.split(":", 1)
            role, text = role.strip().lower(), text.strip()
        if role == "persuader":
            if not isinstance(text, str):
                raise ValueError("Invalid reference Persuader utterance")
            if not text.strip():
                # The frozen corpus has explicit 'persuader:' empty slots.
                # They are not reference utterances; retain the scenario and
                # record the omitted slot rather than inserting 'None'.
                if audit is not None:
                    audit["omitted_empty_turn_indices"].append(turn_index)
                continue
            cleaned = single_text(json.dumps({"persuader_utterance": text}), "persuader_utterance")
            if cleaned is None:
                raise ValueError("Reference answer has no valid current-role utterance")
            if audit is not None and cleaned != text.strip():
                audit["truncated_reference_turn_indices"].append(turn_index)
            replies.append(cleaned)
    return replies


def normalize_item(row: dict[str, Any], index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    scenario = row.get("scenario") if isinstance(row.get("scenario"), dict) else row
    if not all(isinstance(scenario.get(k), str) and scenario[k].strip() for k in ("background", "goal")):
        raise ValueError("Scenario background and goal must be nonempty strings")
    source_index = row.get("cross_domain_source_index")
    identifier = row.get("id", row.get("uid", f"ctompersu_{int(source_index):05d}" if source_index is not None else str(index)))
    domains = scenario.get("domain", ["unknown"])
    domains = domains if isinstance(domains, list) else [str(domains)]
    public = {"id": str(identifier), "background": str(scenario.get("background", "")).strip(),
        "goal": str(scenario.get("goal", "")).strip(),
        "success_criteria": scenario.get("success_criteria", [str(scenario.get("goal", ""))]),
        "persuader": str(scenario.get("persuader", "Persuader")), "persuadee": str(scenario.get("persuadee", "Persuadee")),
        "domains": domains, "domain": str(row.get("cross_domain_primary_domain", scenario.get("cross_domain_primary_domain", domains[0] if domains else "unknown")))}
    # This field is consumed only by judge_dialogues.py after generation.
    # public_context() enumerates its allowed fields and therefore never sends
    # the future reference answer to any generation role.
    public["_offline_reference_audit"] = {"omitted_empty_turn_indices": [], "truncated_reference_turn_indices": []}
    public["_offline_reference_persuader_replies"] = _reference_persuader_replies(row, scenario, public["_offline_reference_audit"])
    public["_offline_reference_dialogue"] = []
    for i, entry in enumerate(row.get("dialog", [])):
        if not isinstance(entry, str) or ":" not in entry:
            raise ValueError("Full reference must contain role-labelled utterances")
        role, text = entry.split(":", 1)
        role, text = role.strip().lower(), text.strip()
        if role != ("persuader" if i % 2 == 0 else "persuadee") or not text:
            raise ValueError("Full reference must be nonempty and alternating")
        public["_offline_reference_dialogue"].append({"role": role, "text": text})
    if not public["background"] or not public["goal"]:
        raise ValueError(f"Missing public background or exact goal: {identifier}")
    if not isinstance(public["success_criteria"], list) or not public["success_criteria"] or not all(isinstance(c, str) and c.strip() for c in public["success_criteria"]):
        raise ValueError(f"Invalid success criteria: {identifier}")
    # Recorded corpus `dialog` is a FUTURE reference, not the live history.
    # Only an explicit, user-supplied initial_history is a usable prefix.
    initial = row.get("initial_history", [])
    if not isinstance(initial, list):
        raise ValueError("initial_history must contain explicit observed turns")
    public["initial_history"] = []
    for t in initial:
        if not isinstance(t, dict) or t.get("role") not in {"persuader", "persuadee"}:
            raise ValueError("Invalid initial history turn")
        text = t.get("text", t.get("content", ""))
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty initial history turn")
        public["initial_history"].append({"role": t["role"], "text": text.strip()})
    private = {k: scenario[k] for k in ("preventive", "generative") if k in scenario}
    return public, private


def parse_world_strategy(raw: str, analysis_budget: int) -> dict[str, Any] | None:
    """The formal strategy protocol never turns a predicted reply into a strategy."""
    return parse_strategy(raw)

def planner_config(cfg: RevisionConfig) -> PlannerConfig:
    return PlannerConfig(root_actions=cfg.root_actions, reaction_samples_per_action=cfg.reaction_samples,
        max_depth=cfg.max_depth, future_actions_per_node=cfg.future_actions, max_candidate_tokens=cfg.max_candidate_tokens,
        max_reaction_tokens=cfg.max_plan_tokens, max_score_tokens=cfg.max_score_tokens,
        proposer_temperature=cfg.proposer_temperature, proposer_top_p=cfg.proposer_top_p,
        world_model_temperature=cfg.world_model_temperature, world_model_top_p=cfg.world_model_top_p,
        analysis_budget=cfg.wm_analysis_budget, structural_retry_limit=cfg.retry_limit)


def select_action(item: dict[str, Any], history: list[dict[str, Any]], ledger: RequestLedger,
                  cfg: RevisionConfig, turn: int, *, node_prefix: str | None = None) -> dict[str, Any]:
    remaining, node = cfg.max_rounds - turn + 1, node_prefix or f"turn-{turn}"
    if cfg.method == "zero_shot":
        strategy = ledger.structured("world_model", strategy_prompt(item, history, turn),
            node + "/zero-shot-strategy", parse_strategy, max_tokens=cfg.max_plan_tokens,
            temperature=cfg.world_model_temperature, top_p=cfg.world_model_top_p)
        text = ledger.structured("persuader", utterance_prompt(item, history, remaining, strategy["strategy"]),
            node + "/zero-shot-realize", lambda raw: single_text(raw, "persuader_utterance"),
            max_tokens=cfg.max_candidate_tokens, temperature=cfg.proposer_temperature,
            top_p=cfg.proposer_top_p)
        return {"selected_action_id": node + "/action-0", "selected_strategy": strategy,
                "selected_text": text, "fallback_used": False, "trace": [],
                "candidate_actions": [{"text": text}],
                "search_summary": {"world_model_prediction_count": 1, "root_actions": 1}}
    if cfg.method == "ma2p":
        return ma2p_action(item, history, ledger, cfg, turn, node_prefix=node)
    if cfg.method == "strategy_selector":
        # Strategy-SFT samples multiple STRATEGIES, then the same backbone
        # scores them. Only the chosen strategy reaches the fixed Persuader.
        strategies, seen, trace = [], set(), []
        for i in range(cfg.root_actions + cfg.retry_limit):
            strategy = ledger.structured("world_model", strategy_prompt(item, history, turn), f"{node}/strategy-{i}",
                parse_strategy, max_tokens=cfg.max_score_tokens, temperature=cfg.world_model_temperature,
                top_p=cfg.world_model_top_p)
            key = hash_object(strategy["strategy"])
            if key not in seen:
                seen.add(key)
                strategies.append(strategy)
            if len(strategies) >= cfg.root_actions:
                break
        scores = []
        for i, strategy in enumerate(strategies):
            assessment = ledger.structured("world_model", strategy_score_prompt(item, history, strategy["strategy"]),
                f"{node}/strategy-{i}/score", parse_score, max_tokens=cfg.max_score_tokens)
            scores.append(assessment["score"])
            trace.append({"strategy_id": f"strategy-{i}", "strategy": strategy, **assessment})
        selected = max(range(len(strategies)), key=lambda i: scores[i])
        text = ledger.structured("persuader", utterance_prompt(item, history, remaining, strategies[selected]["strategy"]),
            f"{node}/strategy-{selected}/execute", lambda raw: single_text(raw, "persuader_utterance"),
            max_tokens=cfg.max_candidate_tokens, temperature=cfg.proposer_temperature, top_p=cfg.proposer_top_p)
        return {"selected_action_id": f"strategy-{selected}", "selected_strategy": strategies[selected],
                "selected_text": text, "trace": trace, "fallback_used": False,
                "search_summary": {"world_model_prediction_count": 0, "root_actions": len(strategies)}}
    # SFT and standard GRPO controls keep their full Analysis + Strategy
    # output. Only the strategy crosses the fixed Persuader boundary.
    structured_methods = {"sft", "grpo_only", "sft_grpo"}
    if cfg.method in structured_methods:
        prompt = structured_strategy_prompt(item, history, turn, cfg.wm_analysis_budget)
        parser = parse_structured_strategy
    else:
        prompt = strategy_prompt(item, history, turn)
        parser = lambda raw: parse_world_strategy(raw, cfg.wm_analysis_budget)
    strategy = ledger.structured("world_model", prompt,
        node + "/strategy", parser,
        max_tokens=cfg.max_plan_tokens,
        temperature=cfg.world_model_temperature, top_p=cfg.world_model_top_p)
    text = ledger.structured("persuader", utterance_prompt(item, history, remaining, strategy["strategy"]),
        node + "/realize", lambda raw: single_text(raw, "persuader_utterance"),
        max_tokens=cfg.max_candidate_tokens, temperature=cfg.proposer_temperature,
        top_p=cfg.proposer_top_p)
    return {"selected_action_id": node + "/action-0", "selected_strategy": strategy,
            "selected_text": text, "fallback_used": False, "trace": [],
            "candidate_actions": [{"text": text}],
            "search_summary": {"world_model_prediction_count": 1, "root_actions": 1}}


def generate_episode(item: dict[str, Any], private: dict[str, Any], cfg: RevisionConfig,
                     runners: dict[str, Any], contract: dict[str, Any],
                     episode_resample_index: int = 0) -> dict[str, Any]:
    reference = list(item.get("_offline_reference_persuader_replies", []))
    reference_audit = item.get("_offline_reference_audit", {})
    # Remove offline answers and arbitrary extra dataset fields at the boundary,
    # before any generation role sees this object.
    full_reference = item.get("_offline_reference_dialogue", [])
    item = {k: item[k] for k in ("id", "background", "goal", "success_criteria", "persuader", "persuadee", "domain", "initial_history", "domains") if k in item}
    episode_id = f"{cfg.method}/{cfg.backbone or 'shared'}/{item['id']}/r{cfg.repeat_id}"
    ledger = RequestLedger(runners, episode_id, item["id"], cfg.repeat_id, cfg.parse_retry_limit,
                           episode_resample_index=episode_resample_index)
    turns = [dict(t) for t in item["initial_history"]]
    steps = []
    started = time.perf_counter()
    record: dict[str, Any] = {"id": item["id"], "scenario_id": item["id"], "episode_id": episode_id,
        "method": cfg.method, "backbone": cfg.backbone, "training_seed": cfg.training_seed,
        "rollout_seed_id": cfg.repeat_id, "item": item, "max_real_turns": cfg.max_rounds,
        "episode_resample_index": episode_resample_index,
        "initial_history_length": len(turns), "protocol_hash": contract["protocol_hash"], "status": "running",
        "prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION,
        "data_manifest_hash": contract["data_sha256"], "model_manifest_hash": contract.get("model_manifest_sha256"),
        "scenario_ids_sha256": contract.get("scenario_ids_sha256"),
        "factuality_version": FACTUALITY_VERSION, "matched_evaluation_version": MATCHED_VERSION,
        "reference_answer": reference, "reference_dialogue": full_reference,
        "reference_answer_audit": reference_audit,
        "reference_answer_source": "eval.dialog.persuader_turns_offline_only"}
    try:
        for turn in range(1, cfg.max_rounds + 1):
            decision_started = time.perf_counter()
            before = [dict(t) for t in turns]
            request_start = len(ledger.requests)
            plan = select_action(item, turns, ledger, cfg, turn, node_prefix=f"turn-{turn}")
            plan["factuality_policy"] = "no_online_audit_or_regeneration; shared_prompt_constraints_only"
            plan["decision_latency_ms"] = (time.perf_counter() - decision_started) * 1000
            text = plan["selected_text"]
            accepted_persuader_requests = [request for request in ledger.requests[request_start:]
                if request.get("role") == "persuader" and request.get("structural_valid") is True]
            if len(accepted_persuader_requests) != 1:
                raise RuntimeError("Each real turn must have exactly one accepted Persuader reply")
            output_tokens = accepted_persuader_requests[0].get("output_tokens")
            plan["persuader_output_tokens"] = output_tokens if isinstance(output_tokens, int) else None
            # Only this single exact root action is ever sent to the environment.
            reply = ledger.structured("persuadee",
                environment_prompt(item, before + [{"role": "persuader", "text": text}], private, turn == cfg.max_rounds),
                f"turn-{turn}/real-reply", lambda raw: single_text(raw, "persuadee_utterance", allow_plain=True),
                max_tokens=cfg.max_reply_tokens, temperature=cfg.persuadee_temperature, top_p=cfg.persuadee_top_p,
                seed_key=f"environment/turn-{turn}")
            turns = append_pair(before, text, reply, simulated=False)
            judgment = ledger.structured("online_judge", matched_goal_prompt(item, turns), f"turn-{turn}/real-judge",
                lambda raw: matched_parse_goal(raw, turns, item), max_tokens=cfg.max_score_tokens)
            plan.update(round=turn, history_before=before, persuader=text, persuadee=reply, real_judgment=judgment)
            steps.append(plan)
            if judgment["success"] or judgment["end_requested"]:
                break
        record["status"] = "completed"
        successes = [s["real_judgment"]["success"] for s in steps]
        reply_token_counts = [s.get("persuader_output_tokens") for s in steps]
        record["metrics"] = {"strict_success": any(successes),
            "success_round": next((s["round"] for s in steps if s["real_judgment"]["success"]), None),
            "final_success": successes[-1] if successes else None,
            "assigned_turn": next((s["round"] for s in steps if s["real_judgment"]["success"]), cfg.max_rounds),
            "persuader_reply_output_tokens": (sum(reply_token_counts)
                if all(isinstance(count, int) for count in reply_token_counts) else None)}
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", metrics={"strict_success": None})
    record.update(dialogue=turns, steps=steps, requests=ledger.requests, costs=ledger.costs(),
        end_to_end_latency_ms=(time.perf_counter() - started) * 1000)
    if parse_failure(record):
        record = excluded_record(record)
    return record


def summarize(records: list[dict[str, Any]], *, expected: int | None = None, **metadata: Any) -> dict[str, Any]:
    rows = last_records(records)
    total = expected if expected is not None else len(rows)
    excluded = [r for r in rows if parse_failure(r)]
    known = [r for r in rows if not parse_failure(r) and r.get("status") == "completed"
             and type(r.get("metrics", {}).get("strict_success")) is bool]
    successes = sum(r["metrics"]["strict_success"] for r in known)
    missing = total - len(known) - len(excluded)
    by_domain = {}
    domains = sorted({r.get("item", {}).get("domain", "unknown") for r in rows})
    for domain in domains:
        group = [r for r in rows if r.get("item", {}).get("domain", "unknown") == domain]
        subset = [r for r in group if r in known]
        subset_excluded = [r for r in group if r in excluded]
        hits = sum(r["metrics"]["strict_success"] for r in subset)
        denom = len(subset)
        by_domain[domain] = {"planned": len(group), "valid_denominator": denom,
            "excluded_parse_failures": len(subset_excluded), "successes": hits,
            "success_percent": 100 * hits / denom if denom and len(subset) + len(subset_excluded) == len(group) else None}
    denom = len(known)
    return {"schema_version": SCHEMA_VERSION, **metadata, "expected": total, "recorded": len(rows),
        "known": denom, "valid_denominator": denom, "excluded_parse_failures": len(excluded),
        "failed_or_missing": missing, "successes": successes,
        "success_percent": 100 * successes / denom if denom and missing == 0 else None,
        "success_bounds_percent": [100 * successes / denom, 100 * (successes + missing) / denom]
            if denom and missing == 0 else None,
        "by_domain": by_domain, "costs": {"input_tokens": sum(r.get("costs", {}).get("input_tokens", 0) for r in rows),
        "output_tokens": sum(r.get("costs", {}).get("output_tokens", 0) for r in rows),
        "usage_complete": all(r.get("costs", {}).get("usage_complete", False) for r in rows)}}


def enforce_execution_contract(lock_path: Path, contract: dict[str, Any], output: Path,
                               parse_retry_limit: int) -> dict[str, Any]:
    """Allow one audited parse-policy update while preserving the frozen protocol lock."""
    policy_path = output / "execution_policy.json"
    current_code = contract["code_sha256"]
    if policy_path.exists():
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if policy.get("current_code_sha256") != current_code:
            raise ValueError("Code changed after the recorded execution-policy amendment")
        if policy.get("parse_retry_limit") != parse_retry_limit:
            raise ValueError("Parse retry policy differs from the recorded execution-policy amendment")

    if not lock_path.exists():
        check_lock(lock_path, contract)
        write_json(policy_path, {
            "protocol_hash": contract.get("protocol_hash"),
            "frozen_code_sha256": current_code,
            "current_code_sha256": current_code,
            "parse_retry_limit": parse_retry_limit,
        })
        return contract

    frozen = json.loads(lock_path.read_text(encoding="utf-8"))
    if policy_path.exists() and policy.get("protocol_hash") != frozen.get("protocol_hash"):
        raise ValueError("Execution-policy amendment belongs to a different frozen protocol")
    if frozen == contract:
        return contract

    old, new = deepcopy(frozen), deepcopy(contract)
    for document in (old, new):
        document.pop("code_sha256", None)
        document.pop("protocol_hash", None)
        document.get("config", {}).pop("parse_retry_limit", None)
    if old != new or parse_retry_limit != 0:
        raise ValueError("Evaluation contract differs beyond the approved zero-parse-retry policy")

    if not policy_path.exists():
        write_json(policy_path, {
            "protocol_hash": frozen.get("protocol_hash"),
            "frozen_code_sha256": frozen.get("code_sha256"),
            "current_code_sha256": current_code,
            "parse_retry_limit": 0,
            "note": "Schema-invalid samples are excluded without regeneration; prompts, models, data, and decoding settings remain frozen.",
        })
    return frozen


def run_modern_evaluation(input_path: str | Path, output_dir: str | Path, cfg: RevisionConfig,
                          runners: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg.validate()
    source, output = Path(input_path), Path(output_dir)
    pairs = [normalize_item(row, i) for i, row in enumerate(read_json_or_jsonl(source))]
    ids = [item["id"] for item, _ in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scenario IDs")
    if cfg.expected_items is not None and len(ids) != cfg.expected_items:
        raise ValueError(f"Expected {cfg.expected_items} items, found {len(ids)}")
    if cfg.expected_items in {169, 249, 500} and any(item["initial_history"] for item, _ in pairs):
        raise ValueError("All formal evaluation scenarios must start with empty dialogue histories")
    normalized = asdict(cfg)
    normalized.pop("resume")
    normalized.pop("allow_partial")
    contract = {"config": normalized, "data_sha256": sha256_file(source), "scenario_ids": ids,
                "scenario_ids_sha256": hash_object(ids), "initial_history_policy": ("empty_for_formal_scenarios" if cfg.expected_items in {169, 249, 500} else "explicit_initial_history"),
                "factuality_version": FACTUALITY_VERSION, "matched_evaluation_version": MATCHED_VERSION,
                "code_sha256": code_fingerprint(ROOT), "prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION}
    if cfg.require_model_manifest and not cfg.model_manifest:
        raise ValueError("A frozen model manifest is required")
    if cfg.model_manifest:
        from utils.model_manifest import verify_manifest
        verify_manifest(cfg.model_manifest, {cfg.persuader_model, cfg.persuadee_model, cfg.judge_model,
                                             cfg.progress_scorer_model} | ({cfg.world_model} if cfg.world_model else set()))
        contract["model_manifest_sha256"] = sha256_file(cfg.model_manifest)
    contract["protocol_hash"] = hash_object(contract)
    output.mkdir(parents=True, exist_ok=True)
    contract = enforce_execution_contract(output / "evaluation_config.json", contract, output,
                                          cfg.parse_retry_limit)
    result_path = output / "results.jsonl"
    if result_path.exists() and not cfg.resume:
        raise FileExistsError(f"Output exists: {result_path}")
    previous = read_json_or_jsonl(result_path) if result_path.exists() else []
    for row in last_records(previous):
        if (row["id"] not in ids or row.get("method") != cfg.method or row.get("backbone") != cfg.backbone
                or row.get("rollout_seed_id") != cfg.repeat_id or row.get("protocol_hash") != contract["protocol_hash"]):
            raise ValueError("Cached dialogue does not match the frozen scenario/method/seed/protocol")
    last = last_records(previous)
    done = {r["id"] for r in last if r.get("status") == "completed" or parse_failure(r)}
    failure_counts = Counter(r["id"] for r in previous if r.get("status") == "failed")
    role_runners = runners or build_runners(cfg)
    processed = 0
    consecutive_failures = 0
    for item, private in pairs:
        if item["id"] in done:
            continue
        resample_index = failure_counts[item["id"]]
        record = None
        while resample_index <= cfg.episode_failure_retries:
            record = generate_episode(item, private, cfg, role_runners, contract,
                                      episode_resample_index=resample_index)
            append_jsonl(result_path, record)
            processed += 1
            if record["status"] == "completed" or parse_failure(record):
                break
            failure_counts[item["id"]] += 1
            resample_index += 1
        if record is None or record["status"] == "failed":
            # Keep isolated failures unknown, but do not discard the remaining
            # scenarios. Exhausted retries stay unknown rather than receiving a
            # fabricated default response.
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_EPISODE_FAILURES:
                break
        else:
            consecutive_failures = 0
    all_rows = read_json_or_jsonl(result_path)
    report = summarize(all_rows, expected=len(ids), method=cfg.method, backbone=cfg.backbone, repeat_id=cfg.repeat_id)
    summary_path = output / "summary.json"
    write_json(summary_path, report)
    if report["failed_or_missing"] and not cfg.allow_partial:
        raise RuntimeError(f"Run incomplete ({report['failed_or_missing']} unknown); see {result_path} and resume after repair")
    return {"result_path": str(result_path), "summary_path": str(summary_path), "processed": processed, "summary": report}
