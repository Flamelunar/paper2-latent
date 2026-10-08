#!/usr/bin/env python3
"""Independently audit the completed GPU4 experiment and its two table rows."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from eclp_elpo import load_tensors, points_for_split, public_history, verify_checkpoint
from evaluate_our import JUDGES, METHOD, verify_generation
from experiment_io import atomic_json, digest, object_hash, read_json, read_jsonl
from update_our_table import replace_our_rows, row_text, verify_summary


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "eclp_runs_gpu4"
TABLE = ROOT / "exp_code_ljj/results/main_experiment_table.md"


def committed_checkpoint(directory: Path, lock: dict, stage: str, step: int):
    last = read_json(directory / "last.json")
    checkpoint = Path(last["checkpoint"])
    if not checkpoint.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"Checkpoint leaves {directory}")
    meta = verify_checkpoint(checkpoint, lock)
    if (meta["checkpoint_id"] != last["checkpoint_id"] or
            meta["stage"] != stage or meta["step"] != step):
        raise ValueError(f"Unexpected {stage} checkpoint identity or step")
    return checkpoint, meta


def audit_sft(checkpoint: Path, meta: dict, lock: dict) -> dict:
    """Prove that the committed SFT consumed every original session once."""
    plan = read_json(ROOT / "my_eclp_final/offline_training_plan.json")
    run = read_json(WORK / "sft_full/run.json")
    source = WORK / "prepared/sft.jsonl"
    sessions = plan["expected_sft_sessions"]
    steps = plan["sft_target_steps"]
    accumulation = run.get("accumulation")
    if (plan.get("sft_epochs") != 1 or sessions != 59868 or steps != 7484 or
            accumulation != 8 or run.get("stage") != "sft" or
            run.get("lock_id") != lock["lock_id"] or
            run.get("sessions") != sessions or
            run.get("architecture") != meta["architecture"] or
            run.get("implementation_sha256") != meta["implementation_sha256"] or
            run.get("data_sha256") != digest(source) or
            meta.get("provenance") != run or
            meta.get("reference_policy_hash") != meta["policy_hash"]):
        raise ValueError("SFT plan, data, run, or checkpoint provenance differs")
    source_count = 0
    scenario_ids = set()
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                source_count += 1
                scenario_ids.add(row["scenario_id"])
    if (source_count != sessions or len(scenario_ids) != 10000 or
            run.get("scenario_ids") != sorted(scenario_ids)):
        raise ValueError("SFT source does not contain the fixed 59,868 sessions / 10,000 scenarios")
    logged = read_jsonl(WORK / "sft_full/train.jsonl")
    if len(logged) != steps:
        raise ValueError("SFT training log does not contain every optimization step")
    for number, row in enumerate(logged, 1):
        if (row.get("step") != number or row.get("epoch") != 0 or
                row.get("next_position") != min(number * accumulation, sessions) or
                any(not math.isfinite(float(row[key])) for key in
                    ("utterance", "observation", "strategy", "reasoning", "grad_norm"))):
            raise ValueError(f"SFT data cursor or numerical values differ at step {number}")
    state = load_tensors(checkpoint / "train_state.pt")
    trainer = state["trainer"]
    if (trainer.get("step") != steps or trainer.get("epoch") != 1 or
            trainer.get("next_position") != 0 or
            trainer.get("run_hash") != object_hash(run) or
            not state.get("optimizer", {}).get("state") or
            set(state.get("rng", {})) != {"python", "cpu", "cuda"}):
        raise ValueError("Final SFT checkpoint lacks the completed epoch or resume state")
    return {"sessions": sessions, "scenarios": len(scenario_ids), "steps": steps,
            "data_cursor_at_last_step": logged[-1]["next_position"],
            "final_checkpoint_epoch": trainer["epoch"],
            "optimizer_and_rng_present": True}


def verify_reward_row(row: dict, point: dict):
    scenario = point["scenario_id"]
    if (row.get("scenario_id") != scenario or
            row.get("source_row_hash") != object_hash(point) or
            row.get("history_hash") != object_hash(public_history(point["history"])) or
            row.get("has_feedback") is not any(
                turn["role"] == "persuadee" for turn in point["history"]) or
            row.get("reward_mode") != "correction" or
            row.get("correction_weight") != 0.1 or
            row.get("policy_samples") != 4 or
            row.get("reference_action_index") != 0 or
            not isinstance(row.get("target_tokens"), int) or row["target_tokens"] < 1 or
            any(row.get(key) != 0 for key in
                ("new_dialogue_calls", "judge_calls", "generated_tokens"))):
        raise ValueError(f"RL reward provenance or configuration differs for {scenario}")
    values = {key: row.get(key) for key in
              ("factual_nll", "counterfactual_nll", "correction_benefit",
               "centered_correction_bonus", "reward")}
    if (any(not isinstance(series, list) or len(series) != 5 for series in values.values()) or
            any(not math.isfinite(float(value)) for series in values.values() for value in series) or
            not isinstance(row.get("paired_reward_delta"), list) or
            len(row["paired_reward_delta"]) != 4 or
            any(not math.isfinite(float(value)) for value in row["paired_reward_delta"])):
        raise ValueError(f"Incomplete or nonfinite RL reward values for {scenario}")
    factual, counter = values["factual_nll"], values["counterfactual_nll"]
    benefit = values["correction_benefit"]
    bonus, reward = values["centered_correction_bonus"], values["reward"]
    reference_benefit = counter[0] - factual[0]
    for index in range(5):
        expected_benefit = counter[index] - factual[index]
        expected_bonus = (math.tanh(expected_benefit - reference_benefit)
                          if row["has_feedback"] else 0.0)
        expected_reward = -factual[index] + 0.1 * bonus[index]
        if (not math.isclose(benefit[index], expected_benefit, abs_tol=5e-6) or
                not math.isclose(bonus[index], expected_bonus, abs_tol=5e-6) or
                not math.isclose(reward[index], expected_reward, abs_tol=5e-6)):
            raise ValueError(f"RL reward formula differs for {scenario}")
    if any(not math.isclose(row["paired_reward_delta"][index], reward[index + 1] - reward[0],
                            abs_tol=5e-6) for index in range(4)):
        raise ValueError(f"Paired RL reward deltas differ for {scenario}")


def audit_rl_rewards(rl: Path, sft_meta: dict, rl_meta: dict):
    run = read_json(rl / "run.json")
    summary = read_json(rl / "summary.json")
    base_run_hash = object_hash(run)
    active_run_hash = summary.get("offline_run_hash")
    if not isinstance(active_run_hash, str):
        raise ValueError("Offline RL summary lacks its active run hash")
    if rl_meta.get("provenance", {}).get("offline_run_hash") != active_run_hash:
        raise ValueError("Offline RL checkpoint and summary lineage differ")
    # The RL trainer can resume after an integrity-preserving implementation
    # repair. In that case committed blocks from the prior run remain valid
    # evidence and are linked by an immutable continuation manifest.
    accepted_run_hashes = {base_run_hash, active_run_hash}
    continued_from = summary.get("continued_from_run_hash")
    if continued_from is not None:
        manifest_path = rl / "continuation_manifest.json"
        manifest = read_json(manifest_path)
        if (manifest.get("legacy_run_hash") != continued_from
                or continued_from != base_run_hash
                or manifest.get("continuation_run_hash") != active_run_hash
                or manifest.get("legacy_trainer_sha256") is None
                or manifest.get("continuation_trainer_sha256") !=
                    rl_meta.get("provenance", {}).get("runtime_trainer_sha256")):
            raise ValueError("Offline continuation manifest does not match the committed run")
    elif active_run_hash != base_run_hash:
        raise ValueError("Offline RL summary changed its run hash without a continuation manifest")
    elif (rl / "continuation_manifest.json").exists():
        raise ValueError("Offline continuation manifest exists without a summary lineage")
    prepared = points_for_split(WORK / "prepared", "rl_train")
    expected_ids = [row["scenario_id"] for row in prepared]
    if (len(expected_ids) != 1800 or len(set(expected_ids)) != 1800 or
            run["scenario_ids"] != expected_ids or
            run["source_sha256"] != digest(WORK / "prepared/rl_train.jsonl") or
            run["initial_checkpoint_id"] != sft_meta["checkpoint_id"] or
            run["plan"]["rl_epochs"] != 1 or run["plan"]["candidates"] != 4 or
            run["plan"]["reward"] != "correction" or
            run["plan"]["correction_weight"] != 0.1 or
            run["training_dialogue_calls"] != 0 or
            run["training_judge_calls"] != 0 or
            run["generated_training_tokens"] != 0):
        raise ValueError("Offline RL source or no-dialogue contract differs")
    if (summary.get("complete") is not True or summary.get("contexts") != 1800 or
            summary.get("unique_scenarios") != 1800 or
            summary.get("offline_run_hash") != active_run_hash or
            summary.get("checkpoint_id") != rl_meta["checkpoint_id"] or
            summary.get("reward_mode") != "correction" or
            summary.get("policy_changed") is not True or
            summary.get("executor_unchanged") is not True or
            any(summary.get(key) != 0 for key in
                ("new_persuadee_calls", "judge_calls", "generated_training_tokens"))):
        raise ValueError("Offline RL completion summary differs")
    if rl_meta["executor_hash"] != sft_meta["executor_hash"]:
        raise ValueError("Frozen executor changed during RL")
    if rl_meta["policy_hash"] == sft_meta["policy_hash"]:
        raise ValueError("RL policy was not updated")
    if rl_meta.get("reference_policy_hash") != sft_meta["policy_hash"]:
        raise ValueError("RL reference policy differs from the completed SFT policy")
    reports = []
    for path_text in summary["blocks"]:
        commit = Path(path_text)
        if not commit.resolve().is_relative_to((rl / "blocks").resolve()):
            raise ValueError("RL block path leaves the experiment")
        block = read_json(commit)
        rewards = commit.parent / "rewards.jsonl"
        if (block["offline_run_hash"] not in accepted_run_hashes or
                block["parent_sft_checkpoint_id"] != sft_meta["checkpoint_id"] or
                block["block_rewards_sha256"] != digest(rewards) or
                block["new_persuadee_calls"] != 0 or block["judge_calls"] != 0 or
                block["autoregressive_generated_tokens"] != 0):
            raise ValueError("RL block provenance differs")
        reports.extend(read_jsonl(rewards))
    if (len(reports) != 1800 or
            sorted(row["scenario_id"] for row in reports) != sorted(expected_ids)):
        raise ValueError("RL reward rows do not cover the original partition exactly")
    by_id = {point["scenario_id"]: point for point in prepared}
    for row in reports:
        verify_reward_row(row, by_id[row["scenario_id"]])
    measured_delta = statistics.fmean(
        abs(value) for row in reports for value in row["paired_reward_delta"])
    measured_nll = statistics.fmean(
        value for row in reports for value in row["factual_nll"][1:])
    if (summary.get("optimizer_updates", 0) < 1 or
            summary.get("contexts_with_feedback") != sum(row["has_feedback"] for row in reports) or
            not math.isclose(summary["mean_absolute_paired_reward_delta"], measured_delta, abs_tol=5e-6) or
            not math.isclose(summary["mean_candidate_nll"], measured_nll, abs_tol=5e-6) or
            measured_delta <= 0):
        raise ValueError("RL summary differs from committed reward records")
    return {"scenarios": len(reports), "blocks": len(summary["blocks"]),
            "optimizer_updates": summary["optimizer_updates"],
            "mean_absolute_paired_reward_delta": summary["mean_absolute_paired_reward_delta"],
            "reward_formula_verified": True, "reference_policy_unchanged": True}


def audit_dev(dev: dict, dev_points: list[dict], checkpoint_id: str):
    rows = dev.get("rows", [])
    if (dev.get("complete") is not True or dev.get("n") != 200 or
            dev.get("checkpoint_id") != checkpoint_id or
            dev.get("proxy_quality_gate_pass") is not True or
            dev.get("data_sha256") != digest(WORK / "prepared/dev.jsonl") or
            len(dev_points) != 200 or len(rows) != 200 or
            sorted(row["scenario_id"] for row in rows) !=
                sorted(point["scenario_id"] for point in dev_points) or
            any(dev.get(key) != 0 for key in
                ("new_persuadee_calls", "judge_calls", "generated_training_tokens"))):
        raise ValueError("200-case dev source or no-dialogue gate differs")
    points = {point["scenario_id"]: point for point in dev_points}
    for row in rows:
        point = points[row["scenario_id"]]
        fields = ("sft_nll", "policy_nll", "nll_delta_policy_minus_sft",
                  "sft_correction_benefit", "policy_correction_benefit",
                  "correction_benefit_delta")
        if (row.get("has_feedback") is not any(
                turn["role"] == "persuadee" for turn in point["history"]) or
                not isinstance(row.get("target_tokens"), int) or row["target_tokens"] < 1 or
                any(not math.isfinite(float(row.get(key, float("nan")))) for key in fields) or
                not math.isclose(row["nll_delta_policy_minus_sft"],
                                 row["policy_nll"] - row["sft_nll"], abs_tol=5e-6) or
                not math.isclose(row["correction_benefit_delta"],
                                 row["policy_correction_benefit"] - row["sft_correction_benefit"],
                                 abs_tol=5e-6)):
            raise ValueError(f"Dev row differs from paired scoring for {row['scenario_id']}")
    feedback = [row for row in rows if row["has_feedback"]]
    mean_delta = statistics.fmean(row["nll_delta_policy_minus_sft"] for row in rows)
    if (not math.isclose(dev.get("sft_mean_nll", float("nan")),
                         statistics.fmean(row["sft_nll"] for row in rows), abs_tol=5e-6) or
            not math.isclose(dev.get("policy_mean_nll", float("nan")),
                             statistics.fmean(row["policy_nll"] for row in rows), abs_tol=5e-6) or
            not math.isclose(dev.get("mean_nll_delta_policy_minus_sft", float("nan")),
                             mean_delta, abs_tol=5e-6) or
            dev.get("feedback_contexts") != len(feedback) or
            not math.isclose(dev.get("mean_correction_benefit_delta_feedback", float("nan")),
                             statistics.fmean(row["correction_benefit_delta"] for row in feedback),
                             abs_tol=5e-6) or
            mean_delta > dev.get("dev_nll_tolerance", float("-inf"))):
        raise ValueError("Dev aggregate or NLL quality gate differs from individual rows")
    return {"n": len(rows), "mean_nll_delta_policy_minus_sft": mean_delta}


def main():
    status = read_json(WORK / "pipeline_status.json")
    if status.get("state") != "complete":
        raise RuntimeError(f"Pipeline is still {status.get('state')} / {status.get('phase')}")
    lock = read_json(WORK / "protocol.lock.json")
    sft, sft_meta = committed_checkpoint(WORK / "sft_full", lock, "sft", 7484)
    sft_evidence = audit_sft(sft, sft_meta, lock)
    rl, rl_meta = committed_checkpoint(WORK / "offline_rl", lock, "rl", 1800)
    rl_evidence = audit_rl_rewards(WORK / "offline_rl", sft_meta, rl_meta)
    dev = read_json(WORK / "offline_dev_validation.json")
    dev_points = points_for_split(WORK / "prepared", "dev")
    dev_evidence = audit_dev(dev, dev_points, rl_meta["checkpoint_id"])
    final = read_json(WORK / "final_checkpoint.json")
    if (Path(final["checkpoint"]).resolve() != rl.resolve() or
            final["metadata"]["checkpoint_id"] != rl_meta["checkpoint_id"] or
            final.get("selected_using_test") is not False):
        raise ValueError("Final checkpoint does not match the committed RL model")
    generation = WORK / "test_eclp_offline_lpo"
    sources = verify_generation(generation, WORK)
    if len(sources) != 1000:
        raise ValueError("Expected 1000 complete five-round test generations")
    summaries = {}
    for judge in ("selene8b", "qwen3_14b"):
        directory = WORK / f"test_eval_{judge}"
        summaries[judge] = verify_summary(directory, generation, WORK, judge)
        if len(read_jsonl(directory / f"{METHOD}.jsonl")) != 1000:
            raise ValueError(f"Expected 1000 original {judge} judgment records")
    expected_rows = {JUDGES[judge]["table_label"]: row_text(summary)
                     for judge, summary in summaries.items()}
    table = TABLE.read_text(encoding="utf-8")
    if replace_our_rows(table, expected_rows) != table:
        raise ValueError("Our rows in the Qwen2.5 table differ from audited scores")
    report = {
        "complete": True, "pipeline_state": status["state"],
        "sft_checkpoint_id": sft_meta["checkpoint_id"], "sft_steps": sft_meta["step"],
        "sft": sft_evidence,
        "rl_checkpoint_id": rl_meta["checkpoint_id"], "rl": rl_evidence,
        "dev": dev_evidence,
        "test_generations": len(sources),
        "judges": {judge: {"records": 1000, "valid_count": value["valid_count"],
                            "success_percent": value["success_percent"],
                            "win_rate": value["win_rate"],
                            "unresolved_cases": value["unresolved_cases"]}
                   for judge, value in summaries.items()},
        "table_sha256": digest(TABLE), "our_rows": expected_rows,
    }
    atomic_json(WORK / "completion_audit.json", report)
    import json
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
