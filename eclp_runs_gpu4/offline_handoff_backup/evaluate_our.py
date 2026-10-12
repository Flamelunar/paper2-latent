#!/usr/bin/env python3
"""Evaluate Our on the original 1000 test cases, using frozen table protocols."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import sys
import time

from experiment_io import (append_jsonl, atomic_json, digest, immutable_json,
                           object_hash, read_json, read_jsonl, require_indexes, writer_lock)

ROOT = Path(__file__).resolve().parents[1]
METHOD = "our_eclp_elpo"
ACTOR = "/app/hzj/models/Qwen2.5-7B-Instruct"
JUDGES = {
    "selene8b": {
        "model": "/app/hzj/models/Selene-1-Mini-Llama-3.1-8B",
        "version": "selene_v17",
        "revision": "joint_selene_success_reference_v17_v7_strict_goal_acceptance",
        "table_label": "Selene-1-Mini-Llama-3.1-8B (BF16, strict V17 v7)",
    },
    "qwen3_14b": {
        "model": "/app/hzj/models/Qwen3-14B",
        "version": "qwen3_v11",
        "revision": "qwen3_gemma_v11_neutral_json_v2",
        "table_label": "Qwen3-14B (BF16)",
    },
}


def protocol_module():
    sys.path.insert(0, str(ROOT / "exp_code_ljj"))
    import final_eval_protocol
    return final_eval_protocol


def verify_generation(generation: Path, work: Path) -> dict[int, dict]:
    """Validate full content alignment, beyond matching only scenario_id."""
    lock = read_json(work / "protocol.lock.json")
    original = ROOT / "exp_code_ljj/data/evaluation_points.jsonl"
    if digest(original) != lock["data"]["evaluation_points"]:
        raise ValueError("Original test data changed after the training lock")
    tests = read_jsonl(original)
    if len(tests) != 1000 or len({row["scenario_id"] for row in tests}) != 1000:
        raise ValueError("Original test set must contain 1000 unique scenarios")
    source = generation / "generation.jsonl"
    summary = read_json(generation / "generation_summary.json")
    run = read_json(generation / "run.json")
    if (summary.get("complete") is not True or summary.get("n") != 1000
            or summary.get("generation_sha256") != digest(source)
            or summary.get("run_hash") != object_hash(run)
            or run.get("split") != "test" or run.get("method") != "ECLP-ELPO"
            or run.get("lock_id") != lock["lock_id"]
            or run.get("judge_used_during_generation") is not False
            or summary.get("judge_calls") != 0):
        raise ValueError("Incomplete generation or mismatched run contract")
    rows = require_indexes(read_jsonl(source))
    if len({r["scenario_id"] for r in rows.values()}) != 1000:
        raise ValueError("Duplicate scenario in generated test dialogues")
    if run["scenario_ids"] != [p["scenario_id"] for p in tests]:
        raise ValueError("Generation selection/order differs from the original test set")
    for index, point in enumerate(tests):
        row = rows[index]
        history = [{"role": t["role"], "content": t["content"]} for t in point["history"]]
        if (row.get("scenario_id") != point["scenario_id"]
                or row.get("scenario") != point["scenario"]
                or row.get("initial_history") != history
                or row.get("reference") != point["reference"]
                or row.get("checkpoint_id") != run["checkpoint_id"]
                or row.get("run_hash") != object_hash(run)
                or row.get("lock_id") != lock["lock_id"]):
            raise ValueError(f"Test content/provenance mismatch at index {index}")
        turns = row.get("trajectory", [])
        if len(turns) != 10 or row.get("rounds") != 5:
            raise ValueError(f"Expected exactly five generated rounds at index {index}")
        for turn_index, turn in enumerate(turns):
            if (turn.get("role") != ("persuader" if turn_index % 2 == 0 else "persuadee")
                    or not isinstance(turn.get("content"), str) or not turn["content"].strip()):
                raise ValueError(f"Missing reply or role mismatch at index {index}, turn {turn_index}")
        if row.get("compute_dtype") != "bfloat16" or row.get("quantization") != "none":
            raise ValueError("Generation precision changed")
    return rows


def judge_record_valid(row: dict, source: dict, judge: str, contract: dict) -> None:
    profile = JUDGES[judge]
    expected = {
        "trial": 1, "method": METHOD, "stratum": "full_test",
        "scenario_id": source["scenario_id"], "goal": source["scenario"]["goal"],
        "prompt_version": profile["version"], "prompt_revision": profile["revision"],
        "judge_model": profile["model"], "judge_device": "cuda:0", "physical_gpu": 4,
        "judge_compute_dtype": "bfloat16", "judge_quantization": "none",
        "max_parse_retries": 3, "evaluation_contract_hash": object_hash(contract),
        "source_row_hash": object_hash(source), "checkpoint_id": source["checkpoint_id"],
        "actor_base_model": ACTOR,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"Judge record {row.get('index')} has mismatched {key}")
    if not isinstance(row.get("raw_outputs"), list) or not row["raw_outputs"]:
        raise ValueError("Judge record lacks raw generations")
    if row.get("model_calls") != len(row["raw_outputs"]):
        raise ValueError("Judge call count does not match raw generations")


def summarize(records: list[dict], sources: dict[int, dict], judge: str, contract: dict) -> dict:
    by_index = require_indexes(records)
    valid = []
    success_unresolved = joint_unresolved = 0
    for index, row in by_index.items():
        judge_record_valid(row, sources[index], judge, contract)
        success_unresolved += bool(row.get("success_parse_error", row.get("parse_error")))
        if row.get("parse_error"):
            joint_unresolved += 1
            continue
        verdict = row.get("verdict")
        if not isinstance(verdict, dict):
            raise ValueError(f"Missing resolved verdict at index {index}")
        rounds = verdict.get("round_success")
        if not isinstance(rounds, list) or len(rounds) != 5 or any(type(v) is not bool for v in rounds):
            raise ValueError(f"Invalid round decisions at index {index}")
        first = next((r for r, value in enumerate(rounds, 1) if value), None)
        if (verdict.get("first_success_round") != first
                or verdict.get("success_within_5_rounds") != (first is not None)):
            raise ValueError(f"Inconsistent first success at index {index}")
        ab, ba = verdict.get("win_ab"), verdict.get("win_ba")
        credit = ((ab == "A") + (ba == "B")) / 2
        if ab not in {"A", "B"} or ba not in {"A", "B"} or verdict.get("win_credit") != credit:
            raise ValueError(f"Invalid swapped Win-Rate at index {index}")
        valid.append(verdict)
    n = len(valid)
    successes = sum(v["success_within_5_rounds"] for v in valid)
    return {
        "expected_count": 1000, "valid_count": n, "success_count": successes,
        "success_percent": 100 * successes / n if n else None,
        "win_rate": 100 * statistics.fmean(v["win_credit"] for v in valid) if n else None,
        "mean_rounds_or_cap5": statistics.fmean(v["first_success_round"] or 5 for v in valid) if n else None,
        # Match the table's denominator: a case must resolve BOTH judgments.
        "unresolved_cases": joint_unresolved, "success_only_unresolved": success_unresolved,
        "success_lower_bound_all_1000_percent": 100 * successes / 1000,
        "success_upper_bound_all_1000_percent": 100 * (successes + joint_unresolved) / 1000,
        "denominator": "jointly_parsed_cases; failures_count_as_round_5",
        "judge": judge, "judge_model": JUDGES[judge]["model"],
        "prompt_version": JUDGES[judge]["version"], "prompt_revision": JUDGES[judge]["revision"],
        "evaluation_contract_hash": object_hash(contract), "complete": True,
    }


def evaluate_qwen_sample(protocol, row: dict, sample: dict, endpoint) -> dict:
    prompt = protocol.joint_prompt(row, "qwen3_v11", "qwen3_14b")
    outputs, errors = [], []
    verdict, error = None, None
    for attempt in range(4):
        retry = ""
        if attempt:
            retry = (f"\n\nThe previous response failed validation: {errors[-1]}. "
                     f"This is correction attempt {attempt} of 3. "
                     "Return a complete JSON object again and follow this exact output contract: "
                     + protocol.prompt_output_contract("qwen3_v11"))
        raw = endpoint.complete(
            system="You are an impartial evaluator of persuasion success and comparative persuasive quality.",
            user=prompt + retry, temperature=0.0, max_tokens=256, json_mode=True)
        outputs.append(raw)
        try:
            verdict = protocol.parse_joint(raw, include_evidence=False)
        except ValueError as exc:
            error = str(exc)
            errors.append(error)
        else:
            error = None
            break
    return {
        **sample, "scenario_id": row["scenario_id"], "goal": row["scenario"]["goal"],
        "model": row["persuader_model"], "prompt_version": "qwen3_v11",
        "prompt_revision": JUDGES["qwen3_14b"]["revision"],
        "judge_model": JUDGES["qwen3_14b"]["model"], "judge_device": "cuda:0",
        "judge_compute_dtype": "bfloat16", "judge_quantization": "none",
        "model_calls": len(outputs), "max_parse_retries": 3,
        "raw_output": outputs[-1], "raw_outputs": outputs, "retry_outputs": outputs[1:],
        "parse_errors_by_attempt": errors, "parse_error": error, "verdict": verdict,
    }


def run_evaluation(judge: str, generation: Path, output: Path, work: Path) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise RuntimeError("Final Our evaluation must run on physical GPU4")
    if not output.resolve().is_relative_to(work.resolve()):
        raise ValueError("Judge outputs must be inside the new Our experiment directory")
    sources = verify_generation(generation, work)
    protocol = protocol_module()
    profile = JUDGES[judge]
    if protocol.PROMPT_REVISIONS[profile["version"]] != profile["revision"]:
        raise ValueError("Frozen Judge profile mismatch")
    from eclp_elpo import model_fingerprint, environment_versions, precision_report
    contract = {
        "method": METHOD, "actor_base_model": ACTOR,
        "generation_sha256": digest(generation / "generation.jsonl"),
        "generation_run_hash": object_hash(read_json(generation / "run.json")),
        "expected_indexes": list(range(1000)), "judge": judge, "judge_model": profile["model"],
        "judge_fingerprint": model_fingerprint(profile["model"]),
        "prompt_version": profile["version"], "prompt_revision": profile["revision"],
        "prompt_file_sha256": digest(protocol.PROMPT_FILE),
        "protocol_source_sha256": digest(Path(protocol.__file__)),
        "evaluator_sha256": digest(Path(__file__)), "judge_device": "cuda:0",
        "physical_gpu": 4, "dtype": "bfloat16", "quantization": "none",
        "max_tokens": 256, "max_parse_retries": 3, "runtime": environment_versions(),
        "simulator_source_sha256": digest(ROOT / "exp_code_ljj/repro/simulation.py"),
    }
    with writer_lock(output / ".writer.lock"):
        immutable_json(output / "run.json", contract)
        path = output / f"{METHOD}.jsonl"
        existing_rows = read_jsonl(path) if path.exists() else []
        existing = {}
        for row in existing_rows:
            index = row.get("index")
            if type(index) is not int or index not in sources or index in existing:
                raise ValueError("Duplicate or unexpected saved Judge index")
            judge_record_valid(row, sources[index], judge, contract)
            existing[index] = row
        pending = [i for i in range(1000) if i not in existing]
        print(f"{judge}: {len(existing)} saved, {len(pending)} pending", flush=True)
        if pending:
            endpoint = protocol.LocalTransformersEndpoint(model=profile["model"], device="cuda:0", quantization="none")
            try:
                precision_report(endpoint.model)
                atomic_json(output / "precision.json", precision_report(endpoint.model))
                for index in pending:
                    row = sources[index]
                    sample = {"trial": 1, "method": METHOD, "index": index, "stratum": "full_test"}
                    start = time.time()
                    if judge == "selene8b":
                        result = protocol.evaluate_selene_sample(
                            row, sample, endpoint, judge_model=profile["model"], judge_device="cuda:0",
                            max_tokens=256, max_parse_retries=3)
                    else:
                        result = evaluate_qwen_sample(protocol, row, sample, endpoint)
                    result.update(physical_gpu=4, actor_base_model=ACTOR,
                                  checkpoint_id=row["checkpoint_id"], source_row_hash=object_hash(row),
                                  evaluation_contract_hash=object_hash(contract), elapsed_seconds=time.time() - start)
                    judge_record_valid(result, row, judge, contract)
                    append_jsonl(path, result)
                    existing[index] = result
                    if (index + 1) % 5 == 0 or index == 0:
                        progress = {"complete": False, "saved": len(existing), "expected": 1000,
                                    "unresolved": sum(bool(v.get("parse_error")) for v in existing.values()),
                                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
                        atomic_json(output / "progress.json", progress)
                        print(f"{judge}: {len(existing)}/1000, unresolved={progress['unresolved']}", flush=True)
            finally:
                endpoint.close()
        summary = summarize(list(existing.values()), sources, judge, contract)
        summary["judgments_sha256"] = digest(path)
        summary["generation_sha256"] = contract["generation_sha256"]
        summary["total_model_calls"] = sum(r["model_calls"] for r in existing.values())
        atomic_json(output / "summary.json", summary)
        atomic_json(output / "progress.json", {"complete": True, "saved": 1000, "expected": 1000,
                                              "unresolved": summary["unresolved_cases"]})
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", choices=JUDGES, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=ROOT / "eclp_runs_gpu4")
    args = parser.parse_args()
    summary = run_evaluation(args.judge, args.generation.resolve(), args.output.resolve(), args.work.resolve())
    import json
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
