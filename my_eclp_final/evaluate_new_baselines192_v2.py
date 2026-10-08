#!/usr/bin/env python3
"""Evaluate new Persuader baselines on the fixed 192 subset using unchanged unified v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

from experiment_io import atomic_json, append_jsonl, digest, immutable_json, object_hash, read_json, read_jsonl, writer_lock
from matched_eval_protocol_v2 import CONFIG, PROMPT_FILE, StrictJSONJudge, order_credit, parse_verdict, success_prompt, text_hash, validate_available_dialogue, validate_trajectory, win_prompt

JUDGES = {
    "selene8b": "/app/hzj/models/Selene-1-Mini-Llama-3.1-8B",
    "qwen3_14b": "/app/hzj/models/Qwen3-14B",
}
METHOD_LABELS = {
    "baseline1_zero_shot": "Baseline 1: Zero-shot",
    "baseline2_dialogue_sft": "Baseline 2: Dialogue-SFT",
    "baseline3_isr_simdpo": "Baseline 3: ISR-SFT + SimDPO",
    "baseline4_ma2p": "Baseline 4: MA2P",
    "baseline5_latentratt": "Baseline 5: LatentRATT",
    "our_eclp_offline_lpo": "Our",
}


def schedule(rows: dict[int, dict]) -> list[int]:
    """A deterministic scenario schedule independent of model outputs."""
    return sorted(rows, key=lambda i: hashlib.sha256(("matched-v2-pilot-42:" + rows[i]["scenario_id"]).encode()).hexdigest())


def validate_record(row: dict, source: dict, golden: dict, contract_hash: str) -> None:
    if (row.get("contract_hash") != contract_hash or row.get("scenario_id") != source["scenario_id"]
            or row.get("source_row_hash") != object_hash(source)
            or row.get("golden_row_hash") != object_hash(golden)):
        raise ValueError("Saved result provenance differs")
    rounds = row.get("round_success")
    if not isinstance(rounds, list) or len(rounds) != 5 or any(type(v) is not bool for v in rounds):
        raise ValueError("Saved success vector lacks five real JSON booleans")
    for metric in ("win1", "win_full"):
        verdict = row.get(metric)
        if not isinstance(verdict, dict):
            raise ValueError("Missing pairwise verdict")
        for order in ("ab", "ba"):
            winner = verdict.get(order)
            if not isinstance(winner, str) or winner not in {"A", "B", "Tie"}:
                raise ValueError("Invalid saved winner")
        expected = statistics.fmean(order_credit(verdict[o], o) for o in ("ab", "ba"))
        if verdict.get("credit") != expected:
            raise ValueError("Saved credit does not match independent order verdicts")


def summarize(records: list[dict], golden: dict[int, dict], contract: dict) -> dict:
    n = len(records)
    first = [next((i for i, v in enumerate(r["round_success"], 1) if v), None) for r in records]
    result = {
        "revision": CONFIG["revision"], "judge": contract["judge"], "judge_model": contract["judge_model"],
        "method": contract.get("method", "Our"), "method_slug": contract.get("method_slug", "our_eclp_offline_lpo"),
        "expected_count": len(golden), "evaluated_count": n,
        "complete": n == len(golden), "pending_count": len(golden) - n,
        "success_count": sum(v is not None for v in first),
        "success_within5_percent": 100 * sum(v is not None for v in first) / n if n else None,
        "final_round5_success_percent": 100 * sum(r["round_success"][4] for r in records) / n if n else None,
        "mean_rounds_or_cap5": statistics.fmean(v or 5 for v in first) if n else None,
        "win1_count": n, "win1_percent": 100 * statistics.fmean(r["win1"]["credit"] for r in records) if n else None,
        "win_full_count": n, "win_full_percent": 100 * statistics.fmean(r["win_full"]["credit"] for r in records) if n else None,
        "format_unresolved_completed_cases": 0,
        "denominator": "all selected completed scenarios; Win@1 and Win-Full use identical scenarios",
        "contract_hash": object_hash(contract),
        "golden_persuadee_source": "original released dialogue; Our uses the fixed simulated Persuadee",
    }
    for metric, group in (("win1", records), ("win_full", records)):
        credits = [[order_credit(r[metric][o], o) for o in ("ab", "ba")] for r in group]
        result[metric + "_order_disagreement_count"] = sum(a != b for a, b in credits)
        result[metric + "_explicit_tie_judgments"] = sum(r[metric][o] == "Tie" for r in group for o in ("ab", "ba"))
        result[metric + "_full_win_count"] = sum(r[metric]["credit"] == 1 for r in group)
        result[metric + "_full_loss_count"] = sum(r[metric]["credit"] == 0 for r in group)
    result["golden_length_groups"] = {}
    for label, condition in (("shorter_than_ours", lambda n: n < 5),
                             ("same_number_of_persuader_replies", lambda n: n == 5),
                             ("longer_than_ours", lambda n: n > 5)):
        group = [r for r in records if condition(golden[r["index"]]["golden_persuader_turns"])]
        result["golden_length_groups"][label] = {"n": len(group),
            "win1_percent": 100 * statistics.fmean(r["win1"]["credit"] for r in group) if group else None,
            "win_full_percent": 100 * statistics.fmean(r["win_full"]["credit"] for r in group) if group else None}
    result["golden_without_final_persuadee_reply_count"] = sum(not golden[r["index"]]["golden_ends_with_persuadee"] for r in records)
    return result


def evaluate(args) -> dict:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.physical_gpu):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r} does not select physical GPU {args.physical_gpu}")
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root / "exp_code_ljj"))
    generation_source = args.generation / "generation.jsonl" if args.generation.is_dir() else args.generation
    selection_path = root / "exp_code_ljj/data/test_selection_192.json"
    selection = read_json(selection_path)
    points = read_jsonl(root / "exp_code_ljj/data/evaluation_points.jsonl")
    selected = selection["original_indexes"]
    if (selection["count"] != 192 or len(set(selected)) != 192
            or selected != [i for i, p in enumerate(points) if p["target_turn"] == 1 and p["history"] == []]
            or selection["source_sha256"] != digest(root / "exp_code_ljj/data/evaluation_points.jsonl")):
        raise ValueError("Fixed 192-scenario selection changed")
    if args.method == "our_eclp_offline_lpo":
        from evaluate_our import verify_generation
        sources = verify_generation(generation_source.parent, root / "eclp_runs_gpu4")
    else:
        from experiment_io import require_indexes
        sources = require_indexes(read_jsonl(generation_source))
        points = read_jsonl(root / "exp_code_ljj/data/evaluation_points.jsonl")
        if len(points) != 1000:
            raise ValueError("Expected the fixed 1,000 test points")
        sources = {i: sources[i] for i in selected}
        for index, row in sources.items():
            point = points[index]
            history = [{"role": t["role"], "content": t["content"]} for t in point["history"]]
            if (row.get("scenario_id") != point["scenario_id"] or row.get("scenario") != point["scenario"]
                    or row.get("initial_history") != history or row.get("reference") != point["reference"]):
                raise ValueError(f"Baseline generation context mismatch at index {index}")
            if (row.get("method") != args.method or row.get("persuadee_model") != "/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged"):
                raise ValueError("Baseline method or fixed Persuadee mismatch")
            validate_trajectory(row["trajectory"])
            generation_protocol = str(row.get("protocol", "")).lower()
            quantization = row.get("quantization", row.get("persuader_quantization", "none"))
            compute_dtype = row.get("compute_dtype", row.get("persuader_compute_dtype"))
            if quantization != "none" or (compute_dtype is not None and compute_dtype != "bfloat16"):
                raise ValueError("Baseline generation precision differs from the required BF16 protocol")
            if args.method == "baseline5_latentratt":
                if row.get("protocol") != "latentRATT_persuasion_paper10k_5rounds_no_judge_v1":
                    raise ValueError(f"Unexpected Baseline 5 generation protocol at index {index}")
            elif "bf16" not in generation_protocol and "bfloat16" not in generation_protocol:
                raise ValueError(f"Baseline generation does not identify a BF16 protocol at index {index}")
    gold_rows = read_jsonl(args.golden / "golden.jsonl")
    golden = {r["index"]: r for r in gold_rows}
    sources = {i: sources[i] for i in selected}
    if len(gold_rows) != 1000 or len(golden) != 1000 or not set(sources).issubset(golden):
        raise ValueError("Golden must exactly cover the original 1,000 scenarios")
    manifest = read_json(args.golden / "manifest.json")
    if (manifest["golden_sha256"] != digest(args.golden / "golden.jsonl") or
            manifest["points_sha256"] != digest(root / "exp_code_ljj/data/evaluation_points.jsonl")):
        raise ValueError("Golden recovery manifest changed")
    for index, source in sources.items():
        g = golden[index]
        if (g["scenario_id"] != source["scenario_id"] or g["scenario"] != source["scenario"]
                or g["initial_history"] != source["initial_history"] or g["golden_first"] != source["reference"]):
            raise ValueError(f"Golden context mismatch at index {index}")
        validate_available_dialogue(g["golden_trajectory"])
    golden = {i: golden[i] for i in selected}
    selected_schedule = schedule(golden)
    model = Path(JUDGES[args.judge])
    model_files = sorted(set(model.glob("*.safetensors")) | set(model.glob("*.json")))
    contract = {
        "revision": CONFIG["revision"], "judge": args.judge, "judge_model": str(model),
        "method_slug": args.method, "method": METHOD_LABELS[args.method],
        "model_metadata": [{"file": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in model_files],
        "prompt_sha256": digest(PROMPT_FILE),
        "protocol_sha256": digest(Path(__file__).with_name("matched_eval_protocol_v2.py")),
        "evaluator_sha256": digest(Path(__file__)),
        "simulation_sha256": digest(root / "exp_code_ljj/repro/simulation.py"),
        "generation_source": str(generation_source.resolve()),
        "generation_sha256": digest(generation_source),
        "checkpoint_id": next(iter(sources.values())).get("checkpoint_id"),
        "generation_precision_provenance": (
            "not_recorded_in_generation_artifact" if args.method == "baseline5_latentratt"
            else "bf16_protocol_tag_and_generation_source"
        ),
        "golden_source": str((args.golden / "golden.jsonl").resolve()),
        "golden_sha256": manifest["golden_sha256"], "golden_source_sha256": manifest["source_sha256"],
        "expected_indexes": selected, "scenario_schedule": selected_schedule,
        "test_selection_sha256": digest(selection_path), "subset_revision": selection["revision"],
        "physical_gpu": args.physical_gpu, "compute_dtype": "bfloat16", "quantization": "none",
        "success_input": "standalone exact round prefix; no reference; no future rounds",
        "win1_input": "first complete replies with common history",
        "win_full_input": "all saved Our turns versus all original golden continuation turns after the common history; lengths may differ",
        "order_swapping": "two standalone requests with separate contexts",
        "explicit_tie_allowed": True, "temperature": 0,
        "output_decoding": "finite_JSON_token_grammar", "max_format_retries": 2,
        "batch_size": args.batch_size,
    }
    if args.validate_only:
        return {"validated": True, "count": len(sources), "generation": str(generation_source),
                "revision": CONFIG["revision"], "test_selection_sha256": digest(selection_path)}
    contract_hash = object_hash(contract)
    with writer_lock(args.output / ".writer.lock"):
        immutable_json(args.output / "run.json", contract)
        records_path = args.output / "judgments.jsonl"
        saved = read_jsonl(records_path) if records_path.exists() else []
        existing = {}
        for row in saved:
            index = row["index"]
            if index not in sources or index in existing:
                raise ValueError("Invalid or duplicate saved result index")
            validate_record(row, sources[index], golden[index], contract_hash)
            existing[index] = row
        pending = [i for i in selected_schedule if i not in existing]
        if args.max_new_scenarios:
            pending = pending[:args.max_new_scenarios]
        requests_dir = args.output / "requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        endpoint = None
        try:
            if pending:
                endpoint = StrictJSONJudge(str(model), memory_cap_gib=args.memory_cap_gib)
            def target_for(index: int, key: str) -> Path:
                return requests_dir / f"{index:04d}_{key}.json"

            def save_response(query: dict, response: dict, attempts: int,
                              errors: list[str], batch_fallback: str | None = None) -> None:
                response.update(index=query["index"], stage=query["key"], contract_hash=contract_hash,
                                independent_request_id=f"{args.judge}:{query['index']}:{query['key']}",
                                format_errors=errors, attempts=attempts,
                                batch_fallback=batch_fallback)
                immutable_json(target_for(query["index"], query["key"]), response)

            for chunk_start in range(0, len(pending), args.scenario_chunk_size):
                scenario_chunk = pending[chunk_start:chunk_start + args.scenario_chunk_size]
                queries = {"success": [], "win": []}
                for index in scenario_chunk:
                    source, g = sources[index], golden[index]
                    prompts = [(f"success_r{r}", success_prompt(source, r), "success") for r in range(1, 6)]
                    prompts.extend((f"{metric}_{order}", win_prompt(source, g, metric, order), "win")
                                   for metric in ("win1", "win_full") for order in ("ab", "ba"))
                    for key, prompt, kind in prompts:
                        target = target_for(index, key)
                        if target.exists():
                            cached = read_json(target)
                            if (cached.get("contract_hash") != contract_hash
                                    or cached.get("prompt_sha256") != text_hash(prompt)):
                                raise ValueError("Cached request does not match frozen protocol")
                            if parse_verdict(cached["raw_output"], kind) != cached["verdict"]:
                                raise ValueError("Cached verdict does not match raw output")
                        else:
                            queries[kind].append({"index": index, "key": key,
                                                  "prompt": prompt, "kind": kind})

                for kind, work_items in queries.items():
                    work_items.sort(key=lambda item: len(item["prompt"]))
                    for offset in range(0, len(work_items), args.batch_size):
                        batch = work_items[offset:offset + args.batch_size]
                        try:
                            responses = endpoint.complete_batch(
                                [item["prompt"] for item in batch], [kind] * len(batch)
                            )
                        except Exception as batch_exc:
                            batch_error = f"{type(batch_exc).__name__}: {batch_exc}"
                            for item in batch:
                                errors = [f"batch fallback: {batch_error}"]
                                for attempt in range(3):
                                    try:
                                        response = endpoint.complete(item["prompt"], kind)
                                    except Exception as exc:
                                        errors.append(f"{type(exc).__name__}: {exc}")
                                        if attempt == 2:
                                            atomic_json(requests_dir / f"{item['index']:04d}_{item['key']}.error.json",
                                                        {"contract_hash": contract_hash, "index": item["index"],
                                                         "stage": item["key"], "prompt_sha256": text_hash(item["prompt"]),
                                                         "errors": errors})
                                            raise
                                    else:
                                        save_response(item, response, attempt + 1, errors, batch_error)
                                        break
                        else:
                            for item, response in zip(batch, responses):
                                save_response(item, response, 1, [])

                for index in scenario_chunk:
                    source, g = sources[index], golden[index]
                    started = time.monotonic()
                    rounds = [read_json(target_for(index, f"success_r{r}"))["verdict"]["success"]
                              for r in range(1, 6)]
                    result = {"index": index, "scenario_id": source["scenario_id"],
                              "contract_hash": contract_hash, "source_row_hash": object_hash(source),
                              "golden_row_hash": object_hash(g), "round_success": rounds,
                              "ours_complete_rounds": 5, "golden_complete_rounds": g["golden_remaining_complete_rounds"],
                              "golden_persuader_turns": g["golden_persuader_turns"],
                              "golden_ends_with_persuadee": g["golden_ends_with_persuadee"]}
                    for metric in ("win1", "win_full"):
                        decisions = {order: read_json(target_for(index, f"{metric}_{order}"))["verdict"]["winner"]
                                     for order in ("ab", "ba")}
                        decisions["credit"] = statistics.fmean(order_credit(decisions[o], o) for o in ("ab", "ba"))
                        result[metric] = decisions
                    result["elapsed_seconds"] = time.monotonic() - started
                    validate_record(result, source, g, contract_hash)
                    append_jsonl(records_path, result)
                    existing[index] = result
                summary = summarize(list(existing.values()), golden, contract)
                atomic_json(args.output / "summary.json", summary)
                atomic_json(args.output / "progress.json", {"saved": len(existing), "expected": len(selected),
                            "complete": summary["complete"], "updated_at": time.strftime("%FT%T%z")})
                print(f"{args.judge}/{args.method}: {len(existing)}/{len(selected)}; "
                      f"Win@1={summary['win1_percent']:.2f}; Win-Full={summary['win_full_percent']:.2f}; "
                      f"Success={summary['success_within5_percent']:.2f}", flush=True)
        finally:
            if endpoint is not None:
                endpoint.close()
        summary = summarize(list(existing.values()), golden, contract)
        summary["judgments_sha256"] = digest(records_path) if records_path.exists() else None
        all_requests = [read_json(p) for p in requests_dir.glob("*.json") if not p.name.endswith(".error.json")]
        summary["total_independent_requests"] = len(all_requests)
        summary["format_retry_count"] = sum(r["attempts"] - 1 for r in all_requests)
        atomic_json(args.output / "summary.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge", choices=JUDGES, required=True)
    parser.add_argument("--method", choices=METHOD_LABELS, default="our_eclp_offline_lpo")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-new-scenarios", type=int, default=0, help="0 completes all; 20 runs a pilot with the same resumable contract")
    parser.add_argument("--memory-cap-gib", type=float)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--scenario-chunk-size", type=int, default=16)
    args = parser.parse_args()
    if args.max_new_scenarios < 0:
        parser.error("max-new-scenarios must be nonnegative")
    if args.physical_gpu < 0 or args.batch_size < 1 or args.scenario_chunk_size < 1:
        parser.error("physical-gpu must be nonnegative and batch/chunk sizes must be positive")
    print(json.dumps(evaluate(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

