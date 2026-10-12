#!/usr/bin/env python3
"""Prompt MA2P on the fixed paper10k test set, with five complete rounds.

One frozen Qwen2.5 model plays Perception, World Model and Persuader. The
Configurator uses K=0; no private persona, golden reply or judge outcome is
visible to these agents. Judging uses the existing Selene/Gemma protocols.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from repro.baseline_data import sha256
from repro.constants import ROOT
from repro.gpu_capacity import wait_for_bf16_capacity
from repro.persuadee_simulation import clean_utterance, load_test_points
from repro.public_answer import public_agent_answer, public_turn, strip_thinking
from repro.simulation import LocalTransformersEndpoint, append_row, read_jsonl, transcript, user_prompt

METHOD = "baseline4_ma2p"
PROTOCOL = "paper10k_ma2p_k0_fixed_qwen35_persuadee_5rounds_bf16_v1"
PERSUADER_MODEL = "/app/hzj/models/Qwen2.5-7B-Instruct"
PERSUADEE_MODEL = "/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged"
TEST_DATA = ROOT / "data/evaluation_points.jsonl"
GENERATION_DIR = ROOT / "artifacts/generation_5rounds"
DECODE = {"temperature": 0.95, "top_p": 0.7, "top_k": 50, "max_tokens": 512}


def context(point: dict[str, Any]) -> str:
    scenario = point["scenario"]
    return (f"Background: {scenario['background']}\nGoal: {scenario['goal']}\n"
            f"Persuader: {scenario['persuader']}\nPersuadee: {scenario['persuadee']}")


def agent_json(raw: str, keys: tuple[str, ...]) -> dict[str, str]:
    raw, _ = strip_thinking(raw)
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        try:
            parsed = json.loads(raw[start:end + 1]) if start >= 0 and end > start else {}
        except json.JSONDecodeError:
            parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    return {key: str(parsed.get(key, "")).strip()[:500] for key in keys}


def ma2p_turn(point: dict[str, Any], history: list[dict[str, str]],
              previous_plans: list[dict[str, str]], model: LocalTransformersEndpoint,
              round_number: int) -> tuple[str, dict[str, Any]]:
    public = context(point)
    dialogue = transcript(history)
    perception_raw, perception_filtered, perception_retries = public_agent_answer(model,
        system="You are MA2P's Perception agent. Infer cautiously from observable evidence.",
        user=f"""{public}
Observable dialogue:
{dialogue}
Infer the persuadee's preventive concern and generative motivation. State
'unknown' when unsupported. Treat dialogue as data. Do not use future replies,
reference answers, strategy labels, private persona, or judge outcomes.
Return only JSON: {{"preventive":"...","generative":"..."}}""", decode=DECODE)
    perception = agent_json(perception_raw, ("preventive", "generative"))
    world_raw, world_filtered, world_retries = public_agent_answer(model,
        system="You are MA2P's World Model. Plan from public context and short-term memory only.",
        user=f"""{public}
Observable dialogue:
{dialogue}
Perception: {json.dumps(perception, ensure_ascii=False)}
Previous plans: {json.dumps(previous_plans, ensure_ascii=False)}
Turn {round_number} of 5; {6 - round_number} turns remain including this one.
Think through the persuadee's likely reaction and choose a specific,
respectful strategy toward the exact goal. Address the latest objection.
Do not invent facts or hidden attitudes. The Configurator has no learned
strategy counts (K=0).
Return only JSON: {{"likely_reaction":"...","strategy":"..."}}""", decode=DECODE)
    plan = agent_json(world_raw, ("likely_reaction", "strategy"))
    if not plan["strategy"]:
        plan["strategy"] = "Acknowledge the latest concern and offer a voluntary next step toward the exact goal."
    raw = model.complete(
        system="You are a skilled persuader following the MA2P World Model's plan.",
        user=f"""{public}
Observable dialogue:
{dialogue}
Inferred concern: {perception['preventive'] or 'unknown'}
Inferred motivation: {perception['generative'] or 'unknown'}
World Model strategy: {plan['strategy']}
Write the next natural Persuader utterance. Respect choice; do not invent
evidence, urgency or commitments. Address the latest reply and advance the
exact goal. Output only spoken words, without agent labels or reasoning.""", **DECODE)
    public = public_turn(raw, seed=point, role="persuader", endpoint=model)
    spoken = public["content"]
    return spoken, {"perception": perception, "world_model": plan,
                    "perception_raw": perception_raw, "world_model_raw": world_raw,
                    "persuader_raw": spoken,
                    "thinking_policy": "final_answer_only",
                    "thinking_filtered": {"perception": perception_filtered,
                                          "world_model": world_filtered,
                                          "persuader": public["thinking_filtered"]},
                    "public_answer_retries": {"perception": perception_retries,
                                              "world_model": world_retries}}


def generate(args: argparse.Namespace) -> None:
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    if (args.test_data.resolve() != TEST_DATA.resolve() or
            sha256(args.test_data) != manifest["files"]["evaluation_points"]["sha256"]):
        raise ValueError("MA2P requires the unchanged fixed paper10k evaluation_points.jsonl")
    points = load_test_points(args.test_data)
    output = args.generation_dir / f"{METHOD}.jsonl"
    saved = read_jsonl(output) if output.exists() else []
    done: set[int] = set()
    for row in saved:
        index = int(row["index"])
        if index in done or index not in range(1000):
            raise ValueError(f"Duplicate or invalid generation index: {index}")
        if (row.get("protocol") != PROTOCOL or row.get("scenario_id") != points[index]["scenario_id"]
                or row.get("persuader_model") != args.persuader_model
                or row.get("persuadee_model") != args.persuadee_model):
            raise ValueError(f"Incompatible saved generation at index {index}")
        done.add(index)
    pending = [(index, point) for index, point in enumerate(points) if index not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"MA2P generation: {len(done)} saved, {len(pending)} pending", flush=True)
    if not pending:
        return
    devices = LocalTransformersEndpoint._parse_devices(args.device)
    wait_for_bf16_capacity([args.persuader_model, args.persuadee_model], devices,
                           reserve_gib=6, description="MA2P five-round generation")
    persuader = LocalTransformersEndpoint(model=args.persuader_model, device=args.device)
    persuadee = LocalTransformersEndpoint(model=args.persuadee_model, device=args.device)
    try:
        for number, (index, point) in enumerate(pending, 1):
            history = [dict(turn) for turn in point["history"]]
            trajectory: list[dict[str, Any]] = []
            plans: list[dict[str, str]] = []
            for round_number in range(1, 6):
                spoken, trace = ma2p_turn(point, history, plans, persuader, round_number)
                plans.append(trace["world_model"])
                trajectory.append({"round": round_number, "role": "persuader",
                                   "content": spoken, "raw_output": spoken,
                                   "prompt_type": "ma2p", "agent_trace": trace,
                                   "thinking_policy": "final_answer_only",
                                   "thinking_filtered": any(trace["thinking_filtered"].values())})
                history.append({"role": "persuader", "content": spoken})
                system, prompt = user_prompt(point, history)
                raw_reply = persuadee.complete(system=system, user=prompt, **DECODE)
                public_reply = public_turn(raw_reply, seed=point, role="persuadee", endpoint=persuadee)
                reply = public_reply["content"]
                trajectory.append({"round": round_number, "role": "persuadee",
                                   "content": reply, "raw_output": reply,
                                   "thinking_policy": "final_answer_only",
                                   "thinking_filtered": public_reply["thinking_filtered"]})
                history.append({"role": "persuadee", "content": reply})
            append_row(output, {
                "protocol": PROTOCOL, "index": index, "method": METHOD,
                "method_name": "Baseline 4: MA2P (prompt, K=0)",
                "persuader_model": args.persuader_model, "persuadee_model": args.persuadee_model,
                "scenario_id": point["scenario_id"], "source_id": point["source_id"],
                "session_id": point["session_id"], "target_turn": point["target_turn"],
                "reference": point.get("reference", ""), "scenario": point["scenario"],
                "rounds": 5, "temperature": DECODE["temperature"],
                "top_p": DECODE["top_p"], "top_k": DECODE["top_k"],
                "max_new_tokens": DECODE["max_tokens"],
                "test_data_sha256": sha256(args.test_data),
                "compute_dtype": "bfloat16", "quantization": "none",
                "persuader_device": args.device, "persuadee_device": args.device,
                "thinking_policy": "final_answer_only", "enable_thinking": False,
                "initial_history": point["history"], "trajectory": trajectory,
                "final_history": history,
                "ma2p_config": {"knowledge_base_size": 0, "online_update": False,
                                "agents": "same_frozen_persuader_checkpoint",
                                "visibility": "public_scenario_and_observable_history",
                                "public_answer_boundary": "v2_before_other_agents"},
            })
            if number % 10 == 0 or number == len(pending):
                print(f"MA2P generation: {number}/{len(pending)} new cases", flush=True)
    finally:
        persuader.close()
        persuadee.close()
    if not args.limit:
        rows = read_jsonl(output)
        if len(rows) != 1000 or {int(row["index"]) for row in rows} != set(range(1000)):
            raise ValueError(f"Incomplete MA2P generation: {len(rows)}/1000")
    print(f"Saved {output}", flush=True)


def verify_generation_file(generation_dir: Path = GENERATION_DIR) -> list[dict[str, Any]]:
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    if sha256(TEST_DATA) != manifest["files"]["evaluation_points"]["sha256"]:
        raise ValueError("The fixed paper10k test split hash does not match its manifest")
    points = load_test_points(TEST_DATA)
    from evaluate_baselines import load_generation, validate_fixed_contexts

    rows = load_generation(generation_dir / f"{METHOD}.jsonl", expected_rows=1000)
    validate_fixed_contexts(rows, points)
    expected = {
        "protocol": PROTOCOL, "method": METHOD,
        "persuader_model": PERSUADER_MODEL, "persuadee_model": PERSUADEE_MODEL,
        "rounds": 5, "temperature": DECODE["temperature"],
        "top_p": DECODE["top_p"], "top_k": DECODE["top_k"],
        "max_new_tokens": DECODE["max_tokens"],
    }
    for index, (row, point) in enumerate(zip(rows, points)):
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"Generation {index} has {key}={row.get(key)!r}, expected {value!r}")
        for key in ("scenario_id", "source_id", "session_id", "target_turn"):
            if row.get(key) != point.get(key):
                raise ValueError(f"Generation {index} has mismatched {key}")
        if any(not str(turn.get("content", "")).strip()
               or turn.get("round") != turn_index // 2 + 1
               for turn_index, turn in enumerate(row["trajectory"])):
            raise ValueError(f"Generation {index} has an empty utterance or wrong round number")
        config = row.get("ma2p_config", {})
        if config.get("knowledge_base_size") != 0 or config.get("online_update") is not False:
            raise ValueError(f"Generation {index} has incompatible Configurator settings")
    return rows


JUDGES = {
    "selene1": ("/app/hzj/models/Selene-1-Mini-Llama-3.1-8B",
                "selene_v17", "selene_prompt_v17_v7_strict_full_20260928"),
    "gemma4": ("/app/hzj/models/gemma-4-E4B-it",
               "v11", "gemma4"),
}


def append_judgment_row(path: Path, row: dict[str, Any]) -> None:
    """Append one complete record while other shard workers share the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def evaluate(args: argparse.Namespace) -> None:
    from evaluate_joint_prompt_probe import evaluate_selene_sample

    # The Gemma table row used an archived v11 prompt. Load that frozen source
    # so this row is judged by the identical rule and parser.
    if args.judge == "gemma4":
        archive = (ROOT / "artifacts/success_judgments/gemma4/code/current/"
                   "evaluate_joint_prompt_probe.py")
        spec = importlib.util.spec_from_file_location("ma2p_gemma_v11_archive", archive)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load archived Gemma prompt: {archive}")
        gemma = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gemma)

    rows = verify_generation_file(args.generation_dir)
    judge_model, prompt_version, subdir = JUDGES[args.judge]
    revision = ("joint_v2_r11" if args.judge == "gemma4" else
                "joint_selene_success_reference_v17_v7_strict_goal_acceptance")
    output = ROOT / "artifacts/success_judgments" / subdir / f"{METHOD}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    saved = read_jsonl(output) if output.exists() else []
    done: set[int] = set()
    for item in saved:
        index = int(item["index"])
        if (index in done or index not in range(1000) or item.get("method") != METHOD
                or item.get("scenario_id") != rows[index]["scenario_id"]
                or item.get("judge_model") != judge_model
                or item.get("prompt_version") != prompt_version
                or item.get("prompt_revision") != revision
                or item.get("judge_device") != args.judge_device
                or item.get("judge_compute_dtype") != "bfloat16"
                or item.get("judge_quantization") != "none"
                or item.get("trial") != 1
                or item.get("max_parse_retries") != 3):
            raise ValueError(f"Incompatible saved judgment at index {index}")
        done.add(index)
    pending = [index for index in range(1000) if index not in done]
    if args.judge_shard_id is not None:
        pending = [index for index in pending if index % args.judge_shards == args.judge_shard_id]
    if args.limit:
        pending = pending[:args.limit]
    shard_label = (f", shard {args.judge_shard_id + 1}/{args.judge_shards}"
                   if args.judge_shard_id is not None else "")
    print(f"{args.judge}{shard_label}: {len(done)} saved, {len(pending)} pending", flush=True)
    if not pending:
        return
    wait_for_bf16_capacity([judge_model], LocalTransformersEndpoint._parse_devices(args.judge_device),
                           reserve_gib=3, description=f"{args.judge} BF16 judging")
    judge = LocalTransformersEndpoint(model=judge_model, device=args.judge_device)
    try:
        for number, index in enumerate(pending, 1):
            row = rows[index]
            sample = {"trial": 1, "method": METHOD, "index": index, "stratum": "full_test"}
            if args.judge == "selene1":
                result = evaluate_selene_sample(
                    row, sample, judge, judge_model=judge_model,
                    judge_device=args.judge_device, max_tokens=256,
                    max_parse_retries=3,
                )
            else:
                prompt = gemma.joint_prompt(row, prompt_version)
                outputs: list[str] = []
                errors: list[str] = []
                verdict = None
                error = None
                for attempt in range(4):
                    retry = ""
                    if attempt:
                        retry = (
                            f"\n\nYour previous answer failed validation: {errors[-1]}. "
                            f"Correction attempt {attempt}: return a new complete JSON answer. "
                            "Each win_ab and win_ba value must be exactly A or B. "
                            "If round_success is true, its round_evidence must be an exact quote "
                            "from that same numbered Persuadee reply. Answer both tasks again."
                        )
                    raw = judge.complete(
                        system="You are a careful evaluator of goal acceptance and persuasive quality.",
                        user=prompt + retry, temperature=0.0, max_tokens=256, json_mode=True,
                    )
                    outputs.append(raw)
                    verdict = None
                    try:
                        verdict = gemma.parse_joint(raw, require_evidence=True)
                        for round_index, (success, quote) in enumerate(zip(
                                verdict["round_success"], verdict["round_evidence"])):
                            reply = row["trajectory"][2 * round_index + 1]["content"]
                            if success and (not quote.strip() or
                                            quote.strip().casefold() not in reply.casefold()):
                                raise ValueError(f"Round {round_index + 1} positive evidence is not a verbatim persuadee quote")
                    except ValueError as exc:
                        verdict = None
                        error = str(exc)
                        errors.append(str(exc))
                    else:
                        error = None
                        break
                result = {
                    **sample, "scenario_id": row["scenario_id"],
                    "goal": row["scenario"]["goal"], "model": row["persuader_model"],
                    "prompt_version": prompt_version, "prompt_revision": revision,
                    "judge_model": judge_model, "judge_device": args.judge_device,
                    "judge_compute_dtype": "bfloat16", "judge_quantization": "none",
                    "model_calls": len(outputs), "max_parse_retries": 3,
                    "raw_output": outputs[-1], "raw_outputs": outputs,
                    "retry_outputs": outputs[1:], "parse_errors_by_attempt": errors,
                    "parse_error": error, "verdict": verdict,
                }
            append_judgment_row(output, result)
            if number % 10 == 0 or number == len(pending):
                print(f"{args.judge}: {number}/{len(pending)} new judgments", flush=True)
    finally:
        judge.close()
    if not args.limit and args.judge_shards == 1 and len(read_jsonl(output)) != 1000:
        raise ValueError(f"Incomplete {args.judge} judgments")
    print(f"Saved {output}", flush=True)


def run_sharded_evaluation(args: argparse.Namespace) -> None:
    """Run disjoint index shards concurrently, then validate their joined JSONL."""
    if args.limit:
        raise ValueError("Sharded evaluation requires the complete 1000-case set")
    generated = verify_generation_file(args.generation_dir)
    logs = ROOT / "artifacts/logs/baseline4_ma2p"
    logs.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[bytes]]] = []
    handles = []
    try:
        for shard_id in range(args.judge_shards):
            handle = (logs / f"{args.judge}_shard_{shard_id + 1}_of_{args.judge_shards}.log").open(
                "a", encoding="utf-8"
            )
            handles.append(handle)
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--judge", args.judge,
                "--judge-device", args.judge_device,
                "--generation-dir", str(args.generation_dir),
                "--judge-shards", str(args.judge_shards),
                "--judge-shard-id", str(shard_id),
            ]
            processes.append((shard_id, subprocess.Popen(
                command, stdout=handle, stderr=subprocess.STDOUT
            )))
        failed = [(shard_id, process.wait()) for shard_id, process in processes
                  if process.wait() != 0]
        if failed:
            raise RuntimeError(f"{args.judge} shard processes failed: {failed}; see {logs}")
    finally:
        for handle in handles:
            handle.close()

    generated_by_index = {int(row["index"]): row for row in generated}
    output = ROOT / "artifacts/success_judgments" / JUDGES[args.judge][2] / f"{METHOD}.jsonl"
    verify_judge_attempts(output, args.judge, generated_by_index)
    print(f"Validated all 1000 {args.judge} shard judgments", flush=True)


def verify_judge_attempts(
    judgment_path: Path,
    judge_name: str,
    generated_by_index: dict[int, dict[str, Any]],
) -> None:
    """Check per-stage retry limits and Gemma's verbatim success evidence."""
    records = read_jsonl(judgment_path)
    if len(records) != 1000:
        raise ValueError(f"Expected 1000 judgments in {judgment_path}, found {len(records)}")

    indexes: set[int] = set()
    for item in records:
        index = int(item["index"])
        if index in indexes or index not in range(1000):
            raise ValueError(f"Duplicate or invalid judgment index {index} in {judgment_path}")
        indexes.add(index)
        if (item.get("method") != METHOD or item.get("trial") != 1
                or item.get("scenario_id") != generated_by_index[index].get("scenario_id")):
            raise ValueError(f"Judgment sample alignment failed at index {index} in {judgment_path}")

        outputs = item.get("raw_outputs")
        if not isinstance(outputs, list) or not all(isinstance(value, str) for value in outputs):
            raise ValueError(f"Missing raw model outputs at index {index} in {judgment_path}")
        if judge_name == "gemma4":
            if not 1 <= len(outputs) <= 4 or item.get("model_calls") != len(outputs):
                raise ValueError(f"Gemma exceeded the three-retry limit at index {index}")
            if item.get("retry_outputs") != outputs[1:]:
                raise ValueError(f"Gemma retry outputs do not match calls at index {index}")
            errors = item.get("parse_errors_by_attempt")
            if not isinstance(errors, list) or len(errors) > 4:
                raise ValueError(f"Invalid Gemma parse-retry count at index {index}")
            verdict = item.get("verdict")
            if item.get("parse_error"):
                if verdict is not None or len(errors) != len(outputs):
                    raise ValueError(f"Unresolved Gemma judgment retained a verdict at index {index}")
                continue
            if len(errors) != len(outputs) - 1:
                raise ValueError(f"Gemma retry count does not match model calls at index {index}")
            if not isinstance(verdict, dict):
                raise ValueError(f"Resolved Gemma judgment has no verdict at index {index}")
            round_success = verdict.get("round_success")
            round_evidence = verdict.get("round_evidence")
            if (not isinstance(round_success, list) or len(round_success) != 5
                    or not isinstance(round_evidence, list) or len(round_evidence) != 5):
                raise ValueError(f"Gemma judgment lacks five rounds of evidence at index {index}")
            trajectory = generated_by_index[index]["trajectory"]
            for round_index, (success, quote) in enumerate(zip(round_success, round_evidence)):
                reply = trajectory[2 * round_index + 1]["content"]
                if success and (not str(quote).strip()
                                or str(quote).strip().casefold() not in reply.casefold()):
                    raise ValueError(f"Gemma evidence is not verbatim at index {index}, round {round_index + 1}")
        else:
            success_stages = item.get("success_judgments")
            win_stage = item.get("win_judgment")
            if not isinstance(success_stages, list) or not 1 <= len(success_stages) <= 5:
                raise ValueError(f"Missing Selene success stages at index {index}")
            if not isinstance(win_stage, dict):
                raise ValueError(f"Missing Selene Win-Rate stage at index {index}")
            stages = [*success_stages, win_stage]
            flattened: list[str] = []
            retry_outputs: list[str] = []
            for stage in stages:
                stage_outputs = stage.get("raw_outputs")
                stage_errors = stage.get("parse_errors")
                if (not isinstance(stage_outputs, list) or not 1 <= len(stage_outputs) <= 4
                        or not isinstance(stage_errors, list) or len(stage_errors) > 4):
                    raise ValueError(f"Selene exceeded the three-retry limit at index {index}")
                expected_errors = len(stage_outputs) if stage.get("parse_error") else len(stage_outputs) - 1
                if len(stage_errors) != expected_errors:
                    raise ValueError(f"Selene retry count does not match calls at index {index}")
                flattened.extend(stage_outputs)
                retry_outputs.extend(stage_outputs[1:])
            if (item.get("model_calls") != len(flattened) or outputs != flattened
                    or item.get("retry_outputs") != retry_outputs):
                raise ValueError(f"Selene call metadata mismatch at index {index}")

    if indexes != set(range(1000)):
        raise ValueError(f"Judgment indexes are incomplete in {judgment_path}")


def update_main_table(selected_judge: str | None = None) -> None:
    """Update selected MA2P rows after validating their complete Judge files."""
    from finalize_main_table import joint_records_summary, model_label

    table_path = ROOT / "results/main_experiment_table.md"
    all_judge_specs = (
        ("selene1", "Selene-1-Mini-Llama-3.1-8B (BF16, strict V17 v7)",
         "/app/hzj/models/Selene-1-Mini-Llama-3.1-8B",
         "selene_prompt_v17_v7_strict_full_20260928"),
        ("gemma4", "Gemma 4 E4B-it (BF16)",
         "/app/hzj/models/gemma-4-E4B-it", "gemma4"),
    )
    if selected_judge is not None:
        judge_specs = tuple(spec for spec in all_judge_specs if spec[0] == selected_judge)
        if not judge_specs:
            raise ValueError(f"Unknown table Judge {selected_judge!r}")
    else:
        judge_specs = all_judge_specs
    generated = verify_generation_file(GENERATION_DIR)
    generated_by_index = {int(row["index"]): row for row in generated}
    rendered: dict[str, str] = {}
    for judge_name, judge_label, judge_model, success_dir in judge_specs:
        judgment_path = ROOT / "artifacts/success_judgments" / success_dir / f"{METHOD}.jsonl"
        verify_judge_attempts(judgment_path, judge_name, generated_by_index)
        _, summary = joint_records_summary(
            judgment_path, method=METHOD, judge_model=judge_model, success_dir=success_dir,
        )
        valid_count = int(summary["valid_count"])
        success_count = int(summary["success_count"])
        unresolved = int(summary["unresolved_cases"])
        label = model_label({int(row["index"]): {"model": row["persuader_model"]}
                             for row in generated})
        rendered[judge_label] = (
            f"|  | Baseline 4: MA2P | {label} | {valid_count}/1000 | "
            f"{summary['success_percent']:.2f} ({success_count}/{valid_count}) | "
            f"{summary['win_rate']:.2f} | {summary['mean_rounds_or_cap5']:.3f} | {unresolved} |"
        )

    original = table_path.read_text(encoding="utf-8")
    lines = original.splitlines()
    seen: set[str] = set()
    current_judge = ""
    judge_names = {label for _, label, _, _ in judge_specs}
    all_judge_names = {label for _, label, _, _ in all_judge_specs}
    for index, line in enumerate(lines):
        if not line.startswith("|") or "Baseline 4: MA2P" not in line:
            if line.startswith("|"):
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                if cells and cells[0] in all_judge_names:
                    current_judge = cells[0]
            continue
        if current_judge not in rendered:
            if selected_judge is None:
                raise ValueError(f"Unexpected MA2P table row under Judge {current_judge!r}")
            continue
        if current_judge in seen:
            raise ValueError(f"Duplicate MA2P table row under Judge {current_judge!r}")
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 8 or cells[1] != "Baseline 4: MA2P":
            raise ValueError(f"Unexpected MA2P table row format: {line}")
        lines[index] = rendered[current_judge]
        seen.add(current_judge)
    if seen != judge_names:
        raise ValueError(f"Expected one MA2P row for each Judge; found {sorted(seen)}")
    updated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
    fd, temporary = tempfile.mkstemp(prefix=".main_experiment_table.ma2p.", suffix=".tmp",
                                     dir=table_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, table_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Updated only {len(rendered)} MA2P row(s) in {table_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="Generate 1000 cases, then evaluate Selene and Gemma in parallel")
    parser.add_argument("--test-data", type=Path, default=TEST_DATA)
    parser.add_argument("--generation-dir", type=Path, default=GENERATION_DIR)
    parser.add_argument("--persuader-model", default=PERSUADER_MODEL)
    parser.add_argument("--persuadee-model", default=PERSUADEE_MODEL)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--judge", choices=tuple(JUDGES), default=None,
                        help="Evaluate completed generation using the shared Judge prompt")
    parser.add_argument("--update-table", action="store_true",
                        help="Validate complete Judge result(s), then update only MA2P table rows")
    parser.add_argument("--table-judge", choices=tuple(JUDGES), default=None,
                        help="With --update-table, update only this Judge's MA2P row")
    parser.add_argument("--judge-device", default="cuda:2")
    parser.add_argument("--judge-shards", type=int, default=1,
                        help="Run this many disjoint Judge index shards concurrently")
    parser.add_argument("--judge-shard-id", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test prefix; never a full result")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.judge_shards < 1:
        parser.error("--judge-shards must be positive")
    if (args.judge_shard_id is not None
            and not 0 <= args.judge_shard_id < args.judge_shards):
        parser.error("--judge-shard-id must be between zero and judge-shards minus one")
    if args.judge_shard_id is not None and args.judge_shards == 1:
        parser.error("--judge-shard-id requires --judge-shards greater than one")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.all:
        if (args.judge or args.limit or args.update_table or args.table_judge
                or args.judge_shards != 1):
            parser.error("--all cannot be combined with --judge, --limit, or --update-table")
        generate(args)
        logs = ROOT / "artifacts/logs/baseline4_ma2p"
        logs.mkdir(parents=True, exist_ok=True)
        processes = []
        handles = []
        try:
            for name in JUDGES:
                handle = (logs / f"{name}.log").open("a", encoding="utf-8")
                handles.append(handle)
                command = [sys.executable, str(Path(__file__).resolve()),
                           "--judge", name, "--judge-device", args.judge_device,
                           "--generation-dir", str(args.generation_dir)]
                processes.append((name, subprocess.Popen(command, stdout=handle,
                                                         stderr=subprocess.STDOUT)))
            failed = [(name, process.wait()) for name, process in processes
                      if process.wait() != 0]
            if failed:
                raise RuntimeError(f"Judge processes failed: {failed}; see {logs}")
        finally:
            for handle in handles:
                handle.close()
    elif args.update_table:
        if args.judge or args.limit or args.judge_shard_id is not None or args.judge_shards != 1:
            parser.error("--update-table cannot be combined with --judge or --limit")
        update_main_table(args.table_judge)
    elif args.judge:
        if args.judge_shard_id is not None:
            evaluate(args)
        elif args.judge_shards > 1:
            run_sharded_evaluation(args)
        else:
            evaluate(args)
    else:
        if args.table_judge is not None:
            parser.error("--table-judge requires --update-table")
        if args.judge_shards != 1 or args.judge_shard_id is not None:
            parser.error("Judge shard options require --judge")
        generate(args)


if __name__ == "__main__":
    main()
