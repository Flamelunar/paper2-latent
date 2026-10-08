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
    selection_path = ROOT / "data/test_selection_192.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["original_indexes"]
    if (len(selected) != 192 or len(set(selected)) != 192 or selection["source_sha256"] != sha256(args.test_data)
            or selected != [i for i,p in enumerate(points) if p["target_turn"] == 1 and p["history"] == []]):
        raise ValueError("The fixed 192 subset changed")
    output = args.generation_dir / f"{METHOD}.jsonl"
    saved = read_jsonl(output) if output.exists() else []
    done: set[int] = set()
    for row in saved:
        index = int(row["index"])
        if index in done or index not in selected:
            raise ValueError(f"Duplicate or invalid generation index: {index}")
        if (row.get("protocol") != PROTOCOL or row.get("scenario_id") != points[index]["scenario_id"]
                or row.get("persuader_model") != args.persuader_model
                or row.get("persuadee_model") != args.persuadee_model):
            raise ValueError(f"Incompatible saved generation at index {index}")
        done.add(index)
    pending = [(index, points[index]) for index in selected if index not in done]
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
                "test_selection_sha256": sha256(selection_path), "test_selection_revision": selection["revision"],
                "selected_count": 192,
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
        if len(rows) != 192 or {int(row["index"]) for row in rows} != set(selected):
            raise ValueError(f"Incomplete MA2P generation: {len(rows)}/192")
    print(f"Saved {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate only the fixed 192 empty-history MA2P scenarios; BF16, no judge.")
    parser.add_argument("--test-data", type=Path, default=TEST_DATA)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--persuader-model", required=True)
    parser.add_argument("--persuadee-model", default=PERSUADEE_MODEL)
    parser.add_argument("--device", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    generate(args)

if __name__ == "__main__":
    main()

