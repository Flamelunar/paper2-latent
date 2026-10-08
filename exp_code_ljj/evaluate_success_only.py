#!/usr/bin/env python3
"""Evaluate five-round strict acceptance with a local BF16 Judge only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import statistics
import time
from pathlib import Path
from typing import Any

from repro.profile import add_profile_argument, configure_profile_from_argv

PROFILE = configure_profile_from_argv()
from repro.constants import ROOT, TEST_SCENARIOS
from repro.simulation import LocalTransformersEndpoint, read_jsonl
from evaluate_baselines import (
    ARCHIVED_METHODS,
    SUCCESS_PROMPTS,
    load_generation,
    parse_success,
    validate_fixed_contexts,
)


JUDGE_MODEL = "/app/hzj/models/Meta-Llama-3.1-8B-Instruct"
GENERATION_DIR = ROOT / "artifacts/generation_5rounds/legacy"
OUTPUT_DIR = ROOT / "artifacts/success_judgments/llama31"
PROTOCOLS = {
    "v1": "strict_acceptance_five_rounds_v1",
    "v2": "goal_acceptance_five_rounds_v2",
}
JUDGE_QUANTIZATION = "none"


def make_judgment(
    row: dict[str, Any], judge: LocalTransformersEndpoint, batch_size: int,
    prompt_version: str = "v1",
) -> list[dict[str, Any]]:
    prompt = SUCCESS_PROMPTS[prompt_version]
    system = (
        "You are a strict, evidence-based judge of explicit goal acceptance."
        if prompt_version == "v1" else
        "You are an evidence-based judge of whether the persuadee accepted the stated goal."
    )
    requests = [
        {
            "system": system,
            "user": prompt(
                row, row["trajectory"][: round_number * 2], round_number
            ),
            "temperature": 0.0,
            "max_tokens": 128,
            "json_mode": True,
        }
        for round_number in range(1, 6)
    ]
    outputs: list[str] = []
    for start in range(0, len(requests), batch_size):
        outputs.extend(judge.complete_batch(requests[start : start + batch_size]))

    judgments: list[dict[str, Any]] = []
    for round_number, raw in enumerate(outputs, 1):
        retries: list[str] = []
        unresolved = False
        try:
            success, evidence = parse_success(raw)
        except ValueError:
            retry_prompt = prompt(
                row, row["trajectory"][: round_number * 2], round_number
            )
            retry_prompt += (
                '\nReturn only JSON: {"success": true} or {"success": false}.'
                if prompt_version == "v1" else
                "\nKeep the same verdict and express it as a JSON Boolean field named success, with a short evidence string."
            )
            for _ in range(2):
                retry = judge.complete_batch([{
                    "system": "Judge explicit acceptance of the stated goal.",
                    "user": retry_prompt,
                    "temperature": 0.0,
                    "max_tokens": 64,
                    "json_mode": True,
                }])[0]
                retries.append(retry)
                try:
                    success, evidence = parse_success(retry)
                    break
                except ValueError:
                    pass
            else:
                success = False
                evidence = "No parseable success verdict after two clarification prompts."
                unresolved = True
        judgments.append({
            "round": round_number,
            "success": success,
            "evidence": evidence,
            "raw_output": raw,
            "retry_outputs": retries,
            "parse_unresolved": unresolved,
        })
    return judgments


def validate_existing(
    path: Path,
    *,
    method: str,
    judge_model: str,
    protocol: str,
    generation: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    existing: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return existing
    generation_by_index = {int(row["index"]): row for row in generation}
    for saved in read_jsonl(path):
        index = int(saved["index"])
        source = generation_by_index.get(index)
        if source is None or index in existing:
            raise ValueError(f"Unexpected or duplicate index {index} in {path}")
        if saved.get("judge_model") != judge_model or saved.get("method") != method:
            raise ValueError(f"Judge or method mismatch in {path} at index {index}")
        if saved.get("protocol") != protocol:
            raise ValueError(f"Prompt protocol mismatch in {path} at index {index}")
        if saved.get("scenario_id") != source.get("scenario_id"):
            raise ValueError(f"Generation mismatch in {path} at index {index}")
        if saved.get("candidate") != str(source["trajectory"][0]["content"]).strip():
            raise ValueError(f"Candidate mismatch in {path} at index {index}")
        if saved.get("reference") != str(source["reference"]).strip():
            raise ValueError(f"Reference mismatch in {path} at index {index}")
        judgments = saved.get("strict_success_by_round")
        if not isinstance(judgments, list) or [x.get("round") for x in judgments] != [1, 2, 3, 4, 5]:
            raise ValueError(f"Incomplete five-round judgments in {path} at index {index}")
        first = next((x["round"] for x in judgments if x.get("success")), None)
        if first != saved.get("first_success_round"):
            raise ValueError(f"Inconsistent first success round in {path} at index {index}")
        if bool(first) != bool(saved.get("strict_success_within_5_rounds")):
            raise ValueError(f"Inconsistent success summary in {path} at index {index}")
        existing[index] = saved
    return existing


def summarize(rows: list[dict[str, Any]], judge_model: str, prompt_version: str) -> dict[str, Any]:
    first_rounds = [row["first_success_round"] for row in rows]
    capped_rounds = [value if value is not None else 5 for value in first_rounds]
    return {
        "n": len(rows),
        "judge_model": judge_model,
        "judge_compute_dtype": "bfloat16",
        "judge_quantization": "none",
        "protocol": PROTOCOLS[prompt_version],
        "prompt_version": prompt_version,
        "strict_success_count": sum(value is not None for value in first_rounds),
        "strict_success_within_5_rounds_percent": 100 * sum(value is not None for value in first_rounds) / len(rows),
        "mean_rounds_to_success_failures_count_as_5": statistics.fmean(capped_rounds),
        "success_parse_unresolved": sum(
            bool(item.get("parse_unresolved"))
            for row in rows for item in row["strict_success_by_round"]
        ),
        "strict_success_rate_by_round_percent": {
            str(round_number): 100 * statistics.fmean(
                bool(row["strict_success_by_round"][round_number - 1]["success"])
                for row in rows
            )
            for round_number in range(1, 6)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_argument(parser)
    parser.add_argument("--generation-dir", type=Path, default=GENERATION_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--judge-device", default="cuda:2")
    parser.add_argument(
        "--judge-quantization", choices=("none",), default=JUDGE_QUANTIZATION
    )
    parser.add_argument("--judge-batch-size", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--startup-delay-seconds", type=float, default=0.0)
    parser.add_argument("--require-gpu-only", action="store_true")
    parser.add_argument("--empty-cache-after-row", action="store_true")
    parser.add_argument("--method", choices=[m.slug for m in ARCHIVED_METHODS], required=True)
    parser.add_argument("--prompt-version", choices=tuple(SUCCESS_PROMPTS), default="v1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if PROFILE != "paper10k":
        raise SystemExit("Success evaluation requires the paper10k held-out split")

    method = next(method for method in ARCHIVED_METHODS if method.slug == args.method)
    generation = load_generation(
        args.generation_dir / f"{method.slug}.jsonl", expected_rows=TEST_SCENARIOS
    )
    test_points = read_jsonl(ROOT / "data/evaluation_points.jsonl")
    validate_fixed_contexts(generation, test_points)

    protocol = PROTOCOLS[args.prompt_version]
    if args.output_dir is None:
        args.output_dir = OUTPUT_DIR if args.prompt_version == "v1" else OUTPUT_DIR.with_name("llama31_v2")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{method.slug}.jsonl"
    summary_path = args.output_dir / f"{method.slug}.summary.json"
    if args.force and output.exists():
        output.unlink()
    existing = validate_existing(
        output,
        method=method.slug,
        judge_model=args.judge_model,
        protocol=protocol,
        generation=generation,
    )
    print(
        f"Starting {method.slug}: {len(existing)}/{len(generation)} rows already complete; "
        f"Judge={args.judge_model}, device={args.judge_device}, "
        f"quantization={args.judge_quantization}, BF16 compute, protocol={protocol}",
        flush=True,
    )

    judge = LocalTransformersEndpoint(
        model=args.judge_model,
        device=args.judge_device,
        quantization=args.judge_quantization,
    )
    device_map = getattr(judge.model, "hf_device_map", {})
    device_counts: dict[str, int] = {}
    for device in device_map.values():
        label = f"cuda:{device}" if isinstance(device, int) else str(device)
        device_counts[label] = device_counts.get(label, 0) + 1
    print(f"Loaded Judge device map: {device_counts or 'single-device placement'}", flush=True)
    offloaded = [
        (name, str(device)) for name, device in device_map.items()
        if str(device).lower() in {"cpu", "disk"}
    ]
    if args.require_gpu_only and offloaded:
        judge.close()
        raise SystemExit(f"Judge has CPU/disk-offloaded modules: {offloaded[:10]}")
    if args.startup_delay_seconds > 0:
        print(
            f"Waiting {args.startup_delay_seconds:g}s before scoring to allow another BF16 replica to load.",
            flush=True,
        )
        time.sleep(args.startup_delay_seconds)
    started = time.monotonic()
    try:
        with output.open("a", encoding="utf-8") as handle:
            for row in generation:
                index = int(row["index"])
                if index in existing:
                    continue
                judgments = make_judgment(
                    row, judge, batch_size=args.judge_batch_size,
                    prompt_version=args.prompt_version,
                )
                first_success = next((item for item in judgments if item["success"]), None)
                result = {
                    "protocol": protocol,
                    "prompt_version": args.prompt_version,
                    "index": index,
                    "scenario_id": row["scenario_id"],
                    "source_id": row["source_id"],
                    "session_id": row["session_id"],
                    "method": method.slug,
                    "model": method.model_path,
                    "judge_model": args.judge_model,
                    "judge_batch_size": args.judge_batch_size,
                    "judge_device": args.judge_device,
                    "judge_compute_dtype": "bfloat16",
                    "judge_quantization": "none",
                    "generated_rounds": 5,
                    "candidate": str(row["trajectory"][0]["content"]).strip(),
                    "reference": str(row["reference"]).strip(),
                    "strict_success_by_round": judgments,
                    "strict_success_within_5_rounds": first_success is not None,
                    "first_success_round": first_success["round"] if first_success else None,
                    "first_success_evidence": first_success["evidence"] if first_success else "",
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                existing[index] = result
                if args.empty_cache_after_row:
                    for device_index in judge.devices:
                        with judge.torch.cuda.device(device_index):
                            judge.torch.cuda.empty_cache()
                if len(existing) % 25 == 0:
                    print(
                        f"{method.slug}: {len(existing)}/{len(generation)} rows, "
                        f"elapsed {time.monotonic() - started:.0f}s",
                        flush=True,
                    )
        ordered = [existing[index] for index in range(len(generation))]
        if len(existing) != len(generation):
            raise ValueError(f"Incomplete evaluation: {len(existing)}/{len(generation)}")
        summary = summarize(ordered, args.judge_model, args.prompt_version)
        summary["method"] = method.slug
        summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"Saved {output}", flush=True)
    finally:
        judge.close()


if __name__ == "__main__":
    main()
