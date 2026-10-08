#!/usr/bin/env python3
"""Run a resumable, paired v1/v2 success-prompt probe on selected trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from repro.profile import configure_profile_from_argv

configure_profile_from_argv()
from repro.constants import ROOT
from repro.simulation import LocalTransformersEndpoint, read_jsonl
from evaluate_baselines import ARCHIVED_METHODS, load_generation
from evaluate_success_only import PROTOCOLS, make_judgment


DEFAULT_OUTPUT_DIR = ROOT / "artifacts/success_prompt_v2_diagnostic"
METHOD_SLUGS = {method.slug for method in ARCHIVED_METHODS}


def read_manifest(path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    for trial in document["trials"]:
        for sample in trial["samples"]:
            method = str(sample["method"])
            if method not in METHOD_SLUGS:
                raise ValueError(f"Unknown method in {path}: {method}")
            samples.append({
                "trial": int(trial["trial"]),
                "method": method,
                "index": int(sample["index"]),
                "stratum": str(sample["stratum"]),
            })
    if len({(s["trial"], s["method"], s["index"]) for s in samples}) != len(samples):
        raise ValueError(f"Duplicate sample within one trial in {path}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "sample_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-name", choices=("llama31", "qwen35", "persugpt"), required=True)
    parser.add_argument("--judge-device", required=True)
    parser.add_argument("--batch-size", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--max-evaluations", type=int, default=0)
    args = parser.parse_args()

    samples = read_manifest(args.manifest)
    generations: dict[str, dict[int, dict[str, Any]]] = {}
    for method in {sample["method"] for sample in samples}:
        rows = load_generation(
            ROOT / "artifacts/generation_5rounds/legacy" / f"{method}.jsonl",
            expected_rows=1000,
        )
        generations[method] = {int(row["index"]): row for row in rows}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.judge_name}.jsonl"
    existing: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    if output.exists():
        for saved in read_jsonl(output):
            key = (int(saved["trial"]), str(saved["method"]), int(saved["index"]), str(saved["prompt_version"]))
            if key in existing or saved.get("judge_model") != args.judge_model:
                raise ValueError(f"Duplicate key or wrong Judge in {output}: {key}")
            if saved.get("protocol") != PROTOCOLS[key[3]]:
                raise ValueError(f"Prompt protocol mismatch in {output}: {key}")
            existing[key] = saved

    pending = [
        (sample, version)
        for sample in samples for version in ("v1", "v2")
        if (sample["trial"], sample["method"], sample["index"], version) not in existing
    ]
    if args.max_evaluations > 0:
        pending = pending[:args.max_evaluations]
    print(f"{args.judge_name}: {len(existing)} saved, {len(pending)} pending evaluations", flush=True)
    if not pending:
        return

    judge = LocalTransformersEndpoint(
        model=args.judge_model, device=args.judge_device, quantization="none"
    )
    device_map = getattr(judge.model, "hf_device_map", {})
    offloaded = {name: str(device) for name, device in device_map.items()
                 if str(device).lower() in {"cpu", "disk"}}
    if offloaded:
        judge.close()
        raise RuntimeError(f"Judge contains CPU/disk offload: {list(offloaded.items())[:5]}")
    print(f"Judge loaded on {args.judge_device}; BF16, no quantization", flush=True)
    try:
        with output.open("a", encoding="utf-8") as handle:
            for number, (sample, version) in enumerate(pending, 1):
                row = generations[sample["method"]][sample["index"]]
                judgments = make_judgment(
                    row, judge, batch_size=args.batch_size, prompt_version=version
                )
                first = next((item["round"] for item in judgments if item["success"]), None)
                result = {
                    **sample,
                    "prompt_version": version,
                    "protocol": PROTOCOLS[version],
                    "judge_model": args.judge_model,
                    "judge_device": args.judge_device,
                    "judge_compute_dtype": "bfloat16",
                    "judge_quantization": "none",
                    "scenario_id": row["scenario_id"],
                    "goal": row["scenario"]["goal"],
                    "persuadee_responses": [turn["content"] for turn in row["trajectory"] if turn["role"] == "persuadee"],
                    "strict_success_by_round": judgments,
                    "first_success_round": first,
                    "strict_success_within_5_rounds": first is not None,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                if number % 10 == 0 or number == len(pending):
                    print(f"{args.judge_name}: {number}/{len(pending)} new evaluations", flush=True)
    finally:
        judge.close()
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
