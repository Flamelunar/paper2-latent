#!/usr/bin/env python3
"""Evaluate Baseline 3 with the exact Gemma 4 v11 protocol used in the table."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from repro.constants import ROOT
from repro.simulation import LocalTransformersEndpoint, read_jsonl
from evaluate_baselines import load_generation


CODE_DIR = ROOT / "artifacts/success_judgments/gemma4/code/current"
sys.path.insert(0, str(CODE_DIR))
import evaluate_joint_prompt_probe as table_evaluator  # noqa: E402


METHOD = "baseline3_isr_simdpo"
JUDGE_MODEL = "/app/hzj/models/gemma-4-E4B-it"
PROMPT_VERSION = "v11"
PROMPT_REVISION = "joint_v2_r11"
OUTPUT_PATH = ROOT / "artifacts/success_judgments/gemma4/baseline3_v11_compatible/baseline3_isr_simdpo.jsonl"
GENERATION_PATH = ROOT / "artifacts/generation_5rounds/baseline3_isr_simdpo.jsonl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-device", default="cuda:2")
    parser.add_argument("--generation", type=Path, default=GENERATION_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--max-parse-retries", type=int, default=3)
    args = parser.parse_args()

    if os.environ.get("ALLOW_JUDGE_EVAL") != "1":
        raise SystemExit("Set ALLOW_JUDGE_EVAL=1 to run the requested Gemma 4 evaluation.")
    if args.max_parse_retries < 0:
        parser.error("--max-parse-retries must be nonnegative")

    rows = load_generation(args.generation, expected_rows=1000)
    generation = {int(row["index"]): row for row in rows}
    if len(generation) != 1000:
        raise ValueError("Baseline 3 generation does not have 1,000 unique test indexes")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[int, dict[str, Any]] = {}
    if args.output.exists():
        for item in read_jsonl(args.output):
            index = int(item["index"])
            row = generation.get(index)
            if (
                row is None
                or item.get("method") != METHOD
                or item.get("judge_model") != JUDGE_MODEL
                or item.get("prompt_version") != PROMPT_VERSION
                or item.get("prompt_revision") != PROMPT_REVISION
                or item.get("judge_compute_dtype") != "bfloat16"
                or item.get("judge_quantization") != "none"
                or int(item.get("max_parse_retries", -1)) != args.max_parse_retries
                or item.get("scenario_id") != row.get("scenario_id")
            ):
                raise ValueError(f"Unexpected or mismatched saved evaluation at index {index}")
            if index in existing:
                raise ValueError(f"Duplicate saved evaluation at index {index}")
            existing[index] = item

    pending = [index for index in range(1000) if index not in existing]
    print(
        f"Gemma 4 {PROMPT_REVISION}: {len(existing)}/1000 saved, {len(pending)} pending",
        flush=True,
    )
    metadata_path = args.output.parent / "run_metadata.json"
    metadata: dict[str, Any] = {
        "method": METHOD,
        "judge_model": JUDGE_MODEL,
        "judge_compute_dtype": "bfloat16",
        "judge_quantization": "none",
        "judge_device": args.judge_device,
        "prompt_version": PROMPT_VERSION,
        "prompt_revision": PROMPT_REVISION,
        "max_tokens": args.max_tokens,
        "max_parse_retries": args.max_parse_retries,
        "generation_path": str(args.generation),
        "generation_sha256": sha256(args.generation),
        "evaluator_path": str(Path(table_evaluator.__file__).resolve()),
        "evaluator_sha256": sha256(Path(table_evaluator.__file__).resolve()),
        "output_path": str(args.output),
        "saved_rows": len(existing),
        "status": "running" if pending else "completed",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_metadata(metadata_path, metadata)
    if not pending:
        return

    judge = LocalTransformersEndpoint(
        model=JUDGE_MODEL,
        device=args.judge_device,
        quantization="none",
    )
    device_map = getattr(judge.model, "hf_device_map", {})
    offloaded = [
        (name, str(device))
        for name, device in device_map.items()
        if str(device).lower() in {"cpu", "disk"}
    ]
    if offloaded:
        judge.close()
        raise RuntimeError(f"Judge has CPU/disk offload: {offloaded[:5]}")

    unresolved_count = sum(bool(item.get("parse_error")) for item in existing.values())
    try:
        with args.output.open("a", encoding="utf-8") as handle:
            for number, index in enumerate(pending, 1):
                row = generation[index]
                prompt = table_evaluator.joint_prompt(row, PROMPT_VERSION)
                outputs: list[str] = []
                parse_errors: list[str] = []
                verdict = None
                error = None
                for attempt in range(args.max_parse_retries + 1):
                    retry_note = ""
                    if attempt:
                        retry_note = (
                            f"\n\nYour previous answer failed validation: {parse_errors[-1]}. "
                            f"Correction attempt {attempt}: return a new complete JSON answer. "
                            "Each win_ab and win_ba value must be exactly A or B. "
                            "If round_success is true, its round_evidence must be an exact quote "
                            "from that same numbered Persuadee reply. Answer both tasks again."
                        )
                    raw = judge.complete(
                        system="You are a careful evaluator of goal acceptance and persuasive quality.",
                        user=prompt + retry_note,
                        temperature=0.0,
                        max_tokens=args.max_tokens,
                        json_mode=True,
                    )
                    outputs.append(raw)
                    try:
                        verdict = table_evaluator.parse_joint(raw, require_evidence=True)
                        for round_index, (is_success, quote) in enumerate(
                            zip(verdict["round_success"], verdict["round_evidence"])
                        ):
                            reply = row["trajectory"][2 * round_index + 1]["content"]
                            if is_success and (
                                not quote.strip()
                                or quote.strip().casefold() not in reply.casefold()
                            ):
                                raise ValueError(
                                    f"Round {round_index + 1} positive evidence is not a verbatim persuadee quote"
                                )
                    except ValueError as exc:
                        verdict = None
                        error = str(exc)
                        parse_errors.append(error)
                    else:
                        error = None
                        break

                unresolved_count += error is not None
                result = {
                    "trial": 1,
                    "method": METHOD,
                    "index": index,
                    "stratum": "full_test",
                    "scenario_id": row["scenario_id"],
                    "goal": row["scenario"]["goal"],
                    "model": row.get("persuader_model", row.get("model", "")),
                    "prompt_version": PROMPT_VERSION,
                    "prompt_revision": PROMPT_REVISION,
                    "judge_model": JUDGE_MODEL,
                    "judge_device": args.judge_device,
                    "judge_compute_dtype": "bfloat16",
                    "judge_quantization": "none",
                    "model_calls": len(outputs),
                    "max_parse_retries": args.max_parse_retries,
                    "raw_output": outputs[-1],
                    "raw_outputs": outputs,
                    "retry_outputs": outputs[1:],
                    "parse_errors_by_attempt": parse_errors,
                    "parse_error": error,
                    "verdict": verdict,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                existing[index] = result
                if number % 5 == 0 or number == len(pending):
                    print(
                        f"Gemma 4 {PROMPT_REVISION}: {len(existing)}/1000 saved, "
                        f"{unresolved_count} unresolved",
                        flush=True,
                    )
                    metadata["saved_rows"] = len(existing)
                    metadata["unresolved_rows"] = unresolved_count
                    metadata["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                    write_metadata(metadata_path, metadata)
    finally:
        judge.close()

    metadata["saved_rows"] = len(existing)
    metadata["unresolved_rows"] = unresolved_count
    metadata["status"] = "completed" if len(existing) == 1000 else "incomplete"
    metadata["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_metadata(metadata_path, metadata)


if __name__ == "__main__":
    main()
