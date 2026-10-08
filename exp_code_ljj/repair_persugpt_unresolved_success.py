#!/usr/bin/env python3
"""Rejudge only PersuGPT strict-success outputs that could not be parsed."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluate_baselines import (
    JUDGE_MODEL,
    parse_success,
    strict_success_prompt,
)
from repro.simulation import LocalTransformersEndpoint, read_jsonl


ROOT = Path(__file__).resolve().parent
GENERATION_DIR = ROOT / "artifacts/generation_5rounds/legacy"
SUCCESS_DIR = ROOT / "artifacts/success_judgments/persugpt"
LOG_DIR = ROOT / "logs"
CHECKPOINT_PATH = LOG_DIR / "persugpt_parse_repair_checkpoint.json"
METHODS = ("qwen25_zero_shot", "qwen25_dialogue_sft")


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def target_key(target: dict[str, Any]) -> str:
    return f"{target['method']}:{target['index']}:{target['round']}"


def save_checkpoint(path: Path, states: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".persugpt_parse_repair.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(states, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_targets(method: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    generations = read_jsonl(GENERATION_DIR / f"{method}.jsonl")
    path = SUCCESS_DIR / f"{method}.jsonl"
    records = read_jsonl(path)
    if len(generations) != 1000 or len(records) != 1000:
        raise ValueError(f"Expected 1,000 rows for {method}; generation={len(generations)}, judge={len(records)}")
    generation_by_index = {int(row["index"]): row for row in generations}
    record_by_index = {int(row["index"]): row for row in records}
    if set(generation_by_index) != set(range(1000)) or set(record_by_index) != set(range(1000)):
        raise ValueError(f"Missing or duplicate indexes for {method}")

    targets: list[dict[str, Any]] = []
    for record in records:
        index = int(record["index"])
        generation = generation_by_index[index]
        if record.get("method") != method or record.get("judge_model") != JUDGE_MODEL:
            raise ValueError(f"Unexpected method or Judge at {method} index {index}")
        if record.get("scenario_id") != generation.get("scenario_id"):
            raise ValueError(f"Generation/Judge scenario mismatch at {method} index {index}")
        judgments = record.get("strict_success_by_round")
        if not isinstance(judgments, list) or [item.get("round") for item in judgments] != [1, 2, 3, 4, 5]:
            raise ValueError(f"Incomplete five-round judgment at {method} index {index}")
        unresolved = [item for item in judgments if item.get("parse_unresolved")]
        if len(unresolved) > 1:
            raise ValueError(f"Expected at most one unresolved round at {method} index {index}")
        if unresolved:
            targets.append({
                "method": method,
                "index": index,
                "record": record,
                "judgment": unresolved[0],
                "generation": generation,
                "round": int(unresolved[0]["round"]),
            })
    return generations, records, targets


def retry_targets(
    targets: list[dict[str, Any]],
    endpoint: LocalTransformersEndpoint,
    *,
    batch_size: int,
    checkpoint_path: Path,
) -> dict[str, dict[str, int]]:
    checkpoint: dict[str, dict[str, Any]] = (
        json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    )
    retryable = list(targets)
    for target in targets:
        request = {
            "system": "You are a strict, evidence-based judge of explicit goal acceptance.",
            "user": strict_success_prompt(
                target["generation"],
                target["generation"]["trajectory"][: target["round"] * 2],
                target["round"],
            ),
            "temperature": 0.0,
            "max_tokens": 128,
            "json_mode": True,
        }
        target["request"] = request
        target["state"] = checkpoint.get(target_key(target))

    initial_targets = [target for target in retryable if target.get("state") is None]
    initial_batches = chunked(initial_targets, batch_size)
    for batch_index, batch in enumerate(initial_batches, 1):
        outputs = endpoint.complete_batch([target["request"] for target in batch])
        if len(outputs) != len(batch):
            raise RuntimeError("Judge returned a different number of initial repair outputs than requests")
        for target, raw in zip(batch, outputs):
            try:
                success, evidence = parse_success(raw)
            except ValueError:
                state = {
                    "status": "pending",
                    "raw_output": raw,
                    "retry_outputs": [],
                    "success": False,
                    "evidence": "",
                }
            else:
                state = {
                    "status": "resolved",
                    "raw_output": raw,
                    "retry_outputs": [],
                    "success": success,
                    "evidence": evidence,
                }
            target["state"] = state
            checkpoint[target_key(target)] = state
        save_checkpoint(checkpoint_path, checkpoint)
        print(f"Initial repair batch {batch_index}/{len(initial_batches)} complete", flush=True)

    for retry_number in range(2):
        pending = [
            target for target in retryable
            if target["state"]["status"] == "pending"
            and len(target["state"]["retry_outputs"]) == retry_number
        ]
        if not pending:
            break
        requests = [
            {
                "system": "Judge explicit acceptance of the stated goal.",
                "user": target["request"]["user"] + "\nAnswer with only TRUE or FALSE.",
                "temperature": 0.0,
                "max_tokens": 64,
            }
            for target in pending
        ]
        outputs: list[str] = []
        retry_batches = chunked(list(zip(pending, requests)), batch_size)
        for batch_index, batch in enumerate(retry_batches, 1):
            outputs.extend(endpoint.complete_batch([request for _, request in batch]))
            print(
                f"Clarification round {retry_number + 1}, batch {batch_index}/{len(retry_batches)} complete",
                flush=True,
            )
        if len(outputs) != len(pending):
            raise RuntimeError("Judge returned a different number of clarification outputs than requests")
        for target, raw in zip(pending, outputs):
            state = target["state"]
            state["retry_outputs"].append(raw)
            try:
                success, evidence = parse_success(raw)
            except ValueError:
                if len(state["retry_outputs"]) >= 2:
                    state["status"] = "unresolved"
                    state["success"] = False
                    state["evidence"] = "No parseable success verdict after two clarification prompts."
            else:
                state.update(status="resolved", success=success, evidence=evidence)
            checkpoint[target_key(target)] = state
        save_checkpoint(checkpoint_path, checkpoint)

    for target in retryable:
        judgment = target["judgment"]
        state = target["state"]
        if state["status"] not in {"resolved", "unresolved"}:
            raise RuntimeError(f"Repair remains incomplete for {target_key(target)}")
        judgment.update(
            raw_output=state["raw_output"],
            retry_outputs=state["retry_outputs"],
            success=state["success"],
            evidence=state["evidence"],
            parse_unresolved=state["status"] == "unresolved",
        )
        judgment.pop("request", None)

    summaries: dict[str, dict[str, int]] = {}
    for method in METHODS:
        selected = [target for target in retryable if target["method"] == method]
        summaries[method] = {
            "rejudged_rounds": len(selected),
            "still_unresolved": sum(target["state"]["status"] == "unresolved" for target in selected),
        }
    return summaries


def refresh_row_summary(record: dict[str, Any]) -> None:
    judgments = record["strict_success_by_round"]
    first = next((item for item in judgments if item.get("success")), None)
    record["strict_success_within_5_rounds"] = first is not None
    record["first_success_round"] = first["round"] if first else None
    record["first_success_evidence"] = first["evidence"] if first else ""


def atomic_replace_jsonl(path: Path, rows: list[dict[str, Any]], backup_stamp: str) -> Path:
    backup = path.with_name(f"{path.name}.before_parse_repair_{backup_stamp}.bak")
    shutil.copy2(path, backup)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.judge_model != JUDGE_MODEL:
        raise SystemExit(f"This repair targets {JUDGE_MODEL}")
    if args.batch_size < 1 or args.batch_size > 4:
        raise SystemExit("--batch-size must be between 1 and 4")

    loaded: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {
        method: load_targets(method) for method in METHODS
    }
    targets = [target for _, _, group in loaded.values() for target in group]
    counts = {method: len(loaded[method][2]) for method in METHODS}
    expected = {"qwen25_zero_shot": 17, "qwen25_dialogue_sft": 21}
    if counts != expected:
        raise SystemExit(f"Unexpected unresolved counts; expected {expected}, found {counts}; files were not changed")
    if not targets:
        print("No unresolved PersuGPT judgments found; nothing to do.")
        return

    import torch

    device_id = int(args.device.split(":", 1)[1])
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
    weight_bytes = sum(path.stat().st_size for path in Path(args.judge_model).glob("*.safetensors"))
    reserve = 4 * 1024**3
    print(
        f"PersuGPT repair preflight: {counts}; device={args.device}; "
        f"weights={weight_bytes / 1024**3:.1f} GiB; "
        f"free={free_bytes / 1024**3:.1f}/{total_bytes / 1024**3:.1f} GiB; "
        f"batch_size={args.batch_size}",
        flush=True,
    )
    if free_bytes < weight_bytes + reserve:
        raise SystemExit("Not enough free GPU memory for the BF16 Judge; files were not changed")

    endpoint = LocalTransformersEndpoint(model=args.judge_model, device=args.device)
    try:
        summaries = retry_targets(
            targets,
            endpoint,
            batch_size=args.batch_size,
            checkpoint_path=CHECKPOINT_PATH,
        )
    finally:
        endpoint.close()

    for target in targets:
        refresh_row_summary(target["record"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backups: dict[str, str] = {}
    for method in METHODS:
        path = SUCCESS_DIR / f"{method}.jsonl"
        records = loaded[method][1]
        backups[method] = str(atomic_replace_jsonl(path, records, stamp))

    report = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "judge_model": args.judge_model,
        "device": args.device,
        "batch_size": args.batch_size,
        "repair_scope": "only strict_success_by_round items previously marked parse_unresolved",
        "methods": summaries,
        "backups": backups,
        "rejudged": [
            {"method": target["method"], "index": target["index"], "round": target["round"]}
            for target in targets
        ],
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"persugpt_parse_repair_{stamp}.json"
    log_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    CHECKPOINT_PATH.unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"Repair log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
