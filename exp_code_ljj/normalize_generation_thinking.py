#!/usr/bin/env python3
"""Normalize completed generation JSONL files to public answer text."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import stat
from pathlib import Path


THINKING_MARKER = re.compile(r"<think\b|</think\b", re.IGNORECASE)


def clean_text(value: str) -> tuple[str, bool]:
    had_marker = bool(THINKING_MARKER.search(value))
    # A closing tag without an opening tag usually means the model generated
    # the continuation of a template-supplied thinking prefix. Its preceding
    # analysis must not become public text merely by deleting the tag.
    if had_marker and not re.search(r"<think\b", value, re.I):
        value = re.split(r"</think\b[^>]*>", value, flags=re.I)[-1]
    value = re.sub(r"<think\b[^>]*>.*?</think\b[^>]*>", "", value, flags=re.I | re.S)
    value = re.sub(r"<think\b[^>]*>.*\Z", "", value, flags=re.I | re.S)
    value = re.sub(r"</think\b[^>]*>", "", value, flags=re.I)
    value = value.strip()
    if had_marker:
        labels = list(re.finditer(
            r"(?im)^\s*(assistant|persuader|persuadee)\s*:?\s*", value
        ))
        segments = []
        for index, match in enumerate(labels):
            end = labels[index + 1].start() if index + 1 < len(labels) else len(value)
            segment = value[match.end():end].strip()
            if segment:
                segments.append(segment)
        if segments:
            value = segments[-1]
    return value, had_marker


def clean_nested(value):
    """Clean agent traces while preserving their JSON structure and types."""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, list):
        cleaned, filtered = [], False
        for item in value:
            item, found = clean_nested(item)
            cleaned.append(item)
            filtered = filtered or found
        return cleaned, filtered
    if isinstance(value, dict):
        cleaned, filtered = {}, False
        for key, item in value.items():
            item, found = clean_nested(item)
            cleaned[key] = item
            filtered = filtered or found
        return cleaned, filtered
    return value, False


def normalize_turn(turn: dict, *, index, require_content: bool) -> None:
    marker = False
    for key in ("raw_output", "content"):
        value = turn.get(key)
        if isinstance(value, str):
            value, found = clean_text(value)
            marker = marker or found
            if key == "content" and require_content and not value:
                raise ValueError(f"empty public content at index {index}")
            turn[key] = value
    if "agent_trace" in turn:
        turn["agent_trace"], found = clean_nested(turn["agent_trace"])
        marker = marker or found
    if marker:
        turn["thinking_filtered"] = True
    turn["thinking_policy"] = "final_answer_only"


def normalize_row(row: dict, *, device: str | None = None,
                  test_data_sha256: str | None = None) -> None:
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, list):
        return
    for turn in trajectory:
        if not isinstance(turn, dict):
            continue
        normalize_turn(turn, index=row.get("index"), require_content=True)
    history = row.get("final_history")
    if isinstance(history, list):
        initial = row.get("initial_history", [])
        for position, turn in enumerate(history):
            if isinstance(turn, dict):
                if position < len(initial):
                    source = initial[position]
                    if (turn.get("role") != source.get("role")
                            or turn.get("content", "").strip() != source.get("content", "").strip()):
                        raise ValueError(f"source history changed at index {row.get('index')}")
                    # Initial context is fixed source data, not a generated
                    # answer. Never trim or reinterpret its original text.
                    turn["content"] = source["content"]
                    continue
                normalize_turn(turn, index=row.get("index"), require_content=True)
    row["thinking_policy"] = "final_answer_only"
    # Generation files are consumed independently of run_config.json during
    # later judging. Keep the fixed full-BF16 policy and device provenance on
    # every normalized row as well.
    row["compute_dtype"] = "bfloat16"
    row["quantization"] = "none"
    row["enable_thinking"] = False
    if device is not None:
        row["persuader_device"] = device
        row["persuadee_device"] = device
    if test_data_sha256 is not None:
        row["test_data_sha256"] = test_data_sha256


def normalize(path: Path) -> tuple[int, int]:
    original_mode = stat.S_IMODE(path.stat().st_mode)
    family_device = {
        "meta_llama31": "cuda:0",
        "deepseek_r1_qwen14b": "cuda:1",
    }.get(path.parent.name)
    test_data_sha256 = None
    manifest = path.parents[3] / "data" / "manifest.json"
    if manifest.is_file():
        try:
            test_data_sha256 = json.loads(manifest.read_text(encoding="utf-8"))["files"]["evaluation_points"]["sha256"]
        except (KeyError, json.JSONDecodeError):
            test_data_sha256 = None
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            normalize_row(row, device=family_device, test_data_sha256=test_data_sha256)
            rows.append(row)
    indices = [row.get("index") for row in rows]
    if len(rows) == 1000 and set(indices) != set(range(1000)):
        raise ValueError(f"completed file has invalid indices: {path}")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.chmod(temporary, original_mode)
    os.replace(temporary, path)
    return len(rows), sum(
        1 for row in rows for turn in row.get("trajectory", [])
        if isinstance(turn, dict) and turn.get("thinking_filtered")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        count, filtered = normalize(path)
        print(f"normalized {path}: rows={count}, filtered_turns={filtered}", flush=True)


if __name__ == "__main__":
    main()
