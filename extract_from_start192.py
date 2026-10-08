#!/usr/bin/env python3
"""Extract the fixed 192 empty-history test rows from existing generations."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "exp_code_ljj"
SELECTION_PATH = EXP / "data/test_selection_192.json"
SOURCE_PATH = EXP / "data/evaluation_points.jsonl"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract(source: Path, target: Path, family: str, method: str):
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    points = read(SOURCE_PATH)
    selected = selection["original_indexes"]
    if (selection.get("count") != 192 or len(selected) != 192 or len(set(selected)) != 192
            or selection.get("source_sha256") != sha(SOURCE_PATH)
            or selected != [i for i, p in enumerate(points) if p.get("target_turn") == 1 and p.get("history") == []]):
        raise ValueError("The fixed 192-scenario manifest or source data changed")
    source_rows = read(source)
    by_index = {}
    for row in source_rows:
        index = row.get("index")
        if type(index) is not int or index in by_index:
            raise ValueError(f"Invalid or duplicate generation index in {source}")
        if index not in range(len(points)):
            raise ValueError(f"Generation index outside the original test split: {index}")
        if row.get("scenario_id") != points[index].get("scenario_id"):
            raise ValueError(f"Scenario ID mismatch at original index {index}")
        by_index[index] = row
    picked = [by_index[i] for i in selected if i in by_index]
    if not picked:
        return {"source": str(source), "extracted": 0, "expected": 192}
    if any(row.get("method") != method for row in picked):
        raise ValueError("Mixed method rows in generation source")
    target.parent.mkdir(parents=True, exist_ok=True)
    original_sha = sha(source)
    metadata = {"revision": selection["revision"], "family": family, "method": method,
                "source_path": str(source), "source_sha256": original_sha,
                "selection_sha256": sha(SELECTION_PATH), "source_row_count": len(source_rows),
                "selected_indexes": selected, "extracted_indexes": [r["index"] for r in picked],
                "expected_count": 192, "extracted_count": len(picked),
                "complete": len(picked) == 192}
    meta_path = target.with_name(target.stem + ".extraction_manifest.json")
    if target.exists() or meta_path.exists():
        if not (target.exists() and meta_path.exists()):
            raise ValueError(f"Partial extraction artifact exists; refusing overwrite: {target}")
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old != metadata or read(target) != picked:
            raise ValueError(f"Existing extracted artifact differs: {target}")
        return metadata
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as f:
        tmp = Path(f.name)
        for row in picked:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, target)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as f:
        tmp_meta = Path(f.name)
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_meta, meta_path)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("meta_llama31", "deepseek_r1_qwen14b"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    args = parser.parse_args()
    target = args.target or ROOT / "exp_code_ljj/artifacts/generation_from_start192" / args.family / f"{args.method}.jsonl"
    print(json.dumps(extract(args.source, target, args.family, args.method), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
