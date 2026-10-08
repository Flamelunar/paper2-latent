#!/usr/bin/env python3
"""Audit the complete ISR run and record its data and BF16 A100 provenance."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import torch

from persuasion_data import ISR_FILE, TEST_FILE, iter_training_rows

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path("/app/hzj/liujianjian/qwen2.5-7b-latentRATT-all-59868")
BASE_MODEL = "/app/hzj/models/Qwen2.5-7B-Instruct"


def atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp",
                                     delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    manifest_path = ROOT / "data/isr_all_59868.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_sha = hashlib.sha256(ISR_FILE.read_bytes()).hexdigest()
    test_sha = hashlib.sha256(TEST_FILE.read_bytes()).hexdigest()
    rows = list(iter_training_rows(ISR_FILE))
    train_scenarios = {str(private["scenario_id"]) for _, _, private in rows}
    test_rows = [json.loads(line) for line in TEST_FILE.open(encoding="utf-8") if line.strip()]
    test_scenarios = {str(row["scenario_id"]) for row in test_rows}
    overlap = train_scenarios & test_scenarios

    if (len(rows) != 59_868 or len(train_scenarios) != 10_000 or len(test_rows) != 1_000
            or len(test_scenarios) != 1_000 or overlap
            or source_sha != manifest.get("source_sha256")
            or test_sha != manifest.get("fixed_split_summary", {}).get("sha256", {}).get("evaluation_points")
            or manifest.get("training_rows") != 59_868
            or manifest.get("training_scenarios") != 10_000
            or manifest.get("training_test_overlap") != 0
            or manifest.get("preference_scenarios_used") != 0
            or manifest.get("dpo_used") is not False):
        raise ValueError("The full ISR manifest, source file, or fixed split failed validation")

    device = torch.device("cuda:2")
    if not torch.cuda.is_available():
        raise RuntimeError("The run audit requires CUDA")
    with torch.cuda.device(device):
        gpu_name = torch.cuda.get_device_name(device)
        if not torch.cuda.is_bf16_supported() or "A100" not in gpu_name:
            raise RuntimeError("The run audit requires BF16 support on an A100 at cuda:2")

    expected_steps = {"sft": math.ceil(len(rows) / 16), "rl": math.ceil((len(rows) - 1) / 8)}
    stage_records = {}
    for stage in ("sft", "rl"):
        config_path = MODEL_ROOT / stage / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if (config.get("stage") != stage or config.get("model") != BASE_MODEL
                or config.get("device") != "cuda:2"
                or config.get("train_rows") != len(rows)
                or config.get("train_scenarios") != len(train_scenarios)
                or config.get("train_sha256") != source_sha
                or config.get("preference_scenarios_used") != 0
                or config.get("completed_steps") != expected_steps[stage]):
            raise ValueError(f"The {stage} run configuration is incomplete or inconsistent")
        if not (MODEL_ROOT / stage / "adapter").is_dir() or not (MODEL_ROOT / stage / "attention.pt").is_file():
            raise FileNotFoundError(f"Missing final {stage} model weights")
        config["compute_dtype"] = "bfloat16"
        config["quantization"] = "none"
        config["gpu_name"] = gpu_name
        atomic_json(config_path, config)
        stage_records[stage] = {
            "completed_steps": config["completed_steps"],
            "train_rows": config["train_rows"],
            "train_scenarios": config["train_scenarios"],
            "compute_dtype": config["compute_dtype"],
            "quantization": config["quantization"],
            "device": config["device"],
            "gpu_name": gpu_name,
        }

    audit = {
        "source_file": str(ISR_FILE),
        "source_sha256": source_sha,
        "training_rows": len(rows),
        "training_scenarios": len(train_scenarios),
        "test_file": str(TEST_FILE),
        "test_sha256": test_sha,
        "test_rows": len(test_rows),
        "training_test_overlap": len(overlap),
        "preference_scenarios_used": 0,
        "dpo_used": False,
        "stages": stage_records,
    }
    atomic_json(MODEL_ROOT / "training_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
