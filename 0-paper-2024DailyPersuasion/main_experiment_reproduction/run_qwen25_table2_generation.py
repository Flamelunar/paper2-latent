#!/usr/bin/env python3
"""Generate five-round Qwen2.5 dialogues for the fixed paper10k test set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from repro.profile import add_profile_argument, configure_profile_from_argv

PROFILE = configure_profile_from_argv()
from repro.constants import ROOT
from repro.persuadee_simulation import PersuaderMethod, run_simulation

BASE_MODEL = "/data1/liujianjian/model/Qwen2.5-7B-Instruct"
SFT_MODEL = "/data1/liujianjian/Persuader/qwen2.5-7b-instruct-dialogue-sft-paper10k"
PERSUADEE_MODEL = "/data1/liujianjian/Persuadee/Qwen3.5-9B-persuadee/merged"
TEST_DATA = ROOT / "runs/paper10k/data/evaluation_points.jsonl"
OUTPUT_DIR = ROOT / "runs/qwen25_dialogue_sft/generation_5rounds"
PROTOCOL = "qwen25_table2_fixed_qwen35_persuadee_5rounds_bf16_contextbatch2_v1"
METHODS = (
    PersuaderMethod(
        slug="qwen25_zero_shot",
        display_name="Qwen2.5-7B-Instruct zero-shot",
        model_path=BASE_MODEL,
        with_isr=False,
    ),
    PersuaderMethod(
        slug="qwen25_dialogue_sft",
        display_name="Qwen2.5-7B-Instruct Dialogue-SFT",
        model_path=SFT_MODEL,
        with_isr=False,
    ),
)


def model_weight_bytes(path: Path) -> int:
    adapter_config = path / "adapter_config.json"
    base = path
    if adapter_config.is_file():
        base = Path(json.loads(adapter_config.read_text(encoding="utf-8"))["base_model_name_or_path"])
    indexes = list(base.glob("*.safetensors.index.json"))
    if indexes:
        return int(json.loads(indexes[0].read_text(encoding="utf-8"))["metadata"]["total_size"])
    return sum(item.stat().st_size for item in base.glob("*.safetensors"))


def check_bf16_capacity(model_path: str, device_spec: str) -> None:
    import torch
    from repro.simulation import LocalTransformersEndpoint

    device_ids = LocalTransformersEndpoint._parse_devices(device_spec)
    for device_id in device_ids:
        with torch.cuda.device(device_id):
            if not torch.cuda.is_bf16_supported():
                raise SystemExit(f"cuda:{device_id} does not support BF16")
    free = sum(torch.cuda.mem_get_info(device_id)[0] for device_id in device_ids)
    weight_bytes = model_weight_bytes(Path(model_path))
    reserve = 2 * 1024**3
    print(
        f"BF16 preflight {model_path}: weights={weight_bytes / 1024**3:.1f} GiB, "
        f"free on {device_spec}={free / 1024**3:.1f} GiB"
    )
    if free < weight_bytes + reserve:
        raise SystemExit("Insufficient free GPU memory for BF16 inference; wait for capacity instead of reducing precision")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_argument(parser)
    parser.add_argument("--test-data", type=Path, default=TEST_DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--persuader-device", default="cuda:1")
    parser.add_argument("--persuadee-device", default="cuda:0,cuda:2")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--context-batch-size", type=int, default=2)
    parser.add_argument("--method", choices=[method.slug for method in METHODS], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if PROFILE != "paper10k":
        raise SystemExit("Qwen Table 2 evaluation requires the paper10k held-out split")
    if args.rounds != 5:
        raise SystemExit("This reproduction protocol fixes five Persuader-Persuadee rounds")
    if args.context_batch_size < 1 or args.context_batch_size > 4:
        raise SystemExit("--context-batch-size must be between 1 and 4")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    selected_methods = tuple(
        method for method in METHODS if args.method is None or method.slug == args.method
    )
    for method in selected_methods:
        if not Path(method.model_path).is_dir():
            raise SystemExit(f"Missing Persuader model: {method.model_path}")
    if not Path(PERSUADEE_MODEL).is_dir():
        raise SystemExit(f"Missing Persuadee model: {PERSUADEE_MODEL}")

    check_bf16_capacity(BASE_MODEL, args.persuader_device)
    check_bf16_capacity(PERSUADEE_MODEL, args.persuadee_device)
    paths = run_simulation(
        methods=selected_methods,
        test_data=args.test_data,
        output_dir=args.output_dir,
        persuadee_model=PERSUADEE_MODEL,
        persuader_device=args.persuader_device,
        persuadee_device=args.persuadee_device,
        persuader_quantization="none",
        rounds=args.rounds,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        force=args.force,
        top_p=args.top_p,
        top_k=args.top_k,
        context_batch_size=args.context_batch_size,
        protocol=PROTOCOL,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
