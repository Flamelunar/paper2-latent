#!/usr/bin/env python3
"""Train baseline 2, Dialogue-SFT, on the fixed paper10k split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from repro.profile import add_profile_argument, configure_profile_from_argv
from repro.constants import ROOT
from repro.baseline_data import sha256
from repro.constants import data_path

ACTIVE_PROFILE = configure_profile_from_argv()
from repro.train_backend import (
    add_sft_continuation_arguments,
    add_train_arguments,
    spec_from_args,
    train_sft,
)

DEFAULT_MODEL = "/app/hzj/models/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_ROOT = ROOT / "models"
STAGE_NAME = "dialogue_sft_adapter"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_argument(parser)
    add_train_arguments(parser)
    add_sft_continuation_arguments(parser)
    parser.set_defaults(
        profile="paper10k",
        model=DEFAULT_MODEL,
        output_dir=DEFAULT_OUTPUT_ROOT,
        stage_name=STAGE_NAME,
        epochs=1.0,
        finetuning_type="lora",
        cutoff_len=2048,
        batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        save_steps=100,
    )
    args = parser.parse_args()
    if ACTIVE_PROFILE != "paper10k":
        raise SystemExit("Baseline 2 Dialogue-SFT uses the paper10k split")
    fixed_train = data_path("dialogue_sft")
    selected_train = args.train_file or fixed_train
    manifest = json.loads((fixed_train.parent / "manifest.json").read_text(encoding="utf-8"))
    if sha256(selected_train) != manifest["files"]["dialogue_sft"]["sha256"]:
        raise SystemExit("Baseline 2 requires the fixed paper10k Dialogue-SFT data")
    if not args.dry_run:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise SystemExit("BF16 SFT requires exactly one visible CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("The selected CUDA GPU does not support BF16")

    output = train_sft(spec_from_args(args), with_isr=False, dry_run=args.dry_run)
    print(output)


if __name__ == "__main__":
    main()
