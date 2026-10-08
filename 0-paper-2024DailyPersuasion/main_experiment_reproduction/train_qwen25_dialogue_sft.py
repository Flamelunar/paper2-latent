#!/usr/bin/env python3
"""Run only the paper10k Dialogue-SFT stage on local Qwen2.5-7B-Instruct."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from repro.profile import add_profile_argument, configure_profile_from_argv

ACTIVE_PROFILE = configure_profile_from_argv()
from repro.train_backend import (
    add_sft_continuation_arguments,
    add_train_arguments,
    spec_from_args,
    train_sft,
)

DEFAULT_MODEL = "/data1/liujianjian/model/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT_ROOT = Path("/data1/liujianjian/Persuader")
STAGE_NAME = "qwen2.5-7b-instruct-dialogue-sft-paper10k"


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
        raise SystemExit("Qwen Dialogue-SFT is pinned to the paper10k split")
    if "qwen2.5" not in args.model.lower():
        raise SystemExit(f"Expected a local Qwen2.5 model, got: {args.model}")
    if not args.dry_run:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise SystemExit("Qwen BF16 SFT requires exactly one visible CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("The selected CUDA GPU does not support BF16")

    output = train_sft(spec_from_args(args), with_isr=False, dry_run=args.dry_run)
    print(output)


if __name__ == "__main__":
    main()
