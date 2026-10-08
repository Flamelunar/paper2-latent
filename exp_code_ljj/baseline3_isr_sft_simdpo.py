#!/usr/bin/env python3
"""Baseline 3: paper10k ISR-SFT, simulated preferences, then DPO.

Run `train-isr`, `simulate`, and `dpo` in order. Simulation reuses the
already fine-tuned local persuadee model; it does not train a user model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from repro.profile import add_profile_argument, configure_profile_from_argv

ACTIVE_PROFILE = configure_profile_from_argv()
from repro.baseline_data import sha256, validate_fixed_splits
from repro.preference_optimization import build_preferences
from repro.train_backend import (
    add_sft_continuation_arguments, add_train_arguments,
    spec_from_args, train_dpo, train_sft,
)

PERSUADER_BASE_MODEL = "/app/hzj/models/Qwen2.5-7B-Instruct"
PERSUADER_OUTPUT_DIR = Path("/data1/liujianjian/Persuader")
ISR_CHECKPOINT = PERSUADER_OUTPUT_DIR / "qwen2.5-7B-Instruct-ISR-sft"
SIMDPO_CHECKPOINT = PERSUADER_OUTPUT_DIR / "qwen2.5-7B-Instruct-ISR-SFT-SimDPO"
LOCAL_PERSUADEE = "/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged"
LOCAL_JUDGE = "/app/hzj/models/Qwen3.5-9B"


def training_parser(subparsers: argparse._SubParsersAction, name: str, help_text: str):
    parser = subparsers.add_parser(name, help=help_text)
    add_train_arguments(parser)
    add_sft_continuation_arguments(parser)
    parser.set_defaults(model=PERSUADER_BASE_MODEL, output_dir=PERSUADER_OUTPUT_DIR,
                        epochs=1.0, finetuning_type="lora", cutoff_len=2048,
                        batch_size=1, gradient_accumulation_steps=16,
                        learning_rate=1e-4, save_steps=100)
    return parser


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_argument(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    training_parser(subparsers, "train-isr", "Supervise reasoning, strategy, and utterance on the fixed SFT split.")

    sim = subparsers.add_parser("simulate", help="Generate ranked responses from the disjoint preference split.")
    sim.add_argument("--persuader-url", default="local://cuda:2")
    sim.add_argument("--persuader-model", default=str(ISR_CHECKPOINT))
    sim.add_argument("--user-url", default="local://cuda:2")
    sim.add_argument("--user-model", default=LOCAL_PERSUADEE,
                     help="Existing fine-tuned local persuadee model; this command does not train it.")
    sim.add_argument("--judge-url", default="local://cuda:2")
    sim.add_argument("--judge-model", default=LOCAL_JUDGE)
    sim.add_argument("--simulation-rounds", type=int, default=4)
    sim.add_argument("--workers", type=int, default=1)
    sim.add_argument("--simulation-batch-size", type=int, default=8,
                     help="Number of preference scenarios decoded together per local generation step.")
    sim.add_argument("--max-retries", type=int, default=6)
    sim.add_argument("--limit", type=int)
    sim.add_argument("--force", action="store_true")
    sim.add_argument("--preference-directory", type=Path)

    dpo = subparsers.add_parser("dpo", help="Optimize the ISR checkpoint using ranked simulated preferences.")
    add_train_arguments(dpo)
    dpo.add_argument("--isr-checkpoint", type=Path, default=ISR_CHECKPOINT)
    dpo.add_argument("--resume-from-checkpoint", type=Path)
    dpo.add_argument("--train-file", type=Path)
    dpo.set_defaults(stage_name=SIMDPO_CHECKPOINT.name)
    dpo.add_argument("--beta", type=float, default=0.1)
    dpo.set_defaults(model=PERSUADER_BASE_MODEL, output_dir=PERSUADER_OUTPUT_DIR,
                     epochs=1.0, finetuning_type="lora", cutoff_len=2048,
                     batch_size=1, gradient_accumulation_steps=16,
                     learning_rate=5e-5, save_steps=100)

    args = parser.parse_args()
    if ACTIVE_PROFILE != "paper10k":
        raise SystemExit("All baselines use the fixed paper10k split")
    summary = validate_fixed_splits()
    print(f"Fixed data: {summary['sft_sessions']} SFT sessions, "
          f"{summary['preference_scenarios']} preference scenarios, "
          f"{summary['test_scenarios']} held-out contexts")

    if args.command == "train-isr":
        if args.train_file and sha256(args.train_file) != summary["sha256"]["isr_sft"]:
            raise SystemExit("Baseline 3 ISR-SFT requires the fixed paper10k ISR data")
        print(train_sft(spec_from_args(args), with_isr=True, dry_run=args.dry_run))
    elif args.command == "simulate":
        print(build_preferences(
            persuader_url=args.persuader_url,
            persuader_model=args.persuader_model,
            user_url=args.user_url,
            user_model_name=args.user_model,
            user_api_key=None,
            judge_url=args.judge_url,
            judge_model=args.judge_model,
            simulation_rounds=args.simulation_rounds,
            workers=args.workers,
            simulation_batch_size=args.simulation_batch_size,
            max_retries=args.max_retries,
            limit=args.limit,
            force=args.force,
            preference_directory=args.preference_directory,
        ))
    else:
        print(train_dpo(spec_from_args(args), isr_checkpoint=args.isr_checkpoint,
                        beta=args.beta, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
