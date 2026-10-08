#!/usr/bin/env python3
"""Main route 4: ISR-SFT, GPT-4o-mini user simulation, and DPO."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from repro.profile import add_profile_argument, configure_profile_from_argv

configure_profile_from_argv()
from repro.constants import OUTPUT_DIR, PREFERENCE_SCENARIOS
from repro.train_backend import (
    add_train_arguments,
    spec_from_args,
    train_dpo,
    train_user_model,
)
from repro.simulation import build_preferences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_argument(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    user_parser = subparsers.add_parser(
        "train-user", help="Optionally fine-tune a local Llama 3.1 persuadee model."
    )
    add_train_arguments(user_parser)

    sim_parser = subparsers.add_parser("simulate", help=f"Build {PREFERENCE_SCENARIOS:,} simulation preferences.")
    sim_parser.add_argument("--persuader-url", default="local://cuda:0")
    sim_parser.add_argument(
        "--persuader-model", default=str(OUTPUT_DIR / "02_llama31_isr_sft")
    )
    sim_parser.add_argument("--user-url", default="https://api.openai.com/v1")
    sim_parser.add_argument("--user-model", default="gpt-4o-mini")
    sim_parser.add_argument(
        "--user-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the GPT-4o-mini API key.",
    )
    sim_parser.add_argument("--judge-url", default="https://api.openai.com/v1")
    sim_parser.add_argument("--judge-model", default="gpt-4o-mini")
    sim_parser.add_argument(
        "--simulation-rounds",
        "--depth",
        dest="simulation_rounds",
        type=int,
        default=4,
        help=(
            "k feedback rounds; each includes a persuadee reply and later rounds "
            "also include a persuader follow-up. The paper does not disclose k."
        ),
    )
    sim_parser.add_argument("--workers", type=int, default=1)
    sim_parser.add_argument("--max-retries", type=int, default=6)
    sim_parser.add_argument("--limit", type=int, default=None)
    sim_parser.add_argument("--force", action="store_true")

    dpo_parser = subparsers.add_parser("dpo", help="Run one epoch of DPO.")
    add_train_arguments(dpo_parser)
    dpo_parser.add_argument("--isr-checkpoint", type=Path, required=True)
    dpo_parser.add_argument("--beta", type=float, default=0.1)

    args = parser.parse_args()
    if args.command == "train-user":
        print(train_user_model(spec_from_args(args), dry_run=args.dry_run))
    elif args.command == "simulate":
        print(
            build_preferences(
                persuader_url=args.persuader_url,
                persuader_model=args.persuader_model,
                user_url=args.user_url,
                user_model_name=args.user_model,
                user_api_key=os.getenv(args.user_api_key_env),
                judge_url=args.judge_url,
                judge_model=args.judge_model,
                simulation_rounds=args.simulation_rounds,
                workers=args.workers,
                max_retries=args.max_retries,
                limit=args.limit,
                force=args.force,
            )
        )
    else:
        print(
            train_dpo(
                spec_from_args(args),
                isr_checkpoint=args.isr_checkpoint,
                beta=args.beta,
                dry_run=args.dry_run,
            )
        )


if __name__ == "__main__":
    main()
