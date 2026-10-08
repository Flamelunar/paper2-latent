#!/usr/bin/env python3
"""Generate and evaluate the paper's zero-shot control and three training routes."""

from __future__ import annotations

import argparse

from repro.profile import add_profile_argument, configure_profile_from_argv

configure_profile_from_argv()
from repro.constants import DEFAULT_MODEL, OUTPUT_DIR
from repro.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_argument(parser)
    parser.add_argument("--zero-shot-url", default="local://cuda:0")
    parser.add_argument("--zero-shot-model", default=DEFAULT_MODEL)
    parser.add_argument("--dialogue-url", default="local://cuda:0")
    parser.add_argument(
        "--dialogue-model", default=str(OUTPUT_DIR / "01_llama31_dialogue_sft")
    )
    parser.add_argument("--isr-url", default="local://cuda:0")
    parser.add_argument("--isr-model", default=str(OUTPUT_DIR / "02_llama31_isr_sft"))
    parser.add_argument("--ours-url", default="local://cuda:0")
    parser.add_argument("--ours-model", default=str(OUTPUT_DIR / "03_llama31_isr_sim_dpo"))
    parser.add_argument("--judge-url", default="https://api.openai.com/v1")
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="OpenAI judge used for this Llama-3.1 adaptation; paper says ChatGPT without a dated snapshot.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--export-merged",
        action="store_true",
        help="Also merge each available LoRA adapter into standalone model weights before generation.",
    )
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--only-method",
        choices=("zero_shot", "dialogue_sft", "isr_sft", "ours"),
        default=None,
        help="Generate/evaluate one method so a single GPU endpoint can be reused.",
    )
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help="Skip model generation and judge all four existing generation files.",
    )
    args = parser.parse_args()
    if args.generate_only and args.judge_only:
        parser.error("--generate-only and --judge-only cannot be used together")
    output = run_evaluation(
        endpoint_specs={
            "zero_shot": (args.zero_shot_url, args.zero_shot_model),
            "dialogue_sft": (args.dialogue_url, args.dialogue_model),
            "isr_sft": (args.isr_url, args.isr_model),
            "ours": (args.ours_url, args.ours_model),
        },
        judge_url=args.judge_url,
        judge_model=args.judge_model,
        workers=args.workers,
        max_retries=args.max_retries,
        force=args.force,
        generate_only=args.generate_only,
        only_method=args.only_method,
        judge_only=args.judge_only,
        export_merged=args.export_merged,
    )
    if output is not None:
        print(output)


if __name__ == "__main__":
    main()
