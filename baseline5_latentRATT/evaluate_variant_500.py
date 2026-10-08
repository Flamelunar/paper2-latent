#!/usr/bin/env python3
"""Run the shared joint success/Win-Rate evaluator on a variant's first 500 cases."""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1] / "exp_code_ljj"
sys.path.insert(0, str(EXP))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-slug", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-name", choices=("selene1", "gemma4"), required=True)
    parser.add_argument("--judge-device", default="cuda:2")
    parser.add_argument("--prompt-version", choices=("selene_v17", "gemma_v11"), required=True)
    args, remainder = parser.parse_known_args()
    if args.judge_name == "selene1" and args.prompt_version != "selene_v17":
        parser.error("Selene evaluation uses the fixed V17 prompt")
    if args.judge_name == "gemma4" and args.prompt_version != "gemma_v11":
        parser.error("Gemma evaluation uses the fixed V11 prompt")

    import evaluate_baselines
    from repro.persuadee_simulation import PersuaderMethod

    method = PersuaderMethod(
        slug=args.method_slug,
        display_name=args.display_name,
        model_path=args.model_path,
        with_isr=False,
    )
    evaluate_baselines.ARCHIVED_METHODS += (method,)
    sys.argv = [
        str(EXP / "evaluate_joint_prompt_probe.py"),
        "--full-evaluation", "--index-start", "0", "--index-end", "500",
        "--methods", args.method_slug,
        "--generation-dir", str(args.generation_dir),
        "--output-dir", str(args.output_dir),
        "--judge-model", args.judge_model,
        "--judge-name", args.judge_name,
        "--judge-device", args.judge_device,
        "--prompt-version", args.prompt_version,
        *remainder,
    ]
    os.environ["ALLOW_JUDGE_EVAL"] = "1"
    evaluator = importlib.import_module("evaluate_joint_prompt_probe")
    evaluator.main()


if __name__ == "__main__":
    main()
