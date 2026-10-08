#!/usr/bin/env python3
"""Dispatch the shared paper10k evaluators for the LatentRATT method."""
from __future__ import annotations

import importlib.util
import fcntl
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP_CODE = ROOT.parent / "exp_code_ljj"
MODEL = "/app/hzj/liujianjian/qwen2.5-7b-latentRATT-all-59868/rl"
METHOD_SLUG = "latentRATT_persuasion"
SELENE = "/app/hzj/models/Selene-1-Mini-Llama-3.1-8B"
GEMMA = "/app/hzj/models/gemma-4-E4B-it"
MAIN_TABLE_EVALUATOR = ROOT / "evaluate_joint_prompt_probe_main_table.py"
CURRENT_JOINT_EVALUATOR = EXP_CODE / "evaluate_joint_prompt_probe.py"
SELENE_CURRENT_REVISION = "joint_selene_success_reference_v17_v7_strict_goal_acceptance"
SELENE_CURRENT_OUTPUT_DIR = EXP_CODE / "artifacts/success_judgments/selene_v17_v7_latentRATT_20260928"
SELENE_SHARD_DIR = SELENE_CURRENT_OUTPUT_DIR / "shards_v7_20260928"
MAIN_TABLE_PROMPTS = {
    "selene-joint": ("v17", "joint_selene_success_literal_v6"),
    "gemma-joint": ("v11", "joint_v2_r11"),
}


def acquire_gemma_lock() -> tuple[object, Path]:
    """Serialize Gemma writers that target the same result file."""
    try:
        output_dir = Path(sys.argv[1 + sys.argv.index("--output-dir")])
    except (ValueError, IndexError):
        raise SystemExit("Gemma evaluation requires --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (output_dir / ".latentRATT_gemma_v11.lock").open("a+")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    return lock_file, output_dir / f"{METHOD_SLUG}.jsonl"


def is_complete_gemma_output(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        indexes = {int(row["index"]) for row in rows}
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return (
        len(rows) == 1000
        and indexes == set(range(1000))
        and all(
            row.get("method") == METHOD_SLUG
            and row.get("prompt_version") == "v11"
            and row.get("prompt_revision") == "joint_v2_r11"
            for row in rows
        )
    )


def wait_for_judge_capacity(model: str, description: str) -> None:
    sys.path.insert(0, str(EXP_CODE))
    from repro.gpu_capacity import wait_for_bf16_capacity

    wait_for_bf16_capacity([model], [2], reserve_gib=4, description=description)


def validate_current_selene_args(*, shard: bool) -> None:
    args = sys.argv[1:]
    if "--full-evaluation" not in args:
        raise SystemExit("selene-current requires --full-evaluation")
    expected = {
        "--judge-model": SELENE,
        "--judge-name": "selene1",
        "--judge-device": "cuda:2",
        "--prompt-version": "selene_v17",
    }
    for flag, value in expected.items():
        try:
            actual = args[args.index(flag) + 1]
        except (ValueError, IndexError):
            raise SystemExit(f"selene-current requires {flag} {value}")
        if actual != value:
            raise SystemExit(f"selene-current requires {flag} {value}, got {actual}")
    try:
        output_dir = Path(args[args.index("--output-dir") + 1]).resolve()
    except (ValueError, IndexError):
        raise SystemExit("selene-current requires --output-dir")
    allowed_output = (
        output_dir != SELENE_CURRENT_OUTPUT_DIR.resolve()
        and output_dir.is_relative_to(SELENE_SHARD_DIR.resolve())
        if shard else output_dir == SELENE_CURRENT_OUTPUT_DIR.resolve()
    )
    if not allowed_output:
        target = SELENE_SHARD_DIR if shard else SELENE_CURRENT_OUTPUT_DIR
        raise SystemExit(f"selene evaluation output must be under/exactly {target}")
    try:
        method_index = args.index("--methods")
        method = args[method_index + 1]
    except (ValueError, IndexError):
        raise SystemExit(f"selene-current requires --methods {METHOD_SLUG}")
    if method != METHOD_SLUG or (method_index + 2 < len(args) and not args[method_index + 2].startswith("--")):
        raise SystemExit(f"selene-current evaluates only --methods {METHOD_SLUG}")
    if shard:
        try:
            start = int(args[args.index("--index-start") + 1])
            end = int(args[args.index("--index-end") + 1])
        except (ValueError, IndexError):
            raise SystemExit("selene-shard requires --index-start and --index-end")
        if not 0 <= start < end <= 1000:
            raise SystemExit(f"Invalid Selene shard range [{start},{end})")
    elif "--index-start" in args or "--index-end" in args:
        raise SystemExit("selene-current does not accept shard index ranges")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"selene-joint", "selene-current", "selene-shard", "gemma-joint"}:
        raise SystemExit(
            "Usage: evaluate_latent_judges.py {selene-joint|selene-current|selene-shard|gemma-joint} [evaluator args]"
        )
    mode = sys.argv.pop(1)
    if mode in {"selene-current", "selene-shard"}:
        validate_current_selene_args(shard=mode == "selene-shard")
    if os.environ.get("ALLOW_JUDGE_EVAL") != "1":
        print("Evaluation deferred until prompts are ready; set ALLOW_JUDGE_EVAL=1 to run.", flush=True)
        return
    gemma_lock = None
    if mode == "gemma-joint":
        gemma_lock, gemma_output = acquire_gemma_lock()
        if is_complete_gemma_output(gemma_output):
            print(f"Gemma v11 output is already complete; skipping duplicate run: {gemma_output}", flush=True)
            gemma_lock.close()
            return
    sys.path.insert(0, str(EXP_CODE))

    import evaluate_baselines
    from repro.persuadee_simulation import PersuaderMethod

    latent_method = PersuaderMethod(
        slug=METHOD_SLUG,
        display_name="our LatentRATT",
        model_path=MODEL,
        with_isr=False,
    )
    evaluate_baselines.ARCHIVED_METHODS += (latent_method,)

    is_selene = mode in {"selene-joint", "selene-current", "selene-shard"}
    judge_model, description = (
        (SELENE, "Selene v17 joint success and Win-Rate evaluation")
        if is_selene else
        (GEMMA, "Gemma v11 joint success and Win-Rate evaluation")
    )
    try:
        wait_for_judge_capacity(judge_model, description)
        use_current = mode in {"selene-current", "selene-shard"}
        evaluator_path = CURRENT_JOINT_EVALUATOR if use_current else MAIN_TABLE_EVALUATOR
        module_name = "latentratt_current_evaluator" if use_current else "latentratt_main_table_evaluator"
        spec = importlib.util.spec_from_file_location(module_name, evaluator_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load the joint evaluator: {evaluator_path}")
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        if use_current:
            prompt_version, expected_revision = "selene_v17", SELENE_CURRENT_REVISION
        else:
            prompt_version, expected_revision = MAIN_TABLE_PROMPTS[mode]
        actual_revision = evaluator.PROMPT_REVISIONS.get(prompt_version)
        if actual_revision != expected_revision:
            raise RuntimeError(
                f"Prompt mismatch for {mode}: expected {expected_revision}, got {actual_revision}"
            )
        print(
            f"Using prompt {prompt_version} ({actual_revision}) from {evaluator_path}", flush=True)
        evaluator.main()
    finally:
        if gemma_lock is not None:
            gemma_lock.close()


if __name__ == "__main__":
    main()
