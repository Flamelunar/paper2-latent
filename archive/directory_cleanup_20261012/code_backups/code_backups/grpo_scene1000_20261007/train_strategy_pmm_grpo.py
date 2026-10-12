#!/usr/bin/env python3
"""Progressively compress analysis while training a strategy World Model."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

from evaluation.revision_prompts import structured_strategy_prompt
from evaluation.revision_schema import STRUCTURED_STRATEGY_SCHEMA_VERSION, parse_structured_strategy
from utils.artifacts import hash_object, sha256_file, write_json
from utils.strategy_pmm import (Cycle, cycle_for_step, mean_strategy_rewards,
                                parse_strategy_plan, score_strategy_completion)
from utils.training_data import read_rows

TASK = "progressive_strategy_wm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(TASK,), default=TASK)
    parser.add_argument("--sft-merged", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merged-output", required=True)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-strategy-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.08)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


class CurriculumState:
    def __init__(self, max_steps: int) -> None:
        self.max_steps, self.step = max_steps, 0

    @property
    def cycle(self) -> Cycle:
        return cycle_for_step(self.step, self.max_steps)


def cycle_instruction(budget: int) -> str:
    if budget == 0:
        return ("\n\nCurrent PMM compression cycle: omit the analysis field entirely. "
                'Return JSON only: {"stage": 1, "strategy": {"key direction": "...", '
                '"recommended_action": "..."}}.')
    return (f"\n\nCurrent PMM compression cycle: include the full analysis object, but limit it to "
            f"at most {budget} tokens. Keep the stage and strategy fields unchanged.")


class CurriculumPromptDataset:
    def __init__(self, rows: list[dict[str, Any]], state: CurriculumState) -> None:
        self.rows, self.state = rows, state

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.rows[index])
        cycle = self.state.cycle
        row["prompt"] = [{"role": "user", "content": row["prompt"].rstrip()
                           + cycle_instruction(cycle.analysis_budget)}]
        row["curriculum_cycle"] = cycle.index
        return row


def _as_list(value: Any, count: int, name: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        value = [value] * count
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{name} must align with the generated completions")
    return list(value)


def main() -> None:
    args = parse_args()
    if args.max_steps < 3:
        raise ValueError("--max-steps must be at least 3 for the 512→256→0 curriculum")
    if args.save_steps < 1:
        raise ValueError("--save-steps must be positive")
    from utils.modeling import load_model, load_tokenizer, make_lora_config, seed_everything
    from utils.peft_compat import disable_bnb_when_unusable
    from utils.trl_compat import load_grpo_components

    import torch
    from peft import get_peft_model
    from transformers import TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from merge_adapter import merge_adapter_model

    disable_bnb_when_unusable()
    if args.precision == "bf16" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16-capable CUDA is required; precision will not be silently changed")
    seed_everything(args.seed)

    sft_contract_path = Path(args.sft_merged) / "training_contract.json"
    if not sft_contract_path.is_file():
        raise FileNotFoundError(f"SFT initialization contract is missing: {sft_contract_path}")
    sft_contract = json.loads(sft_contract_path.read_text(encoding="utf-8"))
    if (sft_contract.get("task") != "structured_strategy"
            or sft_contract.get("schema_version") != STRUCTURED_STRATEGY_SCHEMA_VERSION
            or sft_contract.get("output_analysis_budget") != 512):
        raise ValueError("Strategy PMM must initialize from the structured-strategy SFT checkpoint")
    train_hash = sha256_file(args.train_file)
    if train_hash != sft_contract.get("train_file_sha256"):
        raise ValueError("PMM and SFT must use the identical structured-strategy training file")

    prepared = []
    for row in read_rows(args.train_file):
        if row.get("task") != "structured_strategy":
            raise ValueError("Strategy PMM requires structured_strategy training rows")
        target = parse_structured_strategy(row.get("completion", ""))
        if target is None or target["stage"] != row.get("stage"):
            raise ValueError(f"Invalid strategy target: {row.get('id')}")
        state = row.get("state")
        if not isinstance(state, dict) or not str(state.get("goal", "")).strip():
            raise ValueError(f"Missing public goal: {row.get('id')}")
        expected_prompt = structured_strategy_prompt(state, row.get("history", []), target["stage"])
        if row.get("prompt") != expected_prompt:
            raise ValueError(f"Stored prompt differs from the frozen strategy prompt: {row.get('id')}")
        if parse_strategy_plan(row["completion"], 512) is None:
            raise ValueError(f"Invalid progressive-strategy reference: {row.get('id')}")
        prepared.append({"prompt": row["prompt"], "reference_completion": row["completion"],
                         "goal_text": state["goal"], "stage": target["stage"]})
    if not prepared:
        raise ValueError("Empty strategy PMM training set")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": STRUCTURED_STRATEGY_SCHEMA_VERSION,
        "task": TASK,
        "output_analysis_budget": 0,
        "init_model": str(Path(args.sft_merged).resolve()),
        "init_mode": "sft",
        "sft_contract_sha256": sha256_file(sft_contract_path),
        "train_file_sha256": train_hash,
        "samples": len(prepared),
        "training_seed": args.seed,
        "analysis_budgets": [512, 256, 0],
        "progressive_compression": True,
        "pmm_reward": True,
        "reward": "strategy_pmm_format_length_reference_goal",
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "max_completion_length": 512 + args.max_strategy_tokens,
        "precision": args.precision,
        "arguments": vars(args),
    }
    contract["resume_signature"] = hash_object(contract)
    last_checkpoint = get_last_checkpoint(str(output))
    contract_path = output / "training_contract.json"
    if last_checkpoint:
        if not contract_path.is_file():
            raise ValueError("Cannot resume PMM checkpoint without its training contract")
        previous_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous_contract.get("resume_signature") != contract["resume_signature"]:
            raise ValueError("PMM checkpoint does not match the current training contract")
    write_json(output / "training_contract.json", contract)
    write_json(output / "run_config.json", vars(args))

    tokenizer = load_tokenizer(args.sft_merged, padding_side="left", add_strategy_tokens=False)
    model = load_model(args.sft_merged, precision=args.precision)
    model = get_peft_model(model, make_lora_config(tokenizer, rank=args.lora_rank,
        alpha=args.lora_alpha, dropout=args.lora_dropout, train_special_tokens=False))
    state = CurriculumState(args.max_steps)
    dataset = CurriculumPromptDataset(prepared, state)
    reward_log = output / "reward_log.jsonl"
    reward_calls = 0

    def reward_fn(completions, **kwargs):
        nonlocal reward_calls
        cycles = _as_list(kwargs.get("curriculum_cycle"), len(completions), "curriculum_cycle")
        cycle = state.cycle
        if {int(value) for value in cycles} != {cycle.index}:
            raise RuntimeError("Prompt and reward PMM compression cycles differ")
        references = _as_list(kwargs.get("reference_completion"), len(completions), "reference_completion")
        goals = _as_list(kwargs.get("goal_text"), len(completions), "goal_text")
        completion_ids = kwargs.get("completion_ids")
        if hasattr(completion_ids, "tolist"):
            completion_ids = completion_ids.tolist()
        details = []
        for index, value in enumerate(completions):
            if isinstance(completion_ids, (list, tuple)):
                text = tokenizer.decode(completion_ids[index], skip_special_tokens=True)
            elif isinstance(value, list):
                text = "".join(str(message.get("content", "")) for message in value
                               if isinstance(message, dict))
            else:
                text = str(value)
            details.append(score_strategy_completion(text,
                {"reference_completion": references[index], "goal_text": goals[index]},
                cycle=cycle, tokenizer=tokenizer))
        reward_calls += 1
        if reward_calls % args.logging_steps == 0:
            summary = mean_strategy_rewards(details)
            summary.update({"global_step": state.step, "analysis_budget": cycle.analysis_budget})
            with reward_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        return [detail.total for detail in details]

    class CurriculumCallback(TrainerCallback):
        def on_train_begin(self, trainer_args, trainer_state, control, **kwargs):
            state.step = int(trainer_state.global_step)
            return control

        def on_step_begin(self, trainer_args, trainer_state, control, **kwargs):
            state.step = int(trainer_state.global_step)
            return control

        def on_step_end(self, trainer_args, trainer_state, control, **kwargs):
            state.step = int(trainer_state.global_step)
            return control

    GRPOConfig, GRPOTrainer = load_grpo_components()
    config = GRPOConfig(
        output_dir=str(output), max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate,
        lr_scheduler_type="cosine", warmup_ratio=0.08, max_grad_norm=0.2,
        num_generations=args.num_generations,
        max_completion_length=512 + args.max_strategy_tokens,
        temperature=args.temperature, top_p=args.top_p, epsilon=args.epsilon, beta=args.beta,
        loss_type="dr_grpo", scale_rewards="none", mask_truncated_completions=False,
        bf16=args.precision == "bf16", fp16=args.precision == "fp16",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        logging_steps=args.logging_steps, report_to="none", seed=args.seed, data_seed=args.seed,
        remove_unused_columns=False, dataloader_num_workers=0, dataloader_persistent_workers=False)
    trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, args=config,
        train_dataset=dataset, processing_class=tokenizer, callbacks=[CurriculumCallback()])
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    write_json(output / "trainer_metrics.json", {"global_step": trainer.state.global_step,
        "log_history": trainer.state.log_history, "training_wall_seconds": time.perf_counter() - started})
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    merge_adapter_model(args.sft_merged, str(output), args.merged_output,
                        precision=args.precision, add_strategy_tokens=False)


if __name__ == "__main__":
    main()
