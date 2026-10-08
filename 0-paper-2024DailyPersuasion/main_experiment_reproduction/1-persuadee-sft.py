#!/usr/bin/env python3
"""Fine-tune Qwen3.5-9B as the DailyPersuasion persuadee/user model.

This implements the paper's user-model objective:
    scenario + dialogue history + current persuader utterance -> persuadee reply

The full ``user_model_sft.jsonl`` file stores a complete dialogue in each row.
Only persuadee/assistant spans receive a loss; scenario and persuader text are
strictly conditioning context. The script intentionally uses Qwen's native
chat template rather than the Llama-specific template used by the persuader
training entry points.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


PROJECT_ROOT = Path(__file__).resolve().parent
# The active reduced experiment uses the same 5,000 selected SFT scenarios as
# the persuader baselines.  The paper-scale 10,000-scenario file remains
# available through ``--data`` when a full reproduction is needed.
DEFAULT_DATA = PROJECT_ROOT / "data" / "small5000" / "sft_persuadee.jsonl"
DEFAULT_MODEL = Path("/data1/liujianjian/model/Qwen3.5-9B")
DEFAULT_OUTPUT = Path("/data1/liujianjian/Persuadee/Qwen3.5-9B-persuadee")
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
IGNORE_INDEX = -100


@dataclass(frozen=True)
class DataStats:
    rows: int
    assistant_turns: int
    median_tokens: int
    p95_tokens: int
    max_tokens: int
    truncated_rows: int


class PersuadeeCollator:
    """Right-pad already-tokenized causal-LM examples while preserving labels."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * padding)
            labels.append(feature["labels"] + [IGNORE_INDEX] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    # Two passes provide the requested initial persuadee SFT budget.  An
    # interrupted run can continue from a checkpoint without restarting.
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=2048)
    # A 9B BF16 base plus LoRA and 2,048-token activations fits safely on one
    # 24 GB RTX 3090 with a micro-batch of one.
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--preprocessing-workers", type=int, default=16)
    parser.add_argument("--preprocessing-batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit rows for a smoke test.")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Tokenize and validate data without loading Qwen.")
    parser.add_argument("--skip-merge", action="store_true", help="Save the adapter only.")
    return parser.parse_args()


def row_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Convert the established Alpaca-style row into Qwen chat messages."""
    messages: list[dict[str, str]] = []
    system = str(row.get("system", "")).strip()
    if system:
        messages.append({"role": "system", "content": system})
    for pair in row.get("history") or []:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Malformed history pair: {pair!r}")
        messages.extend(
            (
                {"role": "user", "content": str(pair[0])},
                {"role": "assistant", "content": str(pair[1])},
            )
        )
    instruction = str(row.get("instruction", "")).strip()
    if not instruction:
        raise ValueError("Missing current persuader utterance in instruction")
    user_input = str(row.get("input", "")).strip()
    if user_input:
        instruction = f"{instruction}\n{user_input}"
    output = str(row.get("output", "")).strip()
    if not output:
        raise ValueError("Missing persuadee target output")
    messages.extend(
        (
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        )
    )
    return messages


def _assistant_labels(input_ids: list[int], tokenizer: Any, expected_turns: int) -> list[int]:
    """Mask non-persuadee spans using Qwen's reserved chat-role tokens."""
    start = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not start or end is None or end < 0:
        raise ValueError("Qwen chat-template special tokens are unavailable")
    labels = [IGNORE_INDEX] * len(input_ids)
    assistant_starts: list[int] = []
    width = len(start)
    for index in range(len(input_ids) - width + 1):
        if input_ids[index : index + width] == start:
            assistant_starts.append(index)
    if len(assistant_starts) != expected_turns:
        raise ValueError(
            f"Expected {expected_turns} persuadee spans, found {len(assistant_starts)}. "
            "The Qwen template may have changed."
        )
    for span_start in assistant_starts:
        try:
            span_end = input_ids.index(end, span_start) + 1
        except ValueError as exc:
            raise ValueError("Unterminated Qwen assistant span") from exc
        labels[span_start:span_end] = input_ids[span_start:span_end]
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("No persuadee labels were created")
    return labels


def render_qwen_text_chat(messages: list[dict[str, str]]) -> str:
    """Render the text-only subset of Qwen3.5's native template efficiently."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Persuadee training messages must end with an assistant target")
    last_user = max(index for index, message in enumerate(messages) if message["role"] == "user")
    rendered: list[str] = []
    for index, message in enumerate(messages):
        role = message["role"]
        content = message["content"].strip()
        if role == "system":
            if index != 0:
                raise ValueError("System message must be first")
            rendered.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif role == "user":
            rendered.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            prefix = "<|im_start|>assistant\n"
            if index > last_user:
                prefix += "<think>\n\n</think>\n\n"
            rendered.append(f"{prefix}{content}<|im_end|>\n")
        else:
            raise ValueError(f"Unsupported message role: {role}")
    return "".join(rendered)


def validate_qwen_template(tokenizer: Any) -> None:
    """Fail fast if a future Qwen tokenizer changes this text-only protocol."""
    messages = [
        {"role": "system", "content": "You are a persuadee."},
        {"role": "user", "content": "Would you consider joining?"},
        {"role": "assistant", "content": "I need more information first."},
        {"role": "user", "content": "The event is one hour long."},
        {"role": "assistant", "content": "That sounds manageable."},
    ]
    native = tokenizer.apply_chat_template(messages, tokenize=False)
    direct = render_qwen_text_chat(messages)
    if native != direct:
        raise ValueError("The local Qwen chat template changed; update render_qwen_text_chat before training")


def encode_batch(batch: dict[str, list[Any]], tokenizer: Any, max_length: int) -> dict[str, list[Any]]:
    """Batch BPE tokenization; single-row Qwen tokenization is prohibitively slow."""
    fields = tuple(batch)
    rows = [{field: batch[field][index] for field in fields} for index in range(len(batch[fields[0]]))]
    messages_batch = [row_messages(row) for row in rows]
    input_ids_batch = tokenizer(
        [render_qwen_text_chat(messages) for messages in messages_batch],
        add_special_tokens=False,
    )["input_ids"]
    encoded: dict[str, list[Any]] = {
        "input_ids": [],
        "labels": [],
        "length": [],
        "assistant_turns": [],
        "original_length": [],
        "truncated": [],
    }
    for input_ids, messages in zip(input_ids_batch, messages_batch):
        labels = _assistant_labels(
            input_ids,
            tokenizer,
            expected_turns=sum(message["role"] == "assistant" for message in messages),
        )
        original_length = len(input_ids)
        if original_length > max_length:
            input_ids = input_ids[-max_length:]
            labels = labels[-max_length:]
        if not any(label != IGNORE_INDEX for label in labels):
            raise ValueError("Truncation removed every superviseable persuadee token")
        encoded["input_ids"].append(input_ids)
        encoded["labels"].append(labels)
        encoded["length"].append(len(input_ids))
        encoded["assistant_turns"].append(sum(message["role"] == "assistant" for message in messages))
        encoded["original_length"].append(original_length)
        encoded["truncated"].append(int(original_length > max_length))
    return encoded


def load_and_encode_data(args: argparse.Namespace, tokenizer: Any) -> tuple[Dataset, DataStats]:
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    dataset = load_dataset("json", data_files=str(args.data), split="train")
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    if not len(dataset):
        raise ValueError("No training rows selected")
    workers = max(1, args.preprocessing_workers)
    encoded = dataset.map(
        lambda batch: encode_batch(batch, tokenizer, args.max_length),
        batched=True,
        batch_size=args.preprocessing_batch_size,
        remove_columns=dataset.column_names,
        num_proc=workers,
        desc="Formatting Qwen persuadee conversations",
    )
    lengths = sorted(int(value) for value in encoded["length"])
    stats = DataStats(
        rows=len(encoded),
        assistant_turns=sum(int(value) for value in encoded["assistant_turns"]),
        median_tokens=lengths[len(lengths) // 2],
        p95_tokens=lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))],
        max_tokens=lengths[-1],
        truncated_rows=sum(int(value) for value in encoded["truncated"]),
    )
    return encoded, stats


def ensure_new_output(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    root = args.output_dir.expanduser().resolve()
    adapter_dir = root / "adapter"
    merged_dir = root / "merged"
    checkpoint_dir = root / "checkpoints"
    if args.resume_from_checkpoint is None:
        occupied = [path for path in (adapter_dir, merged_dir, checkpoint_dir) if path.exists() and any(path.iterdir())]
        if occupied:
            joined = ", ".join(str(path) for path in occupied)
            raise FileExistsError(
                f"Refusing to overwrite an existing persuadee run: {joined}. "
                "Use --resume-from-checkpoint for an interrupted job or choose a new --output-dir."
            )
    elif not args.resume_from_checkpoint.is_dir():
        raise FileNotFoundError(args.resume_from_checkpoint)
    root.mkdir(parents=True, exist_ok=True)
    return root, adapter_dir, merged_dir


def build_model(args: argparse.Namespace) -> Any:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Qwen3.5-9B persuadee SFT requires a CUDA GPU with bfloat16 support")
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = sorted(set(LORA_TARGETS) - available)
    if missing:
        raise ValueError(f"Qwen model is missing expected LoRA modules: {missing}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(LORA_TARGETS),
        ),
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def merge_adapter(
    trainer: Trainer, tokenizer: Any, merged_dir: Path, root: Path, args: argparse.Namespace
) -> None:
    trainer.accelerator.wait_for_everyone()
    if not trainer.is_world_process_zero:
        return
    if merged_dir.exists() and any(merged_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing merged model: {merged_dir}")
    model = trainer.accelerator.unwrap_model(trainer.model)
    if not isinstance(model, PeftModel):
        raise TypeError("Expected a PEFT model before merge")
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(str(merged_dir))
    write_json(
        merged_dir / "merge_metadata.json",
        {
            "base_model": str(args.model),
            "adapter": str(root / "adapter"),
            "standalone": True,
            "dtype": "bfloat16",
        },
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.max_length < 64 or args.batch_size < 1:
        raise ValueError("epochs must be positive, max-length >= 64, and batch-size >= 1")
    if args.gradient_accumulation_steps < 1 or args.preprocessing_workers < 1 or args.preprocessing_batch_size < 1:
        raise ValueError("gradient accumulation and preprocessing workers must be positive")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    validate_qwen_template(tokenizer)
    encoded, data_stats = load_and_encode_data(args, tokenizer)
    run_summary = {
        "objective": "scenario + dialogue history + current persuader utterance -> persuadee response",
        "paper_alignment": "Supervised user-model training only; no ISR target and no DPO.",
        "data": str(args.data),
        "model": str(args.model),
        "output_dir": str(args.output_dir),
        "arguments": vars(args),
        "data_stats": asdict(data_stats),
    }
    print(json.dumps(run_summary, ensure_ascii=False, indent=2, default=str))
    if args.dry_run:
        return

    root, adapter_dir, merged_dir = ensure_new_output(args)
    write_json(root / "run_config.json", run_summary)
    write_json(root / "training_status.json", {"status": "running", **run_summary})
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = build_model(args)
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(root / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=False,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        use_cache=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded,
        data_collator=PersuadeeCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    result = trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    )
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero:
        write_json(root / "train_metrics.json", result.metrics)
    if not args.skip_merge:
        merge_adapter(trainer, tokenizer, merged_dir, root, args)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero:
        write_json(
            root / "training_status.json",
            {
                "status": "completed",
                **run_summary,
                "adapter": str(adapter_dir),
                "merged_model": None if args.skip_merge else str(merged_dir),
                "metrics": result.metrics,
            },
        )


if __name__ == "__main__":
    main()
