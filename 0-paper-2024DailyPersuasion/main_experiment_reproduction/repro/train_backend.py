"""Training backend for the isolated DailyPersuasion reproduction project.

The released 2024 PersuGPT trainer pins an older Transformers stack that cannot
parse the Llama 3.1 RoPE configuration. This module keeps the paper's data
protocol while using the installed Transformers/PEFT/TRL APIs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
from trl.chat_template_utils import llama3_training_chat_template

from .constants import CUTOFF_LEN, DEFAULT_MODEL, OUTPUT_DIR, RUN_CONFIG_DIR, RUN_ROOT, SAVE_STEPS, data_path
from .data import prepare_all

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainSpec:
    model_name_or_path: str = DEFAULT_MODEL
    output_dir: Path = OUTPUT_DIR
    finetuning_type: str = "lora"
    epochs: float = 1.0
    cutoff_len: int = CUTOFF_LEN
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float | None = None
    seed: int = 42
    deepspeed: str | None = None
    save_steps: int | None = None
    overwrite_output_dir: bool = False
    max_samples: int | None = None
    train_file: Path | None = None
    adapter_checkpoint: Path | None = None
    resume_from_checkpoint: Path | None = None
    stage_name: str | None = None


def _dataset_path(name: str) -> Path:
    if name not in {"dialogue_sft", "isr_sft", "user_model_sft", "simulation_preferences"}:
        raise ValueError(f"Unknown reproduction dataset: {name}")
    return data_path(name)


def _load_rows(name: str, max_samples: int | None = None, train_file: Path | None = None) -> Dataset:
    path = train_file or _dataset_path(name)
    if not path.is_file():
        raise FileNotFoundError(path)
    dataset = load_dataset("json", data_files=str(path), split="train")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def row_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = str(row.get("system", "")).strip()
    if system:
        messages.append({"role": "system", "content": system})
    for pair in row.get("history") or []:
        if len(pair) != 2:
            raise ValueError(f"Malformed history pair: {pair!r}")
        messages.extend(
            [
                {"role": "user", "content": str(pair[0])},
                {"role": "assistant", "content": str(pair[1])},
            ]
        )
    prompt = str(row.get("instruction", ""))
    if str(row.get("input", "")).strip():
        prompt = f"{prompt}\n{row['input']}"
    messages.extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": str(row["output"])},
        ]
    )
    return messages


def preference_messages(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row.get("output"), list) or len(row["output"]) != 2:
        raise ValueError("A DPO row must contain output=[chosen, rejected]")
    messages: list[dict[str, str]] = []
    system = str(row.get("system", "")).strip()
    if system:
        messages.append({"role": "system", "content": system})
    for pair in row.get("history") or []:
        if len(pair) != 2:
            raise ValueError(f"Malformed preference history pair: {pair!r}")
        messages.extend(
            [
                {"role": "user", "content": str(pair[0])},
                {"role": "assistant", "content": str(pair[1])},
            ]
        )
    prompt = str(row.get("instruction", ""))
    if str(row.get("input", "")).strip():
        prompt = f"{prompt}\n{row['input']}"
    prompt_messages = messages + [{"role": "user", "content": prompt}]
    return {
        "prompt": prompt_messages,
        "chosen": [{"role": "assistant", "content": str(row["output"][0])}],
        "rejected": [{"role": "assistant", "content": str(row["output"][1])}],
    }


def _map_messages(dataset: Dataset) -> Dataset:
    return dataset.map(
        lambda row: {"messages": row_messages(row)},
        remove_columns=dataset.column_names,
        desc="Preparing conversational SFT rows",
    )


def _map_preferences(dataset: Dataset) -> Dataset:
    return dataset.map(
        preference_messages,
        remove_columns=dataset.column_names,
        desc="Preparing DPO preference rows",
    )


def _qwen_assistant_mask_template(template: str) -> str:
    """Keep Qwen's native format while marking assistant tokens for TRL loss masking."""
    if "{% generation %}" in template or "{%- generation %}" in template:
        return template
    target = "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    replacement = (
        "{{- '<|im_start|>' + message.role + '\\n' }}"
        "{%- if message.role == 'assistant' %}"
        "{% generation %}{{- message.content + '<|im_end|>' }}{% endgeneration %}"
        "{%- else %}{{- message.content + '<|im_end|>' }}{%- endif %}"
        "{{- '\\n' }}"
    )
    if template.count(target) != 1:
        raise ValueError("Unrecognized Qwen chat template; cannot add assistant-only loss markers safely")
    return template.replace(target, replacement)


def _tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model_hint = str(model_name_or_path).lower()
    if "llama" in model_hint:
        # The released Llama 3.1 template lacks TRL's generation markers. The
        # standard TRL Llama-3 template preserves the header/eot protocol.
        tokenizer.chat_template = llama3_training_chat_template
    elif "qwen" in model_hint:
        if not tokenizer.chat_template:
            raise ValueError(f"Model has no native chat template: {model_name_or_path}")
        tokenizer.chat_template = _qwen_assistant_mask_template(tokenizer.chat_template)
    elif not tokenizer.chat_template or "{% generation %}" not in tokenizer.chat_template:
        raise ValueError(f"No assistant-masked chat template is available for {model_name_or_path}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _model(model_name_or_path: str):
    import torch

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )


def _lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )


def _common_config(spec: TrainSpec, output: Path, *, learning_rate: float) -> dict[str, Any]:
    save_steps = spec.save_steps if spec.save_steps is not None else SAVE_STEPS
    if save_steps < 1:
        raise ValueError("save_steps must be positive")
    config: dict[str, Any] = {
        "output_dir": str(output),
        "num_train_epochs": spec.epochs,
        "max_length": spec.cutoff_len,
        "per_device_train_batch_size": spec.batch_size,
        "gradient_accumulation_steps": spec.gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "save_steps": save_steps,
        "save_total_limit": 2,
        "report_to": [],
        "seed": spec.seed,
        "assistant_only_loss": True,
        "deepspeed": spec.deepspeed,
    }
    return config


def _write_run_config(output: Path, spec: TrainSpec, resolved: dict[str, Any], disclosure: str) -> None:
    RUN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": "Transformers + PEFT + TRL (native model chat template with assistant-only loss)",
        "train_spec": {**asdict(spec), "output_dir": str(spec.output_dir)},
        "resolved_args": resolved,
        "disclosure": disclosure,
    }
    try:
        name = "__".join(output.resolve().relative_to(RUN_ROOT.resolve()).parts)
    except ValueError:
        name = output.resolve().name
    (RUN_CONFIG_DIR / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _sft_config(spec: TrainSpec, output: Path, *, learning_rate: float) -> SFTConfig:
    kwargs = _common_config(spec, output, learning_rate=learning_rate)
    kwargs.update(
        {
            "dataset_text_field": "text",
            "completion_only_loss": False,
            "assistant_only_loss": True,
            "packing": False,
            "remove_unused_columns": False,
            "use_cache": False,
        }
    )
    return SFTConfig(**kwargs)


def _dpo_config(spec: TrainSpec, output: Path, beta: float) -> DPOConfig:
    kwargs = _common_config(
        spec,
        output,
        learning_rate=spec.learning_rate if spec.learning_rate is not None else 5e-5,
    )
    kwargs.update(
        {
            "beta": beta,
            "loss_type": "sigmoid",
            "disable_dropout": True,
            "remove_unused_columns": False,
            "use_cache": False,
        }
    )
    kwargs.pop("assistant_only_loss", None)
    return DPOConfig(**kwargs)


def dpo_reference_mode(finetuning_type: str) -> str:
    if finetuning_type == "lora":
        return "trl_reference_adapter"
    if finetuning_type == "full":
        return "separate_full_model"
    raise ValueError("finetuning_type must be full or lora")


def train_sft(spec: TrainSpec, *, with_isr: bool, dry_run: bool = False) -> Path:
    if spec.adapter_checkpoint and spec.resume_from_checkpoint:
        raise ValueError("Choose adapter continuation OR checkpoint resume, not both")
    if spec.adapter_checkpoint and spec.finetuning_type != "lora":
        raise ValueError("Adapter continuation requires LoRA")
    if spec.train_file is None:
        prepare_all(seed=spec.seed)
    dataset_name = "isr_sft" if with_isr else "dialogue_sft"
    default_stage = "02_llama31_isr_sft" if with_isr else "01_llama31_dialogue_sft"
    output = spec.output_dir / (spec.stage_name or default_stage)
    if not dry_run:
        if spec.adapter_checkpoint:
            if not (spec.adapter_checkpoint / "adapter_config.json").is_file():
                raise FileNotFoundError(spec.adapter_checkpoint / "adapter_config.json")
            if output.resolve() == spec.adapter_checkpoint.resolve():
                raise ValueError("Use a new --output-dir to preserve the source adapter")
        if spec.resume_from_checkpoint:
            for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
                if not (spec.resume_from_checkpoint / filename).is_file():
                    raise FileNotFoundError(spec.resume_from_checkpoint / filename)
        elif output.exists() and any(output.iterdir()) and not spec.overwrite_output_dir:
            raise ValueError("Output is nonempty; choose a new output directory or resume a checkpoint")
    lr = spec.learning_rate if spec.learning_rate is not None else (2e-5 if spec.finetuning_type == "full" else 1e-4)
    resolved = {
        "stage": "sft",
        "dry_run": dry_run,
        "dataset": dataset_name,
        "train_file": str(spec.train_file or _dataset_path(dataset_name)),
        "adapter_checkpoint": str(spec.adapter_checkpoint) if spec.adapter_checkpoint else None,
        "resume_from_checkpoint": str(spec.resume_from_checkpoint) if spec.resume_from_checkpoint else None,
        "model_name_or_path": spec.model_name_or_path,
        "finetuning_type": spec.finetuning_type,
        **_common_config(spec, output, learning_rate=lr),
        "lora_target": LORA_TARGETS,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    }
    if dry_run:
        print(json.dumps(resolved, ensure_ascii=False, indent=2, default=str))
        return output
    _write_run_config(
        output, spec, resolved,
        "The paper specifies one SFT epoch but does not disclose PEFT, optimizer, or batch details; these follow the recorded reproduction defaults.",
    )
    if spec.finetuning_type not in {"full", "lora"}:
        raise ValueError("finetuning_type must be full or lora")
    set_seed(spec.seed)
    dataset = _map_messages(_load_rows(dataset_name, spec.max_samples, spec.train_file))
    tokenizer = _tokenizer(spec.model_name_or_path)
    model = _model(spec.model_name_or_path)
    if spec.adapter_checkpoint:
        model = PeftModel.from_pretrained(model, str(spec.adapter_checkpoint), is_trainable=True)
    trainer = SFTTrainer(
        model=model,
        args=_sft_config(spec, output, learning_rate=lr),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config() if spec.finetuning_type == "lora" and not spec.adapter_checkpoint else None,
    )
    result = trainer.train(resume_from_checkpoint=str(spec.resume_from_checkpoint) if spec.resume_from_checkpoint else None)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "train_metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return output


def train_user_model(spec: TrainSpec, *, dry_run: bool = False) -> Path:
    prepare_all(seed=spec.seed)
    output = spec.output_dir / "03_user_model_sft"
    lr = spec.learning_rate if spec.learning_rate is not None else (2e-5 if spec.finetuning_type == "full" else 1e-4)
    resolved = {
        "stage": "sft",
        "dry_run": dry_run,
        "dataset": "user_model_sft",
        "model_name_or_path": spec.model_name_or_path,
        "finetuning_type": spec.finetuning_type,
        **_common_config(spec, output, learning_rate=lr),
    }
    if dry_run:
        print(json.dumps(resolved, ensure_ascii=False, indent=2, default=str))
        return output
    _write_run_config(
        output,
        spec,
        resolved,
        "Optional paper-style learned user model. The default simulation route uses GPT-4o-mini API instead.",
    )
    set_seed(spec.seed)
    dataset = _map_messages(_load_rows("user_model_sft", spec.max_samples))
    tokenizer = _tokenizer(spec.model_name_or_path)
    model = _model(spec.model_name_or_path)
    trainer = SFTTrainer(
        model=model,
        args=_sft_config(spec, output, learning_rate=lr),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_lora_config() if spec.finetuning_type == "lora" else None,
    )
    result = trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "train_metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return output


def train_dpo(
    spec: TrainSpec,
    *,
    isr_checkpoint: Path,
    beta: float = 0.1,
    dry_run: bool = False,
) -> Path:
    prepare_all(seed=spec.seed)
    preference_file = data_path("simulation_preferences")
    if not preference_file.is_file() and not dry_run:
        raise FileNotFoundError(
            f"{preference_file} is missing; run 04_Llama31_Ours_ISR_SimDPO_DPO.py simulate first"
        )
    output = spec.output_dir / "03_llama31_isr_sim_dpo"
    lr = spec.learning_rate if spec.learning_rate is not None else 5e-5
    resolved_training = _common_config(spec, output, learning_rate=lr)
    resolved_training.pop("assistant_only_loss", None)
    resolved = {
        "stage": "dpo",
        "dry_run": dry_run,
        "dataset": "simulation_preferences",
        "model_name_or_path": spec.model_name_or_path,
        "isr_checkpoint": str(isr_checkpoint),
        "finetuning_type": spec.finetuning_type,
        "reference_mode": dpo_reference_mode(spec.finetuning_type),
        "beta": beta,
        **resolved_training,
    }
    if dry_run:
        print(json.dumps(resolved, ensure_ascii=False, indent=2, default=str))
        return output
    _write_run_config(
        output,
        spec,
        resolved,
        "DPO continues the Llama 3.1 ISR adapter. GPT-4o-mini-generated preferences are an adaptation, not the paper's GPT-4/LLaMA-2 setting.",
    )
    if not isr_checkpoint.is_dir():
        raise FileNotFoundError(isr_checkpoint)
    if spec.finetuning_type not in {"full", "lora"}:
        raise ValueError("finetuning_type must be full or lora")
    set_seed(spec.seed)
    dataset = _map_preferences(_load_rows("simulation_preferences", spec.max_samples))
    tokenizer = _tokenizer(spec.model_name_or_path)
    if spec.finetuning_type == "lora" and (isr_checkpoint / "adapter_config.json").is_file():
        # TRL creates a frozen reference adapter from the policy adapter when
        # ref_model is omitted. This keeps one Llama 3.1 base model in memory
        # instead of loading a second 8B copy just for reference log-probs.
        policy = PeftModel.from_pretrained(
            _model(spec.model_name_or_path), str(isr_checkpoint), is_trainable=True
        )
        reference = None
    else:
        policy = _model(str(isr_checkpoint))
        reference = _model(str(isr_checkpoint))
    trainer = DPOTrainer(
        model=policy,
        ref_model=reference,
        args=_dpo_config(spec, output, beta),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "train_metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return output


def add_train_arguments(parser: Any) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--finetuning-type", choices=("full", "lora"), default="lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--cutoff-len", type=int, default=CUTOFF_LEN)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Checkpoint cadence; defaults to 100 for small5000 and 500 for paper profile.",
    )
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--stage-name", default=None, help="Override the output stage directory name.")
    parser.add_argument("--dry-run", action="store_true")


def add_sft_continuation_arguments(parser: Any) -> None:
    parser.add_argument("--train-file", type=Path, help="Explicit SFT JSON/JSONL; skips automatic dataset preparation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--adapter-checkpoint", type=Path, help="Continue adapter weights with a fresh optimizer on selected data")
    group.add_argument("--resume-from-checkpoint", type=Path, help="Restore Trainer state using the same dataset and settings")


def spec_from_args(args: Any) -> TrainSpec:
    return TrainSpec(
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        finetuning_type=args.finetuning_type,
        epochs=args.epochs,
        cutoff_len=args.cutoff_len,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        deepspeed=args.deepspeed,
        save_steps=args.save_steps,
        overwrite_output_dir=args.overwrite_output_dir,
        max_samples=args.max_samples,
        train_file=getattr(args, "train_file", None),
        adapter_checkpoint=getattr(args, "adapter_checkpoint", None),
        resume_from_checkpoint=getattr(args, "resume_from_checkpoint", None),
        stage_name=getattr(args, "stage_name", None),
    )
