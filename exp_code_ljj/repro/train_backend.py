"""Shared SFT and DPO backend for the paper10k baseline methods."""

from __future__ import annotations
import json
import hashlib
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
from trl.chat_template_utils import llama3_training_chat_template
from .constants import CUTOFF_LEN, DEFAULT_MODEL, OUTPUT_DIR, RUN_CONFIG_DIR, RUN_ROOT, SAVE_STEPS, DATA_DIR, data_path
from .gpu_capacity import wait_for_bf16_capacity
from .public_answer import model_preference_directory
from .baseline_data import sha256

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

def _map_messages(dataset: Dataset) -> Dataset:
    return dataset.map(
        lambda row: {"messages": row_messages(row)},
        remove_columns=dataset.column_names,
        desc="Preparing conversational SFT rows",
    )

def preference_messages(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one ordered preference pair into TRL conversational DPO fields."""
    answers = row.get("output")
    if not isinstance(answers, list) or len(answers) != 2 or any(not str(x).strip() for x in answers):
        raise ValueError("DPO output must be [chosen, rejected] with two nonempty strings")
    prompt = str(row.get("instruction", ""))
    if str(row.get("input", "")).strip():
        prompt += "\n" + str(row["input"])
    messages = []
    if str(row.get("system", "")).strip():
        messages.append({"role": "system", "content": str(row["system"])})
    for pair in row.get("history") or []:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"Malformed DPO history pair: {pair!r}")
        messages.extend([
            {"role": "user", "content": str(pair[0])},
            {"role": "assistant", "content": str(pair[1])},
        ])
    messages.append({"role": "user", "content": prompt})
    return {
        "prompt": messages,
        "chosen": [{"role": "assistant", "content": str(answers[0])}],
        "rejected": [{"role": "assistant", "content": str(answers[1])}],
    }

def _map_preferences(dataset: Dataset) -> Dataset:
    return dataset.map(preference_messages, remove_columns=dataset.column_names,
                       desc="Preparing conversational DPO pairs")

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
    if template.count(target) == 1:
        return template.replace(target, replacement)

    # Qwen3.5 renders assistant messages through a richer branch that can add
    # reasoning and tool-call fields. Mark the whole assistant branch so TRL
    # masks user/system context while retaining the model's native rendering.
    assistant_branch = next((
        branch for branch in (
            "{%- elif message.role == 'assistant' %}",
            '{%- elif message.role == "assistant" %}',
        ) if template.count(branch) == 1
    ), None)
    tool_branch = next((
        branch for branch in (
            "{%- elif message.role == 'tool' %}",
            '{%- elif message.role == "tool" %}',
        ) if template.count(branch) == 1
    ), None)
    if assistant_branch and tool_branch:
        start = template.index(assistant_branch)
        end = template.index(tool_branch, start)
        body_start = start + len(assistant_branch)
        branch_body = template[body_start:end]
        marked = (
            assistant_branch + "\n{% generation %}"
            + branch_body
            + "{% endgeneration %}\n"
        )
        updated = template[:start] + marked + template[end:]
        # Training targets and the local inference endpoint both use plain
        # dialogue. Qwen3.5 otherwise inserts empty <think> wrappers by default.
        if "enable_thinking = false" not in updated:
            updated = (
                "{%- if enable_thinking is not defined %}"
                "{%- set enable_thinking = false %}{%- endif %}\n"
                + updated
            )
        thinking_branch = "{%- if loop.index0 > ns.last_query_index %}"
        if updated.count(thinking_branch) == 1:
            updated = updated.replace(
                thinking_branch,
                "{%- if loop.index0 > ns.last_query_index and enable_thinking %}",
            )
        return updated

    raise ValueError("Unrecognized Qwen chat template; cannot add assistant-only loss markers safely")


def _deepseek_assistant_mask_template() -> str:
    """Use DeepSeek's role tokens while marking only assistant content for TRL."""
    return """{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
{{ bos_token }}{{ message['content'] }}
{%- elif message['role'] == 'user' -%}
{{ '<｜User｜>' + message['content'] }}
{%- elif message['role'] == 'assistant' -%}
{{ '<｜Assistant｜>' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{ '<｜end▁of▁sentence｜>' }}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{ '<｜Assistant｜>' }}{%- endif -%}"""

def _tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model_hint = str(model_name_or_path).lower()
    if "llama" in model_hint:
        tokenizer.chat_template = llama3_training_chat_template
    elif "deepseek-r1" in model_hint:
        tokenizer.chat_template = _deepseek_assistant_mask_template()
    elif "qwen" in model_hint:
        if not tokenizer.chat_template:
            raise ValueError(f"Missing Qwen chat template: {model_name_or_path}")
        tokenizer.chat_template = _qwen_assistant_mask_template(tokenizer.chat_template)
    elif not tokenizer.chat_template or "generation" not in tokenizer.chat_template:
        raise ValueError(f"Assistant-masked chat template required: {model_name_or_path}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def _model(model_name_or_path: str):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; refusing to fall back from BF16 to FP32")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support BF16 training")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if not gpu_names or any("A100" not in name for name in gpu_names):
        raise RuntimeError(f"Training is restricted to A100 GPUs; visible devices are {gpu_names}")
    wait_for_bf16_capacity(
        [model_name_or_path],
        list(range(torch.cuda.device_count())),
        reserve_gib=float(os.getenv("LJJ_TRAIN_RESERVE_GIB", "16")),
        description="BF16 training",
    )
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
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

def _atomic_json_record(path: Path, payload: dict[str, Any]) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink training record: {path}")
    content = json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_run_config(output: Path, spec: TrainSpec, resolved: dict[str, Any], disclosure: str) -> None:
    resolved_args = dict(resolved)
    train_file = resolved_args.get("train_file")
    if train_file:
        digest = sha256(Path(train_file))
        previous = resolved_args.get("train_sha256")
        if previous is not None and previous != digest:
            raise ValueError("Training data changed before recording provenance")
        resolved_args["train_sha256"] = digest
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    resolved_args.update({
        "cuda_visible_devices": mask,
        "physical_gpu": int(mask) if mask is not None and mask.isdecimal() else None,
        "training_process_pid": os.getpid(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "quantization": "none",
    })
    if resolved_args.get("bf16") is True:
        resolved_args["compute_dtype"] = "bfloat16"
    payload = {
        "backend": "Transformers + PEFT + TRL (native model chat template with assistant-only loss)",
        "train_spec": {**asdict(spec), "output_dir": str(spec.output_dir)},
        "resolved_args": resolved_args,
        "disclosure": disclosure,
        "recording_phase": "training_process_pre_model_load",
        "resolved_output_dir": str(output.resolve()),
    }
    # A basename is not an identity: both external family roots have the same
    # stage names. Resolve symlink aliases and include the full output identity
    # so concurrent families never overwrite each other's central records.
    identity = output.resolve()
    readable = "__".join(identity.parts[-2:])
    digest = hashlib.sha256(str(identity).encode("utf-8")).hexdigest()
    central = RUN_CONFIG_DIR / f"{readable}--{digest}.json"
    _atomic_json_record(output / "run_config.json", resolved_args)
    _atomic_json_record(output / "training_provenance.json", payload)
    _atomic_json_record(central, payload)

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

def _dpo_config(spec: TrainSpec, output: Path, *, beta: float, learning_rate: float) -> DPOConfig:
    kwargs = _common_config(spec, output, learning_rate=learning_rate)
    kwargs.pop("assistant_only_loss")
    kwargs.update({"beta": beta, "loss_type": "sigmoid", "disable_dropout": True,
                   "remove_unused_columns": False, "use_cache": False})
    return DPOConfig(**kwargs)

def train_sft(spec: TrainSpec, *, with_isr: bool, dry_run: bool = False,
              dataset_name_override: str | None = None) -> Path:
    if spec.adapter_checkpoint and spec.resume_from_checkpoint:
        raise ValueError("Choose adapter continuation OR checkpoint resume, not both")
    if spec.adapter_checkpoint and spec.finetuning_type != "lora":
        raise ValueError("Adapter continuation requires LoRA")
    dataset_name = dataset_name_override or ("isr_sft" if with_isr else "dialogue_sft")
    if dataset_name not in {"dialogue_sft", "isr_sft", "user_model_sft"}:
        raise ValueError(f"Unsupported SFT dataset: {dataset_name}")
    if spec.train_file is None:
        spec.train_file = data_path(dataset_name)
    default_stage = {"dialogue_sft": "dialogue_sft_adapter",
                     "isr_sft": "baseline3_isr_sft_adapter",
                     "user_model_sft": "baseline3_user_model_adapter"}[dataset_name]
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

def train_dpo(spec: TrainSpec, *, isr_checkpoint: Path, beta: float = 0.1,
              dry_run: bool = False) -> Path:
    """Continue the ISR policy from ordered simulated preference pairs."""
    if spec.finetuning_type not in {"full", "lora"}:
        raise ValueError("finetuning_type must be full or lora")
    if beta <= 0:
        raise ValueError("DPO beta must be positive")
    preference_file = spec.train_file or (model_preference_directory(isr_checkpoint, DATA_DIR) / "simulation_preferences.jsonl")
    if not preference_file.is_file() and not dry_run:
        raise FileNotFoundError(f"Missing preference pairs: {preference_file}; run simulate first")
    simulation_config = preference_file.with_name("simulation_preferences_config.json")
    if not dry_run:
        if not simulation_config.is_file():
            raise FileNotFoundError(f"Missing simulation provenance: {simulation_config}")
        used_model = json.loads(simulation_config.read_text(encoding="utf-8"))["persuader_model"]
        if Path(used_model).expanduser().resolve() != isr_checkpoint.expanduser().resolve():
            raise ValueError("DPO ISR checkpoint differs from the model used to generate preferences")
        if spec.max_samples is None and len(_load_rows("simulation_preferences", train_file=preference_file)) != 2000:
            raise ValueError("Expected all 2,000 preference scenarios; use --max-samples only for a trial run")
    output = spec.output_dir / (spec.stage_name or "baseline3_isr_simdpo_adapter")
    learning_rate = spec.learning_rate if spec.learning_rate is not None else 5e-5
    config = {
        "stage": "dpo", "dataset": "simulation_preferences",
        "train_file": str(preference_file), "isr_checkpoint": str(isr_checkpoint),
        "model_name_or_path": spec.model_name_or_path,
        "finetuning_type": spec.finetuning_type, "beta": beta,
        **{k: v for k, v in _common_config(spec, output, learning_rate=learning_rate).items()
           if k != "assistant_only_loss"},
    }
    if not dry_run:
        config["train_sha256"] = sha256(preference_file)
        config["simulation_config"] = json.loads(simulation_config.read_text(encoding="utf-8"))
    if dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2, default=str))
        return output
    if not isr_checkpoint.is_dir():
        raise FileNotFoundError(isr_checkpoint)
    adapter_config = isr_checkpoint / "adapter_config.json"
    if adapter_config.is_file():
        adapter_base = json.loads(adapter_config.read_text(encoding="utf-8"))["base_model_name_or_path"]
        if Path(str(adapter_base)).expanduser().resolve() != Path(spec.model_name_or_path).expanduser().resolve():
            raise ValueError("DPO base model differs from the ISR adapter base model")
    if spec.resume_from_checkpoint:
        for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt"):
            if not (spec.resume_from_checkpoint / filename).is_file():
                raise FileNotFoundError(spec.resume_from_checkpoint / filename)
    elif output.exists() and any(output.iterdir()) and not spec.overwrite_output_dir:
        raise ValueError("DPO output is nonempty; use a new stage name or resume from a checkpoint")
    if spec.output_dir.resolve() == Path("/app/hzj/liujianjian/persuader_atapter_save/meta_llama31").resolve():
        # The running simulator stays untouched. This new DPO process owns
        # GPU 0 after simulation exits, so it can repair the three audited B2
        # scenes first, release inference models, then start normal SimDPO.
        from repair_baseline2_history import repair_llama_baseline2
        set_seed(spec.seed)
        repair_llama_baseline2(device="cuda:0")
    _write_run_config(output, spec, config,
                      "Paper-style ISR policy preference optimization; simulation settings are recorded separately.")
    set_seed(spec.seed)
    dataset = _map_preferences(_load_rows("simulation_preferences", spec.max_samples, preference_file))
    tokenizer = _tokenizer(spec.model_name_or_path)
    if spec.finetuning_type == "lora" and (isr_checkpoint / "adapter_config.json").is_file():
        policy = PeftModel.from_pretrained(
            _model(spec.model_name_or_path), str(isr_checkpoint), is_trainable=True)
        reference = None  # TRL freezes a copy of the ISR adapter as the reference.
    else:
        policy = _model(str(isr_checkpoint))
        reference = _model(str(isr_checkpoint))
    trainer = DPOTrainer(
        model=policy, ref_model=reference,
        args=_dpo_config(spec, output, beta=beta, learning_rate=learning_rate),
        train_dataset=dataset, processing_class=tokenizer)
    result = trainer.train(
        resume_from_checkpoint=(
            str(spec.resume_from_checkpoint) if spec.resume_from_checkpoint else None
        )
    )
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    (output / "train_metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
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
        help="Checkpoint cadence; defaults to 100 for this baseline run.",
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
