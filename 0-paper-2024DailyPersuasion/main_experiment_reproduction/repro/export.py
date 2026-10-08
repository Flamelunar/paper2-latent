"""Export trained LoRA adapters as standalone merged Transformers checkpoints."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .constants import DEFAULT_MODEL, OUTPUT_DIR


ADAPTER_NAMES = (
    "01_llama31_dialogue_sft",
    "02_llama31_isr_sft",
    "03_llama31_isr_sim_dpo",
)


def _is_complete_model(path: Path) -> bool:
    if not (path / "config.json").is_file() or not (path / "merge_metadata.json").is_file():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def _load_and_merge(adapter_dir: Path, merged_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    base_model = str(config.get("base_model_name_or_path") or DEFAULT_MODEL)
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if use_cuda:
        # The launcher exposes the intended GPU as cuda:0 inside the process.
        load_kwargs["device_map"] = {"": "cuda:0"}
    base = None
    model = None
    merged = None
    base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    try:
        model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
        merged = model.merge_and_unload()
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(
            str(merged_dir),
            safe_serialization=True,
            max_shard_size="5GB",
        )
        tokenizer_dir = adapter_dir if (adapter_dir / "tokenizer.json").is_file() else Path(base_model)
        AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True).save_pretrained(
            str(merged_dir)
        )
        (merged_dir / "merge_metadata.json").write_text(
            json.dumps(
                {
                    "adapter_checkpoint": str(adapter_dir),
                    "base_model": base_model,
                    "dtype": str(dtype),
                    "standalone": True,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        del base
        if model is not None:
            del model
        if merged is not None:
            del merged
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def export_merged_models(
    *,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
    strict: bool = False,
) -> dict[str, str]:
    """Export every available adapter; return status per route.

    Missing adapters are reported as ``missing`` so zero-shot-only evaluation can
    still run. Export errors are retained in the status map and only raised in
    strict mode.
    """
    statuses: dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in tqdm(ADAPTER_NAMES, desc="Export merged adapters", unit="adapter"):
        adapter_dir = output_dir / name
        merged_dir = output_dir / f"{name}_merged"
        if not (adapter_dir / "adapter_config.json").is_file():
            statuses[name] = "missing"
            continue
        if not force and _is_complete_model(merged_dir):
            statuses[name] = "exists"
            continue
        try:
            _load_and_merge(adapter_dir, merged_dir)
            statuses[name] = "exported"
        except Exception as exc:  # keep evaluation usable if optional export fails
            statuses[name] = f"error: {type(exc).__name__}: {exc}"
            if strict:
                raise
    (output_dir / "merged_export_status.json").write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return statuses
