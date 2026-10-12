"""Shared local model endpoints and persuadee prompts for the baseline runs."""

from __future__ import annotations
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any
from openai import OpenAI
from .constants import MAX_NEW_TOKENS

WRITE_LOCK = threading.Lock()

def endpoint_api_key(base_url: str, fallback: str | None = None) -> str | None:
    """Select the provider key that matches an OpenAI-compatible endpoint."""
    if "api.openai.com" in base_url.lower():
        return os.getenv("OPENAI_API_KEY") or fallback
    if "chatanywhere" in base_url.lower():
        return os.getenv("CHATANYWHERE_API") or fallback
    return fallback or os.getenv("OPENAI_API_KEY") or os.getenv("CHATANYWHERE_API")

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL {path}:{line_number}") from exc
    return rows

def transcript(history: list[dict[str, str]]) -> str:
    if not history:
        return "(empty)"
    return "\n".join(f"{turn['role'].title()}: {turn['content']}" for turn in history)

def extract_utterance(text: str) -> str:
    match = re.search(r"(?:^|\n)\s*Utterance\s*:\s*(.+)\s*$", text, re.I | re.S)
    if match:
        return match.group(1).strip()
    return text.strip()

class ChatEndpoint:
    def __init__(self, *, base_url: str, model: str, api_key: str, max_retries: int) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.max_retries = max(1, max_retries)

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int = MAX_NEW_TOKENS,
        top_p: float | None = None,
        top_k: int | None = None,
        json_mode: bool = False,
    ) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=300,
                max_retries=0,
            )
            request: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if top_p is not None:
                request["top_p"] = top_p
            if json_mode:
                request["response_format"] = {"type": "json_object"}
            try:
                response = client.chat.completions.create(**request)
                text = (response.choices[0].message.content or "").strip()
                if text:
                    return text
                raise ValueError("empty completion")
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2**attempt, 8))
            finally:
                client.close()
        raise RuntimeError(f"Endpoint {self.model} failed: {last_error}")

    def close(self) -> None:
        """API clients are created per request, so there is no persistent state."""

class LocalTransformersEndpoint:
    """OpenAI-like completion interface for a local BF16 model or LoRA checkpoint."""

    def __init__(self, *, model: str, device: str, quantization: str = "none") -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if quantization != "none":
            raise ValueError("This reproduction requires full BF16 weights; quantization is disabled")
        if not torch.cuda.is_available():
            raise RuntimeError("Local inference requires CUDA; refusing to change precision")
        self.torch = torch
        self.model_name = model
        self.quantization = quantization
        self.devices = self._parse_devices(device)
        for device_id in self.devices:
            with torch.cuda.device(device_id):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(f"cuda:{device_id} does not support BF16 inference")
        model_path = Path(model).expanduser().resolve()
        adapter_config = model_path / "adapter_config.json"
        base_path = model_path
        if adapter_config.is_file():
            adapter = json.loads(adapter_config.read_text(encoding="utf-8"))
            base_path = Path(str(adapter["base_model_name_or_path"])).expanduser().resolve()
        tokenizer_path = model_path if (model_path / "tokenizer_config.json").is_file() else base_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        if len(self.devices) > 1:
            # Place layers in order so uneven free memory can be used without
            # unnecessarily offloading later layers to CPU.
            gpu_memory: dict[int | str, int | str] = {}
            for gpu_index in self.devices:
                free_bytes, _ = torch.cuda.mem_get_info(gpu_index)
                # Keep room for generation KV cache and activations, especially
                # when another evaluation replica shares one of the devices.
                reserve = 3 * 1024**3
                gpu_memory[gpu_index] = max(1 * 1024**3, int(free_bytes - reserve))
            gpu_memory["cpu"] = "64GiB"
            load_kwargs.update(device_map="sequential", max_memory=gpu_memory)
        base = AutoModelForCausalLM.from_pretrained(str(base_path), **load_kwargs)
        if adapter_config.is_file():
            base = PeftModel.from_pretrained(base, str(model_path), is_trainable=False)
            # PEFT saves LoRA tensors as FP32; keep their inference math in BF16.
            for name, parameter in base.named_parameters():
                if ".lora_" in name and parameter.is_floating_point():
                    parameter.data = parameter.data.to(dtype=torch.bfloat16)
        if len(self.devices) == 1:
            self.model = base.to(f"cuda:{self.devices[0]}")
            self.model = self.model.eval()
            self.input_device = f"cuda:{self.devices[0]}"
        else:
            self.model = base.eval()
            first_parameter = next(self.model.parameters())
            self.input_device = str(first_parameter.device)
        text_config = getattr(self.model.config, "text_config", None)
        context_limit = getattr(text_config, "max_position_embeddings", None)
        if context_limit is None:
            context_limit = getattr(self.model.config, "max_position_embeddings", 32768)
        self.context_limit = int(context_limit)
        self._generation_lock = threading.Lock()

    @staticmethod
    def _parse_devices(device: str) -> list[int]:
        """Parse ``cuda:0`` or a comma-separated multi-GPU device list."""
        values = [part.strip() for part in device.split(",") if part.strip()]
        if not values or any(not value.startswith("cuda:") for value in values):
            raise ValueError(
                "LocalTransformersEndpoint expects cuda:N or comma-separated cuda:N devices"
            )
        try:
            indexes = [int(value.split(":", 1)[1]) for value in values]
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid CUDA device specification: {device}") from exc
        if len(set(indexes)) != len(indexes):
            raise ValueError(f"Duplicate CUDA device in specification: {device}")
        return indexes

    def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int = MAX_NEW_TOKENS,
        top_p: float | None = None,
        top_k: int | None = None,
        json_mode: bool = False,
    ) -> str:
        if json_mode:
            user += "\nReturn only a valid JSON object, without markdown or surrounding text."
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if getattr(self.tokenizer, "chat_template", None):
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                # Qwen3.5 otherwise emits long private reasoning traces for every
                # role-play turn. The public dialogue should contain only speech.
                enable_thinking=False,
            )
        else:
            # The released PersuGPT checkpoint is Llama-2 Chat without saved template metadata.
            prompt = (
                f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n"
                f"{user} [/INST]"
            )
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
        prompt_tokens = int(input_ids.shape[-1])
        context_limit = getattr(self, "context_limit", 2048)
        if prompt_tokens + max_tokens > context_limit:
            raise ValueError(
                f"Local prompt and completion exceed the model context: "
                f"{prompt_tokens}+{max_tokens}>{context_limit}"
            )
        input_ids = input_ids.to(self.input_device)
        generation_args: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": self.torch.ones_like(input_ids),
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            generation_args.update(do_sample=True, temperature=temperature)
            if top_p is not None:
                generation_args["top_p"] = top_p
            if top_k is not None:
                generation_args["top_k"] = top_k
        else:
            generation_args["do_sample"] = False
        with self._generation_lock, self.torch.inference_mode(), self.torch.autocast(
            device_type="cuda", dtype=self.torch.bfloat16
        ):
            generated = self.model.generate(**generation_args)
        return self.tokenizer.decode(
            generated[0, input_ids.shape[-1] :], skip_special_tokens=True
        ).strip()

    def complete_batch(self, requests: list[dict[str, Any]]) -> list[str]:
        """Decode independent local prompts together using left-padded inputs."""
        if not requests:
            return []

        options = [
            (
                float(request.get("temperature", 0.0)),
                int(request.get("max_tokens", MAX_NEW_TOKENS)),
                request.get("top_p"),
                request.get("top_k"),
            )
            for request in requests
        ]
        if any(option != options[0] for option in options[1:]):
            raise ValueError("A local generation batch must share decode settings")
        temperature, max_tokens, top_p, top_k = options[0]

        sequences: list[Any] = []
        for request in requests:
            user = str(request["user"])
            if request.get("json_mode"):
                user += "\nReturn only a valid JSON object, without markdown or surrounding text."
            messages = [
                {"role": "system", "content": str(request["system"])},
                {"role": "user", "content": user},
            ]
            if getattr(self.tokenizer, "chat_template", None):
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    enable_thinking=False,
                )
            else:
                # The released PersuGPT checkpoint is Llama-2 Chat without saved template metadata.
                prompt = (
                    f"[INST] <<SYS>>\n{messages[0]['content']}\n<</SYS>>\n\n"
                    f"{messages[1]['content']} [/INST]"
                )
                encoded = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            if hasattr(encoded, "input_ids"):
                encoded = encoded.input_ids
            sequence = encoded[0]
            if int(sequence.shape[-1]) + max_tokens > self.context_limit:
                raise ValueError(
                    f"Local prompt and completion exceed the model context: "
                    f"{sequence.shape[-1]}+{max_tokens}>{self.context_limit}"
                )
            sequences.append(sequence)

        width = max(int(sequence.shape[-1]) for sequence in sequences)
        batch = self.torch.full(
            (len(sequences), width),
            self.tokenizer.pad_token_id,
            dtype=sequences[0].dtype,
        )
        attention_mask = self.torch.zeros_like(batch)
        for row, sequence in enumerate(sequences):
            length = int(sequence.shape[-1])
            batch[row, width - length :] = sequence
            attention_mask[row, width - length :] = 1

        generation_args: dict[str, Any] = {
            "input_ids": batch.to(self.input_device),
            "attention_mask": attention_mask.to(self.input_device),
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_args["temperature"] = temperature
            if top_p is not None:
                generation_args["top_p"] = top_p
            if top_k is not None:
                generation_args["top_k"] = top_k
        with self._generation_lock, self.torch.inference_mode(), self.torch.autocast(
            device_type="cuda", dtype=self.torch.bfloat16
        ):
            generated = self.model.generate(**generation_args)
        return [
            self.tokenizer.decode(generated[row, width:], skip_special_tokens=True).strip()
            for row in range(len(sequences))
        ]

    def close(self) -> None:
        del self.model
        for gpu_index in self.devices:
            with self.torch.cuda.device(gpu_index):
                self.torch.cuda.empty_cache()

def generation_endpoint(
    *, base_url: str, model: str, api_key: str, max_retries: int,
    quantization: str = "none",
) -> ChatEndpoint | LocalTransformersEndpoint:
    if base_url.startswith("local://"):
        device = base_url.removeprefix("local://") or "cuda:0"
        # The local endpoint rejects any quantization request and uses BF16 weights.
        if quantization == "none":
            return LocalTransformersEndpoint(model=model, device=device)
        return LocalTransformersEndpoint(model=model, device=device, quantization=quantization)
    return ChatEndpoint(
        base_url=base_url, model=model, api_key=api_key, max_retries=max_retries
    )

def user_prompt(seed: dict[str, Any], history: list[dict[str, str]]) -> tuple[str, str]:
    scenario = seed["scenario"]
    system = (
        f"You are {scenario['persuadee']}, speaking with {scenario['persuader']} in a "
        "realistic persuasion conversation. Use only the personal context stated in "
        "the background and the beliefs or concerns shown in the dialogue. Stay in "
        "character and react naturally to the latest message. You may resist, hesitate, "
        "ask questions, make a qualified commitment, or accept when the conversation "
        "earns it. Never act as the persuader, mention this simulation, or invent "
        "personal facts; output one natural next utterance only."
    )
    user = f"""Persuasion background: {scenario['background']}
Persuasion goal: {scenario['goal']}

Dialogue so far:
{transcript(history)}

Write only the next natural persuadee utterance. Do not add a speaker label."""
    return system, user

def append_row(path: Path, row: dict[str, Any]) -> None:
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
