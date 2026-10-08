from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm.auto import tqdm

from .constants import DATA_DIR, MAX_NEW_TOKENS, SYSTEM_PERSUADER, PREFERENCE_SCENARIOS, data_path
from .data import prepare_all

WRITE_LOCK = threading.Lock()
SIMULATION_PROTOCOL = "persugpt_simulation_table5_reconstruction_v2"


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


def latest_by_source(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["metadata"]["scenario_id"]): row for row in read_jsonl(path)}


def transcript(history: list[dict[str, str]]) -> str:
    if not history:
        return "(empty)"
    return "\n".join(f"{turn['role'].title()}: {turn['content']}" for turn in history)


def isr_prompt(seed: dict[str, Any], history: list[dict[str, str]] | None = None) -> str:
    scenario = seed["scenario"]
    history = seed["history"] if history is None else history
    return f"""The persuasion background: {scenario['background']}
The persuasion goal: {scenario['goal']}
The persuader: {scenario['persuader']}
The persuadee: {scenario['persuadee']}
Candidate persuasion strategies: {'; '.join(seed['strategies'])}

Historical dialogue:
{transcript(history)}

Analyze the persuadee's current intent, select one candidate strategy, and
write the next persuader response. Return exactly this three-line format:
Intent to Strategy Reasoning: <analysis>
Selected Strategy: <strategy>
Utterance: <persuader response>"""


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

        if quantization not in {"none", "4bit"}:
            raise ValueError(f"Unsupported local quantization: {quantization}")
        self.torch = torch
        self.model_name = model
        self.quantization = quantization
        self.devices = self._parse_devices(device)
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
        if quantization == "4bit":
            # The available GPUs may have only 6--8 GiB free after the fixed
            # persuadee and unrelated jobs are loaded.  4-bit NF4 keeps the
            # 8B persuader on one 3090 and avoids multi-GPU CPU offloading.
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        if len(self.devices) > 1:
            # A Llama 8B checkpoint is too close to the free capacity of the
            # occupied 3090. Let Accelerate balance layers across the two
            # explicitly requested GPUs instead of silently spilling weights.
            gpu_memory: dict[int | str, int | str] = {}
            for gpu_index in self.devices:
                free_bytes, _ = torch.cuda.mem_get_info(gpu_index)
                reserve = 512 * 1024**2
                gpu_memory[gpu_index] = max(1 * 1024**3, int(free_bytes - reserve))
            gpu_memory["cpu"] = "64GiB"
            load_kwargs.update(device_map="balanced", max_memory=gpu_memory)
        elif quantization == "4bit":
            # Quantized modules must be placed during loading; calling
            # ``model.to`` afterwards is unsupported by bitsandbytes.
            load_kwargs["device_map"] = {"": f"cuda:{self.devices[0]}"}
        base = AutoModelForCausalLM.from_pretrained(str(base_path), **load_kwargs)
        if adapter_config.is_file():
            base = PeftModel.from_pretrained(base, str(model_path), is_trainable=False)
            # PEFT saves LoRA tensors as FP32; keep their inference math in BF16.
            for name, parameter in base.named_parameters():
                if ".lora_" in name and parameter.is_floating_point():
                    parameter.data = parameter.data.to(dtype=torch.bfloat16)
        if len(self.devices) == 1:
            self.model = (base if quantization == "4bit" else base.to(f"cuda:{self.devices[0]}"))
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
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            # Qwen3.5 otherwise emits long private reasoning traces for every
            # role-play turn. The public dialogue should contain only speech.
            enable_thinking=False,
        )
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
        # Preserve the lightweight constructor contract for the unquantized
        # path while forwarding the optional 4-bit setting when requested.
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


def simulate_path(
    seed: dict[str, Any],
    candidate_full: str,
    persuader: ChatEndpoint | LocalTransformersEndpoint,
    user_model: ChatEndpoint | LocalTransformersEndpoint,
    simulation_rounds: int,
    top_p: float | None = None,
    top_k: int | None = None,
) -> list[dict[str, str]]:
    path = [{"role": "persuader", "content": extract_utterance(candidate_full)}]
    combined = [dict(turn) for turn in seed["history"]] + path
    # A reward prefix must contain user feedback. Round 1 is the candidate and
    # its reply; later rounds add one persuader follow-up and one user reply.
    for round_index in range(simulation_rounds):
        if round_index > 0:
            response = persuader.complete(
                system=SYSTEM_PERSUADER,
                user=isr_prompt(seed, combined),
                temperature=0.7,
                top_p=top_p,
                top_k=top_k,
            )
            turn = {"role": "persuader", "content": extract_utterance(response)}
            path.append(turn)
            combined.append(turn)

        system, prompt = user_prompt(seed, combined)
        response = user_model.complete(
            system=system,
            user=prompt,
            temperature=0.7,
            top_p=top_p,
            top_k=top_k,
        )
        turn = {"role": "persuadee", "content": response}
        path.append(turn)
        combined.append(turn)
    return path


def comparison_prompt(
    seed: dict[str, Any], path_x: list[dict[str, str]], path_y: list[dict[str, str]]
) -> str:
    scenario = seed["scenario"]
    return f"""I will provide a persuasion background, goal, persuader, persuadee,
historical conversation, and continuations in two parallel universes, Uni-X
and Uni-Y. Judge which universe the persuader performs better.

Rules adapted from Table 5 of Persuading across Diverse Domains:
1. Evaluate persuasiveness, user feedback, semantic relevance, emotional
   factors, factual correctness, and overall quality.
2. Summarize the history, then summarize Uni-X and Uni-Y separately.
3. Compare the two universes and decide where the persuader performs better.
4. Do not be affected by universe order; use only conversation content.
5. Choose exactly one universe; do not answer tie, both, or neither.

Background: {scenario['background']}
Goal: {scenario['goal']}
Persuader: {scenario['persuader']}
Persuadee: {scenario['persuadee']}

Historical dialogue:
{transcript(seed['history'])}

Uni-X continuation:
{transcript(path_x)}

Uni-Y continuation:
{transcript(path_y)}

Return only a JSON object:
{{"summary_history":"...","summary_X":"...","summary_Y":"...",
"explain":"objective comparison, ending with the decision",
"result":"Uni-X"}}
The result value must be exactly "Uni-X" or "Uni-Y"."""


def parse_comparison(raw: str) -> dict[str, str]:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"Judge output is not JSON: {raw[:300]}")
    value = json.loads(raw[start : end + 1])
    if value.get("result") not in {"Uni-X", "Uni-Y"}:
        raise ValueError(f"Invalid comparison result: {value.get('result')}")
    return {key: str(value.get(key, "")) for key in (
        "summary_history", "summary_X", "summary_Y", "explain", "result"
    )}


def compare_prefixes(
    seed: dict[str, Any],
    path_x: list[dict[str, str]],
    path_y: list[dict[str, str]],
    judge: ChatEndpoint | LocalTransformersEndpoint,
    weights: list[float],
) -> tuple[float, float, list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    score_x = 0.0
    score_y = 0.0
    for round_number, weight in enumerate(weights, 1):
        prefix_utterances = 2 * round_number
        raw = judge.complete(
            system="You are a careful path-comparison evaluator following the Table 5 criteria.",
            user=comparison_prompt(
                seed, path_x[:prefix_utterances], path_y[:prefix_utterances]
            ),
            temperature=0.0,
            json_mode=True,
        )
        parsed = parse_comparison(raw)
        x_wins = parsed["result"] == "Uni-X"
        score_x += weight * int(x_wins)
        score_y += weight * int(not x_wins)
        decisions.append(
            {
                "simulation_round": round_number,
                "prefix_utterances": prefix_utterances,
                "weight": weight,
                "judgment": parsed,
            }
        )
    return score_x, score_y, decisions


def build_one_preference(
    seed: dict[str, Any],
    persuader: ChatEndpoint | LocalTransformersEndpoint,
    user_model: ChatEndpoint | LocalTransformersEndpoint,
    judge: ChatEndpoint | LocalTransformersEndpoint,
    weights: list[float],
) -> dict[str, Any]:
    prompt = isr_prompt(seed)
    candidate_x = persuader.complete(
        system=SYSTEM_PERSUADER, user=prompt, temperature=0.9
    )
    candidate_y = persuader.complete(
        system=SYSTEM_PERSUADER, user=prompt, temperature=0.9
    )
    if extract_utterance(candidate_x) == extract_utterance(candidate_y):
        candidate_y = persuader.complete(
            system=SYSTEM_PERSUADER,
            user=prompt + "\nUse a substantially different valid strategy from the first sample.",
            temperature=1.0,
        )
    path_x = simulate_path(seed, candidate_x, persuader, user_model, len(weights))
    path_y = simulate_path(seed, candidate_y, persuader, user_model, len(weights))
    score_x, score_y, decisions = compare_prefixes(
        seed, path_x, path_y, judge, weights
    )
    tied = score_x == score_y
    # The paper does not define reward-tie handling; prefer the judgment with
    # the longest available continuation instead of always favoring Uni-X.
    x_chosen = score_x > score_y or (
        tied and decisions[-1]["judgment"]["result"] == "Uni-X"
    )
    return {
        "instruction": prompt,
        "input": "",
        "output": [candidate_x, candidate_y] if x_chosen else [candidate_y, candidate_x],
        "history": [],
        "system": SYSTEM_PERSUADER,
        "metadata": {
            "scenario_id": seed["scenario_id"],
            "source_id": seed["source_id"],
            "session_id": seed["session_id"],
            "target_turn": seed["target_turn"],
            "protocol": SIMULATION_PROTOCOL,
            "weights": weights,
            "score_x": score_x,
            "score_y": score_y,
            "reward_tied": tied,
            "tie_break": "longest_prefix_judgment" if tied else None,
            "chosen_universe": "X" if x_chosen else "Y",
            "path_x": path_x,
            "path_y": path_y,
            "prefix_decisions": decisions,
        },
    }


def append_row(path: Path, row: dict[str, Any]) -> None:
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def build_preferences(
    *,
    persuader_url: str,
    persuader_model: str,
    user_url: str,
    user_model_name: str,
    user_api_key: str | None,
    judge_url: str,
    judge_model: str,
    simulation_rounds: int,
    workers: int,
    max_retries: int,
    limit: int | None,
    force: bool,
) -> Path:
    if simulation_rounds < 1:
        raise ValueError("simulation_rounds must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if max_retries < 1:
        raise ValueError("max_retries must be positive")
    prepare_all()
    seeds = read_jsonl(data_path("preference_seeds"))
    if len(seeds) != PREFERENCE_SCENARIOS:
        raise ValueError(f"Expected {PREFERENCE_SCENARIOS} preference seeds, found {len(seeds)}")
    if limit is not None:
        seeds = seeds[: max(0, limit)]
    output = data_path("simulation_preferences")
    if force and output.exists():
        output.unlink()
    if output.exists():
        protocols = {row.get("metadata", {}).get("protocol") for row in read_jsonl(output)}
        if protocols != {SIMULATION_PROTOCOL}:
            raise ValueError(
                f"Stale simulation protocol in {output}: {protocols}. Rerun with --force."
            )
    done = latest_by_source(output)
    pending = [seed for seed in seeds if seed["scenario_id"] not in done]
    local_key = os.getenv("LOCAL_OPENAI_API_KEY", "local-not-secret")
    local_user = user_url.startswith("local://")
    local_judge = judge_url.startswith("local://")
    user_key = local_key if local_user else endpoint_api_key(user_url, user_api_key)
    judge_key = local_key if local_judge else endpoint_api_key(judge_url)
    if not judge_key:
        raise RuntimeError("The selected judge API endpoint requires a configured API key")
    if not user_key:
        raise RuntimeError("The selected persuadee API endpoint requires a configured API key")
    persuader = generation_endpoint(
        base_url=persuader_url,
        model=persuader_model,
        api_key=local_key,
        max_retries=max_retries,
    )
    user_model = generation_endpoint(
        base_url=user_url,
        model=user_model_name,
        api_key=user_key,
        max_retries=max_retries,
    )
    judge = generation_endpoint(
        base_url=judge_url,
        model=judge_model,
        api_key=judge_key,
        max_retries=max_retries,
    )
    weights = [float(simulation_rounds - index) for index in range(simulation_rounds)]

    def task(seed: dict[str, Any]) -> None:
        append_row(
            output,
            build_one_preference(seed, persuader, user_model, judge, weights),
        )

    progress = tqdm(total=len(pending), desc="Simulation preferences", unit="history")
    if workers == 1:
        for seed in pending:
            task(seed)
            progress.update(1)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(task, seed) for seed in pending]
            for future in as_completed(futures):
                future.result()
                progress.update(1)
    progress.close()
    rows = latest_by_source(output)
    if limit is None and len(rows) != PREFERENCE_SCENARIOS:
        raise ValueError(f"Expected {PREFERENCE_SCENARIOS} preference rows, found {len(rows)}")
    summary = {
        "protocol": SIMULATION_PROTOCOL,
        "rows": len(rows),
        "simulation_rounds_k": simulation_rounds,
        "path_semantics": (
            "Each reward prefix contains the initial candidate, a persuadee reply, "
            "and for later rounds one additional persuader/persuadee exchange."
        ),
        "weights": weights,
        "persuader_url": persuader_url,
        "persuader_model": persuader_model,
        "user_url": user_url,
        "user_model": user_model_name,
        "judge_url": judge_url,
        "judge_model": judge_model,
        "user_compute_dtype": "bfloat16" if local_user else None,
        "judge_compute_dtype": "bfloat16" if local_judge else None,
        "disclosure": (
            "The paper does not publish k or numerical w_j values. This run uses "
            "the recorded configurable reconstruction values."
        ),
    }
    (DATA_DIR / "simulation_preferences_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    persuader.close()
    user_model.close()
    judge.close()
    return output
