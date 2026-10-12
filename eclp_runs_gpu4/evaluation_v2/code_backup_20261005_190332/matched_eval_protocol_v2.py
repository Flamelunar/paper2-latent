#!/usr/bin/env python3
"""Shared standalone rubrics and finite JSON decoding for both final judges."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

PROMPT_FILE = Path(__file__).with_name("matched_judge_prompts_v2.json")
CONFIG = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
SUCCESS_OUTPUTS = ('{"success":true}', '{"success":false}')
WIN_OUTPUTS = ('{"winner":"A"}', '{"winner":"B"}', '{"winner":"Tie"}')


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_verdict(raw: str, kind: str) -> dict[str, Any]:
    """Reject coerced booleans, duplicate keys, fences, and trailing text."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Expected exactly one JSON object") from exc
    if kind == "success":
        if not isinstance(value, dict) or set(value) != {"success"} or type(value["success"]) is not bool:
            raise ValueError("success must be the only key with a bare JSON boolean")
    elif kind == "win":
        if (not isinstance(value, dict) or set(value) != {"winner"}
                or not isinstance(value["winner"], str) or value["winner"] not in {"A", "B", "Tie"}):
            raise ValueError("winner must be the only key and equal A, B, or Tie")
    else:
        raise ValueError("Unknown verdict kind")
    return value


def transcript(turns: list[dict]) -> str:
    if not turns:
        return "(No preceding dialogue.)"
    return "\n".join(f"{turn['role'].title()}: {turn['content']}" for turn in turns)


def validate_trajectory(turns: list[dict]) -> None:
    if len(turns) != 10:
        raise ValueError("Expected exactly five complete rounds")
    for i, turn in enumerate(turns):
        if turn.get("role") != ("persuader" if i % 2 == 0 else "persuadee"):
            raise ValueError("Invalid alternating dialogue roles")
        if not isinstance(turn.get("content"), str) or not turn["content"].strip():
            raise ValueError("Missing dialogue content")


def validate_available_dialogue(turns: list[dict]) -> None:
    if not isinstance(turns, list) or not turns:
        raise ValueError("Golden must have at least its recorded first Persuader reply")
    for i, turn in enumerate(turns):
        if turn.get("role") != ("persuader" if i % 2 == 0 else "persuadee"):
            raise ValueError("Invalid alternating golden roles")
        if not isinstance(turn.get("content"), str) or not turn["content"].strip():
            raise ValueError("Missing golden dialogue content")


def success_prompt(row: dict, round_number: int) -> str:
    validate_trajectory(row["trajectory"])
    if round_number not in range(1, 6):
        raise ValueError("round_number must be 1 through 5")
    return f"""Assess goal acceptance at generated round {round_number}.

Goal: {row['scenario']['goal']}
Background: {row['scenario']['background']}

Earlier dialogue (context only):
{transcript(row['initial_history'])}

Generated conversation through the current round:
{transcript(row['trajectory'][:2 * round_number])}

Rubric:
{CONFIG['success_rule']}

Output:
{CONFIG['success_contract']}"""


def win_prompt(row: dict, golden: dict, metric: str, order: str) -> str:
    validate_trajectory(row["trajectory"])
    if order not in {"ab", "ba"}:
        raise ValueError("Unknown candidate order")
    if metric == "win1":
        ours, reference = row["trajectory"][0]["content"], golden["golden_first"]
        description = "Two alternative first complete Persuader replies after the preceding history"
    elif metric == "win_full":
        validate_available_dialogue(golden["golden_trajectory"])
        ours, reference = transcript(row["trajectory"]), transcript(golden["golden_trajectory"])
        description = "All available dialogue in two conversations after the same preceding history"
    else:
        raise ValueError("Unknown metric")
    a, b = (ours, reference) if order == "ab" else (reference, ours)
    return f"""{description}.

Goal: {row['scenario']['goal']}
Background: {row['scenario']['background']}

Common preceding dialogue:
{transcript(row['initial_history'])}

Candidate A:
{a}

Candidate B:
{b}

Rubric:
{CONFIG[metric + '_rule']}

Output:
{CONFIG['win_contract']}"""


def order_credit(winner: str, order: str) -> float:
    if winner == "Tie":
        return 0.5
    if winner not in {"A", "B"} or order not in {"ab", "ba"}:
        raise ValueError("Invalid winner or order")
    return float(winner == ("A" if order == "ab" else "B"))


class TokenTrie:
    """Only permit token sequences of the schema's finite legal JSON outputs."""
    def __init__(self, sequences: list[list[int]], eos: int):
        self.root: dict = {}
        self.eos = eos
        self.max_length = max(map(len, sequences))
        for sequence in sequences:
            if not sequence:
                raise ValueError("Empty output token sequence")
            node = self.root
            for token in sequence:
                node = node.setdefault(token, {})
            node[None] = True

    def allowed(self, generated: list[int]) -> list[int]:
        node = self.root
        for token in generated:
            if token == self.eos and None in node:
                return [self.eos]
            if token not in node:
                raise ValueError("Generated token left the JSON grammar")
            node = node[token]
        allowed = [token for token in node if token is not None]
        if None in node:
            allowed.append(self.eos)
        return allowed


class StrictJSONJudge:
    """Keep native model chat templates, constrain only the output schema."""
    def __init__(self, model: str, device: str = "cuda:0", memory_cap_gib: float | None = None):
        import torch
        if memory_cap_gib is not None:
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(memory_cap_gib * 1024 ** 3 / total, 0)
        from repro.simulation import LocalTransformersEndpoint
        self.endpoint = LocalTransformersEndpoint(model=model, device=device, quantization="none")
        self.torch = self.endpoint.torch
        self.tokenizer = self.endpoint.tokenizer
        self.eos = self.tokenizer.eos_token_id
        if self.eos is None:
            raise ValueError("Judge tokenizer has no EOS token")
        self.tries = {}
        for kind, outputs in (("success", SUCCESS_OUTPUTS), ("win", WIN_OUTPUTS)):
            sequences = [self.tokenizer.encode(text, add_special_tokens=False) for text in outputs]
            for tokens, text in zip(sequences, outputs):
                if self.tokenizer.decode(tokens, skip_special_tokens=True) != text:
                    raise ValueError("JSON tokenization does not round-trip exactly")
            self.tries[kind] = TokenTrie(sequences, self.eos)

    def complete(self, prompt: str, kind: str) -> dict:
        messages = [{"role": "system", "content": CONFIG["system"]},
                    {"role": "user", "content": prompt}]
        inputs = self.endpoint._encode_messages(messages)
        if inputs is None:
            raise ValueError("These judges must have native chat templates")
        if hasattr(inputs, "input_ids"):
            inputs = inputs.input_ids
        width = int(inputs.shape[-1])
        trie = self.tries[kind]
        if width + trie.max_length + 1 > self.endpoint.context_limit:
            raise ValueError("Full evaluation prompt exceeds judge context; truncation is forbidden")
        inputs = inputs.to(self.endpoint.input_device)
        generation_config = copy.deepcopy(self.endpoint.model.generation_config)
        generation_config.do_sample = False
        generation_config.num_beams = 1
        generation_config.min_length = 0
        generation_config.min_new_tokens = 0
        generation_config.eos_token_id = self.eos
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.forced_bos_token_id = None
        generation_config.forced_eos_token_id = None
        generation_config.repetition_penalty = 1.0
        generation_config.no_repeat_ngram_size = 0
        # Some released configs specify sampling options even in greedy mode.
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
        started = time.monotonic()
        with self.torch.inference_mode():
            generated = self.endpoint.model.generate(
                input_ids=inputs, attention_mask=self.torch.ones_like(inputs),
                generation_config=generation_config,
                max_new_tokens=trie.max_length + 1,
                prefix_allowed_tokens_fn=lambda batch, ids: trie.allowed(ids[width:].tolist()),
            )
        raw = self.tokenizer.decode(generated[0, width:], skip_special_tokens=True).strip()
        tokens = int(generated.shape[-1] - width)
        del generated, inputs
        verdict = parse_verdict(raw, kind)
        return {"raw_output": raw, "verdict": verdict, "prompt_sha256": text_hash(prompt),
                "input_tokens": width, "output_tokens": tokens,
                "elapsed_seconds": time.monotonic() - started,
                "output_decoding": "finite_JSON_token_grammar", "parse_error": None}

    def close(self) -> None:
        self.endpoint.close()
