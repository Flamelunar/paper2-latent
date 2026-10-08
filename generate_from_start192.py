#!/usr/bin/env python3
"""Generate only the fixed 192 empty-history dialogues, preserving original indexes; no judge."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import os
from pathlib import Path

import torch
from tqdm.auto import tqdm

from persuasion_data import BASELINES, TEST_FILE, format_prompt
from run_latent_ratt import BASE_MODEL, load_model
from direct_answer_prompt import direct_answer_ids
from repro.baseline_data import sha256
from repro.persuadee_simulation import clean_utterance, load_test_points, simulate_case, PersuaderMethod
from repro.simulation import generation_endpoint, append_row, read_jsonl

DEFAULT_PERSUADEE = "/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged"
PROTOCOL = "latentRATT_persuasion_paper10k_5rounds_no_judge_v1"
V2 = Path(__file__).resolve().parent.parent / "baseline5_latentRATT-v2"


class LatentEndpoint:
    def __init__(self, checkpoint: Path, base_model: str, max_prompt_tokens: int, device: str):
        self.tokenizer, self.model = load_model(base_model, stage="eval", checkpoint=checkpoint,
                                                device=device)
        self.model.backbone.config.use_cache = True
        self.max_prompt_tokens = max_prompt_tokens
        model_hint = str(base_model).lower()
        self.disable_thinking = any(
            marker in model_hint for marker in ("deepseek-r1", "deepseek_r1", "deepseekr1")
        )

    def complete(self, *, system, user, temperature, max_tokens, top_p=None, top_k=None):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = direct_answer_ids(
            self.tokenizer, messages,
            model_hint="deepseek-r1" if self.disable_thinking else None,
        )
        prompt = prompt[-self.max_prompt_tokens:]
        ids = torch.tensor([prompt], device=self.model.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens = self.model.generate_text(
                ids, max_new_tokens=max_tokens, eos_token_id=self.tokenizer.eos_token_id,
                temperature=temperature, top_p=top_p or 1.0, top_k=top_k or 0)
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()


class SeededEndpoint:
    """Give each test case and dialogue turn an independent, repeatable RNG stream."""

    def __init__(self, endpoint, *, seed: int, devices: list[int]):
        self.endpoint = endpoint
        self.seed = seed
        self.devices = devices
        self.case_index = 0
        self.call_index = 0

    def begin_case(self, index: int) -> None:
        self.case_index = index
        self.call_index = 0

    def complete(self, **kwargs) -> str:
        seed = self.seed + self.case_index * 16 + self.call_index
        self.call_index += 1
        with torch.random.fork_rng(devices=self.devices):
            torch.random.default_generator.manual_seed(seed)
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            return self.endpoint.complete(**kwargs)

    def close(self) -> None:
        self.endpoint.close()


def cuda_device_indices(device_spec: str) -> list[int]:
    devices = [part.strip() for part in device_spec.split(",") if part.strip()]
    if not devices:
        raise ValueError("At least one CUDA device is required")
    indices = []
    for device in devices:
        parsed = torch.device(device)
        if parsed.type != "cuda" or parsed.index is None:
            raise ValueError(f"Expected a CUDA device, got {device!r}")
        indices.append(parsed.index)
    return indices


def wait_for_device_memory(devices: list[str], minimum_free_mib: int) -> None:
    """Wait until the combined selected devices have enough free memory."""
    indices = sorted({index for device in devices for index in cuda_device_indices(device)})
    while True:
        free_by_index = {}
        for index in indices:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(index), "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True)
            free_by_index[index] = int(result.stdout.strip())
        total_free = sum(free_by_index.values())
        if total_free >= minimum_free_mib:
            print(
                f"GPU memory preflight passed: {free_by_index} MiB free "
                f"({total_free} MiB combined)", flush=True)
            return
        print(
            f"Waiting for GPU memory before generation: {free_by_index} MiB free "
            f"({total_free} MiB combined); need {minimum_free_mib} MiB. "
            "Other jobs remain untouched.",
            flush=True,
        )
        time.sleep(60)


def validate_generation_rows(
    rows: list[dict],
    points: list[dict],
    *,
    checkpoint: Path,
    persuadee_model: str,
    require_all: bool,
    selected_indexes: list[int],
) -> set[int]:
    """Check fixed indices and five complete persuader/persuadee exchanges."""
    if not 0 < len(points) <= 1000:
        raise ValueError(f"Expected 1-1,000 fixed test points, got {len(points)}")
    indices: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Generation output contains a non-object JSON row")
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index not in selected_indexes:
            raise ValueError(f"Invalid test index in generation row: {index!r}")
        if index in indices:
            raise ValueError(f"Duplicate test index in generation output: {index}")
        indices.add(index)
        point = points[index]
        if row.get("scenario_id") != point["scenario_id"]:
            raise ValueError(f"Scenario mismatch for test index {index}")
        if row.get("rounds") != 5:
            raise ValueError(f"Generation row {index} does not declare five rounds")
        if row.get("persuader_model") != str(checkpoint):
            raise ValueError(f"Unexpected persuader checkpoint for test index {index}")
        if row.get("persuadee_model") != persuadee_model:
            raise ValueError(f"Unexpected persuadee model for test index {index}")

        trajectory = row.get("trajectory")
        if (not isinstance(trajectory, list) or len(trajectory) != 10
                or any(not isinstance(turn, dict) for turn in trajectory)):
            raise ValueError(f"Test index {index} must contain ten dialogue turns")
        for round_number in range(1, 6):
            pair = trajectory[(round_number - 1) * 2 : round_number * 2]
            expected_roles = ("persuader", "persuadee")
            for turn, expected_role in zip(pair, expected_roles):
                if turn.get("round") != round_number or turn.get("role") != expected_role:
                    raise ValueError(
                        f"Test index {index} has an invalid turn sequence in round {round_number}"
                    )
                if not isinstance(turn.get("content"), str) or not turn["content"].strip():
                    raise ValueError(
                        f"Test index {index} has an empty {expected_role} turn in round {round_number}"
                    )

    if require_all and indices != set(selected_indexes):
        missing = sorted(set(selected_indexes) - indices)
        raise ValueError(
            f"Incomplete generation: {len(indices)}/{len(selected_indexes)}; missing indices {missing[:10]}"
        )
    return indices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--persuader-device", default="cuda:2")
    p.add_argument("--persuadee-model", default=DEFAULT_PERSUADEE)
    p.add_argument("--persuadee-device", default="cuda:2")
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "generation/baseline5_latentRATT.json")
    p.add_argument("--max-prompt-tokens", type=int, default=1536)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.95)
    p.add_argument("--top-p", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--method-slug", default="latentRATT_persuasion")
    p.add_argument("--display-name", default="LatentRATT persuasion")
    p.add_argument("--protocol", default=PROTOCOL)
    p.add_argument("--architecture", choices=("v0", "v2"), default="v0")
    p.add_argument("--min-free-memory-mib", type=int, default=50000)
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    if args.max_new_tokens != 512:
        raise ValueError("The shared paper10k five-round protocol fixes max_new_tokens=512")
    persuader_devices = cuda_device_indices(args.persuader_device)
    if len(persuader_devices) != 1:
        raise ValueError("The LatentRATT persuader must use one A100 device")
    persuader_index = persuader_devices[0]
    if "A100" not in torch.cuda.get_device_name(persuader_index):
        raise ValueError(f"The LatentRATT persuader requires an A100: {args.persuader_device}")
    for index in sorted(set(cuda_device_indices(args.persuadee_device))):
        if torch.cuda.get_device_capability(index) < (8, 0):
            raise ValueError(f"BF16 persuadee inference requires Ampere or newer: cuda:{index}")
    if args.min_free_memory_mib <= 0:
        raise ValueError("--min-free-memory-mib must be positive")
    wait_for_device_memory(
        [args.persuader_device, args.persuadee_device], args.min_free_memory_mib)
    manifest = json.loads((BASELINES / "data/manifest.json").read_text())
    test_sha256 = sha256(TEST_FILE)
    if test_sha256 != manifest["files"]["evaluation_points"]["sha256"]:
        raise ValueError("Test set differs from the fixed paper10k split")
    points = load_test_points(TEST_FILE)
    if not 1 <= args.limit <= 1000:
        raise ValueError("--limit must select between 1 and 1,000 fixed cases")
    selection_path = BASELINES / "data/test_selection_192.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["original_indexes"]
    if (selection["count"] != 192 or len(set(selected)) != 192
            or selection["source_sha256"] != test_sha256
            or selected != [i for i,p in enumerate(points) if p["target_turn"] == 1 and p["history"] == []]):
        raise ValueError("Fixed empty-history subset changed")
    if args.limit != 1000:
        raise ValueError("Subset generator always uses all fixed 192; do not use prefix --limit")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(args.output) if args.output.exists() else []
    training_config = json.loads((args.checkpoint / "run_config.json").read_text(encoding="utf-8"))
    if args.method_slug == "baseline5_latentratt":
        if (training_config.get("stage") != "rl"
                or training_config.get("training_complete") is not True
                or training_config.get("completed_steps") != 29934
                or training_config.get("model") != args.base_model):
            raise ValueError("Extended Baseline 5 requires its own completed full-data LR-GRPO checkpoint")
    if any(row.get("protocol") != args.protocol for row in existing):
        raise ValueError("Existing generation file has a different protocol or checkpoint")
    done = validate_generation_rows(
        existing, points, checkpoint=args.checkpoint,
        persuadee_model=args.persuadee_model, require_all=False, selected_indexes=selected)
    if args.architecture == "v2":
        sys.path.insert(0, str(V2))
        from dual_slots import DualSlotPersuader
        import run_latent_ratt
        run_latent_ratt.LatentPersuader = DualSlotPersuader
    persuader = LatentEndpoint(args.checkpoint, args.base_model, args.max_prompt_tokens,
                               args.persuader_device)
    persuadee = generation_endpoint(base_url=f"local://{args.persuadee_device}",
                                    model=args.persuadee_model, api_key="local-not-secret", max_retries=1)
    if args.seed is not None:
        persuadee_devices = cuda_device_indices(args.persuadee_device)
        persuader = SeededEndpoint(
            persuader, seed=args.seed, devices=[persuader_index],
        )
        persuadee = SeededEndpoint(
            persuadee, seed=args.seed, devices=persuadee_devices,
        )
    method = PersuaderMethod(args.method_slug, args.display_name,
                            str(args.checkpoint), False)
    try:
        for index in tqdm(selected, desc="Generate 192 test dialogues"):
            point = points[index]
            if index in done:
                continue
            if args.seed is not None:
                persuader.begin_case(index)
                persuadee.begin_case(index)
            simulation = simulate_case(point=point, method=method, persuader=persuader,
                                       persuadee=persuadee, rounds=5,
                                       temperature=args.temperature,
                                       max_new_tokens=args.max_new_tokens,
                                       top_p=args.top_p, top_k=args.top_k)
            append_row(args.output, {
                "protocol": args.protocol, "index": index, "method": method.slug,
                "method_name": method.display_name, "persuader_model": str(args.checkpoint),
                "persuadee_model": args.persuadee_model, "scenario_id": point["scenario_id"],
                "source_id": point["source_id"], "session_id": point["session_id"],
                "target_turn": point["target_turn"], "reference": point.get("reference", ""),
                "scenario": point["scenario"], "strategies": point.get("strategies", []),
                "rounds": 5, "temperature": args.temperature, "top_p": args.top_p,
                "top_k": args.top_k, "max_new_tokens": args.max_new_tokens,
                "test_data": str(TEST_FILE), "test_sha256": test_sha256,
                "test_selection_sha256": sha256(selection_path), "test_selection_revision": selection["revision"],
                "selected_count": 192,
                "train_sha256": training_config.get("train_sha256"),
                "adapter_path": str(args.checkpoint / "adapter"),
                "latent_attention_path": str(args.checkpoint / "attention.pt"),
                "base_model": args.base_model,
                "persuader_device": args.persuader_device,
                "persuadee_device": args.persuadee_device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "persuader_compute_dtype": "bfloat16", "persuadee_compute_dtype": "bfloat16",
                "quantization": "none", "thinking_policy": "final_answer_only",
                **({"sampling_seed": args.seed} if args.seed is not None else {}),
                **simulation,
            })
    finally:
        persuadee.close()
    rows = read_jsonl(args.output)
    validate_generation_rows(
        rows, points, checkpoint=args.checkpoint,
        persuadee_model=args.persuadee_model, require_all=True, selected_indexes=selected)
    print(f"Generated 192 test dialogues at {args.output}; no evaluation performed.")


if __name__ == "__main__":
    main()

