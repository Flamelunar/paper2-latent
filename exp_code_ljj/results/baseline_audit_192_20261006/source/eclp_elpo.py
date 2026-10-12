#!/usr/bin/env python3
"""ECLP + ELPO research runner. Core implementation in one file.

Production neural forwards: BF16, CUDA, no quantization. Scalar losses,
Gaussian algebra, SVD and optimizer master states: FP32 for numerical stability.
This is a new implementation, NOT the original LatentR3 code or a reproduced result.
See FINAL_PLAN.md and README.md for supported scope and unverified GPU steps.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
from typing import Any, Iterable, Iterator

import torch
from torch import nn
from torch.nn import functional as F

VERSION = "eclp-elpo-bf16-v2.1"
BF16 = torch.bfloat16
DEFAULT_PERSUADEE = "/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged"
SOURCE_FILES = (
    "exp_code_ljj/repro/simulation.py",
    "exp_code_ljj/repro/persuadee_simulation.py",
    "exp_code_ljj/repro/constants.py",
    "exp_code_ljj/repro/data.py",
    "exp_code_ljj/repro/profile.py",
    "exp_code_ljj/repro/judge_parse.py",
    "exp_code_ljj/repro/gpu_capacity.py",
    "exp_code_ljj/evaluate_baselines.py",
)
DATA_NAMES = ("dialogue_sft", "isr_sft", "user_model_sft", "preference_seeds", "evaluation_points")
PRIVATE_MARKERS = re.compile(r"Intent to Strategy Reasoning\s*:|Selected Strategy\s*:|<think>", re.I)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Bind provenance to this process's implementation, even if the source file is
# replaced while a long training run is still alive.
IMPLEMENTATION_SHA256 = sha256(Path(__file__))


def object_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            if line.startswith("version https://git-lfs.github.com/spec/"):
                raise ValueError(f"Git LFS pointer, not data: {path}; recover the ORIGINAL object first")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from e
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            yield row


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())


def save_tensors(path: str | Path, obj: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(obj, tmp); os.replace(tmp, path)


def load_tensors(path: str | Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


@contextlib.contextmanager
def exclusive_writer(directory: Path):
    """A process lock, not a stale on-disk sentinel. Linux experiment servers only."""
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".writer.lock").open("a") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f"Another writer owns {directory}") from e
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@dataclasses.dataclass
class Config:
    repo_root: str = "."
    work_dir: str = "eclp_runs_v2"
    base_model: str = "/data1/liujianjian/model/Qwen2.5-7B-Instruct"
    persuadee_model: str = DEFAULT_PERSUADEE
    judge_model: str = ""  # MUST resolve the existing local evaluation checkpoint before lock.
    actor_system_prompt: str = "You are a skilled persuader."
    actor_device: str = "cuda:0"
    persuadee_device: str = "cuda:1"
    judge_device: str = "cuda:1"
    dtype: str = "bfloat16"
    quantization: str = "none"
    seed: int = 42
    width: int = 256
    feature_dim: int = 128
    state_slots: int = 2
    strategy_slots: int = 2
    attention_heads: int = 4
    action_dim: int = 64
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    max_prompt_tokens: int = 1536
    max_target_tokens: int = 512
    feature_context_tokens: int = 1536
    feature_response_tokens: int = 256
    max_new_tokens: int = 512
    rounds: int = 5
    temperature: float = 0.95
    top_p: float = 0.7
    top_k: int = 50
    test_action: str = "mean"
    obs_weight: float = 1.0
    strategy_weight: float = 0.1
    reasoning_weight: float = 0.1
    ppl_shaping_weight: float = 0.1
    state_mode: str = "eclp"  # eclp / no_error / no_correction / stateless
    isolate_state: bool = True
    gradient_checkpointing: bool = True
    public_supervision: str = "last"  # matches the supplied latent training's last-turn target
    dev_scenarios: int = 200
    exploration_radius: float = 0.5
    effect_alpha: float = 0.5
    effect_rank: int = 8
    clip_epsilon: float = 0.2
    kl_beta: float = 0.02
    max_kl: float = 0.1
    judge_prompt_version: str = "v2"
    judge_max_new_tokens: int = 256
    judge_parse_retries: int = 2
    feature_cache_entries: int = 64

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = read_json(path)
        unknown = set(raw) - {x.name for x in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = cls(**raw); cfg.validate(); return cfg

    def validate(self) -> None:
        if self.dtype != "bfloat16" or self.quantization != "none":
            raise ValueError("Only BF16, non-quantized neural training/inference is supported")
        for name in ("width", "feature_dim", "state_slots", "strategy_slots", "attention_heads",
                     "action_dim", "lora_rank", "lora_alpha", "max_prompt_tokens", "max_target_tokens",
                     "feature_context_tokens", "feature_response_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("obs_weight", "strategy_weight", "reasoning_weight", "ppl_shaping_weight", "exploration_radius", "effect_alpha",
                     "clip_epsilon", "kl_beta", "max_kl", "lora_dropout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if min(self.obs_weight, self.strategy_weight, self.kl_beta) < 0 or self.reasoning_weight < 0 or self.ppl_shaping_weight < 0:
            raise ValueError("Loss weights cannot be negative")
        if not 0 < self.clip_epsilon < 1 or self.max_kl <= 0:
            raise ValueError("Require clip_epsilon in (0,1) and max_kl > 0")
        for name in ("judge_max_new_tokens", "judge_parse_retries", "feature_cache_entries"):
            if type(getattr(self, name)) is not int or getattr(self, name) < (1 if name == "judge_max_new_tokens" else 0):
                raise ValueError(f"Invalid {name}")
        if self.effect_rank > self.feature_dim:
            raise ValueError("effect_rank cannot exceed the feedback feature dimension")
        if self.lora_dropout != 0:
            raise ValueError("This reproducible collector requires lora_dropout=0")
        if self.width % self.attention_heads:
            raise ValueError("width must be divisible by attention_heads")
        if not 0 <= self.effect_alpha < 1 or not 1 <= self.effect_rank <= self.action_dim:
            raise ValueError("Effect covariance must remain full rank: alpha in [0,1), valid rank")
        if self.exploration_radius <= 0 or self.feature_response_tokens < 1:
            raise ValueError("Invalid exploration radius or feature response budget")
        if self.state_mode not in {"eclp", "no_error", "no_correction", "stateless"}:
            raise ValueError("Unknown state ablation")
        if self.public_supervision != "last":
            raise ValueError("This release implements LAST-turn public supervision only")
        if not isinstance(self.actor_system_prompt, str) or not self.actor_system_prompt.strip():
            raise ValueError("actor_system_prompt must be a nonempty string")
        if self.test_action not in {"mean", "sample"}:
            raise ValueError("test_action must be mean or sample")
        if self.rounds != 5 or self.max_new_tokens != 512:
            raise ValueError("Fixed evaluation protocol: five rounds and max_new_tokens=512")
        if (self.temperature, self.top_p, self.top_k) != (0.95, 0.7, 50):
            raise ValueError("Keep existing fixed generation sampling: 0.95 / 0.7 / 50")
        if not 1 <= self.dev_scenarios < 2000:
            raise ValueError("Reserve a nonempty dev set INSIDE the existing preference split")
        if self.judge_prompt_version not in {"v1", "v2"}:
            raise ValueError("Only existing source SUCCESS_PROMPTS v1/v2 are supported")

    @property
    def root(self) -> Path:
        return Path(self.repo_root).expanduser().resolve()

    @property
    def work(self) -> Path:
        p = Path(self.work_dir).expanduser()
        return (self.root / p).resolve() if not p.is_absolute() else p.resolve()

    def architecture(self) -> dict:
        names = ["width", "feature_dim", "state_slots", "strategy_slots", "attention_heads",
                 "action_dim", "lora_rank", "lora_alpha", "lora_dropout", "state_mode",
                 "isolate_state", "obs_weight", "strategy_weight", "reasoning_weight", "public_supervision",
                 "max_prompt_tokens", "max_target_tokens", "feature_context_tokens",
                 "feature_response_tokens"]
        return {n: getattr(self, n) for n in names}

    def protocol(self) -> dict:
        names = ["dtype", "quantization", "rounds", "max_new_tokens", "temperature", "top_p",
                 "top_k", "ppl_shaping_weight", "judge_prompt_version", "judge_max_new_tokens", "judge_parse_retries",
                 "dev_scenarios", "max_prompt_tokens", "actor_system_prompt"]
        return {n: getattr(self, n) for n in names}


def seed_all(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_bf16(device: str) -> torch.device:
    d = torch.device(device)
    if d.type != "cuda" or d.index is None or not torch.cuda.is_available():
        raise RuntimeError(f"Production requires explicit BF16-capable cuda:N, got {device}")
    with torch.cuda.device(d):
        if not torch.cuda.is_bf16_supported() or torch.cuda.get_device_capability(d)[0] < 8:
            raise RuntimeError(f"Native BF16 support is required on {device}")
    return d


def precision_report(module: nn.Module, require_cuda: bool = True) -> dict:
    counts: dict[str, int] = {}; bad = []
    for name, p in module.named_parameters():
        counts[str(p.dtype)] = counts.get(str(p.dtype), 0) + p.numel()
        if p.is_floating_point() and p.dtype != BF16:
            bad.append((name, str(p.dtype)))
        if require_cuda and p.device.type != "cuda":
            bad.append((name, str(p.device)))
    if getattr(module, "is_quantized", False) or bad:
        raise RuntimeError(f"Precision/device contract violation: {bad[:12]}")
    return {"parameter_dtypes": counts, "quantized": False,
            "losses_covariance_svd_and_master_optimizer": "float32"}


def model_fingerprint(directory: str) -> dict:
    p = Path(directory).expanduser().resolve()
    if not (p / "config.json").is_file():
        raise FileNotFoundError(f"Expected a LOCAL model directory with config.json: {p}")
    config = read_json(p / "config.json")
    if config.get("quantization_config"):
        raise ValueError(f"Quantized model config forbidden: {p}")
    wanted = [x for x in p.iterdir() if x.is_file() and
              (x.suffix in {".json", ".safetensors", ".bin", ".model", ".tiktoken", ".jinja"}
               or x.name in {"vocab.txt", "merges.txt"})]
    if not any(x.suffix in {".safetensors", ".bin"} for x in wanted):
        raise ValueError(f"No model weight files in {p}")
    # Content hashes rather than directory names: moving identical files is allowed.
    files = {x.name: {"sha256": sha256(x), "bytes": x.stat().st_size} for x in sorted(wanted)}
    return {"files": files, "digest": object_hash(files)}


def audit_sources(cfg: Config) -> dict:
    data = cfg.root / "exp_code_ljj/data"
    manifest = read_json(data / "manifest.json")
    report = {"source": str(cfg.root), "files": {}, "blockers": []}
    for name in DATA_NAMES:
        p = data / f"{name}.jsonl"
        if not p.is_file():
            report["blockers"].append(f"Missing {p}"); continue
        with p.open("rb") as f:
            pointer = f.read(100).startswith(b"version https://git-lfs.github.com/spec/")
        digest = sha256(p)
        good = digest == manifest["files"][name]["sha256"]
        report["files"][name] = {"bytes": p.stat().st_size, "lfs_pointer": pointer,
                                 "sha256": digest, "matches_manifest": good,
                                 "expected_rows": manifest["files"][name]["rows"]}
        if pointer or not good:
            report["blockers"].append(f"{name}: {'LFS pointer' if pointer else 'checksum mismatch'}")
    for rel in SOURCE_FILES:
        if not (cfg.root / rel).is_file():
            report["blockers"].append(f"Missing source {rel}")
    return report


def create_lock(cfg: Config) -> dict:
    report = audit_sources(cfg)
    if report["blockers"]:
        raise RuntimeError("Cannot freeze incomplete source data: " + "; ".join(report["blockers"]))
    if not cfg.judge_model:
        raise ValueError("Set judge_model to the existing local Judge checkpoint; no silent replacement")
    # Source and model contents are hashed below. Absolute mount paths are
    # provenance, rather than a substitute for checkpoint identity.
    mod = ast.parse((cfg.root / "exp_code_ljj/repro/persuadee_simulation.py").read_text())
    values = {}
    for node in mod.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value.value
    source_default = values.get("DEFAULT_PERSUADEE_MODEL")
    if not isinstance(source_default, str) or not source_default.strip():
        raise ValueError("Missing source default Persuadee; review the experiment contract")
    if not cfg.persuadee_model:
        raise ValueError("Set persuadee_model to the locked local merged checkpoint")
    lock = {"version": VERSION, "protocol": cfg.protocol(),
            "data": {n: report["files"][n]["sha256"] for n in DATA_NAMES},
            "sources": {r: sha256(cfg.root / r) for r in SOURCE_FILES},
            "models": {n: model_fingerprint(getattr(cfg, n))
                       for n in ("base_model", "persuadee_model", "judge_model")},
            "original_paths": {n: getattr(cfg, n) for n in ("base_model", "persuadee_model", "judge_model")},
            "source_default_persuadee": source_default}
    lock["lock_id"] = object_hash(lock)
    return lock


def verify_lock(cfg: Config, lock_path: Path, models: Iterable[str] = ()) -> dict:
    lock = read_json(lock_path)
    if object_hash({k: v for k, v in lock.items() if k != "lock_id"}) != lock["lock_id"]:
        raise ValueError("Corrupt protocol lock")
    if lock.get("version") != VERSION:
        raise ValueError("Implementation version changed; create a new lock in a new work directory")
    if cfg.protocol() != lock["protocol"]:
        raise ValueError("Protocol changed; do not combine incompatible runs")
    for r, digest in lock["sources"].items():
        if sha256(cfg.root / r) != digest:
            raise ValueError(f"Locked source changed: {r}")
    for n, digest in lock["data"].items():
        if sha256(cfg.root / f"exp_code_ljj/data/{n}.jsonl") != digest:
            raise ValueError(f"Locked source data changed: {n}")
    for n in models:
        if model_fingerprint(getattr(cfg, n))["digest"] != lock["models"][n]["digest"]:
            raise ValueError(f"Locked checkpoint identity changed: {n}")
    return lock


def parse_isr(text: str) -> tuple[str, str, str]:
    a, marker, rest = text.partition("\nSelected Strategy:")
    b, utterance, c = rest.partition("\nUtterance:")
    if not a.startswith("Intent to Strategy Reasoning:") or not marker or not utterance:
        raise ValueError("Malformed original ISR target; do not silently fabricate labels")
    a = a.removeprefix("Intent to Strategy Reasoning:").strip()
    if not b.strip() or not c.strip():
        raise ValueError("Empty ISR strategy / utterance")
    return a, b.strip(), c.strip()


def parse_session(row: dict) -> dict:
    pairs = list(row.get("history") or []) + [[row["instruction"], row["output"]]]
    first = pairs[0][0]
    marker_b, marker_g = "The persuasion background: ", "\nThe persuasion goal: "
    if marker_b not in first or marker_g not in first:
        raise ValueError("Cannot recover ORIGINAL background/goal")
    background, rest = first.split(marker_b, 1)[1].split(marker_g, 1)
    goal = rest.split("\n", 1)[0]
    turns = []
    for i, (user, assistant) in enumerate(pairs):
        if i:
            turns.append({"role": "persuadee", "content": str(user).strip()})
        reasoning, strategy, utt = parse_isr(assistant)
        turns.append({"role": "persuader", "content": utt, "reasoning": reasoning, "strategy": strategy})
    meta = row["metadata"]
    if int(meta.get("persuader_turns", len(pairs))) != len(pairs):
        raise ValueError("Session turn count mismatch")
    return {"scenario_id": str(meta["scenario_id"]), "source_id": str(meta.get("source_id", "")),
            "session_id": str(meta["session_id"]), "scenario": {"background": background, "goal": goal},
            "turns": turns}


def public_history(turns: list[dict]) -> list[dict]:
    return [{"role": t["role"], "content": t["content"]} for t in turns]


def prepare_data(cfg: Config, lock: dict, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {destination}")
    tmp = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    tmp.mkdir(parents=True)
    source = cfg.root / "exp_code_ljj/data"
    manifest = read_json(source / "manifest.json")
    training_ids = set(); sessions = set(); count = 0; feedback_count = 0
    unique_feedback = set()
    try:
        with (tmp / "sft.jsonl").open("w", encoding="utf-8") as f:
            for row in jsonl(source / "isr_sft.jsonl"):
                s = parse_session(row)
                key = (s["scenario_id"], s["session_id"])
                if key in sessions:
                    raise ValueError(f"Duplicate scenario/session key: {key}")
                sessions.add(key); training_ids.add(s["scenario_id"]); count += 1
                prefix_key = object_hash([s["scenario_id"], s["scenario"]["background"]])
                for turn in s["turns"]:
                    prefix_key = object_hash([prefix_key, turn["role"], turn["content"]])
                    if turn["role"] == "persuadee":
                        feedback_count += 1
                        unique_feedback.add(prefix_key)
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        prefs = list(jsonl(source / "preference_seeds.jsonl"))
        tests = list(jsonl(source / "evaluation_points.jsonl"))
        pref_ids = {x["scenario_id"] for x in prefs}; test_ids = {x["scenario_id"] for x in tests}
        if (count != manifest["files"]["isr_sft"]["rows"] or len(training_ids) != 10000
                or len(pref_ids) != 2000 or len(prefs) != 2000 or len(test_ids) != 1000 or len(tests) != 1000):
            raise ValueError("Fixed split size does not match the supplied manifest")
        if training_ids & pref_ids or training_ids & test_ids or pref_ids & test_ids:
            raise ValueError("Scenario leakage across source partitions")
        # Independent of training seed: all methods get exactly the same dev split.
        ordered = sorted(pref_ids, key=lambda x: object_hash(["ECLP-dev-v1", x]))
        dev_ids = set(ordered[:cfg.dev_scenarios])
        partitions = {"rl_train": [x for x in prefs if x["scenario_id"] not in dev_ids],
                      "dev": [x for x in prefs if x["scenario_id"] in dev_ids], "test": tests}
        for name, points in partitions.items():
            with (tmp / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                for p in points:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
        out = {"lock_id": lock["lock_id"], "sft_sessions": count, "sft_scenarios": len(training_ids),
               "observed_feedback_pairs": feedback_count, "public_targets": count,
               "unique_public_feedback_prefixes": len(unique_feedback),
               "observation_weighting": "session_mean; repeated source prefixes counted as exposures",
               "terminal_feedback_synthesized": 0,
               "partitions": {k: len(v) for k, v in partitions.items()},
               "dev_ids": sorted(dev_ids),
               "files": {p.name: sha256(p) for p in tmp.glob("*.jsonl")}}
        atomic_json(tmp / "manifest.json", out)
        os.replace(tmp, destination); return out
    except Exception:
        # Keep diagnostics, never overwrite the originals.
        raise


def verified_prepared(path: Path, lock: dict) -> dict:
    manifest = read_json(path / "manifest.json")
    if manifest["lock_id"] != lock["lock_id"]:
        raise ValueError("Prepared data uses a different protocol lock")
    for name, digest in manifest["files"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Prepared file changed: {name}")
    return manifest


def tensor_digest(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, t in sorted(state.items()):
        t = t.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def finite(t: torch.Tensor, name: str) -> torch.Tensor:
    if not bool(torch.isfinite(t).all()):
        raise FloatingPointError(f"Nonfinite {name}")
    return t


class FP32MasterAdamW(torch.optim.Optimizer):
    """BF16 trainable weights/gradients, FP32 master weights and Adam moments.

    No FP32 neural forward fallback. Without a master copy, sub-BF16-ULP updates
    can disappear. State loading explicitly restores FP32 master/moment dtypes.
    """
    def __init__(self, params, lr: float = 1e-4, weight_decay: float = 0.01,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = finite(p.grad.detach().float(), "optimizer gradient")
                if g.is_sparse:
                    raise ValueError("Sparse gradients unsupported")
                st = self.state[p]
                if not st:
                    st.update(step=0, master=p.detach().float().clone(),
                              exp_avg=torch.zeros_like(p, dtype=torch.float32),
                              exp_avg_sq=torch.zeros_like(p, dtype=torch.float32))
                st["step"] += 1; n = st["step"]
                st["exp_avg"].mul_(b1).add_(g, alpha=1-b1)
                st["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1-b2)
                st["master"].mul_(1-group["lr"]*group["weight_decay"])
                denom = st["exp_avg_sq"].sqrt() / math.sqrt(1-b2**n) + group["eps"]
                st["master"].addcdiv_(st["exp_avg"], denom, value=-group["lr"]/(1-b1**n))
                finite(st["master"], "optimizer master")
                p.copy_(st["master"].to(p.dtype))
        return loss

    def load_state_dict(self, state_dict):
        source = copy.deepcopy(state_dict)
        result = super().load_state_dict(state_dict)
        for old_group, new_group in zip(source["param_groups"], self.param_groups):
            for idx, p in zip(old_group["params"], new_group["params"]):
                if idx not in source["state"]:
                    continue
                for k in ("master", "exp_avg", "exp_avg_sq"):
                    self.state[p][k] = source["state"][idx][k].to(device=p.device, dtype=torch.float32)
                self.state[p]["step"] = int(source["state"][idx]["step"])
        return result


class CrossBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.0)
        self.norm1 = nn.LayerNorm(width); self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(query, memory, memory, need_weights=False)
        out = self.norm1(query + out)
        return self.norm2(out + self.ff(out))


class EvidenceState(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__(); w, f = cfg.width, cfg.feature_dim
        self.slots = nn.Parameter(torch.randn(1, cfg.state_slots, w) * 0.02)
        self.read = CrossBlock(w, cfg.attention_heads)
        self.to_memory = nn.Linear(f, w)
        self.transition = nn.Sequential(nn.Linear(w+f, w*2), nn.SiLU(), nn.Linear(w*2, w))
        self.prior_norm = nn.LayerNorm(w)
        self.predict = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, f))
        self.error_projection = nn.Linear(f, w)
        self.correction = nn.Sequential(nn.Linear(w*3, w*2), nn.SiLU(), nn.Linear(w*2, w))
        self.gate = nn.Linear(w*3, w); self.posterior_norm = nn.LayerNorm(w)
        self.mode = cfg.state_mode

    def initial(self, background_tokens: torch.Tensor) -> torch.Tensor:
        return self.read(self.slots.expand(background_tokens.shape[0], -1, -1),
                         self.to_memory(background_tokens))

    def prior(self, state: torch.Tensor, spoken_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = spoken_feature.unsqueeze(1).expand(-1, state.shape[1], -1)
        prior = self.prior_norm(state + self.transition(torch.cat([state, y], -1)))
        predicted = self.predict(prior.mean(1))
        return prior, predicted

    def policy_prior(self, state: torch.Tensor, spoken_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Freeze prior parameters, retaining its actual input-state Jacobian.

        Functional calls use the same weights and arithmetic as the observation
        path. This avoids BF16 cancellation and an identity straight-through
        Jacobian, while allowing later losses to reach earlier corrections.
        """
        y = spoken_feature.detach().unsqueeze(1).expand(-1, state.shape[1], -1)
        transition = torch.func.functional_call(
            self.transition, {n: p.detach() for n, p in self.transition.named_parameters()},
            (torch.cat([state, y], -1),), strict=True)
        prior = torch.func.functional_call(
            self.prior_norm, {n: p.detach() for n, p in self.prior_norm.named_parameters()},
            (state + transition,), strict=True)
        with torch.no_grad():
            predicted = self.predict(prior.mean(1))
        return prior, predicted

    def correct(self, prior: torch.Tensor, prediction: torch.Tensor,
                observed_tokens: torch.Tensor, observed_feature: torch.Tensor) -> torch.Tensor:
        if self.mode == "no_correction":
            return prior
        memory = self.to_memory(observed_tokens)
        if self.mode == "stateless":
            return self.read(self.slots.expand(prior.shape[0], -1, -1), memory)
        obs = self.read(prior, memory)
        # Mechanistic control: keep the actual observation readout and update,
        # remove only the prediction-error channel. Unlike no_correction, this
        # does NOT make the ablation blind to the new reply.
        if self.mode == "no_error":
            error = torch.zeros_like(prior)
        else:
            error = self.error_projection(observed_feature - prediction).unsqueeze(1).expand_as(prior)
        joint = torch.cat([prior, obs, error], -1)
        return self.posterior_norm(prior + torch.sigmoid(self.gate(joint)) * self.correction(joint))

    def policy_correct(self, prior: torch.Tensor, prediction: torch.Tensor,
                       observed_tokens: torch.Tensor, observed_feature: torch.Tensor) -> torch.Tensor:
        """Update policy state without public gradients into feedback readout."""
        if self.mode == "no_correction":
            return prior
        if self.mode == "stateless":
            return self.correct(prior, prediction, observed_tokens, observed_feature).detach()
        # Match observation-path kernel selection before cutting the gradient.
        memory = self.to_memory(observed_tokens).detach()
        obs = self.read(prior.detach(), memory).detach()
        if self.mode == "no_error":
            error = torch.zeros_like(prior)
        else:
            delta = observed_feature.detach() - prediction.detach()
            error = self.error_projection(delta).unsqueeze(1).expand_as(prior)
        joint = torch.cat([prior, obs, error], -1)
        return self.posterior_norm(prior + torch.sigmoid(self.gate(joint)) * self.correction(joint))


class StrategyPolicy(nn.Module):
    def __init__(self, cfg: Config, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, cfg.width)
        self.query = nn.Parameter(torch.randn(1, cfg.strategy_slots, cfg.width) * 0.02)
        self.attn = CrossBlock(cfg.width, cfg.attention_heads)
        self.mean_head = nn.Linear(cfg.width * cfg.strategy_slots, cfg.action_dim)

    def forward(self, frozen_context: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(frozen_context.shape[0], -1, -1)
        h = self.attn(q, self.proj(frozen_context))
        return torch.tanh(self.mean_head(h.flatten(1)))


class LatentHeads(nn.Module):
    def __init__(self, cfg: Config, hidden: int, embedding_rms: float):
        super().__init__(); self.cfg = cfg; self.hidden = hidden
        self.state = EvidenceState(cfg)
        self.state_readout = nn.Linear(cfg.width, hidden)
        self.policy = StrategyPolicy(cfg, hidden)
        self.action_expand = nn.Linear(cfg.action_dim, cfg.strategy_slots * cfg.width)
        self.action_readout = nn.Linear(cfg.width, hidden)
        self.strategy_align = nn.Linear(hidden, cfg.feature_dim)
        # Fixed representation of the frozen backbone: never updated by any objective.
        gen = torch.Generator(device="cpu").manual_seed(1729)
        p = torch.randn(hidden, cfg.feature_dim, generator=gen) / math.sqrt(hidden)
        self.register_buffer("feature_projection", p)
        self.register_buffer("embedding_rms", torch.tensor(max(embedding_rms, 1e-4)))

    def normalize_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        rms = (tokens.float().square().mean(-1, keepdim=True) + 1e-8).sqrt()
        return (tokens.float() / rms * self.embedding_rms.float()).to(tokens.dtype)

    def state_tokens(self, state: torch.Tensor, allow_state_grad: bool = False) -> torch.Tensor:
        x = state if (allow_state_grad or not self.cfg.isolate_state) else state.detach()
        return self.normalize_embeddings(self.state_readout(x))

    def strategy_tokens(self, actions: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.action_expand(actions.to(self.action_expand.weight.dtype)))
        x = x.reshape(actions.shape[0], self.cfg.strategy_slots, self.cfg.width)
        return self.normalize_embeddings(self.action_readout(x))

    def frozen_feature(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden.to(self.feature_projection.dtype) @ self.feature_projection
        return F.normalize(x.float(), dim=-1).to(x.dtype)


class EffectModel(nn.Module):
    def __init__(self, context_dim: int, action_dim: int, feature_dim: int, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(context_dim+action_dim, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, feature_dim))

    def raw(self, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        return self.net(torch.cat([context, action], -1).to(dtype))

    def forward(self, context: torch.Tensor, action: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return self.raw(context, action) - self.raw(context, reference)


def gaussian_log_prob(actions: torch.Tensor, means: torch.Tensor,
                      chol: torch.Tensor) -> torch.Tensor:
    """Joint density over ALL action dimensions (sum, never mean)."""
    if actions.ndim < 1 or actions.shape[-1] != means.shape[-1]:
        raise ValueError("Action/mean dimension mismatch")
    if chol.shape[-2:] != (actions.shape[-1], actions.shape[-1]):
        raise ValueError("Cholesky/action dimension mismatch")
    for value, name in ((actions, "actions"), (means, "means"), (chol, "Cholesky")):
        finite(value, name)
    if bool((torch.diagonal(chol, dim1=-2, dim2=-1) <= 0).any()):
        raise ValueError("Cholesky diagonal must be positive")
    diff = (actions.float() - means.float()).unsqueeze(-1)
    white = torch.linalg.solve_triangular(chol.float(), diff, upper=False).squeeze(-1)
    log_det_half = torch.diagonal(chol.float(), dim1=-2, dim2=-1).log().sum(-1)
    return -0.5 * (white.square().sum(-1) + actions.shape[-1]*math.log(2*math.pi)) - log_det_half


def equal_covariance_kl(mean: torch.Tensor, reference: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
    white = torch.linalg.solve_triangular(chol.float(),
                                         (mean.float()-reference.float()).unsqueeze(-1), upper=False)
    return 0.5 * white.square().sum(dim=(-2, -1))


def exploration_cholesky(cfg: Config, mode: str, effect: EffectModel | None,
                         context: torch.Tensor, reference: torch.Tensor,
                         seed: int) -> tuple[torch.Tensor, dict]:
    m = cfg.action_dim; device = reference.device
    I = torch.eye(m, dtype=torch.float32, device=device)
    diagnostic: dict[str, Any] = {"mode": mode, "action_noise_energy": cfg.exploration_radius**2}
    if mode == "isotropic":
        cov = cfg.exploration_radius**2 * I / m
    else:
        k = cfg.effect_rank
        if mode == "random":
            g = torch.Generator(device="cpu").manual_seed(seed)
            U = torch.linalg.qr(torch.randn(m, k, generator=g, dtype=torch.float32))[0].to(device)
        elif mode == "effect":
            if effect is None:
                raise ValueError("Effect-aligned sampling requires an ACTUALLY fitted effect checkpoint")
            with torch.enable_grad(), torch.autocast(device_type=device.type, dtype=BF16):
                a0 = reference.detach().float().reshape(-1).requires_grad_(True)
                def call(a):
                    return effect(context.detach(), a.unsqueeze(0), reference.detach()).float().squeeze(0)
                J = torch.autograd.functional.jacobian(call, a0, vectorize=False).float()
            finite(J, "effect Jacobian")
            _, singular, vh = torch.linalg.svd(J, full_matrices=False)
            if k > vh.shape[0]:
                raise ValueError("effect_rank exceeds observable Jacobian rank bound")
            U = vh[:k].T
            diagnostic["singular_values"] = singular[:k].detach().cpu().tolist()
            if float(singular.max()) < 1e-7:
                raise RuntimeError("Effect Jacobian collapsed; do not silently call random exploration ELPO")
        else:
            raise ValueError(f"Unknown exploration: {mode}")
        cov = cfg.exploration_radius**2*((1-cfg.effect_alpha)*I/m + cfg.effect_alpha*(U @ U.T)/k)
    chol = torch.linalg.cholesky(cov)
    diagnostic["min_eigenvalue"] = float(torch.linalg.eigvalsh(cov).min())
    return chol, diagnostic


def clipped_policy_loss(means: torch.Tensor, actions: torch.Tensor, old_logp: torch.Tensor,
                        advantages: torch.Tensor, references: torch.Tensor, chol: torch.Tensor,
                        clip: float, beta: float) -> tuple[torch.Tensor, dict]:
    new_logp = gaussian_log_prob(actions, means, chol)
    ratio = finite(torch.exp(new_logp - old_logp.float()), "importance ratio")
    advantage = advantages.detach().float()
    surrogate = torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage)
    kl = equal_covariance_kl(means, references, chol)
    loss = -surrogate.mean() + beta*kl.mean()
    return finite(loss, "policy loss"), {"ratio_mean": float(ratio.detach().mean()),
             "ratio_min": float(ratio.detach().min()), "ratio_max": float(ratio.detach().max()),
             "kl": float(kl.detach().mean()), "advantage_mean": float(advantage.mean()),
             "clip_fraction": float(((ratio-1).abs()>clip).float().mean())}


class BaselineBridge:
    """Import ORIGINAL prompt, cleanup and local Persuadee implementation unchanged."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        base = cfg.root / "exp_code_ljj"
        sys.path.insert(0, str(base))
        os.environ["DAILYPERSUASION_PROFILE"] = "paper10k"
        self.sim = importlib.import_module("repro.simulation")
        self.persuadee = importlib.import_module("repro.persuadee_simulation")
        # Source evaluate_baselines parses sys.argv on import: avoid our CLI conflicting.
        saved = sys.argv
        try:
            sys.argv = [saved[0]]
            self.evaluator = importlib.import_module("evaluate_baselines")
        finally:
            sys.argv = saved
        if Path(self.sim.__file__).resolve() != (base / "repro/simulation.py").resolve():
            raise RuntimeError("A different project's 'repro' module is already imported")
        self.user_endpoint = None; self.judge_endpoint = None
        self.ledger = None

    def _invoke(self, kind: str, endpoint, **kwargs):
        audit = self.ledger.track(kind, {"request_sha256": object_hash(kwargs)}) if self.ledger else contextlib.nullcontext()
        with audit:
            return endpoint.complete(**kwargs)

    def actor_prompt(self, point: dict, history: list[dict]) -> str:
        return self.persuadee.dialogue_prompt(point, history)

    def clean_actor(self, raw: str, point: dict) -> str:
        return self.persuadee.clean_utterance(raw, is_isr=False,
                                            speaker_name=point["scenario"].get("persuader"))

    def respond(self, point: dict, history: list[dict], seed: int) -> tuple[str, str]:
        if self.user_endpoint is None:
            self.user_endpoint = self.sim.LocalTransformersEndpoint(
                model=self.cfg.persuadee_model, device=self.cfg.persuadee_device, quantization="none")
            precision_report(self.user_endpoint.model)
        system, prompt = self.sim.user_prompt(point, history)
        with independent_rng(seed, self.user_endpoint.devices):
            raw = self._invoke("persuadee", self.user_endpoint, system=system, user=prompt, temperature=self.cfg.temperature,
                 max_tokens=self.cfg.max_new_tokens, top_p=self.cfg.top_p, top_k=self.cfg.top_k)
        text = self.persuadee.clean_utterance(raw, is_isr=False,
                                             speaker_name=point["scenario"].get("persuadee"))
        return text, raw

    def score(self, row: dict, round_number: int) -> dict:
        if self.judge_endpoint is None:
            if not self.cfg.judge_model:
                raise ValueError("judge_model was not configured")
            self.judge_endpoint = self.sim.LocalTransformersEndpoint(
                model=self.cfg.judge_model, device=self.cfg.judge_device, quantization="none")
            precision_report(self.judge_endpoint.model)
        prompt = self.evaluator.SUCCESS_PROMPTS[self.cfg.judge_prompt_version](
            row, row["trajectory"][:round_number*2], round_number)
        raws = []
        for _ in range(self.cfg.judge_parse_retries + 1):
            # Identical rubric on every retry. No alternate, more permissive prompt.
            raw = self._invoke("judge", self.judge_endpoint,
                system="You are an evidence-based judge of whether the persuadee accepted the stated goal.",
                user=prompt, temperature=0.0, max_tokens=self.cfg.judge_max_new_tokens, json_mode=True)
            raws.append(raw)
            try:
                success = parse_judge_boolean(raw)
                return {"round": round_number, "success": success, "parse_ok": True, "raw_outputs": raws}
            except ValueError:
                continue
        return {"round": round_number, "success": None, "parse_ok": False, "raw_outputs": raws}

    def close(self) -> None:
        for name in ("user_endpoint", "judge_endpoint"):
            endpoint = getattr(self, name)
            if endpoint is not None:
                close = getattr(endpoint, "close", None)
                if callable(close): close()
                setattr(self, name, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_judge_boolean(raw: str) -> bool:
    """Only a JSON Boolean; no permissive prose inference. Report unresolved separately."""
    if "</think>" in raw.lower():
        raw = re.split(r"</think>", raw, maxsplit=1, flags=re.I)[1]
    decoder = json.JSONDecoder(); labels = []
    for i, char in enumerate(raw):
        if char != "{": continue
        try:
            obj, _ = decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and type(obj.get("success")) is bool:
            labels.append(obj["success"])
    if not labels or len(set(labels)) != 1:
        raise ValueError("Unparseable or contradictory success Booleans")
    return labels[0]


@contextlib.contextmanager
def independent_rng(seed: int, devices: Iterable[int]):
    """Separate action RNG and public/environment generation RNG streams."""
    ds = sorted(set(devices))
    with torch.random.fork_rng(devices=ds):
        torch.random.default_generator.manual_seed(seed)
        for d in ds:
            with torch.cuda.device(d): torch.cuda.manual_seed(seed)
        yield



def public_identity(point: dict) -> str:
    """No private goal: changing an unobserved goal must not change evidence state."""
    return object_hash([point.get("scenario_id", ""), point["scenario"]["background"]])


def scenario_seed(global_seed: int, scenario_id: str, stream: str, replica: int = 0) -> int:
    """Stable under dataset order, sharding and resume. Candidate is not a CRN key."""
    return int(object_hash(["eclp-v2-rng", global_seed, scenario_id, stream, replica])[:15], 16)


class CallLedger:
    """Append BEFORE a request, so crashes cannot hide already attempted calls.

    Counts endpoint.complete invocations, not any internal retry of an external
    endpoint. A start without a terminal record remains unknown, never zero cost.
    """
    def __init__(self, path: Path):
        self.path = Path(path)

    @contextlib.contextmanager
    def track(self, kind: str, metadata: dict | None = None):
        ident = object_hash([kind, os.getpid(), time.time_ns()])
        t = time.monotonic()
        append_jsonl(self.path, {"call_id":ident,"kind":kind,"event":"started", **(metadata or {})})
        try:
            yield
        except BaseException as error:
            append_jsonl(self.path, {"call_id":ident,"kind":kind,"event":"failed",
                                    "error_type":type(error).__name__,"seconds":time.monotonic()-t})
            raise
        else:
            append_jsonl(self.path, {"call_id":ident,"kind":kind,"event":"completed",
                                    "seconds":time.monotonic()-t})

    def summary(self) -> dict:
        rows = list(jsonl(self.path)) if self.path.exists() else []
        starts = {r["call_id"]:r for r in rows if r["event"]=="started"}
        ends = {r["call_id"]:r for r in rows if r["event"] in {"failed","completed"}}
        return {"attempted":len(starts), "completed":sum(r["event"]=="completed" for r in ends.values()),
                "failed":sum(r["event"]=="failed" for r in ends.values()),
                "unresolved_after_interruption":len(set(starts)-set(ends)),
                "by_kind":{k:sum(r["kind"]==k for r in starts.values()) for k in sorted({r["kind"] for r in starts.values()})}}


def guarded_mean_step(policy: nn.Module, optimizer: FP32MasterAdamW,
                      batch: dict, cfg: Config) -> dict:
    """One actual BF16 mean-policy PPO step; restore weights AND optimizer on KL violation.

    Recorded actions and Cholesky factors stay fixed. Post-step monitoring matters:
    a pre-step check alone cannot prevent the final large update from escaping.
    This uses standard PPO/rollback engineering, not a novel RL theorem.
    """
    device = next(policy.parameters()).device
    actions, chol = batch["actions"].detach(), batch["chol"].detach()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=BF16):
        means = policy(batch["context"].detach())
    loss, info = clipped_policy_loss(means, actions, batch["old_logp"].detach(),
                                    batch["advantages"].detach(), batch["reference"].detach(),
                                    chol, cfg.clip_epsilon, cfg.kl_beta)
    finite(loss, "PPO loss")
    if info["kl"] > cfg.max_kl:
        return {**info,"accepted":False,"reason":"pre_step_reference_kl","loss":float(loss.detach())}
    weights = {k:v.detach().clone() for k,v in policy.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    loss.backward()
    params = [p for p in policy.parameters() if p.requires_grad]
    grad = torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad(), torch.autocast(device_type=device.type,dtype=BF16):
        new_mean = policy(batch["context"].detach())
    post_ref = float(equal_covariance_kl(new_mean,batch["reference"],chol).mean())
    post_old = float(equal_covariance_kl(new_mean,batch["old_mean"],chol).mean())
    accepted = math.isfinite(post_ref) and math.isfinite(post_old) and max(post_ref,post_old) <= cfg.max_kl
    if not accepted:
        policy.load_state_dict(weights)
        optimizer.load_state_dict(optimizer_state)
    optimizer.zero_grad(set_to_none=True)
    return {**info,"loss":float(loss.detach()),"grad_norm":float(grad),"accepted":accepted,
            "reason":"accepted" if accepted else "post_step_kl_rollback",
            "post_reference_kl":post_ref if math.isfinite(post_ref) else None,
            "post_behavior_kl":post_old if math.isfinite(post_old) else None}


@dataclasses.dataclass
class LiveState:
    B: torch.Tensor
    history: list[dict]
    prediction: torch.Tensor | None = None
    identity: str = ""  # public scenario identity; deliberately excludes private goal
    policy_B: torch.Tensor | None = None
    policy_prediction: torch.Tensor | None = None

    def branch(self) -> "LiveState":
        return LiveState(
            B=self.B.detach().clone(), history=copy.deepcopy(self.history),
            prediction=None if self.prediction is None else self.prediction.detach().clone(),
            identity=self.identity,
            policy_B=None if self.policy_B is None else self.policy_B.detach().clone(),
            policy_prediction=None if self.policy_prediction is None else self.policy_prediction.detach().clone(),
        )


class PromptRepetitionPenalty:
    """Apply the actor's penalty to public prompt and generated token IDs.

    HF inputs_embeds-only generation has no prompt IDs in its processor state.
    Keep continuous slots out of the vocabulary history, but retain the same
    public prompt penalty as an ordinary input_ids-based actor.
    """
    def __init__(self, prompt_ids: torch.Tensor, penalty: float):
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError("Expected one public prompt for single-actor generation")
        if not math.isfinite(penalty) or penalty <= 0:
            raise ValueError("Repetition penalty must be finite and positive")
        self.prompt_ids = prompt_ids
        self.penalty = penalty

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        history = torch.cat([self.prompt_ids.expand(input_ids.shape[0], -1), input_ids], dim=1)
        selected = torch.gather(scores, 1, history)
        selected = torch.where(selected < 0, selected * self.penalty, selected / self.penalty)
        return scores.scatter(1, history, selected)


class ECLPRunner:
    """Single local base LLM, generation LoRA, evidence state and strategy heads."""
    def __init__(self, cfg: Config, bridge: BaselineBridge, checkpoint: Path | None = None,
                 sft: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, PeftModel
        # PEFT 0.19 eagerly considers bitsandbytes if installed. This
        # experiment explicitly forbids quantization, so disable only its
        # optional dispatcher and keep the native Linear LoRA path.
        if cfg.quantization == "none":
            import peft.tuners.lora.model as _lora_model
            _lora_model.is_bnb_available = lambda: False
            _lora_model.is_bnb_4bit_available = lambda: False
        self.cfg = cfg; self.bridge = bridge; self.device = cuda_bf16(cfg.actor_device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, local_files_only=True, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=BF16, local_files_only=True, low_cpu_mem_usage=True,
            attn_implementation="sdpa").to(self.device)
        if base.config.model_type not in {"qwen2", "llama", "mistral"}:
            raise ValueError("Actor implementation targets Qwen2.5/Llama/Mistral decoder-only backbones")
        base.requires_grad_(False)
        if checkpoint:
            self.llm = PeftModel.from_pretrained(base, str(checkpoint / "adapter"), is_trainable=sft,
                                                autocast_adapter_dtype=False)
        else:
            self.llm = get_peft_model(base, LoraConfig(
                r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias="none",
                task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                       "gate_proj", "up_proj", "down_proj"]),
                autocast_adapter_dtype=False)
        self.llm.to(dtype=BF16)
        self.llm.config.use_cache = False
        hidden = int(base.config.hidden_size)
        with torch.no_grad():
            emb = self.llm.get_input_embeddings().weight[:4096].float()
            rms = float(emb.square().mean().sqrt())
        self.heads = LatentHeads(cfg, hidden, rms).to(device=self.device, dtype=BF16)
        self.reference_policy = None
        if checkpoint:
            metadata = read_json(checkpoint / "checkpoint.json")
            if metadata["architecture"] != cfg.architecture():
                raise ValueError("Checkpoint architecture/training contract differs from config")
            self.heads.load_state_dict(load_tensors(checkpoint / "heads.pt"))
            self.reference_policy = copy.deepcopy(self.heads.policy)
            self.reference_policy.load_state_dict(load_tensors(checkpoint / "reference_policy.pt"))
            self.reference_policy.eval().requires_grad_(False)
        from collections import OrderedDict
        self.feature_cache = OrderedDict()
        self.set_stage("sft" if sft else "eval")
        self.precision = {"llm": precision_report(self.llm), "heads": precision_report(self.heads)}

    def set_stage(self, stage: str) -> None:
        self.llm.requires_grad_(False); self.heads.requires_grad_(False)
        self.llm.eval(); self.heads.eval()
        if stage == "sft":
            for name, p in self.llm.named_parameters():
                if "lora_" in name: p.requires_grad_(True)
            self.heads.requires_grad_(True); self.heads.train(); self.llm.train()
            if self.cfg.gradient_checkpointing:
                self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        elif stage == "rl":
            self.heads.policy.requires_grad_(True)
        elif stage != "eval":
            raise ValueError(stage)
        if stage != "sft":
            self.llm.gradient_checkpointing_disable()
        self.stage = stage

    def _decoder(self):
        return self.llm.get_base_model().model

    def _hidden(self, *, ids: torch.Tensor | None = None,
                embeddings: torch.Tensor | None = None, allow_grad: bool = False) -> torch.Tensor:
        # Always disable LoRA for evidence and state-conditioned context.
        prior_mode = self.llm.training
        self.llm.eval()
        try:
            with self.llm.disable_adapter(), (contextlib.nullcontext() if allow_grad else torch.no_grad()):
                kwargs = {"input_ids": ids} if embeddings is None else {"inputs_embeds": embeddings}
                n = ids.shape[-1] if embeddings is None else embeddings.shape[1]
                kwargs.update(attention_mask=torch.ones(1, n, device=self.device, dtype=torch.long),
                              position_ids=torch.arange(n, device=self.device).unsqueeze(0),
                              use_cache=False, return_dict=True)
                return self._decoder()(**kwargs).last_hidden_state
        finally:
            self.llm.train(prior_mode)

    def encode_public(self, background: str, history: list[dict], role: str,
                      text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool ONLY the actual reply span; never pool the entire history as target."""
        key = object_hash([background, history, role, text])
        if key in self.feature_cache:
            tokens, feature = self.feature_cache.pop(key)
            self.feature_cache[key] = (tokens, feature)
            return tokens.to(self.device), feature.to(self.device)
        tok = self.tokenizer
        bg = tok.encode("Public background: " + background + "\n", add_special_tokens=False)
        hist = tok.encode("\n".join(t["role"] + ": " + t["content"] for t in history)
                          + "\n" + role + ": ", add_special_tokens=False)
        span = tok.encode(text, add_special_tokens=False)[:self.cfg.feature_response_tokens]
        if not span:
            raise ValueError("Cannot encode an empty actual utterance")
        budget = self.cfg.feature_context_tokens - len(bg) - len(span)
        if budget < 0:
            raise ValueError("Background exceeds feature budget; do not silently delete source context")
        prefix = bg + (hist[-budget:] if budget else [])
        ids = torch.tensor([prefix + span], device=self.device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=BF16):
            h = self._hidden(ids=ids)[:, len(prefix):]
            tokens = self.heads.frozen_feature(h).detach()
            feature = F.normalize(tokens.float().mean(1), dim=-1).to(BF16).detach()
        if self.cfg.feature_cache_entries:
            self.feature_cache[key] = (tokens.cpu(), feature.cpu())
            while len(self.feature_cache) > self.cfg.feature_cache_entries:
                self.feature_cache.popitem(last=False)
        return tokens, feature

    def initialize(self, point: dict) -> LiveState:
        bg = point["scenario"]["background"]
        tokens, _ = self.encode_public(bg, [], "background", bg)
        state = self.heads.state.initial(tokens)
        return LiveState(B=state, policy_B=state.detach(), history=[], identity=public_identity(point))

    def consume(self, live: LiveState, point: dict, turn: dict) -> torch.Tensor | None:
        if live.identity and live.identity != public_identity(point):
            raise ValueError("Cross-scenario state reuse is forbidden; initialize/reset the state")
        role, text = turn["role"], turn["content"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty observed utterance; do not synthesize a replacement")
        tokens, feature = self.encode_public(point["scenario"]["background"], live.history, role, text)
        loss = None
        if role == "persuader":
            if live.prediction is not None:
                raise ValueError("Two persuader turns without an observed persuadee reply")
            live.B, live.prediction = self.heads.state.prior(live.B, feature)
            if live.policy_B is None:
                raise ValueError("Missing policy state; initialize the live state before consuming turns")
            live.policy_B, live.policy_prediction = self.heads.state.policy_prior(live.policy_B, feature)
        elif role == "persuadee":
            if live.prediction is None or live.policy_prediction is None or live.policy_B is None:
                raise ValueError("Persuadee feedback without a preceding observed persuader action")
            loss = F.mse_loss(live.prediction.float(), feature.float())
            live.B = self.heads.state.correct(live.B, live.prediction, tokens, feature)
            live.policy_B = self.heads.state.policy_correct(
                live.policy_B, live.policy_prediction, tokens, feature)
            live.prediction = None
            live.policy_prediction = None
        else:
            raise ValueError(f"Unexpected public role: {role}")
        live.history.append({"role": role, "content": text})
        return loss

    def replay(self, point: dict, history: list[dict]) -> tuple[LiveState, list[torch.Tensor]]:
        live = self.initialize(point); losses = []
        for turn in history:
            loss = self.consume(live, point, turn)
            if loss is not None: losses.append(loss)
        if live.prediction is not None or live.policy_prediction is not None:
            raise ValueError("Decision context must end before the next persuader utterance")
        return live, losses

    def prompt_ids(self, point: dict, history: list[dict]) -> list[int]:
        kept = public_history(history)
        while True:
            prompt = self.bridge.actor_prompt(point, kept)
            ids = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": self.cfg.actor_system_prompt},
                 {"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
            if hasattr(ids, "input_ids"): ids = ids.input_ids
            if len(ids) <= self.cfg.max_prompt_tokens:
                return ids
            if len(kept) > 2:
                kept = kept[2:]  # preserve background, goal and the latest exchange
            else:
                raise ValueError("Prompt over budget after dropping old exchanges; preserve goal and newest feedback")

    def context(self, point: dict, live: LiveState, *, detach_state: bool | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = torch.tensor([self.prompt_ids(point, live.history)], device=self.device)
        prompt_emb = self.llm.get_input_embeddings()(ids).detach()
        if detach_state is None:
            detach_state = self.cfg.isolate_state
        state_for_policy = live.policy_B if detach_state else live.B
        if state_for_policy is None:
            raise ValueError("Missing policy state; initialize/replay the public history first")
        bt = self.heads.state_tokens(state_for_policy, allow_state_grad=self.stage == "sft")
        prefix = torch.cat([prompt_emb, bt], 1)
        hidden = self._hidden(embeddings=prefix, allow_grad=self.stage == "sft")
        ctx = hidden[:, -self.cfg.state_slots:]
        effect_context = torch.cat([self.heads.frozen_feature(ctx.detach()).mean(1),
                                    live.B.detach().mean(1)], -1)
        return prefix, ctx, effect_context

    def utterance_loss(self, prefix: torch.Tensor, action: torch.Tensor, target: str) -> tuple[torch.Tensor, int]:
        s = self.heads.strategy_tokens(action)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)[:self.cfg.max_target_tokens-1]
        target_ids.append(self.tokenizer.eos_token_id)
        tid = torch.tensor([target_ids], device=self.device)
        x = torch.cat([prefix, s, self.llm.get_input_embeddings()(tid)], 1)
        start = prefix.shape[1] + s.shape[1] - 1
        logits = self.llm(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], device=self.device, dtype=torch.long),
                          position_ids=torch.arange(x.shape[1], device=self.device).unsqueeze(0),
                          use_cache=False, return_dict=True).logits[:, start:start+tid.shape[1]]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tid.reshape(-1))
        return finite(loss, "utterance NLL"), len(target_ids)

    def session_losses(self, session: dict) -> tuple[dict[str, torch.Tensor], dict]:
        point = {"scenario": session["scenario"], "scenario_id": session["scenario_id"]}; history = public_history(session["turns"][:-1])
        live, observation_losses = self.replay(point, history)
        prefix, ctx, _ = self.context(point, live)
        action = self.heads.policy(ctx)
        target = session["turns"][-1]
        utt, ntok = self.utterance_loss(prefix, action, target["content"])
        _, strategy_target = self.encode_public("", [], "strategy", target["strategy"])
        s = self.heads.strategy_tokens(action)
        aligned = self.heads.strategy_align(s.mean(1))
        strat = (1-F.cosine_similarity(aligned.float(), strategy_target.float(), dim=-1)).mean()
        # Private ISR reasoning is an auxiliary teacher target. It trains the
        # implicit state/latent path during SFT but is never exposed at inference.
        private_prefix, private_ctx, _ = self.context(point, live, detach_state=False)
        private_action = self.heads.policy(private_ctx)
        reasoning = target.get("reasoning", "").strip()
        if not reasoning:
            raise ValueError("Missing ISR reasoning target; do not silently drop the private supervision")
        reason_loss, reason_tokens = self.utterance_loss(private_prefix, private_action, reasoning)
        obs = torch.stack(observation_losses).mean() if observation_losses else utt.new_zeros(())
        return {"utterance": utt, "observation": obs, "strategy": strat, "reasoning": reason_loss}, {
            "supervised_tokens": ntok, "reasoning_tokens": reason_tokens, "feedback_pairs": len(observation_losses),
            "scenario_id": session["scenario_id"], "state_norm": float(live.B.detach().float().norm()),
            "action_norm": float(action.detach().float().norm())}

    @torch.no_grad()
    def speak(self, point: dict, live: LiveState, seed: int, action: torch.Tensor | None = None,
              policy: StrategyPolicy | None = None) -> tuple[dict, torch.Tensor, torch.Tensor]:
        prefix, ctx, eff = self.context(point, live)
        if action is None:
            action = (policy or self.heads.policy)(ctx)
            if self.cfg.test_action == "sample":
                g = torch.Generator(device="cpu").manual_seed(seed + 700000001)
                noise = torch.randn(action.shape, generator=g).to(self.device)
                action = action.float() + self.cfg.exploration_radius / math.sqrt(self.cfg.action_dim) * noise
        st = self.heads.strategy_tokens(action)
        embeddings = torch.cat([prefix, st], 1)
        stop = self.llm.generation_config.eos_token_id or self.tokenizer.eos_token_id
        penalty = float(self.llm.generation_config.repetition_penalty)
        prompt_ids = torch.tensor([self.prompt_ids(point, live.history)], device=self.device)
        processors = [PromptRepetitionPenalty(prompt_ids, penalty)] if penalty != 1.0 else []
        with independent_rng(seed, [self.device.index]), torch.autocast(device_type="cuda", dtype=BF16):
            generated = self.llm.generate(
                inputs_embeds=embeddings, attention_mask=torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long),
                max_new_tokens=self.cfg.max_new_tokens, do_sample=self.cfg.temperature>0,
                temperature=self.cfg.temperature, top_p=self.cfg.top_p, top_k=self.cfg.top_k,
                repetition_penalty=1.0, logits_processor=processors,
                eos_token_id=stop, pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True, return_dict_in_generate=True)
        # With inputs_embeds-only decoder generation, sequences contain NEW ids,
        # not virtual prefix ids. Assert that contract instead of blindly slicing.
        ids = generated.sequences[0]
        if len(ids) > self.cfg.max_new_tokens:
            raise RuntimeError("Transformers inputs_embeds generation contract changed; audit before slicing")
        raw = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        text = self.bridge.clean_actor(raw, point)
        return {"role": "persuader", "content": text, "raw_output": raw,
                "private_marker_leak": bool(PRIVATE_MARKERS.search(text)),
                "actor_repetition_penalty": penalty,
                "actor_repetition_penalty_scope": "public_prompt_and_generated_ids",
                "public_tokens": len(ids), "input_tokens_including_latents": embeddings.shape[1]}, ctx, eff

    @torch.no_grad()
    def effect_feedback_feature(self, point: dict, initial_history: list[dict], reply: str) -> torch.Tensor:
        # Different candidate y must NOT change the embedding of an identical u.
        # Keep the pre-action context fixed across branches. State correction and
        # Judge still see the complete actual conversation including y.
        _, feature = self.encode_public(point["scenario"]["background"], initial_history, "persuadee", reply)
        return feature.squeeze(0)

    @torch.no_grad()
    def rollout(self, point: dict, initial: LiveState, horizon: int, first_action: torch.Tensor,
                execution_seed: int, continuation: StrategyPolicy) -> tuple[dict, torch.Tensor]:
        live = initial.branch(); initial_history = copy.deepcopy(live.history)
        trajectory = []; feedback_features = []
        for t in range(horizon):
            action = first_action if t == 0 else None
            spoken, _, _ = self.speak(point, live, execution_seed+t*2, action=action, policy=continuation)
            spoken["round"] = t+1; trajectory.append(spoken)
            self.consume(live, point, spoken)
            text, raw = self.bridge.respond(point, live.history, execution_seed+t*2+1)
            reply = {"round": t+1, "role": "persuadee", "content": text, "raw_output": raw}
            trajectory.append(reply)
            if t == 0:
                feedback_features.append(self.effect_feedback_feature(point, initial_history, text).cpu())
            self.consume(live, point, reply)
        row = {"scenario_id": point["scenario_id"], "scenario": point["scenario"],
               "source_id": point.get("source_id", ""), "session_id": point.get("session_id", ""),
               "target_turn": point.get("target_turn", 1), "reference": point.get("reference", point.get("gold_response", "")),
               "initial_history": initial_history, "trajectory": trajectory, "final_history": live.history,
               "rounds": horizon, "persuadee_model": self.cfg.persuadee_model,
               "compute_dtype": "bfloat16", "quantization": "none", "execution_seed": execution_seed,
               "effect_target_context": "fixed_pre_action_prefix_reply_span_v2",
               "format_valid": not any(x.get("private_marker_leak", False) for x in trajectory)}
        return row, feedback_features[0]

    def close(self):
        del self.llm; self.feature_cache.clear()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


# -------------------------- Checkpoints / experiment commands --------------------------

def environment_versions() -> dict:
    versions = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available()}
    for name in ("transformers", "peft", "accelerate", "openai"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    if torch.cuda.is_available():
        versions["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return versions


def capture_rng() -> dict:
    return {"python": random.getstate(), "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"]); torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the same visible CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])


def trainable_parameters(runner: ECLPRunner) -> list[nn.Parameter]:
    return [p for module in (runner.llm, runner.heads) for p in module.parameters() if p.requires_grad]


def frozen_executor_hash(runner: ECLPRunner) -> str:
    heads = {n: t for n, t in runner.heads.state_dict().items() if not n.startswith("policy.")}
    lora = {n: p for n, p in runner.llm.named_parameters() if "lora_" in n}
    return object_hash([tensor_digest(heads), tensor_digest(lora)])


def save_checkpoint(runner: ECLPRunner, out: Path, lock: dict, stage: str, step: int,
                    optimizer: FP32MasterAdamW | None = None, trainer_state: dict | None = None,
                    provenance: dict | None = None) -> Path:
    if (provenance or {}).get("implementation_sha256", IMPLEMENTATION_SHA256) != IMPLEMENTATION_SHA256:
        raise ValueError("Checkpoint provenance disagrees with this process's implementation")
    name = f"checkpoint-{step:06d}"
    destination = out / name
    if destination.exists():
        raise FileExistsError(f"Checkpoint exists, refusing overwrite: {destination}")
    tmp = out / (name + f".tmp-{os.getpid()}"); tmp.mkdir(parents=True)
    runner.llm.save_pretrained(tmp / "adapter", safe_serialization=True)
    runner.tokenizer.save_pretrained(tmp / "adapter")
    save_tensors(tmp / "heads.pt", {k: v.detach().cpu() for k, v in runner.heads.state_dict().items()})
    ref = runner.heads.policy if stage == "sft" or runner.reference_policy is None else runner.reference_policy
    save_tensors(tmp / "reference_policy.pt", {k: v.detach().cpu() for k, v in ref.state_dict().items()})
    if optimizer is not None:
        save_tensors(tmp / "train_state.pt", {"optimizer": optimizer.state_dict(),
                    "trainer": trainer_state or {}, "rng": capture_rng()})
    files = {str(p.relative_to(tmp)): sha256(p) for p in sorted(tmp.rglob("*")) if p.is_file()}
    metadata = {"version": VERSION, "lock_id": lock["lock_id"], "stage": stage, "step": step,
                "architecture": runner.cfg.architecture(), "files": files,
                "implementation_sha256": IMPLEMENTATION_SHA256,
                "executor_hash": frozen_executor_hash(runner),
                "policy_hash": tensor_digest(runner.heads.policy.state_dict()),
                "reference_policy_hash": tensor_digest(ref.state_dict()),
                "environment": environment_versions(), "precision": runner.precision,
                "provenance": provenance or {}}
    metadata["checkpoint_id"] = object_hash(metadata)
    atomic_json(tmp / "checkpoint.json", metadata)
    os.replace(tmp, destination)
    atomic_json(out / "last.json", {"checkpoint": str(destination.resolve()),
                                   "checkpoint_id": metadata["checkpoint_id"]})
    return destination


def verify_checkpoint(checkpoint: Path, lock: dict) -> dict:
    m = read_json(checkpoint / "checkpoint.json")
    if m.get("version") != VERSION:
        raise ValueError("Checkpoint implementation version differs; retrain the repaired method in a new directory")
    if m["lock_id"] != lock["lock_id"]:
        raise ValueError("Checkpoint uses a different protocol lock")
    if m.get("implementation_sha256") != IMPLEMENTATION_SHA256:
        raise ValueError("Implementation changed since checkpoint; review migration or retrain, not silent resume")
    source_hash = m.get("provenance", {}).get("implementation_sha256")
    if source_hash is not None and source_hash != m["implementation_sha256"]:
        raise ValueError("Checkpoint implementation hash disagrees with its training provenance")
    if object_hash({k: v for k, v in m.items() if k != "checkpoint_id"}) != m["checkpoint_id"]:
        raise ValueError("Corrupt checkpoint metadata")
    for p, digest in m["files"].items():
        if sha256(checkpoint / p) != digest:
            raise ValueError(f"Checkpoint content mismatch: {p}")
    return m


def immutable_run_manifest(out: Path, meta: dict) -> None:
    # Mutation is intentional: downstream object_hash(meta) covers the exact code/runtime.
    meta["implementation_sha256"] = IMPLEMENTATION_SHA256
    meta["runtime"] = environment_versions()
    out.mkdir(parents=True, exist_ok=True)
    p = out / "run.json"
    if p.exists():
        if read_json(p) != meta:
            raise ValueError(f"Resume provenance differs in {out}; use a new run directory")
    else:
        atomic_json(p, meta)


def reconcile_sft_log(path: Path, saved_step: int) -> None:
    """Preserve, but exclude from active history, logged steps after an unsaved crash."""
    if not path.exists():return
    rows=list(jsonl(path))
    steps=[r["step"] for r in rows]
    if steps!=sorted(set(steps)):raise ValueError("Training log steps duplicated/out of order; manually audit")
    committed=[r for r in rows if r["step"]<=saved_step]
    lost=[r for r in rows if r["step"]>saved_step]
    if not lost:return
    backup=path.with_name(f"uncommitted_after_step_{saved_step}_{time.time_ns()}.jsonl")
    shutil.copy2(path,backup)  # full original log retained, including actual spent compute
    tmp=path.with_name(path.name+f".resume-{os.getpid()}")
    with tmp.open("w",encoding="utf-8") as f:
        for row in committed:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    atomic_json(backup.with_suffix(".json"),{"resume_from_step":saved_step,"uncommitted_steps":[r["step"] for r in lost],
                "original_log_sha256":sha256(backup),"note":"Compute was spent, but model updates were not in a committed checkpoint; keep in resource accounting"})


def command_sft(args, cfg: Config, lock: dict) -> None:
    prepared = verified_prepared(args.prepared, lock)
    source = args.prepared / "sft.jsonl"
    rows = list(jsonl(source))
    # Pilot samples are whole scenarios, not silently sliced across dialogue history.
    if args.limit_scenarios:
        ids = sorted({r["scenario_id"] for r in rows}, key=lambda x: object_hash(["sft-pilot", x]))
        chosen = set(ids[:args.limit_scenarios])
        rows = [r for r in rows if r["scenario_id"] in chosen]
    if not rows:
        raise ValueError("No SFT sessions")
    if args.resume and not args.checkpoint:
        raise ValueError("--resume requires --checkpoint")
    if args.checkpoint:
        meta = verify_checkpoint(args.checkpoint, lock)
        if meta["stage"] != "sft":
            raise ValueError("SFT resume/initialization requires an SFT checkpoint in this release")
    run = {"version": VERSION, "lock_id": lock["lock_id"], "stage": "sft", "seed": cfg.seed,
           "architecture": cfg.architecture(), "data_sha256": prepared["files"]["sft.jsonl"],
           "sessions": len(rows), "scenario_ids": sorted({r["scenario_id"] for r in rows}),
           "learning_rate": args.lr, "accumulation": args.accumulation}
    with exclusive_writer(args.output):
        if not args.resume and ((args.output/"train.jsonl").exists() or (args.output/"last.json").exists()):
            raise FileExistsError("Existing SFT run requires explicit --resume or a new output directory")
        if args.resume and (args.output/"last.json").exists():
            if read_json(args.output/"last.json")["checkpoint_id"] != meta["checkpoint_id"]:
                raise ValueError("Resume must use the latest committed checkpoint in this run; fork older checkpoints into a new directory")
        immutable_run_manifest(args.output, run)
        seed_all(cfg.seed); bridge = BaselineBridge(cfg)
        runner = ECLPRunner(cfg, bridge, checkpoint=args.checkpoint, sft=True)
        params = trainable_parameters(runner)
        optimizer = FP32MasterAdamW(params, lr=args.lr, weight_decay=0.01)
        step = epoch = position = 0
        if args.resume:
            state = load_tensors(args.checkpoint / "train_state.pt")
            tr = state["trainer"]
            if tr["run_hash"] != object_hash(run):
                raise ValueError("Resume data, seed or optimizer contract changed")
            optimizer.load_state_dict(state["optimizer"]); restore_rng(state["rng"])
            step, epoch, position = tr["step"], tr["epoch"], tr["next_position"]
            reconcile_sft_log(args.output/"train.jsonl", step)
        if args.max_steps <= step:
            raise ValueError("max_steps is an absolute target and must exceed the saved step")
        last_saved = -1
        start_time = time.time()
        try:
            while epoch < args.epochs and step < args.max_steps:
                order = list(range(len(rows))); random.Random(cfg.seed+epoch).shuffle(order)
                while position < len(order) and step < args.max_steps:
                    batch = order[position:position+args.accumulation]
                    optimizer.zero_grad(set_to_none=True)
                    sums = {k: 0.0 for k in ("utterance", "observation", "strategy", "reasoning")}
                    tokens = feedback_pairs = 0
                    for idx in batch:
                        with torch.autocast(device_type="cuda", dtype=BF16):
                            losses, stats = runner.session_losses(rows[idx])
                            loss = (losses["utterance"] + cfg.obs_weight*losses["observation"] + cfg.strategy_weight*losses["strategy"] + cfg.reasoning_weight*losses["reasoning"])
                        finite(loss, "SFT total").div(len(batch)).backward()
                        for k in sums: sums[k] += float(losses[k].detach())/len(batch)
                        tokens += stats["supervised_tokens"]; feedback_pairs += stats["feedback_pairs"]
                    grad = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0, error_if_nonfinite=True)
                    optimizer.step(); position += len(batch); step += 1
                    row = {"step": step, "epoch": epoch, "next_position": position, **sums,
                           "supervised_tokens": tokens, "feedback_pairs": feedback_pairs,
                           "grad_norm": float(grad), "seconds_since_start": time.time()-start_time,
                           "max_memory_allocated": torch.cuda.max_memory_allocated(runner.device)}
                    append_jsonl(args.output / "train.jsonl", row)
                    print(json.dumps(row), flush=True)
                    if step % args.save_every == 0:
                        tr = {"step": step, "epoch": epoch, "next_position": position, "run_hash": object_hash(run)}
                        save_checkpoint(runner, args.output, lock, "sft", step, optimizer, tr, run)
                        last_saved = step
                if position >= len(order): epoch += 1; position = 0
            if step != last_saved:
                tr = {"step": step, "epoch": epoch, "next_position": position, "run_hash": object_hash(run)}
                save_checkpoint(runner, args.output, lock, "sft", step, optimizer, tr, run)
        finally:
            bridge.close()


def points_for_split(prepared: Path, split: str, limit: int = 0, offset: int = 0) -> list[dict]:
    if min(limit, offset) < 0: raise ValueError("Selection limit/offset must be nonnegative")
    points = list(jsonl(prepared / f"{split}.jsonl"))
    ids=[p["scenario_id"] for p in points]
    if len(ids)!=len(set(ids)): raise ValueError("Duplicate scenario in selection")
    if split != "test":
        points.sort(key=lambda p: object_hash(["pilot-order-v1", p["scenario_id"]]))
    points=points[offset:]
    return points[:limit] if limit else points


def load_effect(path: Path, cfg: Config, lock: dict, actor_meta: dict, device: torch.device) -> tuple[EffectModel, dict]:
    m = read_json(path / "effect.json")
    if object_hash({k:v for k,v in m.items() if k != "effect_id"}) != m["effect_id"]:
        raise ValueError("Corrupt effect metadata")
    if m.get("effect_target_context") != "fixed_pre_action_prefix_reply_span_v2":
        raise ValueError("Effect target semantics changed; v1 predictor cannot be reused")
    if not m.get("beats_zero_predictor", False):
        raise ValueError("Effect predictor failed held-out zero-predictor gate; not deployable")
    if m["lock_id"] != lock["lock_id"] or m["executor_hash"] != actor_meta["executor_hash"]:
        raise ValueError("Effect model trained under a different state/decoder/action mapping")
    if sha256(path / "effect.pt") != m["sha256"]:
        raise ValueError("Effect checkpoint checksum mismatch")
    effect = EffectModel(cfg.feature_dim+cfg.width, cfg.action_dim, cfg.feature_dim, cfg.width)
    effect.load_state_dict(load_tensors(path / "effect.pt"))
    effect.to(device=device, dtype=BF16).eval().requires_grad_(False)
    precision_report(effect)
    return effect, m


def command_collect(args, cfg: Config, lock: dict) -> None:
    verified_prepared(args.prepared, lock); meta = verify_checkpoint(args.checkpoint, lock)
    if not 1 <= args.horizon <= 5 or args.candidates < 2 or args.repeats < 1:
        raise ValueError("Require 1<=H<=5, G>=2, M>=1")
    points = points_for_split(args.prepared, "rl_train", args.limit, getattr(args,"offset",0))
    if not points: raise ValueError("Empty training rollout selection")
    run = {"version": VERSION, "lock_id": lock["lock_id"], "checkpoint_id": meta["checkpoint_id"],
           "executor_hash": meta["executor_hash"], "seed": cfg.seed, "split": "rl_train",
           "scenario_ids": [p["scenario_id"] for p in points], "horizon": args.horizon,
           "candidates": args.candidates, "repeats": args.repeats, "reward": args.reward,
           "exploration": args.exploration, "exploration_radius": cfg.exploration_radius,
           "effect_alpha": cfg.effect_alpha, "effect_rank": cfg.effect_rank,
           "effect_checkpoint": str(args.effect.resolve()) if args.effect else None,
           "effect_id": read_json(args.effect / "effect.json")["effect_id"] if args.effect else None,
           "test_action_used_for_continuation": cfg.test_action}
    with exclusive_writer(args.output):
        immutable_run_manifest(args.output, run)
        seed_all(cfg.seed); bridge = BaselineBridge(cfg)
        bridge.ledger = CallLedger(args.output / "calls.jsonl")
        runner = ECLPRunner(cfg, bridge, args.checkpoint)
        effect = None
        if args.exploration == "effect":
            if not args.effect: raise ValueError("--effect is required")
            effect, _ = load_effect(args.effect, cfg, lock, meta, runner.device)
        continuation = copy.deepcopy(runner.heads.policy).eval().requires_grad_(False)
        records_path = args.output / "episodes.jsonl"
        existing = {}
        if records_path.exists():
            for row in jsonl(records_path):
                key = row["episode_id"]
                if key in existing: raise ValueError(f"Duplicate episode_id: {key}")
                existing[key] = row
        completed = 0; calls = 0
        try:
            for gi, point in enumerate(points):
                gid = point["scenario_id"]
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", gid) or gid in {".",".."}:
                    raise ValueError("Scenario ID cannot be used safely as an artifact name")
                group_path = args.output / "groups" / f"{gid}.pt"
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=BF16):
                    initial, _ = runner.replay(point, public_history(point["history"]))
                    prefix, ctx, eff_ctx = runner.context(point, initial)
                    mu = runner.heads.policy(ctx).float()
                    ref = runner.reference_policy(ctx).float()
                if group_path.exists():
                    group = load_tensors(group_path)
                    if group["run_hash"] != object_hash(run):
                        raise ValueError("Stale group tensor file")
                    if not torch.equal(group["context"], ctx.detach().cpu()):
                        raise ValueError("Fixed context recomputation differs on resume; investigate precision/library drift")
                    actions = group["actions"].to(runner.device)
                    chol = group["chol"].to(runner.device)
                else:
                    chol, cov_diag = exploration_cholesky(cfg, args.exploration, effect, eff_ctx, ref,
                                                         scenario_seed(cfg.seed, gid, "covariance"))
                    gen = torch.Generator(device="cpu").manual_seed(scenario_seed(cfg.seed, gid, "latent_action"))
                    noise = torch.randn(args.candidates, cfg.action_dim, generator=gen).to(runner.device)
                    actions = mu + noise @ chol.T
                    old_logp = gaussian_log_prob(actions, mu, chol)
                    with torch.no_grad():
                        sdelta = runner.heads.strategy_tokens(actions) - runner.heads.strategy_tokens(ref)
                        candidate_noise_s = runner.heads.strategy_tokens(actions)-runner.heads.strategy_tokens(mu)
                    group = {"run_hash": object_hash(run), "scenario_id": gid,
                             "context": ctx.detach().cpu(), "effect_context": eff_ctx.detach().cpu(),
                             "state": initial.B.detach().cpu(), "actions": actions.detach().cpu(),
                             "old_mean": mu.detach().cpu(), "reference_mean": ref.detach().cpu(),
                             "old_logp": old_logp.detach().cpu(), "chol": chol.cpu(), "covariance": cov_diag,
                             "strategy_change_rms_vs_reference": sdelta.float().square().mean((1,2)).sqrt().cpu(),
                             "strategy_noise_rms_vs_mean": candidate_noise_s.float().square().mean((1,2)).sqrt().cpu(),
                             "feedback": {}, "episode_ids": [], "complete": False}
                    save_tensors(group_path, group)
                # Reference action is never treated as an on-policy sampled action.
                all_actions = [ref] + [a.unsqueeze(0) for a in actions]
                for ai, action in enumerate(all_actions):
                    label = "ref" if ai == 0 else str(ai-1)
                    for repeat in range(args.repeats):
                        eid = f"{gid}:{label}:{repeat}"
                        seed = scenario_seed(cfg.seed, gid, "execution", repeat)
                        feature_path = args.output / "feedback" / f"{gid}_{label}_{repeat}.pt"
                        if eid in existing:
                            if args.reward == "interaction":
                                if not feature_path.is_file():
                                    raise ValueError("Episode exists without its committed feedback tensor")
                                if sha256(feature_path) != existing[eid]["feedback_sha256"]:
                                    raise ValueError("Feedback feature changed after episode commit")
                                group["feedback"][eid] = load_tensors(feature_path)
                            continue
                        if args.reward == "interaction":
                            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=BF16):
                                row, feature = runner.rollout(point, initial, args.horizon, action, seed, continuation)
                            save_tensors(feature_path, feature)
                            group["feedback"][eid] = feature
                            row.update(feedback_file=str(feature_path.relative_to(args.output)),
                                       feedback_sha256=sha256(feature_path))
                            # Cache reference-answer NLL during collection so the
                            # offline Judge process can add a small PPL shaping term.
                            reference = point.get("gold_response", point.get("reference", ""))
                            if reference:
                                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=BF16):
                                    reference_nll, _ = runner.utterance_loss(prefix, action, reference)
                                row["reference_nll"] = float(reference_nll)
                                row["ppl_shaping_available"] = True
                            else:
                                row["reference_nll"] = None
                                row["ppl_shaping_available"] = False
                            calls += args.horizon
                        else:
                            reference = point.get("gold_response", point.get("reference"))
                            if not reference: raise ValueError("PPL control needs an existing TRAIN reference")
                            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=BF16):
                                nll, _ = runner.utterance_loss(prefix, action, reference)
                            reward = -math.exp(float(nll))
                            if not math.isfinite(reward): raise FloatingPointError("PPL overflow")
                            row = {"scenario_id": gid, "reward": reward, "reward_type": "negative_ppl",
                                   "nll": float(nll), "format_valid": True}
                        row.update(episode_id=eid, action_index=ai-1, repeat=repeat,
                                   is_reference=ai==0, run_hash=object_hash(run), lock_id=lock["lock_id"])
                        append_jsonl(records_path, row); existing[eid] = row
                        save_tensors(group_path, group)
                expected_ids = [f"{gid}:{label}:{r}" for label in ["ref"]+[str(i) for i in range(args.candidates)]
                                for r in range(args.repeats)]
                group["episode_ids"] = expected_ids
                group["complete"] = all(e in existing for e in expected_ids)
                save_tensors(group_path, group); completed += 1
                print(f"Collected {completed}/{len(points)} contexts, {len(existing)} episodes", flush=True)
            expected_all = {f"{p['scenario_id']}:{label}:{r}" for p in points
                            for label in ["ref"]+[str(i) for i in range(args.candidates)]
                            for r in range(args.repeats)}
            if set(existing) != expected_all:
                raise ValueError("Collection has missing/unexpected episode IDs")
            atomic_json(args.output / "collection.json", {"complete": completed==len(points),
                        "contexts": completed, "episodes": len(existing), "run_hash": object_hash(run),
                        "new_persuadee_calls_this_invocation": calls,
                        "total_persuadee_calls": len(existing)*args.horizon if args.reward=="interaction" else 0,
                        "episodes_sha256": sha256(records_path),
                        "groups": {p.name: sha256(p) for p in sorted((args.output/"groups").glob("*.pt"))},
                        "request_ledger": bridge.ledger.summary()})
        finally:
            bridge.close()


def verify_collection(directory: Path, lock: dict) -> tuple[dict, dict]:
    run = read_json(directory / "run.json"); complete = read_json(directory / "collection.json")
    if run["lock_id"] != lock["lock_id"] or not complete["complete"]:
        raise ValueError("Collection incomplete or uses a different protocol")
    if object_hash(run) != complete["run_hash"] or sha256(directory/"episodes.jsonl") != complete["episodes_sha256"]:
        raise ValueError("Rollout provenance/episodes changed")
    for name, digest in complete["groups"].items():
        if sha256(directory / "groups" / name) != digest: raise ValueError(f"Group changed: {name}")
    return run, complete


def within_h_success(judgments: list[dict]) -> bool | None:
    """Exact three-valued OR; do not impute unparseable replies as failures."""
    if not judgments: raise ValueError("No prefix judgments")
    if any(j["parse_ok"] and j["success"] is True for j in judgments): return True
    if any(not j["parse_ok"] for j in judgments): return None
    return False


def command_score(args, cfg: Config, lock: dict) -> None:
    # Offline scoring: ONLY the Judge is resident, not Actor or Persuadee.
    run, collection = verify_collection(args.rollouts, lock)
    if run["reward"] != "interaction": raise ValueError("PPL controls already contain their numeric rewards")
    episodes = list(jsonl(args.rollouts / "episodes.jsonl"))
    with exclusive_writer(args.output):
        manifest = {"lock_id": lock["lock_id"], "rollout_run_hash": object_hash(run),
                    "episodes_sha256": collection["episodes_sha256"], "judge": lock["models"]["judge_model"]["digest"],
                    "prompt_version": cfg.judge_prompt_version, "parse_rule": "json-boolean-v1",
                    "reward_aggregation":"within_h_plus_ppl", "ppl_shaping_weight": cfg.ppl_shaping_weight}
        immutable_run_manifest(args.output, manifest)
        existing = {r["episode_id"]: r for r in jsonl(args.output/"scores.jsonl")} if (args.output/"scores.jsonl").exists() else {}
        reference_nll = {r["scenario_id"]: r.get("reference_nll") for r in episodes
                        if r.get("is_reference") and r.get("reference_nll") is not None}
        bridge = BaselineBridge(cfg)
        bridge.ledger = CallLedger(args.output / "calls.jsonl")
        try:
            for row in episodes:
                if row["episode_id"] in existing: continue
                judgments = [bridge.score(row, r) for r in range(1,run["horizon"]+1)]
                # Match the original success-within-five metric, restricted to H.
                # A known success resolves the union even if another prefix is unknown.
                accepted = within_h_success(judgments)
                interaction_reward = float(accepted and row["format_valid"]) if accepted is not None else None
                ref_nll = reference_nll.get(row["scenario_id"])
                candidate_nll = row.get("reference_nll")
                if ref_nll is not None and candidate_nll is not None:
                    # Lower reference NLL is better; tanh bounds shaping so
                    # interaction success remains the dominant signal.
                    ppl_delta = math.tanh(float(ref_nll) - float(candidate_nll))
                else:
                    ppl_delta = 0.0
                reward = (interaction_reward + cfg.ppl_shaping_weight * ppl_delta
                          if interaction_reward is not None else None)
                result = {"episode_id": row["episode_id"], "reward": reward,
                          "interaction_reward": interaction_reward, "ppl_delta": ppl_delta,
                          "ppl_shaping_weight": cfg.ppl_shaping_weight,
                          "judgments": judgments, "reward_aggregation":"within_h_plus_ppl",
                          "format_valid": row["format_valid"], "run_hash": object_hash(manifest)}
                append_jsonl(args.output/"scores.jsonl", result); existing[row["episode_id"]] = result
            atomic_json(args.output/"scoring.json", {"complete": len(existing)==len(episodes),
                        "episodes": len(existing), "unresolved": sum(x["reward"] is None for x in existing.values()),
                        "scores_sha256": sha256(args.output/"scores.jsonl"), "run_hash": object_hash(manifest),
                        "request_ledger": bridge.ledger.summary()})
        finally: bridge.close()


def load_reward_map(rollouts: Path, scores: Path | None, run: dict) -> dict[str, float | None]:
    if run["reward"] == "ppl":
        return {r["episode_id"]: r["reward"] for r in jsonl(rollouts/"episodes.jsonl")}
    if scores is None: raise ValueError("Interaction update requires --scores")
    sm = read_json(scores/"run.json"); done = read_json(scores/"scoring.json")
    if (sm["rollout_run_hash"] != object_hash(run) or not done["complete"] or
            object_hash(sm) != done["run_hash"] or sha256(scores/"scores.jsonl") != done["scores_sha256"]):
        raise ValueError("Scored rollout mismatch/incomplete")
    items = list(jsonl(scores/"scores.jsonl"))
    expected_ids = {r["episode_id"] for r in jsonl(rollouts/"episodes.jsonl")}
    if {r["episode_id"] for r in items} != expected_ids:
        raise ValueError("Scored episode set does not match collection")
    out = {r["episode_id"]: r["reward"] for r in items}
    if len(items) != len(out): raise ValueError("Duplicate score IDs")
    return out


def group_advantages(group: dict, rewards: dict, candidates: int, repeats: int,
                     scale: float = 1.0) -> torch.Tensor:
    if not math.isfinite(scale) or scale <= 0: raise ValueError("Advantage scale must be finite, positive, fixed before the batch")
    gid = group["scenario_id"]
    ref = [rewards[f"{gid}:ref:{j}"] for j in range(repeats)]
    if any(x is None for x in ref): raise ValueError("Unresolved reference")
    vals = []
    for i in range(candidates):
        current = [rewards[f"{gid}:{i}:{j}"] for j in range(repeats)]
        if any(x is None for x in current): raise ValueError("Unresolved candidate")
        vals.append(sum(float(x)-float(y) for x,y in zip(current,ref))/repeats/scale)
    return finite(torch.tensor(vals, dtype=torch.float32), "paired advantages")


def command_update(args, cfg: Config, lock: dict) -> None:
    run, collection = verify_collection(args.rollouts, lock); meta = verify_checkpoint(args.checkpoint, lock)
    if run["checkpoint_id"] != meta["checkpoint_id"]:
        raise ValueError("Stale rollout: update MUST start from the checkpoint that generated it")
    if not 0 <= args.max_unresolved_fraction <= 1:
        raise ValueError("max_unresolved_fraction must be in [0,1]")
    rewards = load_reward_map(args.rollouts, args.scores, run)
    groups = [load_tensors(args.rollouts/"groups"/name) for name in sorted(collection["groups"])]
    accepted = []; rejected = []
    for g in groups:
        if not g["complete"]: raise ValueError("Incomplete root group")
        if any(e not in rewards or rewards[e] is None for e in g["episode_ids"]):
            rejected.append(g["scenario_id"])
        else: accepted.append(g)
    if len(rejected) / max(1, len(groups)) > args.max_unresolved_fraction:
        raise RuntimeError(f"Unresolved groups {len(rejected)}/{len(groups)}: repair scoring before RL")
    if not accepted: raise ValueError("No valid groups")
    advantages = torch.cat([group_advantages(g, rewards, run["candidates"], run["repeats"], args.advantage_scale) for g in accepted])
    if advantages.abs().max().item() == 0:
        raise RuntimeError("All paired advantages are zero. Log sparse-reward diagnostic; do not call a KL-only step successful RL")
    seed_all(cfg.seed); bridge = BaselineBridge(cfg)
    runner = ECLPRunner(cfg, bridge, args.checkpoint); runner.set_stage("rl")
    params = trainable_parameters(runner)
    optimizer = FP32MasterAdamW(params, lr=args.lr, weight_decay=0.0)
    executor_before = frozen_executor_hash(runner)
    current = []
    for g in accepted:
        G = run["candidates"]
        ctx = g["context"].expand(G,-1,-1)
        current.append({"context": ctx, "actions": g["actions"], "old_logp": g["old_logp"],
                        "reference": g["reference_mean"].expand(G,-1),
                        "chol": g["chol"].unsqueeze(0).expand(G,-1,-1),
                        "old_mean": g["old_mean"].expand(G,-1)})
    data = {k: torch.cat([x[k] for x in current]) for k in current[0]}
    data["advantages"] = advantages
    with exclusive_writer(args.output):
        if (args.output/"update.json").exists() or (args.output/"train.jsonl").exists():
            raise FileExistsError("Use a fresh update directory. Interrupted short PPO updates restart from the unchanged parent in a NEW directory")
        immutable_run_manifest(args.output, {"version": VERSION, "lock_id": lock["lock_id"],
            "stage": "elpo", "rollout_run_hash": object_hash(run), "checkpoint_id": meta["checkpoint_id"],
            "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
            "advantage_scale": args.advantage_scale, "clip": cfg.clip_epsilon, "kl_beta": cfg.kl_beta,
            "rejected_groups": rejected})
        # Validate behavior density before modifying anything.
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=BF16):
            max_mean_error = 0.0; max_logp_error = 0.0
            for start in range(0,len(advantages),args.batch_size):
                b = {k:v[start:start+args.batch_size].to(runner.device) for k,v in data.items()}
                means = runner.heads.policy(b["context"])
                max_mean_error = max(max_mean_error, float((means.float()-b["old_mean"].float()).abs().max()))
                logp = gaussian_log_prob(b["actions"],means,b["chol"])
                max_logp_error = max(max_logp_error,float((logp-b["old_logp"]).abs().max()))
        if max_mean_error > 0.005 or max_logp_error > 0.1:
            raise RuntimeError(f"Behavior replay mismatch: mean {max_mean_error}, logp {max_logp_error}. "
                               "Try batch_size=1 and audit kernels; do NOT rewrite old logp")
        step = 0; stop = False
        for epoch in range(args.epochs):
            order = list(range(len(advantages))); random.Random(cfg.seed+epoch).shuffle(order)
            for start in range(0,len(order),args.batch_size):
                indexes = order[start:start+args.batch_size]
                b = {k:v[indexes].to(runner.device) for k,v in data.items()}
                info = guarded_mean_step(runner.heads.policy, optimizer, b, cfg)
                if not info["accepted"]:
                    append_jsonl(args.output/"rejected_steps.jsonl", {"step_before":step,"epoch":epoch,**info})
                    stop = True; break
                step += 1
                append_jsonl(args.output/"train.jsonl", {"step":step,"epoch":epoch,**info})
            if stop: break
        if not step: raise RuntimeError("No policy update passed trust-region checks")
        if frozen_executor_hash(runner) != executor_before:
            raise RuntimeError("ELPO changed a frozen state/decoder/action mapping")
        provenance = {"rollout_run_hash":object_hash(run),"parent_checkpoint_id":meta["checkpoint_id"],
                      "exploration":run["exploration"],"reward":run["reward"],
                      "accepted_groups":len(accepted),"rejected_groups":rejected,
                      "behavior_mean_max_error":max_mean_error,"behavior_logp_max_error":max_logp_error,
                      "kl_early_stop":stop,"executor_unchanged":True}
        save_checkpoint(runner,args.output,lock,"rl",step,optimizer,{"step":step},provenance)
        atomic_json(args.output/"update.json",provenance)
    bridge.close()


def command_effect(args,cfg:Config,lock:dict)->None:
    run,collection = verify_collection(args.rollouts,lock)
    if run["reward"]!="interaction": raise ValueError("Effect learning requires actual feedback")
    groups=[load_tensors(args.rollouts/"groups"/name) for name in sorted(collection["groups"])]
    if len(groups)<10: raise ValueError("Need at least 10 training contexts for an effect train/validation split")
    # The split is by scenario, NEVER by candidate or repetition.
    order=sorted(range(len(groups)),key=lambda i:object_hash(["effect-val-v1",groups[i]["scenario_id"]]))
    val_ids=set(order[:max(2,len(order)//5)])
    train=[];valid=[]
    for idx,g in enumerate(groups):
        gid=g["scenario_id"]
        references=torch.stack([g["feedback"][f"{gid}:ref:{r}"].float() for r in range(run["repeats"])]).mean(0)
        for i in range(run["candidates"]):
            responses=torch.stack([g["feedback"][f"{gid}:{i}:{r}"].float() for r in range(run["repeats"])]).mean(0)
            item=(g["effect_context"].squeeze(0),g["actions"][i],g["reference_mean"].squeeze(0),responses-references)
            (valid if idx in val_ids else train).append(item)
    device=cuda_bf16(cfg.actor_device);seed_all(cfg.seed)
    model=EffectModel(cfg.feature_dim+cfg.width,cfg.action_dim,cfg.feature_dim,cfg.width).to(device=device,dtype=BF16)
    optimizer=FP32MasterAdamW(model.parameters(),lr=args.lr,weight_decay=0.01)
    def tensors(items): return [torch.stack([x[i] for x in items]).to(device) for i in range(4)]
    tx,ta,tr,ty=tensors(train);vx,va,vr,vy=tensors(valid)
    best=float("inf");best_state=None;zero_mse=float(vy.float().square().mean())
    with exclusive_writer(args.output):
        if (args.output/"effect.json").exists() or (args.output/"train.jsonl").exists():
            raise FileExistsError("Effect fitting requires a fresh output directory; do not overwrite diagnostics")
        immutable_run_manifest(args.output,{"lock_id":lock["lock_id"],"rollout_run_hash":object_hash(run),
                                           "epochs":args.epochs,"lr":args.lr,"seed":cfg.seed,
                                           "batch_size":args.batch_size})
        for epoch in range(args.epochs):
            model.train();order=torch.randperm(len(train),device=device)
            for start in range(0,len(train),args.batch_size):
                b=order[start:start+args.batch_size];optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda",dtype=BF16): pred=model(tx[b],ta[b],tr[b])
                loss=F.mse_loss(pred.float(),ty[b].float());finite(loss,"effect loss").backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            model.eval()
            with torch.no_grad(),torch.autocast(device_type="cuda",dtype=BF16):
                pred=model(vx,va,vr);mse=float(F.mse_loss(pred.float(),vy.float()))
            append_jsonl(args.output/"train.jsonl",{"epoch":epoch,"val_mse":mse,"zero_predictor_mse":zero_mse})
            if mse<best:
                best=mse;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        save_tensors(args.output/"effect.pt",best_state)
        metadata={"version":VERSION,"lock_id":lock["lock_id"],"executor_hash":run["executor_hash"],
                  "source_rollout_run_hash":object_hash(run),"train_contexts":len(groups)-len(val_ids),
                  "source_scenario_ids":[g["scenario_id"] for g in groups],
                  "effect_target_context":"fixed_pre_action_prefix_reply_span_v2",
                  "validation_contexts":len(val_ids),"validation_scenario_ids":[groups[i]["scenario_id"] for i in sorted(val_ids)],
                  "val_mse":best,"zero_predictor_mse":zero_mse,"beats_zero_predictor":best<zero_mse,
                  "sha256":sha256(args.output/"effect.pt"),"precision":precision_report(model)}
        metadata["effect_id"]=object_hash(metadata);atomic_json(args.output/"effect.json",metadata)
        if not metadata["beats_zero_predictor"]:
            raise RuntimeError("Effect predictor failed its zero-predictor gate. Artifact saved as FAILED diagnostic; do not deploy it")



def effect_prediction_metrics(predicted: torch.Tensor, observed: torch.Tensor) -> dict:
    """Observed vector prediction diagnostics, NOT a persuasion outcome metric."""
    if predicted.shape!=observed.shape or predicted.ndim!=2 or not predicted.numel():
        raise ValueError("Expected aligned nonempty [examples,features] tensors")
    p=finite(predicted.detach().float(),"effect predictions")
    y=finite(observed.detach().float(),"observed effects")
    mse=float((p-y).square().mean());zero=float(y.square().mean())
    valid=(y.norm(dim=1)>1e-8)&(p.norm(dim=1)>1e-8)
    cosine=float(F.cosine_similarity(p[valid],y[valid],dim=-1).mean()) if valid.any() else None
    x=p.norm(dim=-1);t=y.norm(dim=-1);x=x-x.mean();t=t-t.mean()
    denom=x.norm()*t.norm()
    correlation=float((x*t).sum()/denom) if denom>1e-12 else None
    return {"examples":len(y),"mse":mse,"zero_predictor_mse":zero,
            "relative_mse_gain":1-mse/zero if zero>1e-12 else None,
            "nonzero_pair_cosine":cosine,"effect_norm_pearson":correlation,
            "not_a_success_rate":True}


def command_check_effect(args,cfg:Config,lock:dict)->None:
    run,collection=verify_collection(args.rollouts,lock)
    if run["reward"]!="interaction":raise ValueError("Need observed replies, not PPL scores")
    groups=[load_tensors(args.rollouts/"groups"/name) for name in sorted(collection["groups"])]
    meta=read_json(args.effect/"effect.json")
    overlap=set(meta["source_scenario_ids"])&{g["scenario_id"] for g in groups}
    if overlap:raise ValueError("Effect validation contains fit/selection scenarios; collect a disjoint training-pool offset")
    device=cuda_bf16(cfg.actor_device)
    model,meta=load_effect(args.effect,cfg,lock,{"executor_hash":run["executor_hash"]},device)
    predictions=[];observed=[];scenario_metrics=[]
    for g in groups:
        gid=g["scenario_id"];refs=torch.stack([g["feedback"][f"{gid}:ref:{j}"].float() for j in range(run["repeats"])]).mean(0)
        ys=torch.stack([torch.stack([g["feedback"][f"{gid}:{i}:{j}"].float() for j in range(run["repeats"])]).mean(0)-refs for i in range(run["candidates"])])
        with torch.no_grad(),torch.autocast(device_type=device.type,dtype=BF16):
            ps=model(g["effect_context"].to(device).expand(run["candidates"],-1),g["actions"].to(device),g["reference_mean"].to(device).expand(run["candidates"],-1))
        predictions.append(ps.cpu());observed.append(ys)
        scenario_metrics.append({"scenario_id":gid,**effect_prediction_metrics(ps.cpu(),ys)})
    result=effect_prediction_metrics(torch.cat(predictions),torch.cat(observed))
    result.update(effect_id=meta["effect_id"],rollout_run_hash=object_hash(run),
                  scenario_count=len(groups),scenarios_disjoint_from_fitting_and_selection=True,
                  per_scenario=scenario_metrics,
                  note="Do not count candidates as independent scenarios; this is a diagnostic, not geometry causality or task success.")
    if args.output.exists():raise FileExistsError("Do not overwrite effect-check results")
    atomic_json(args.output,result);print(json.dumps(result,ensure_ascii=False,indent=2))


def command_validate(args, cfg: Config, lock: dict) -> None:
    """Offline DEV teacher-forcing: no new Persuadee/Judge calls, never on test."""
    verified_prepared(args.prepared, lock); meta=verify_checkpoint(args.checkpoint,lock)
    points=points_for_split(args.prepared,"dev",args.limit)
    seed_all(cfg.seed);bridge=BaselineBridge(cfg);runner=ECLPRunner(cfg,bridge,args.checkpoint)
    run={"lock_id":lock["lock_id"],"checkpoint_id":meta["checkpoint_id"],"split":"dev",
         "seed":cfg.seed,"scenario_ids":[p["scenario_id"] for p in points]}
    with exclusive_writer(args.output):
        immutable_run_manifest(args.output,run)
        path=args.output/"validation.jsonl"
        existing={r["scenario_id"]:r for r in jsonl(path)} if path.exists() else {}
        try:
            for point in points:
                if point["scenario_id"] in existing:continue
                target=point.get("gold_response")
                if not target:raise ValueError("DEV requires existing source gold_response")
                with torch.no_grad(),torch.autocast(device_type="cuda",dtype=BF16):
                    live,obs=runner.replay(point,public_history(point["history"]))
                    prefix,ctx,_=runner.context(point,live)
                    loss,ntok=runner.utterance_loss(prefix,runner.heads.policy(ctx),target)
                row={"scenario_id":point["scenario_id"],"nll":float(loss),"target_tokens":ntok,
                     "observation_squared_error_sum":sum(float(x) for x in obs),"observation_pairs":len(obs),
                     "run_hash":object_hash(run)}
                append_jsonl(path,row);existing[row["scenario_id"]]=row
            if set(existing)!=set(run["scenario_ids"]):raise ValueError("Validation IDs missing or duplicated")
            values=list(existing.values());tokens=sum(r["target_tokens"] for r in values)
            pairs=sum(r["observation_pairs"] for r in values)
            nll=sum(r["nll"]*r["target_tokens"] for r in values)/tokens
            result={"run":run,"n":len(values),"token_weighted_nll":nll,"token_weighted_ppl":math.exp(nll),
                    "observed_pair_mse":sum(r["observation_squared_error_sum"] for r in values)/pairs if pairs else None,
                    "observation_pairs":pairs,"new_persuadee_calls":0,"judge_calls":0,
                    "validation_sha256":sha256(path)}
            atomic_json(args.output/"summary.json",result);print(json.dumps(result,ensure_ascii=False,indent=2))
        finally:bridge.close()


def command_generate(args,cfg:Config,lock:dict)->None:
    verified_prepared(args.prepared,lock);meta=verify_checkpoint(args.checkpoint,lock)
    if args.split=="test" and not args.allow_test:
        raise ValueError("Test is not a development set. Freeze checkpoint/config first, then explicitly pass --allow-test")
    points=points_for_split(args.prepared,args.split,args.limit)
    full_points=list(jsonl(args.prepared/f"{args.split}.jsonl"))
    original_index={p["scenario_id"]:i for i,p in enumerate(full_points)}
    run={"version":VERSION,"lock_id":lock["lock_id"],"checkpoint_id":meta["checkpoint_id"],
         "method":args.method,"split":args.split,"seed":cfg.seed,"test_action":cfg.test_action,
         "test_action_radius":cfg.exploration_radius if cfg.test_action=="sample" else None,
         "scenario_ids":[p["scenario_id"] for p in points],"judge_used_during_generation":False}
    with exclusive_writer(args.output):
        immutable_run_manifest(args.output,run)
        output=args.output/"generation.jsonl"
        existing={r["scenario_id"]:r for r in jsonl(output)} if output.exists() else {}
        seed_all(cfg.seed);bridge=BaselineBridge(cfg);bridge.ledger=CallLedger(args.output/"calls.jsonl");runner=ECLPRunner(cfg,bridge,args.checkpoint)
        try:
            for point in points:
                if point["scenario_id"] in existing: continue
                index=original_index[point["scenario_id"]]
                # Scenario-based seeds survive sharding/order changes.
                seed=scenario_seed(cfg.seed, point["scenario_id"], "generation")
                start=time.time()
                with torch.no_grad(),torch.autocast(device_type="cuda",dtype=BF16):
                    live,_=runner.replay(point,public_history(point["history"]))
                    _,ctx,_=runner.context(point,live);action=runner.heads.policy(ctx)
                    if cfg.test_action=="sample":
                        g=torch.Generator(device="cpu").manual_seed(seed+700000001)
                        action=action.float()+cfg.exploration_radius/math.sqrt(cfg.action_dim)*torch.randn(action.shape,generator=g).to(runner.device)
                    row,_=runner.rollout(point,live,cfg.rounds,action,seed,runner.heads.policy)
                row.update(index=index,method=args.method,persuader_model=str(args.checkpoint.resolve()),
                           lock_id=lock["lock_id"],run_hash=object_hash(run),
                           elapsed_seconds=time.time()-start,checkpoint_id=meta["checkpoint_id"])
                append_jsonl(output,row);existing[point["scenario_id"]]=row
                print(f"Generated {len(existing)}/{len(points)}",flush=True)
            atomic_json(args.output/"generation_summary.json",{"complete":len(existing)==len(points),
                        "n":len(existing),"generation_sha256":sha256(output),"run_hash":object_hash(run),
                        "private_marker_turns":sum(t.get("private_marker_leak",False) for r in existing.values() for t in r["trajectory"]),
                        "neural_dtype":"bfloat16","judge_calls":0,"request_ledger":bridge.ledger.summary()})
        finally:bridge.close()


def judgment_summary(rows:list[dict],expected_n:int)->dict:
    if expected_n < 1 or len(rows)!=expected_n: raise ValueError("Fixed denominator requires ALL expected judgment rows")
    by_id={r["scenario_id"]:r for r in rows}
    if len(by_id)!=expected_n: raise ValueError("Duplicate judged scenario")
    successes=final=unknown=final_unknown=0
    for r in rows:
        js=r["judgments"]
        if len(js) != 5: raise ValueError("Expected five prefix judgments per scenario")
        known_success=any(x["parse_ok"] and x["success"] for x in js)
        unresolved=any(not x["parse_ok"] for x in js)
        successes+=known_success;unknown+=(unresolved and not known_success)
        final+=bool(js[-1]["parse_ok"] and js[-1]["success"])
        final_unknown+=not js[-1]["parse_ok"]
    return {"n":expected_n,"success_within_five_lower_percent":100*successes/expected_n,
            "success_within_five_upper_percent":100*(successes+unknown)/expected_n,
            "final_success_lower_percent":100*final/expected_n,
            "final_success_upper_percent":100*(final+final_unknown)/expected_n,
            "unresolved_scenarios_affecting_ever_success":unknown,"unresolved_final":final_unknown,
            "parse_unresolved_turns":sum(not j["parse_ok"] for r in rows for j in r["judgments"])}


def command_evaluate(args,cfg:Config,lock:dict)->None:
    genrun=read_json(args.generation/"run.json");done=read_json(args.generation/"generation_summary.json")
    if (genrun["lock_id"]!=lock["lock_id"] or not done["complete"] or
            done["run_hash"]!=object_hash(genrun) or
            sha256(args.generation/"generation.jsonl")!=done["generation_sha256"]):
        raise ValueError("Generation missing/corrupt/protocol mismatch")
    rows=list(jsonl(args.generation/"generation.jsonl"))
    if (len(rows)!=len(genrun["scenario_ids"]) or
            {r["scenario_id"] for r in rows}!=set(genrun["scenario_ids"])):
        raise ValueError("Generation scenario IDs missing or duplicated")
    with exclusive_writer(args.output):
        run={"lock_id":lock["lock_id"],"generation_run_hash":object_hash(genrun),
             "generation_sha256":done["generation_sha256"],"method":genrun["method"],
             "split":genrun["split"],"seed":genrun["seed"],"expected_n":len(rows),
             "test_action":genrun["test_action"],"test_action_radius":genrun["test_action_radius"],
             "judge_model":str(Path(cfg.judge_model).expanduser().resolve()),
             "judge_model_hash":model_fingerprint(cfg.judge_model)["digest"],
             "prompt_version":cfg.judge_prompt_version,"parse_rule":"json-boolean-v1"}
        immutable_run_manifest(args.output,run)
        p=args.output/"judgments.jsonl";existing={r["scenario_id"]:r for r in jsonl(p)} if p.exists() else {}
        bridge=BaselineBridge(cfg)
        bridge.ledger=CallLedger(args.output/"calls.jsonl")
        try:
            for row in rows:
                if row["scenario_id"] in existing:continue
                js=[bridge.score(row,i) for i in range(1,cfg.rounds+1)]
                result={"scenario_id":row["scenario_id"],"index":row["index"],"judgments":js,
                        "run_hash":object_hash(run),"format_valid":row["format_valid"]}
                append_jsonl(p,result);existing[row["scenario_id"]]=result
            summary=judgment_summary(list(existing.values()),len(rows));summary["run"]=run
            summary["judgments_sha256"]=sha256(p)
            summary["request_ledger"]=bridge.ledger.summary()
            atomic_json(args.output/"summary.json",summary)
            print(json.dumps(summary,ensure_ascii=False,indent=2))
        finally:bridge.close()


def paired_bootstrap(a:list[float],b:list[float],seed:int=17,repetitions:int=10000)->dict:
    if len(a)!=len(b) or not a:raise ValueError("Paired nonempty observations required")
    g=torch.Generator().manual_seed(seed);delta=torch.tensor(a)-torch.tensor(b)
    estimates=[]
    for start in range(0,repetitions,250):
        ix=torch.randint(len(a),(min(250,repetitions-start),len(a)),generator=g)
        estimates.append(delta[ix].mean(1))
    x=torch.cat(estimates)
    return {"n":len(a),"difference_percentage_points":float(delta.mean()*100),
            "bootstrap_95_percent_interval_pp":[float(torch.quantile(x,0.025)*100),float(torch.quantile(x,0.975)*100)],
            "bootstrap_repetitions":repetitions,"pairing_unit":"scenario"}


def command_compare(args):
    ma=read_json(args.a/"run.json");mb=read_json(args.b/"run.json")
    for key in ("lock_id","split","seed","expected_n","judge_model_hash","prompt_version","parse_rule","test_action","test_action_radius"):
        if ma[key]!=mb[key]:raise ValueError(f"Comparison contract differs: {key}")
    aa={x["scenario_id"]:x for x in jsonl(args.a/"judgments.jsonl")}
    bb={x["scenario_id"]:x for x in jsonl(args.b/"judgments.jsonl")}
    if set(aa)!=set(bb) or len(aa)!=ma["expected_n"]:raise ValueError("Paired scenario IDs differ")
    def label(row):
        js=row["judgments"]
        relevant=js if args.metric=="ever" else js[-1:]
        if any(not j["parse_ok"] for j in relevant):
            raise ValueError("Resolve parsing before statistical comparison; never silently drop cases")
        return float(any(j["success"] for j in relevant))
    ids=sorted(aa);result=paired_bootstrap([label(aa[i]) for i in ids],[label(bb[i]) for i in ids])
    result.update(method_a=ma["method"],method_b=mb["method"],metric=args.metric)
    atomic_json(args.output,result);print(json.dumps(result,indent=2))


def command_smoke(args,cfg:Config,lock:dict)->None:
    """REAL pretrained CUDA BF16 gate; separate from CPU unit tests."""
    verified_prepared(args.prepared,lock)
    if args.checkpoint:verify_checkpoint(args.checkpoint,lock)
    session=next(s for s in jsonl(args.prepared/"sft.jsonl") if len(s["turns"])>=5)
    seed_all(cfg.seed);bridge=BaselineBridge(cfg);bridge.ledger=CallLedger(args.output.with_suffix(".calls.jsonl"))
    runner=ECLPRunner(cfg,bridge,args.checkpoint,sft=True)
    report={"version":VERSION,"implementation_sha256":IMPLEMENTATION_SHA256,
            "environment":environment_versions(),"precision":runner.precision,"tests":{}}
    try:
        with torch.autocast(device_type="cuda",dtype=BF16):
            losses,stats=runner.session_losses(session)
        state_named=list(runner.heads.state.named_parameters())
        state_params=[p for _,p in state_named]
        public=losses["utterance"]+cfg.strategy_weight*losses["strategy"]
        grad=torch.autograd.grad(public,state_params,allow_unused=True,retain_graph=True)
        public_grad_norms={name:0.0 if g is None else float(g.float().norm())
                           for (name,_),g in zip(state_named,grad)}
        predictor_isolated=all(
            public_grad_norms[name] <= 1e-12
            for name in public_grad_norms
            if name == "slots" or name.startswith(("read.","to_memory.","transition.","prior_norm.","predict."))
        )
        correction_active=any(
            public_grad_norms[name] > 1e-12
            for name in public_grad_norms
            if name.startswith(("correction.","gate.","posterior_norm."))
        )
        error_projection_active=any(
            public_grad_norms[name] > 1e-12
            for name in public_grad_norms if name.startswith("error_projection.")
        )
        if cfg.isolate_state and not predictor_isolated:
            raise RuntimeError("Public loss leaked into the evidence predictor/readout path")
        if cfg.state_mode in {"eclp","no_error"} and not correction_active:
            raise RuntimeError("Public loss failed to reach the ECLP correction path")
        if cfg.state_mode in {"no_correction","stateless"} and correction_active:
            raise RuntimeError("A correction-disabled ablation received a public correction gradient")
        if cfg.state_mode == "eclp" and not error_projection_active:
            raise RuntimeError("Public loss failed to reach the ECLP error projection")
        if cfg.state_mode != "eclp" and error_projection_active:
            raise RuntimeError("An error-disabled ablation received an error projection gradient")
        obs_grad=torch.autograd.grad(losses["observation"],state_params,allow_unused=True,retain_graph=True)
        if not any(g is not None and float(g.float().norm())>0 for g in obs_grad):
            raise RuntimeError("Observation objective fails to train evidence state")
        reason_grad=torch.autograd.grad(losses["reasoning"],state_params,allow_unused=True,retain_graph=True)
        if not any(g is not None and float(g.float().norm())>0 for g in reason_grad):
            raise RuntimeError("Reasoning objective fails to train evidence state")
        total=public+cfg.obs_weight*losses["observation"]+cfg.reasoning_weight*losses["reasoning"]
        total.backward();params=trainable_parameters(runner)
        opt=FP32MasterAdamW(params,lr=1e-4);torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
        opt.zero_grad(set_to_none=True)
        report["tests"].update(
            public_predictor_gradient_isolated=predictor_isolated,
            public_correction_gradient_nonzero=correction_active,
            public_error_projection_gradient_nonzero=error_projection_active,
            public_state_gradient_norms=public_grad_norms,
            observation_gradient_nonzero=True,
            reasoning_gradient_nonzero=True,
            bf16_forward_backward_optimizer_step=True,
        )
        report["losses"]={k:float(v.detach()) for k,v in losses.items()};report["session_stats"]=stats
        runner.set_stage("eval")
        point=points_for_split(args.prepared,"dev",1)[0]
        with torch.no_grad(),torch.autocast(device_type="cuda",dtype=BF16):
            live,_=runner.replay(point,public_history(point["history"]))
            spoken,_,_=runner.speak(point,live,cfg.seed)
            runner.consume(live,point,spoken)
            # Smoke can stage the two real models separately when another run
            # occupies the same GPU. Every neural forward still uses CUDA BF16.
            # This is transient placement during verification, not offloaded training.
            sequential = getattr(args, "sequential_models", False)
            if sequential:
                runner.llm.to("cpu")
                torch.cuda.empty_cache()
            try:
                reply,raw=bridge.respond(point,live.history,cfg.seed+1)
            finally:
                if sequential:
                    bridge.close()
                    runner.llm.to(runner.device)
            runner.consume(live,point,{"role":"persuadee","content":reply})
        report["transient_actor_cpu_placement_between_smoke_stages"] = sequential
        report["tests"].update(pretrained_generation=True,fixed_persuadee_reply=True,state_feedback_update=True)
        report["example"]={"persuader":spoken,"persuadee":reply,"persuadee_raw":raw}
        report["status"]="PASS_GPU_SMOKE_NOT_AN_EXPERIMENT_RESULT"
        atomic_json(args.output,report)
    finally:bridge.close()


def make_parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=Path(__file__).with_name("config.json"))
    p.add_argument("--lock",type=Path,default=None,help="Defaults to work_dir/protocol.lock.json")
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("audit",help="Read-only source/LFS/hash audit; no GPU")
    a.add_argument("--output",type=Path,default=Path("source_audit.json"))
    sub.add_parser("lock",help="Freeze original model/source/data/protocol identities; expensive full weight hashes")
    a=sub.add_parser("prepare");a.add_argument("--output",type=Path,required=True)
    a=sub.add_parser("sft")
    a.add_argument("--prepared",type=Path,required=True);a.add_argument("--output",type=Path,required=True)
    a.add_argument("--checkpoint",type=Path);a.add_argument("--resume",action="store_true")
    a.add_argument("--max-steps",type=int,default=100);a.add_argument("--epochs",type=int,default=1)
    a.add_argument("--accumulation",type=int,default=8);a.add_argument("--lr",type=float,default=1e-4)
    a.add_argument("--save-every",type=int,default=50);a.add_argument("--limit-scenarios",type=int,default=0)
    a=sub.add_parser("collect")
    a.add_argument("--prepared",type=Path,required=True);a.add_argument("--checkpoint",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True);a.add_argument("--effect",type=Path)
    a.add_argument("--limit",type=int,default=128);a.add_argument("--offset",type=int,default=0)
    a.add_argument("--horizon",type=int,default=3)
    a.add_argument("--candidates",type=int,default=4);a.add_argument("--repeats",type=int,default=2)
    a.add_argument("--exploration",choices=["isotropic","random","effect"],default="isotropic")
    a.add_argument("--reward",choices=["interaction","ppl"],default="interaction")
    a=sub.add_parser("score");a.add_argument("--rollouts",type=Path,required=True);a.add_argument("--output",type=Path,required=True)
    a=sub.add_parser("fit-effect")
    a.add_argument("--rollouts",type=Path,required=True);a.add_argument("--output",type=Path,required=True)
    a.add_argument("--epochs",type=int,default=30);a.add_argument("--batch-size",type=int,default=32)
    a.add_argument("--lr",type=float,default=1e-3)
    a=sub.add_parser("check-effect",help="Effect prediction on NEW training-pool scenarios, not used for fit/selection")
    a.add_argument("--rollouts",type=Path,required=True);a.add_argument("--effect",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    a=sub.add_parser("update")
    a.add_argument("--rollouts",type=Path,required=True);a.add_argument("--scores",type=Path)
    a.add_argument("--checkpoint",type=Path,required=True);a.add_argument("--output",type=Path,required=True)
    a.add_argument("--epochs",type=int,default=2);a.add_argument("--batch-size",type=int,default=1)
    a.add_argument("--lr",type=float,default=1e-5);a.add_argument("--advantage-scale",type=float,default=1.0)
    a.add_argument("--max-unresolved-fraction",type=float,default=0.0)
    a=sub.add_parser("validate",help="Offline DEV NLL and observed-feedback prediction, no simulator/Judge calls")
    a.add_argument("--prepared",type=Path,required=True);a.add_argument("--checkpoint",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True);a.add_argument("--limit",type=int,default=32)
    a=sub.add_parser("generate")
    a.add_argument("--prepared",type=Path,required=True);a.add_argument("--checkpoint",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True);a.add_argument("--method",default="eclp")
    a.add_argument("--split",choices=["dev","test"],default="dev");a.add_argument("--limit",type=int,default=32)
    a.add_argument("--allow-test",action="store_true")
    a=sub.add_parser("evaluate")
    a.add_argument("--generation",type=Path,required=True);a.add_argument("--output",type=Path,required=True)
    a.add_argument("--judge-model",type=str,default=None,
                   help="Optional alternate local Judge. The generation lock remains unchanged.")
    a=sub.add_parser("compare")
    a.add_argument("--a",type=Path,required=True);a.add_argument("--b",type=Path,required=True)
    a.add_argument("--metric",choices=["ever","final"],default="ever");a.add_argument("--output",type=Path,required=True)
    a=sub.add_parser("smoke")
    a.add_argument("--prepared",type=Path,required=True);a.add_argument("--checkpoint",type=Path)
    a.add_argument("--output",type=Path,required=True)
    a.add_argument("--sequential-models",action="store_true",
                   help="Smoke only: stage actor in CPU memory between CUDA BF16 model calls")
    return p


def main(argv:list[str]|None=None)->int:
    args=make_parser().parse_args(argv)
    cfg=Config.load(args.config)
    if hasattr(args,"output"):
        target=args.output.expanduser().resolve()
        protected=[cfg.root/"exp_code_ljj",cfg.root/"baseline5_latentRATT"]
        protected += [Path(x).expanduser().resolve() for x in (cfg.base_model,cfg.persuadee_model,cfg.judge_model) if x]
        if any(target==p or p in target.parents for p in protected):
            raise ValueError("Output points inside an original baseline/model directory; choose a new eclp_runs path")
    if args.command=="compare":command_compare(args);return 0
    lock_path=args.lock or cfg.work/"protocol.lock.json"
    resolved_lock=lock_path.expanduser().resolve()
    for base in (cfg.root/"exp_code_ljj", cfg.root/"baseline5_latentRATT"):
        if resolved_lock==base or base in resolved_lock.parents:
            raise ValueError("Lock points inside an original baseline directory")
    if args.command=="audit":
        report=audit_sources(cfg);report["environment"]=environment_versions();atomic_json(args.output,report)
        print(json.dumps(report,ensure_ascii=False,indent=2));return 2 if report["blockers"] else 0
    if args.command=="lock":
        if lock_path.exists():raise FileExistsError("Existing lock must not be overwritten")
        atomic_json(lock_path,create_lock(cfg));print(lock_path);return 0
    required={"prepare":(),"sft":("base_model",),"smoke":("base_model","persuadee_model"),
              "collect":("base_model","persuadee_model") if getattr(args,"reward",None)=="interaction" else ("base_model",),
              "score":("judge_model",),"fit-effect":(),"check-effect":(),"update":("base_model",),
              "generate":("base_model","persuadee_model"),"evaluate":("judge_model",),"validate":("base_model",)}
    alternate_judge = getattr(args, "judge_model", None)
    if alternate_judge is not None and args.command != "evaluate":
        raise ValueError("--judge-model is only supported by evaluate")
    # Alternate Judges are evaluation-only consumers of an already generated
    # trajectory. Verify the source/data/protocol lock, then fingerprint the
    # requested Judge without pretending it was the Judge used for training.
    lock=verify_lock(cfg,lock_path,() if alternate_judge else required[args.command])
    if alternate_judge:
        cfg.judge_model = str(Path(alternate_judge).expanduser().resolve())
        model_fingerprint(cfg.judge_model)
    if args.command=="prepare":
        result=prepare_data(cfg,lock,args.output);print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    for k in ("max_steps","epochs","accumulation","save_every","batch_size"):
        if hasattr(args,k) and getattr(args,k)<1:raise ValueError(f"{k} must be positive")
    for k in ("limit","limit_scenarios","offset"):
        if hasattr(args,k) and getattr(args,k)<0:raise ValueError(f"{k} must be nonnegative")
    functions={"sft":command_sft,"collect":command_collect,"score":command_score,
               "fit-effect":command_effect,"check-effect":command_check_effect,"update":command_update,"generate":command_generate,
               "evaluate":command_evaluate,"smoke":command_smoke,"validate":command_validate}
    functions[args.command](args,cfg,lock);return 0


if __name__=="__main__":
    try:sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted. Resume from the last committed checkpoint/episode; no partial checkpoint was promoted.",file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"BLOCKED: {type(e).__name__}: {e}",file=sys.stderr)
        raise
