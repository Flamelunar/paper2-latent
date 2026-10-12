"""Revision runtime configuration; no secrets and no automatic experiment start."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any
from evaluation.runtime import LocalHFRunner, OpenAICompatibleRunner
from evaluation.revision_schema import SCHEMA_VERSION, STRUCTURED_STRATEGY_SCHEMA_VERSION

METHODS = ("zero_shot", "sft", "ma2p", "grpo_only", "sft_grpo", "ours")


class ExclusiveRunnerPool:
    """Keep one auto-placed model resident when sequential roles share GPUs."""

    def __init__(self, runners: list[LocalHFRunner]) -> None:
        self.runners = {id(runner): runner for runner in runners}
        self.active: LocalHFRunner | None = None
        self.lock = RLock()

    def generate(self, runner: LocalHFRunner, *args: Any, **kwargs: Any) -> tuple[str, int]:
        with self.lock:
            if self.active is not runner:
                if self.active is not None:
                    self.active.unload()
                self.active = runner
            return runner.generate(*args, **kwargs)


class _PooledRoleRunner:
    def __init__(self, pool: ExclusiveRunnerPool, runner: LocalHFRunner) -> None:
        self.pool, self.runner = pool, runner

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float = 0.0,
                 top_p: float = 1.0, seed: int | None = None) -> tuple[str, int]:
        return self.pool.generate(self.runner, prompt, max_new_tokens=max_new_tokens,
                                  temperature=temperature, top_p=top_p, seed=seed)


@dataclass
class RevisionConfig:
    method: str = "ours"
    world_model_mode: str = "strategy"
    world_model: str = ""
    persuader_model: str = "/app/hzj/models/Qwen3.5-9B"
    persuadee_model: str = "/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged"
    judge_model: str = "/app/hzj/models/Qwen3-14B"
    progress_scorer_model: str = "/app/hzj/models/Qwen3-14B"
    backbone: str = ""
    repeat_id: int = 0
    training_seed: int | None = None
    max_rounds: int = 3
    root_actions: int = 3
    reaction_samples: int = 2
    max_depth: int = 2
    future_actions: int = 1
    max_candidate_tokens: int = 128
    max_plan_tokens: int = 128
    max_reply_tokens: int = 128
    max_score_tokens: int = 192
    max_input_tokens: int = 4096
    wm_analysis_budget: int = 0
    proposer_temperature: float = 0.7
    proposer_top_p: float = 0.9
    world_model_temperature: float = 0.7
    world_model_top_p: float = 0.95
    persuadee_temperature: float = 0.7
    persuadee_top_p: float = 0.95
    # Candidate expansion and schema retries are separate protocol controls.
    retry_limit: int = 1
    parse_retry_limit: int = 1
    factual_retry_limit: int = 3
    episode_failure_retries: int = 3
    precision: str = "bf16"
    device: str = "auto"
    role_devices: dict[str, str] = field(default_factory=dict)
    resume: bool = True
    expected_items: int | None = 500
    allow_partial: bool = False
    ma2p_root: str = "baselines/ma2p"
    ma2p_kb: str = "baselines/ma2p/method/kb_highlevel_500.json"
    model_manifest: str = ""
    require_model_manifest: bool = False
    shared_local_endpoint: str = ""

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unknown method: {self.method}")
        if not 1 <= self.max_rounds <= 3:
            raise ValueError("max_rounds must be in [1,3]")
        if min(self.root_actions, self.reaction_samples, self.max_depth, self.future_actions) < 1:
            raise ValueError("Planning counts must be positive")
        if self.factual_retry_limit < 0:
            raise ValueError("factual_retry_limit must be nonnegative")
        if self.episode_failure_retries < 0:
            raise ValueError("episode_failure_retries must be nonnegative")
        if not self.world_model:
            raise ValueError("world_model is required for this method")
        if self.world_model_mode not in {"strategy", "action_conditioned_reply"}:
            raise ValueError("Unknown world-model interface")
        if self.world_model_mode == "action_conditioned_reply":
            if self.method not in {"zero_shot", "ours"} or self.wm_analysis_budget != 0:
                raise ValueError("Prediction priority comparison requires zero-shot/PMM with reply-only outputs")
            return
        if self.method == "ours" and self.wm_analysis_budget != 0:
            raise ValueError("ours uses the analysis-free final Compact-WM checkpoint")
        if self.method != "ours" and self.wm_analysis_budget != 512:
            raise ValueError("non-PMM methods use the full 512-token strategy schema")

    @classmethod
    def from_experiment(cls, raw: dict[str, Any]) -> "RevisionConfig":
        roles = raw["roles"]
        planning, inf = raw["planning"], raw["inference"]
        return cls(world_model_mode=raw.get("world_model_mode", "strategy"), persuader_model=roles["persuader"]["path"], persuadee_model=roles["persuadee"]["path"],
            judge_model=roles["online_judge"]["path"], progress_scorer_model=roles["progress_scorer"]["path"],
            role_devices={k: v.get("device", "auto") for k, v in roles.items() if isinstance(v, dict)},
            max_rounds=raw["interaction"]["max_real_turns"], root_actions=planning["root_actions"],
            reaction_samples=planning["reaction_samples_per_action"], max_depth=planning["max_depth"],
            future_actions=planning["future_actions_per_node"], precision=inf["precision"],
            max_input_tokens=inf["max_input_tokens"], max_candidate_tokens=inf["candidate_max_new_tokens"],
            max_plan_tokens=inf["reaction_max_new_tokens"], max_reply_tokens=inf["reply_max_new_tokens"],
            max_score_tokens=inf["scorer_max_new_tokens"], expected_items=raw["data"]["expected_eval_items"],
            model_manifest=raw.get("model_manifest", ""), require_model_manifest=raw.get("require_model_manifest", False),
            ma2p_root=raw["baselines"]["ma2p_root"], ma2p_kb=raw["baselines"]["ma2p_kb"],
            proposer_temperature=inf["proposer_temperature"], proposer_top_p=inf["proposer_top_p"],
            world_model_temperature=inf["world_model_temperature"], world_model_top_p=inf["world_model_top_p"],
            persuadee_temperature=inf["persuadee_temperature"], persuadee_top_p=inf["persuadee_top_p"],
            retry_limit=inf.get("structural_retry_limit", 1),
            parse_retry_limit=int(os.environ.get("AAAI_PARSE_RETRY_LIMIT", inf["structural_retry_limit"])),
            factual_retry_limit=inf.get("factual_retry_limit", 3),
            episode_failure_retries=inf.get("episode_failure_retries", 3),
            shared_local_endpoint=raw.get("shared_local_endpoint", ""))


def build_runners(cfg: RevisionConfig) -> dict[str, Any]:
    if cfg.world_model_mode == "action_conditioned_reply" and cfg.method == "ours":
        checkpoint = Path(cfg.world_model)
        contract_path = checkpoint / "training_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if (contract.get("task") != "compact_wm" or contract.get("schema_version") != "action-conditioned-reply-v1.1"
                or contract.get("output_analysis_budget") != 0
                or contract.get("analysis_budgets") != [512, 256, 0]
                or contract.get("optimism_reward") is not False):
            raise ValueError("Prediction PMM requires the preserved truthful Compact-WM checkpoint")
        source = Path(contract["sft_merged"]) / "training_contract.json"
        parent = json.loads(source.read_text(encoding="utf-8"))
        if (parent.get("task") != "structured_wm"
                or hashlib.sha256(source.read_bytes()).hexdigest() != contract["sft_contract_sha256"]):
            raise ValueError("Prediction PMM SFT provenance mismatch")
        if contract.get("arguments", {}).get("precision") != "bf16" or cfg.precision != "bf16":
            raise ValueError("Prediction PMM comparison requires BF16")
    expected_task = {"sft": "structured_strategy", "grpo_only": "standard_grpo",
                     "sft_grpo": "standard_grpo", "ours": "progressive_strategy_wm"}.get(cfg.method)
    if expected_task and cfg.world_model_mode != "action_conditioned_reply":
        contract_path = Path(cfg.world_model) / "training_contract.json"
        if not contract_path.is_file():
            raise FileNotFoundError(f"Missing trained checkpoint contract: {contract_path}")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        expected_schema = STRUCTURED_STRATEGY_SCHEMA_VERSION
        expected_budget = 0 if cfg.method == "ours" else cfg.wm_analysis_budget
        if (contract.get("schema_version") != expected_schema or contract.get("task") != expected_task
                or contract.get("output_analysis_budget") != expected_budget):
            raise ValueError(f"{cfg.method} requires a {expected_task} checkpoint with analysis budget {cfg.wm_analysis_budget}")
        if expected_task == "standard_grpo":
            reward = contract.get("reward")
            if (reward != "structured_strategy_format_plus_reference_token_f1"
                    or contract.get("reward_family") != "generic_baseline_task_match"):
                raise ValueError("standard GRPO controls require the frozen generic task-match reward")
            progressive = contract.get("progressive_compression")
            if contract.get("pmm_reward") is True or (progressive is not None and progressive is not False):
                raise ValueError("standard GRPO controls cannot use PMM rewards or compression")
            one_epoch = contract.get("epochs") == 1 and contract.get("max_steps") is None
            one_update = contract.get("max_steps") == 1 and contract.get("epochs") is None
            if (not (one_epoch or one_update) or contract.get("max_completion_length") != 512
                    or contract.get("num_generations") != 8):
                raise ValueError("standard GRPO controls require one configured round, fixed 512-token output and eight generations")
            init_model = str(contract.get("init_model", ""))
            init_mode = contract.get("init_mode")
            if cfg.method == "grpo_only" and init_mode != "base":
                raise ValueError("GRPO-only must initialize from the untouched backbone")
            if cfg.method == "sft_grpo" and (init_mode != "sft" or "structured_strategy_merged" not in init_model):
                raise ValueError("SFT+GRPO must initialize from the structured-strategy SFT checkpoint")
            if cfg.method == "grpo_only" and "_merged" in init_model:
                raise ValueError("GRPO-only must initialize from the untouched backbone")
        if cfg.method == "ours":
            if (contract.get("progressive_compression") is not True
                    or contract.get("pmm_reward") is not True
                    or contract.get("analysis_budgets") != [512, 256, 0]
                    or contract.get("max_steps", 0) < 3
                    or contract.get("num_generations") != 8):
                raise ValueError("PMM requires its progressive 512→256→0 strategy curriculum")
            init_model = Path(contract.get("init_model", ""))
            sft_path = init_model / "training_contract.json"
            if not sft_path.is_file():
                raise ValueError("PMM's structured-strategy SFT initialization is missing")
            sft_contract = json.loads(sft_path.read_text(encoding="utf-8"))
            if (contract.get("init_mode") != "sft"
                    or sft_contract.get("task") != "structured_strategy"
                    or sft_contract.get("schema_version") != STRUCTURED_STRATEGY_SCHEMA_VERSION
                    or sft_contract.get("train_file_sha256") != contract.get("train_file_sha256")
                    or hashlib.sha256(sft_path.read_bytes()).hexdigest() != contract.get("sft_contract_sha256")):
                raise ValueError("PMM and SFT must share the exact structured-strategy SFT checkpoint and training data")
    specs = {"persuader": cfg.persuader_model, "persuadee": cfg.persuadee_model,
             "online_judge": cfg.judge_model, "progress_scorer": cfg.progress_scorer_model}
    if cfg.world_model:
        specs["world_model"] = cfg.world_model
    cache = {}
    result = {}
    pooled_roles = []
    for role, path in specs.items():
        configured_device = cfg.role_devices.get(role, cfg.device)
        pooled = configured_device == "auto_shared"
        device = "auto" if pooled else configured_device
        key = (path, device)
        if key not in cache:
            if cfg.shared_local_endpoint and role != "world_model":
                from evaluation.shared_local import SharedLocalRunner
                cache[key] = SharedLocalRunner(path, cfg.shared_local_endpoint,
                                              cfg.precision, cfg.max_input_tokens)
            else:
                cache[key] = LocalHFRunner(path, device, cfg.precision, cfg.max_input_tokens)
        result[role] = cache[key]
        if pooled:
            pooled_roles.append(role)
    if pooled_roles:
        # Reserve the fixed dialogue models before Transformers places a
        # rotating World Model / judge into the remaining GPU memory.
        for role in ("persuader", "persuadee"):
            runner = result[role]
            if isinstance(runner, LocalHFRunner):
                runner._load()
        pool = ExclusiveRunnerPool([result[role] for role in pooled_roles])
        for role in pooled_roles:
            result[role] = _PooledRoleRunner(pool, result[role])
    return result
