#!/usr/bin/env python3
"""Offline continuous-action RL, with no autoregressive or simulator calls.

The original ECLP implementation is imported unchanged, preserving the running
SFT checkpoint contract. This trainer is an action-density policy optimizer,
not a line-for-line reproduction of the paper's token-ratio LR-GRPO.
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/app/hzj/liujianjian/ljj-2-paper/my_eclp_final')

import argparse
import copy
import math
import os
from pathlib import Path
import random
import statistics
import time

import torch
import torch.nn.functional as F

from eclp_elpo import (
    BF16, BaselineBridge, Config, ECLPRunner, FP32MasterAdamW,
    clipped_policy_loss, equal_covariance_kl, exploration_cholesky,
    frozen_executor_hash, gaussian_log_prob, load_tensors, points_for_split, public_history,
    restore_rng, save_checkpoint, save_tensors, scenario_seed, seed_all,
    tensor_digest, trainable_parameters, verified_prepared, verify_checkpoint,
    verify_lock,
)
from experiment_io import (
    append_jsonl, atomic_json, digest, immutable_json, object_hash, read_json,
    writer_lock,
)

VERSION = "eclp-offline-lpo-v1"
CODE = Path(__file__).resolve().parents[1]


class OfflineBridge(BaselineBridge):
    """Fail immediately if training accidentally reaches a dialogue endpoint."""
    def _invoke(self, *args, **kwargs):
        raise RuntimeError("Offline latent RL forbids simulator and Judge calls")

    def respond(self, *args, **kwargs):
        raise RuntimeError("Offline latent RL forbids persuadee generation")

    def score(self, *args, **kwargs):
        raise RuntimeError("Offline latent RL forbids training Judge generation")


def prohibit_generation(*args, **kwargs):
    raise RuntimeError("Offline latent RL forbids autoregressive generation")


def paired_contexts(runner: ECLPRunner, point: dict):
    """Intervene only on the latest observed reply's prediction-error term.

    Both branches consume exactly the same public history. Their state before
    the latest reply is identical. The counterfactual still reads that reply;
    it only removes the error vector, rather than removing feedback or history.
    """
    history = public_history(point["history"])
    live = runner.initialize(point)
    indices = [i for i, turn in enumerate(history) if turn["role"] == "persuadee"]
    last_feedback = indices[-1] if indices else None
    shadow = None
    original_mode = runner.heads.state.mode
    if original_mode != "eclp":
        raise ValueError("The paired correction reward requires the factual ECLP state")
    for index, turn in enumerate(history):
        if index == last_feedback:
            shadow = live.branch()
            try:
                runner.heads.state.mode = "no_error"
                runner.consume(shadow, point, turn)
            finally:
                runner.heads.state.mode = original_mode
        elif shadow is not None:
            runner.consume(shadow, point, turn)
        runner.consume(live, point, turn)
    if live.prediction is not None or live.policy_prediction is not None:
        raise ValueError("Offline decision history must end after a persuadee reply")
    prefix, context, _ = runner.context(point, live)
    if shadow is None:
        return prefix, context, prefix, False
    if live.history != shadow.history:
        raise RuntimeError("Counterfactual changed the observed public history")
    if runner.prompt_ids(point, live.history) != runner.prompt_ids(point, shadow.history):
        raise RuntimeError("Counterfactual changed the public prompt")
    counter_prefix, _, _ = runner.context(point, shadow)
    if prefix.shape != counter_prefix.shape:
        raise RuntimeError("Counterfactual changed the decoder prefix length")
    return prefix, context, counter_prefix, True


def target_nlls(runner: ECLPRunner, prefix: torch.Tensor,
                actions: torch.Tensor, target: str, batch_size: int):
    """One teacher-forced pass per batch of latent actions; target-only CE."""
    ids = runner.tokenizer.encode(target, add_special_tokens=False)[:runner.cfg.max_target_tokens - 1]
    if not ids:
        raise ValueError("Offline reward requires a nonempty existing training answer")
    ids.append(runner.tokenizer.eos_token_id)
    target_ids = torch.tensor([ids], device=runner.device, dtype=torch.long)
    target_embeddings = runner.llm.get_input_embeddings()(target_ids).detach()
    values = []
    for start in range(0, len(actions), batch_size):
        block = actions[start:start + batch_size]
        count = len(block)
        strategy = runner.heads.strategy_tokens(block)
        x = torch.cat([prefix.expand(count, -1, -1), strategy,
                       target_embeddings.expand(count, -1, -1)], dim=1)
        offset = prefix.shape[1] + strategy.shape[1] - 1
        result = runner.llm(
            inputs_embeds=x,
            attention_mask=torch.ones(x.shape[:2], device=runner.device, dtype=torch.long),
            position_ids=torch.arange(x.shape[1], device=runner.device).unsqueeze(0).expand(count, -1),
            use_cache=False, return_dict=True,
        )
        logits = result.logits[:, offset:offset + len(ids)]
        losses = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            target_ids.expand(count, -1).reshape(-1), reduction="none",
        ).reshape(count, len(ids)).mean(dim=1)
        if not torch.isfinite(losses).all():
            raise FloatingPointError("Nonfinite teacher-forced target NLL")
        values.append(losses.detach().float())
        del result, logits, losses, x
    return torch.cat(values), len(ids)


def dense_rewards(factual: torch.Tensor, counter: torch.Tensor,
                  has_feedback: bool, mode: str, weight: float):
    """Index zero is the fixed SFT reference action, never a policy sample."""
    if mode == "ppl":
        reward = -torch.exp(factual.float())
        bonus = torch.zeros_like(reward)
    elif mode in {"nll", "correction"}:
        benefit = counter.float() - factual.float()
        # Center against the SAME context's fixed SFT reference. The same
        # sampled action is used in both state branches; no resampling here.
        bonus = torch.tanh(benefit - benefit[0])
        if not has_feedback or mode == "nll":
            bonus = torch.zeros_like(bonus)
        reward = -factual.float() + weight * bonus
    else:
        raise ValueError(f"Unknown offline reward mode {mode}")
    if not torch.isfinite(reward).all():
        raise FloatingPointError("Nonfinite offline reward (PPL may overflow)")
    if float(bonus[0]) != 0.0:
        raise RuntimeError("The fixed reference must have zero correction bonus")
    return reward, bonus


def score_group(runner: ECLPRunner, point: dict, plan: dict, epoch: int):
    started = time.monotonic()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=BF16):
        prefix, context, counter_prefix, has_feedback = paired_contexts(runner, point)
        mean = runner.heads.policy(context).float()
        reference = runner.reference_policy(context).float()
        chol, _ = exploration_cholesky(
            runner.cfg, "isotropic", None, context, reference,
            scenario_seed(runner.cfg.seed, point["scenario_id"], "offline_covariance"),
        )
        gen = torch.Generator(device="cpu").manual_seed(
            scenario_seed(runner.cfg.seed, point["scenario_id"], f"offline_action_epoch_{epoch}"))
        noise = torch.randn(plan["candidates"], runner.cfg.action_dim, generator=gen).to(runner.device)
        actions = mean + noise @ chol.T
        old_logp = gaussian_log_prob(actions, mean, chol)
        all_actions = torch.cat([reference, actions], dim=0)
        target = point.get("gold_response", "")
        factual, tokens = target_nlls(runner, prefix, all_actions, target, plan["score_batch_size"])
        if has_feedback and plan["reward"] == "correction" and plan["correction_weight"] > 0:
            counter, _ = target_nlls(runner, counter_prefix, all_actions, target, plan["score_batch_size"])
        else:
            counter = factual.clone()
        reward, bonus = dense_rewards(factual, counter, has_feedback,
                                     plan["reward"], plan["correction_weight"])
    group = {
        "context": context.detach().cpu().expand(plan["candidates"], -1, -1).clone(),
        "actions": actions.detach().cpu(), "old_logp": old_logp.detach().cpu(),
        "chol": chol.detach().cpu().unsqueeze(0).expand(plan["candidates"], -1, -1).clone(),
        "reference": reference.detach().cpu().expand(plan["candidates"], -1).clone(),
        "old_mean": mean.detach().cpu().expand(plan["candidates"], -1).clone(),
        "deltas": (reward[1:] - reward[0]).detach().cpu(),
    }
    report = {
        "scenario_id": point["scenario_id"], "source_row_hash": object_hash(point),
        "history_hash": object_hash(public_history(point["history"])),
        "has_feedback": has_feedback, "target_tokens": tokens,
        "reward_mode": plan["reward"], "correction_weight": plan["correction_weight"],
        "factual_nll": factual.cpu().tolist(), "counterfactual_nll": counter.cpu().tolist(),
        "correction_benefit": (counter - factual).cpu().tolist(),
        "centered_correction_bonus": bonus.cpu().tolist(), "reward": reward.cpu().tolist(),
        "paired_reward_delta": group["deltas"].tolist(),
        "reference_action_index": 0, "policy_samples": plan["candidates"],
        "new_dialogue_calls": 0, "judge_calls": 0, "generated_tokens": 0,
        "seconds": time.monotonic() - started,
    }
    return group, report


def policy_data(groups: list[dict], plan: dict):
    data = {key: torch.cat([g[key] for g in groups]) for key in groups[0] if key != "deltas"}
    delta = torch.cat([g["deltas"] for g in groups]).float()
    if not torch.isfinite(delta).all():
        raise FloatingPointError("Nonfinite paired advantages")
    scale = max(float(delta.std(unbiased=False)), plan["advantage_scale_floor"])
    data["advantages"] = (delta / scale).clamp(-plan["advantage_clip"], plan["advantage_clip"])
    return data, scale


def validate_plan(plan: dict):
    for key in ["rl_epochs", "contexts_per_update", "candidates", "score_batch_size",
                "policy_update_epochs", "policy_update_batch_size"]:
        if type(plan[key]) is not int or plan[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if plan["candidates"] < 2 or plan["reward"] not in {"ppl", "nll", "correction"}:
        raise ValueError("Require at least two policy samples and a known reward")
    if not 0 <= plan["correction_weight"] <= 0.25:
        raise ValueError("Keep correction shaping bounded relative to target fit")
    for key in ["advantage_scale_floor", "advantage_clip", "policy_learning_rate"]:
        if not math.isfinite(plan[key]) or plan[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if plan["online_training_calls"] != 0:
        raise ValueError("This trainer only supports zero online training calls")
    if (type(plan["max_lr_backoffs"]) is not int or not 0 <= plan["max_lr_backoffs"] <= 10
            or not 0 < plan["min_policy_learning_rate"] <= plan["policy_learning_rate"]):
        raise ValueError("Invalid trust-region step-size backoff limits")


def load_runner(config: Path, checkpoint: Path):
    cfg = Config.load(config)
    cfg.validate()
    lock = verify_lock(cfg, cfg.work / "protocol.lock.json", ("base_model",))
    verify_checkpoint(checkpoint, lock)
    bridge = OfflineBridge(cfg)
    runner = ECLPRunner(cfg, bridge, checkpoint)
    runner.llm.generate = prohibit_generation
    return cfg, lock, bridge, runner


def replay_density_check(runner: ECLPRunner, data: dict, batch_size: int):
    largest = 0.0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=BF16):
        for start in range(0, len(data["actions"]), batch_size):
            batch = {k: v[start:start + batch_size].to(runner.device) for k, v in data.items()}
            mean = runner.heads.policy(batch["context"])
            replay = gaussian_log_prob(batch["actions"], mean, batch["chol"])
            largest = max(largest, float((replay - batch["old_logp"]).abs().max()))
    if largest > 0.1:
        raise RuntimeError(f"Stored behavior-density replay mismatch: {largest}")
    return largest


def behavior_guarded_mean_step(policy, optimizer, batch: dict, cfg: Config) -> dict:
    """Take one PPO step with the batch behavior policy as the hard trust region.

    The fixed SFT policy remains the KL anchor in the loss.  It is not the
    behavior policy for later batches, so treating its cumulative KL as a hard
    per-step constraint eventually makes every subsequent update impossible.
    """
    device = next(policy.parameters()).device
    actions, chol = batch["actions"].detach(), batch["chol"].detach()
    old_mean = batch["old_mean"].detach()
    reference = batch["reference"].detach()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=BF16):
        means = policy(batch["context"].detach())
    loss, info = clipped_policy_loss(
        means, actions, batch["old_logp"].detach(), batch["advantages"].detach(),
        reference, chol, cfg.clip_epsilon, cfg.kl_beta,
    )
    pre_ref = equal_covariance_kl(means, reference, chol).float()
    pre_behavior = equal_covariance_kl(means, old_mean, chol).float()
    if not torch.isfinite(loss).all() or not torch.isfinite(pre_ref).all() or not torch.isfinite(pre_behavior).all():
        raise FloatingPointError("Nonfinite policy trust-region diagnostic")
    if float(pre_behavior.max()) > cfg.max_kl:
        return {
            **info, "loss": float(loss.detach()), "accepted": False,
            "reason": "pre_step_behavior_kl", "pre_reference_kl": float(pre_ref.mean()),
            "pre_behavior_kl": float(pre_behavior.mean()),
            "pre_reference_kl_max": float(pre_ref.max()),
            "pre_behavior_kl_max": float(pre_behavior.max()),
        }
    weights = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    loss.backward()
    params = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    grad = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=BF16):
        new_mean = policy(batch["context"].detach())
    post_ref = equal_covariance_kl(new_mean, reference, chol).float()
    post_behavior = equal_covariance_kl(new_mean, old_mean, chol).float()
    finite_post = torch.isfinite(post_ref).all() and torch.isfinite(post_behavior).all()
    accepted = bool(finite_post) and float(post_behavior.max()) <= cfg.max_kl
    if not accepted:
        policy.load_state_dict(weights)
        optimizer.load_state_dict(optimizer_state)
    optimizer.zero_grad(set_to_none=True)
    return {
        **info, "loss": float(loss.detach()), "grad_norm": float(grad), "accepted": accepted,
        "reason": "accepted" if accepted else "post_step_behavior_kl_rollback",
        "pre_reference_kl": float(pre_ref.mean()), "pre_behavior_kl": float(pre_behavior.mean()),
        "pre_reference_kl_max": float(pre_ref.max()), "pre_behavior_kl_max": float(pre_behavior.max()),
        "post_reference_kl": float(post_ref.mean()) if bool(torch.isfinite(post_ref).all()) else None,
        "post_behavior_kl": float(post_behavior.mean()) if bool(torch.isfinite(post_behavior).all()) else None,
        "post_reference_kl_max": float(post_ref.max()) if bool(torch.isfinite(post_ref).all()) else None,
        "post_behavior_kl_max": float(post_behavior.max()) if bool(torch.isfinite(post_behavior).all()) else None,
        "reference_kl_exceeds_behavior_budget": bool(
            bool(torch.isfinite(post_ref).all()) and float(post_ref.max()) > cfg.max_kl
        ),
    }


def policy_step(runner: ECLPRunner, optimizer, batch: dict, plan: dict):
    """Retry only a rolled-back behavior-policy step, without changing rewards."""
    attempts = []
    for index in range(plan["max_lr_backoffs"] + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        info = behavior_guarded_mean_step(runner.heads.policy, optimizer, batch, runner.cfg)
        attempts.append({"learning_rate": learning_rate, **info})
        if info["accepted"]:
            return {**info, "learning_rate": learning_rate, "backoff_attempts": attempts}
        if (info["reason"] != "post_step_behavior_kl_rollback"
                or index == plan["max_lr_backoffs"] or learning_rate <= plan["min_policy_learning_rate"]):
            break
        for group in optimizer.param_groups:
            group["lr"] = max(group["lr"] / 2, plan["min_policy_learning_rate"])
    return {**info, "learning_rate": learning_rate, "backoff_attempts": attempts}


def train(args):
    plan = read_json(args.plan)
    validate_plan(plan)
    cfg, lock, bridge, runner = load_runner(args.config, args.checkpoint)
    verified_prepared(args.prepared, lock)
    points = points_for_split(args.prepared, "rl_train")
    if len(points) != plan["expected_rl_scenarios"]:
        raise ValueError("Offline RL must use the entire fixed 1800-scenario training partition")
    parent_meta = read_json(args.checkpoint / "checkpoint.json")
    if parent_meta["stage"] != "sft" or parent_meta["step"] != plan["sft_target_steps"]:
        raise ValueError("Full offline RL requires the completed one-epoch SFT checkpoint")
    run = {
        "version": VERSION, "plan": plan, "lock_id": lock["lock_id"],
        "initial_checkpoint_id": parent_meta["checkpoint_id"],
        "source_sha256": digest(args.prepared / "rl_train.jsonl"),
        "trainer_sha256": digest(Path(__file__)), "scenario_ids": [p["scenario_id"] for p in points],
        "counterfactual": "same_public_history; zero_latest_feedback_prediction_error_only",
        "reward_formula": {"ppl": "-exp(target_mean_nll)", "nll": "-target_mean_nll",
                           "correction": "-target_mean_nll + lambda*tanh((cf_nll-nll)-(cf_ref_nll-ref_nll))"}[plan["reward"]],
        "policy_ratio": "joint_Gaussian_action_density; sampled_actions_detached",
        "training_dialogue_calls": 0, "training_judge_calls": 0, "generated_training_tokens": 0,
    }
    with writer_lock(args.output / ".writer.lock"):
        # Preserve committed blocks from an earlier trainer and record an
        # explicit continuation contract instead of rewriting their hashes.
        legacy_run_hash = None
        manifest = args.output / "run.json"
        if manifest.exists():
            previous = read_json(manifest)
            identity_keys = [
                "version", "lock_id", "initial_checkpoint_id", "source_sha256",
                "scenario_ids", "counterfactual", "reward_formula", "policy_ratio",
                "training_dialogue_calls", "training_judge_calls", "generated_training_tokens",
            ]
            if any(previous.get(key) != run.get(key) for key in identity_keys):
                raise ValueError("Existing offline run uses a different data/reward contract")
            if previous != run:
                legacy_run_hash = object_hash(previous)
                continuation = {
                    "legacy_run_hash": legacy_run_hash,
                    "legacy_trainer_sha256": previous.get("trainer_sha256"),
                    "legacy_plan": previous.get("plan"),
                    "continuation_run_hash": object_hash(run),
                    "continuation_trainer_sha256": run["trainer_sha256"],
                    "continuation_plan": run["plan"],
                    "reason": "trust_region_semantics_and_batched_policy_update_repair",
                }
                repair_manifest = args.output / "continuation_manifest.json"
                if repair_manifest.exists() and read_json(repair_manifest) != continuation:
                    raise ValueError("Offline continuation manifest changed")
                if not repair_manifest.exists():
                    immutable_json(repair_manifest, continuation)
        else:
            immutable_json(manifest, run)
        accepted_run_hashes = {object_hash(run)} | ({legacy_run_hash} if legacy_run_hash else set())
        seed_all(cfg.seed)
        optimizer = None
        cursor = updates = 0
        final = None
        source_policy_hash = parent_meta["policy_hash"]
        runner.set_stage("rl")
        if (args.output / "last.json").exists():
            final = Path(read_json(args.output / "last.json")["checkpoint"])
            meta = verify_checkpoint(final, lock)
            provenance = meta["provenance"]
            if provenance.get("offline_run_hash") not in accepted_run_hashes:
                raise ValueError("Offline resume belongs to another data/reward contract")
            runner.close()
            runner = ECLPRunner(cfg, bridge, final)
            runner.llm.generate = prohibit_generation
            runner.set_stage("rl")
            state = load_tensors(final / "train_state.pt")
            trainer = state["trainer"]
            cursor, updates = trainer["cursor"], trainer["optimizer_updates"]
            optimizer = FP32MasterAdamW(trainable_parameters(runner), lr=plan["policy_learning_rate"], weight_decay=0)
            optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"])
        else:
            optimizer = FP32MasterAdamW(trainable_parameters(runner), lr=plan["policy_learning_rate"], weight_decay=0)
        frozen = frozen_executor_hash(runner)
        if frozen != parent_meta["executor_hash"]:
            raise RuntimeError("Offline RL executor differs from the completed SFT executor")
        scheduled = []
        for epoch in range(plan["rl_epochs"]):
            order = list(range(len(points)))
            random.Random(cfg.seed + epoch).shuffle(order)
            scheduled.extend((epoch, points[i]) for i in order)
        if not 0 <= cursor <= len(scheduled):
            raise ValueError("Invalid offline resume cursor")
        started = time.monotonic()
        initial_cursor = cursor
        while cursor < len(scheduled):
            end = min(cursor + plan["contexts_per_update"], len(scheduled))
            root = args.output / "blocks" / f"block-{cursor:06d}-{end:06d}"
            attempts = sorted(root.glob("attempt-*")) if root.exists() else []
            attempt = root / f"attempt-{len(attempts) + 1:03d}"
            attempt.mkdir(parents=True)
            # An interrupted block is recomputed from the last COMMITTED policy.
            # Earlier attempts stay on disk for cost accounting and diagnosis.
            immutable_json(attempt / "run.json", {
                "offline_run_hash": object_hash(run), "cursor": cursor, "end": end,
                "behavior_policy_hash": tensor_digest(runner.heads.policy.state_dict()),
                "scenario_ids": [point["scenario_id"] for _, point in scheduled[cursor:end]],
            })
            groups = []
            for epoch, point in scheduled[cursor:end]:
                group, report = score_group(runner, point, plan, epoch)
                groups.append(group)
                append_jsonl(attempt / "rewards.jsonl", report)
                atomic_json(args.output / "progress.json", {
                    "state": "collecting_offline_rewards", "committed_contexts": cursor,
                    "scored_in_current_block": len(groups), "target_contexts": len(scheduled),
                    "optimizer_updates": updates, "last_scenario": point["scenario_id"],
                    "last_group_seconds": report["seconds"], "last_rewards": report["reward"],
                    "last_correction_bonus": report["centered_correction_bonus"],
                    "dialogue_calls": 0, "judge_calls": 0,
                })
            data, scale = policy_data(groups, plan)
            save_tensors(attempt / "batch.pt", data)
            # Keep replay in the same single-example BF16 kernel used when
            # old_logp was recorded; larger optimizer batches can change
            # matmul rounding enough to look like a false density mismatch.
            replay_error = replay_density_check(runner, data, plan["score_batch_size"])
            nonzero = int(torch.count_nonzero(data["advantages"]).item())
            if not nonzero:
                atomic_json(attempt / "diagnostic.json", {"reason": "all_latent_rewards_identical", "cursor": cursor})
                raise RuntimeError("All latent reward differences are zero; diagnose decoder sensitivity before claiming RL")
            before = updates
            early_stop = False
            for epoch in range(plan["policy_update_epochs"]):
                order = list(range(len(data["actions"])))
                random.Random(cfg.seed + cursor * 7 + epoch).shuffle(order)
                for start in range(0, len(order), plan["policy_update_batch_size"]):
                    indices = order[start:start + plan["policy_update_batch_size"]]
                    batch = {k: v[indices].to(runner.device) for k, v in data.items()}
                    info = policy_step(runner, optimizer, batch, plan)
                    append_jsonl(attempt / "updates.jsonl", {"update_before": updates, "epoch": epoch, **info})
                    if not info["accepted"]:
                        early_stop = True
                        break
                    updates += 1
                if early_stop:
                    break
            if frozen_executor_hash(runner) != frozen:
                raise RuntimeError("Offline RL modified frozen state, decoder or latent execution mapping")
            if updates == before:
                atomic_json(attempt / "diagnostic.json", {"reason": "no_accepted_policy_step", "cursor": cursor})
                raise RuntimeError("No update passed the trust region; retain diagnostics instead of silently skipping data")
            provenance = {
                "offline_trainer_version": VERSION, "offline_run_hash": object_hash(run),
                "trainer_sha256": run["trainer_sha256"], "parent_sft_checkpoint_id": parent_meta["checkpoint_id"],
                "runtime_trainer_sha256": digest(Path(__file__)),
                "reward": plan["reward"], "correction_weight": plan["correction_weight"],
                "processed_contexts": end, "optimizer_updates": updates,
                "block": str(attempt), "block_rewards_sha256": digest(attempt / "rewards.jsonl"),
                "block_batch_sha256": digest(attempt / "batch.pt"), "advantage_scale": scale,
                "behavior_logp_max_error": replay_error, "kl_early_stop": early_stop,
                "executor_unchanged": True, "new_persuadee_calls": 0,
                "judge_calls": 0, "autoregressive_generated_tokens": 0,
            }
            final = save_checkpoint(runner, args.output, lock, "rl", end, optimizer,
                                    {"cursor": end, "optimizer_updates": updates, "run_hash": object_hash(run)},
                                    provenance)
            atomic_json(attempt / "commit.json", {"checkpoint": str(final), **provenance})
            cursor = end
            atomic_json(args.output / "progress.json", {
                "state": "committed_offline_update", "committed_contexts": cursor,
                "target_contexts": len(scheduled), "optimizer_updates": updates,
                "seconds_this_invocation": time.monotonic() - started,
                "contexts_this_invocation": cursor - initial_cursor,
                "dialogue_calls": 0, "judge_calls": 0, "checkpoint": str(final),
            })
            print(f"OFFLINE committed {cursor}/{len(scheduled)} contexts; {updates} policy steps; zero dialogue/Judge calls", flush=True)
        if final is None or updates < 1:
            raise RuntimeError("No committed offline policy checkpoint")
        final_meta = verify_checkpoint(final, lock)
        if final_meta["policy_hash"] == source_policy_hash:
            raise RuntimeError("Policy parameters did not change; do not claim successful RL")
        all_rewards = []
        committed = []
        for commit in sorted((args.output / "blocks").glob("block-*/attempt-*/commit.json")):
            info = read_json(commit)
            if info["offline_run_hash"] not in accepted_run_hashes:
                raise ValueError("Mixed offline block contracts")
            rewards_path = commit.parent / "rewards.jsonl"
            if digest(rewards_path) != info["block_rewards_sha256"]:
                raise ValueError("Committed offline rewards changed")
            import json
            all_rewards.extend(json.loads(line) for line in rewards_path.read_text().splitlines())
            committed.append(str(commit))
        # If a crash separated checkpoint commit from block commit, recover
        # the missing marker only from the checkpoint's recorded provenance.
        if len(all_rewards) != len(scheduled):
            for checkpoint in sorted(args.output.glob("checkpoint-*/checkpoint.json")):
                prov = read_json(checkpoint).get("provenance", {})
                block = Path(prov.get("block", "/nonexistent"))
                if prov.get("offline_run_hash") in accepted_run_hashes and not (block / "commit.json").exists():
                    verify_checkpoint(checkpoint.parent, lock)
                    if digest(block / "rewards.jsonl") != prov["block_rewards_sha256"]:
                        raise ValueError("Interrupted block's reward hash changed")
                    atomic_json(block / "commit.json", {"checkpoint": str(checkpoint.parent), **prov})
                    import json
                    all_rewards.extend(json.loads(line) for line in (block / "rewards.jsonl").read_text().splitlines())
                    committed.append(str(block / "commit.json"))
        seen = [row["scenario_id"] for row in all_rewards]
        expected = [point["scenario_id"] for _, point in scheduled]
        if sorted(seen) != sorted(expected):
            raise RuntimeError("Offline RL did not cover exactly the scheduled training partition")
        summary = {
            "complete": True, "version": VERSION, "offline_run_hash": object_hash(run),
            "checkpoint": str(final), "checkpoint_id": final_meta["checkpoint_id"],
            "contexts": cursor, "unique_scenarios": len(set(seen)),
            "optimizer_updates": updates, "blocks": committed,
            "reward_mode": plan["reward"], "correction_weight": plan["correction_weight"],
            "contexts_with_feedback": sum(row["has_feedback"] for row in all_rewards),
            "mean_candidate_nll": statistics.fmean(v for row in all_rewards for v in row["factual_nll"][1:]),
            "mean_absolute_paired_reward_delta": statistics.fmean(abs(v) for row in all_rewards for v in row["paired_reward_delta"]),
            "new_persuadee_calls": 0, "judge_calls": 0, "generated_training_tokens": 0,
            "executor_unchanged": True, "policy_changed": True,
            "continued_from_run_hash": legacy_run_hash,
            "runtime_trainer_sha256": digest(Path(__file__)),
            "seconds_this_invocation": time.monotonic() - started,
            "note": "Offline proxy reward is not observed persuasion success; final dialogue evaluation is separate.",
        }
        atomic_json(args.output / "summary.json", summary)
        atomic_json(args.output / "progress.json", {"state": "complete", **summary})
    runner.close()
    bridge.close()


def diagnose(args):
    """Real GPU verification against an existing checkpoint; never an experiment score."""
    plan = read_json(args.plan)
    validate_plan(plan)
    cfg, lock, bridge, runner = load_runner(args.config, args.checkpoint)
    points = points_for_split(args.prepared, "rl_train", args.limit)
    before = frozen_executor_hash(runner)
    policy_before = tensor_digest(runner.heads.policy.state_dict())
    groups, rows = [], []
    scalar_error = 0.0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=BF16):
        for point in points:
            group, row = score_group(runner, point, plan, 0)
            groups.append(group)
            rows.append(row)
        # Check batch score alignment against the core's scalar target NLL.
        prefix, _, _, _ = paired_contexts(runner, points[-1])
        target = points[-1]["gold_response"]
        for i, action in enumerate(groups[-1]["actions"]):
            scalar, _ = runner.utterance_loss(prefix, action.unsqueeze(0).to(runner.device), target)
            scalar_error = max(scalar_error, abs(float(scalar) - rows[-1]["factual_nll"][i + 1]))
    if scalar_error > 0.005:
        atomic_json(args.output.with_suffix(".failed.json"), {
            "status": "FAIL_OFFLINE_NLL_ALIGNMENT", "rows": rows,
            "batch_scalar_nll_max_error": scalar_error, "plan": plan,
            "trainer_sha256": digest(Path(__file__)),
        })
        raise RuntimeError(f"Batched target NLL differs from the core scalar path: {scalar_error}")
    data, scale = policy_data(groups, plan)
    runner.set_stage("rl")
    replay = replay_density_check(runner, data, 1)
    optimizer = FP32MasterAdamW(trainable_parameters(runner), lr=plan["policy_learning_rate"], weight_decay=0)
    index = int(data["advantages"].abs().argmax())
    if not float(data["advantages"][index].abs()) > 0:
        raise RuntimeError("No dense reward signal in the real-model diagnostic")
    batch = {k: v[index:index + 1].to(runner.device) for k, v in data.items()}
    info = policy_step(runner, optimizer, batch, plan)
    if not info["accepted"] or tensor_digest(runner.heads.policy.state_dict()) == policy_before:
        atomic_json(args.output.with_suffix(".failed.json"), {
            "status": "FAIL_OFFLINE_POLICY_UPDATE", "rows": rows, "plan": plan,
            "policy_update": info, "batch_scalar_nll_max_error": scalar_error,
            "policy_changed": tensor_digest(runner.heads.policy.state_dict()) != policy_before,
            "trainer_sha256": digest(Path(__file__)),
        })
        print(f"FAILED_POLICY_UPDATE {info}", flush=True)
        raise RuntimeError("Real offline diagnostic did not change the mean policy within the trust region")
    if frozen_executor_hash(runner) != before:
        raise RuntimeError("Diagnostic changed a frozen execution component")
    for function in [bridge.respond, bridge.score, runner.llm.generate]:
        try:
            function()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Offline generation prohibition failed")
    atomic_json(args.output, {
        "status": "PASS_OFFLINE_GPU_DIAGNOSTIC_NOT_AN_EXPERIMENT_RESULT",
        "checkpoint": str(args.checkpoint), "trainer_sha256": digest(Path(__file__)),
        "groups": len(groups), "rows": rows, "batch_scalar_nll_max_error": scalar_error,
        "behavior_logp_replay_error": replay, "advantage_scale": scale,
        "policy_update": info, "executor_unchanged": True, "policy_changed": True,
        "new_persuadee_calls": 0, "judge_calls": 0, "generated_training_tokens": 0,
        "mean_seconds_per_context": statistics.fmean(row["seconds"] for row in rows),
        "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(runner.device),
    })
    runner.close()
    bridge.close()


def validate(args):
    """Held-out proxy diagnostic; compare final policy to its fixed SFT reference."""
    plan = read_json(args.plan)
    cfg, lock, bridge, runner = load_runner(args.config, args.checkpoint)
    points = points_for_split(args.prepared, "dev")
    if len(points) != plan["expected_dev_scenarios"]:
        raise ValueError("Use the entire existing 200-scenario dev partition")
    rows = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=BF16):
        for point in points:
            prefix, context, counter_prefix, has_feedback = paired_contexts(runner, point)
            mean = runner.heads.policy(context).float()
            reference = runner.reference_policy(context).float()
            actions = torch.cat([reference, mean])
            factual, tokens = target_nlls(runner, prefix, actions, point["gold_response"], plan["score_batch_size"])
            if has_feedback:
                counter, _ = target_nlls(runner, counter_prefix, actions, point["gold_response"], plan["score_batch_size"])
            else:
                counter = factual.clone()
            benefit = counter - factual
            rows.append({
                "scenario_id": point["scenario_id"], "target_tokens": tokens,
                "has_feedback": has_feedback, "sft_nll": float(factual[0]),
                "policy_nll": float(factual[1]), "nll_delta_policy_minus_sft": float(factual[1] - factual[0]),
                "sft_correction_benefit": float(benefit[0]), "policy_correction_benefit": float(benefit[1]),
                "correction_benefit_delta": float(benefit[1] - benefit[0]),
            })
    nll_delta = statistics.fmean(row["nll_delta_policy_minus_sft"] for row in rows)
    feedback = [row for row in rows if row["has_feedback"]]
    atomic_json(args.output, {
        "complete": True, "checkpoint_id": read_json(args.checkpoint / "checkpoint.json")["checkpoint_id"],
        "data_sha256": digest(args.prepared / "dev.jsonl"), "n": len(rows),
        "sft_mean_nll": statistics.fmean(row["sft_nll"] for row in rows),
        "policy_mean_nll": statistics.fmean(row["policy_nll"] for row in rows),
        "mean_nll_delta_policy_minus_sft": nll_delta,
        "beats_sft_nll": nll_delta < 0,
        "proxy_quality_gate_pass": nll_delta <= plan["dev_nll_tolerance"],
        "dev_nll_tolerance": plan["dev_nll_tolerance"],
        "feedback_contexts": len(feedback),
        "mean_correction_benefit_delta_feedback": statistics.fmean(row["correction_benefit_delta"] for row in feedback) if feedback else None,
        "new_persuadee_calls": 0, "judge_calls": 0, "generated_training_tokens": 0,
        "note": "Held-out answer likelihood is a proxy, not measured persuasion success or proof of innovation.",
        "rows": rows,
    })
    runner.close()
    bridge.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CODE / "config_gpu4.json")
    parser.add_argument("--plan", type=Path, default=CODE / "offline_training_plan.json")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["train", "diagnose", "validate"]:
        part = sub.add_parser(command)
        part.add_argument("--prepared", type=Path, required=True)
        part.add_argument("--checkpoint", type=Path, required=True)
        part.add_argument("--output", type=Path, required=True)
        if command == "diagnose":
            part.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Offline trainer must run only on physical GPU1")
    cfg = Config.load(args.config)
    if not args.output.resolve().is_relative_to(cfg.work.resolve()):
        raise ValueError("Offline artifacts must stay inside the current Our work directory")
    if args.command == "train":
        train(args)
    elif args.command == "diagnose":
        if args.limit < 2:
            raise ValueError("Diagnostic needs several histories, not a single empty context")
        diagnose(args)
    else:
        validate(args)


if __name__ == "__main__":
    main()
