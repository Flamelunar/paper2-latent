"""Finite-horizon action selection: max over actions, mean over reactions."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from typing import Any
from evaluation.ledger import RequestLedger
from evaluation.revision_prompts import utterance_prompt, transition_prompt, progress_prompt
from evaluation.revision_schema import single_text, parse_transition, parse_progress


@dataclass(frozen=True)
class PlannerConfig:
    root_actions: int = 3
    reaction_samples_per_action: int = 2
    max_depth: int = 2
    future_actions_per_node: int = 1
    max_candidate_tokens: int = 128
    max_reaction_tokens: int = 128
    max_score_tokens: int = 192
    proposer_temperature: float = 0.7
    proposer_top_p: float = 0.9
    world_model_temperature: float = 0.7
    world_model_top_p: float = 0.95
    analysis_budget: int = 0
    structural_retry_limit: int = 1

    def validate(self) -> None:
        if min(self.root_actions, self.reaction_samples_per_action, self.max_depth, self.future_actions_per_node) < 1:
            raise ValueError("B, K, D and B_future must be positive")
        if self.structural_retry_limit < 0 or self.analysis_budget < 0:
            raise ValueError("Retry/analysis budgets must be non-negative")
        if self.reaction_samples_per_action > 1 and self.world_model_temperature <= 0:
            raise ValueError("K>1 requires stochastic world-model decoding")


def append_pair(turns: list[dict[str, Any]], action: str, reply: str, *, simulated: bool) -> list[dict[str, Any]]:
    return [dict(t) for t in turns] + [{"role": "persuader", "text": action, "simulated": simulated},
                                     {"role": "persuadee", "text": reply, "simulated": simulated}]


class RolloutPlanner:
    def __init__(self, ledger: RequestLedger, config: PlannerConfig) -> None:
        config.validate()
        self.ledger, self.config = ledger, config
        self.trace: list[dict[str, Any]] = []

    def propose(self, item: dict[str, Any], turns: list[dict[str, Any]], count: int,
                remaining: int, node_id: str) -> list[dict[str, str]]:
        # Individual samples with a count-independent prompt keep candidate 0
        # fixed when B changes. Exact duplicate actions alone are deduplicated.
        context_hash = hashlib.sha256(json.dumps(turns, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        actions: list[dict[str, str]] = []
        seen = set()
        requested = count
        for index in range(count + self.config.structural_retry_limit):
            text = self.ledger.structured("persuader", utterance_prompt(item, turns, remaining),
                f"{node_id}/propose-{index}", lambda raw: single_text(raw, "persuader_utterance"),
                max_tokens=self.config.max_candidate_tokens, temperature=self.config.proposer_temperature,
                top_p=self.config.proposer_top_p,
                seed_key=f"propose/{context_hash}/{remaining}/{index}")
            key = " ".join(text.split()).casefold()
            if key in seen:
                self.trace.append({"kind": "duplicate_action", "node_id": node_id, "candidate_text": text})
                continue
            seen.add(key)
            actions.append({"action_id": f"{node_id}/action-{len(actions)}", "text": text})
            if len(actions) == count:
                break
        if not actions:
            raise RuntimeError(f"No valid actions at {node_id}")
        self.trace.append({"kind": "action_pool", "node_id": node_id, "requested": requested,
                           "valid": len(actions), "actions": actions, "context_hash": context_hash})
        return actions

    def score(self, item: dict[str, Any], turns: list[dict[str, Any]], node_id: str) -> dict[str, Any]:
        score = self.ledger.structured("progress_scorer", progress_prompt(item, turns), f"{node_id}/progress",
            lambda raw: parse_progress(raw, turns), max_tokens=self.config.max_score_tokens)
        self.trace.append({"kind": "progress_score", "node_id": node_id, **score})
        return score

    def action_value(self, item: dict[str, Any], turns: list[dict[str, Any]], action: dict[str, str],
                     depth: int, remaining: int) -> float:
        values = []
        for sample in range(self.config.reaction_samples_per_action):
            node = f"{action['action_id']}/reaction-{sample}"
            # The K samples see exactly the same prompt, with independent RNG.
            prediction = self.ledger.structured("world_model",
                transition_prompt(item, turns, action["text"], self.config.analysis_budget), f"{node}/predict",
                lambda raw: parse_transition(raw, self.config.analysis_budget),
                max_tokens=self.config.analysis_budget + self.config.max_reaction_tokens,
                temperature=self.config.world_model_temperature, top_p=self.config.world_model_top_p)
            branch = append_pair(turns, action["text"], prediction["predicted_reply"], simulated=True)
            assessment = self.score(item, branch, node)
            terminal = assessment["goal_achieved"] or assessment["end_requested"]
            row = {"kind": "world_model_prediction", "node_id": node, "parent_node_id": action["action_id"],
                "action_id": action["action_id"], "candidate_text": action["text"], "reaction_sample_id": sample,
                "depth_remaining": depth, "branch_history": branch, **prediction, **assessment,
                "terminal_in_simulation": terminal}
            self.trace.append(row)
            if depth == 1 or terminal:
                value = assessment["progress_level"] / 4.0
            else:
                future = self.propose(item, branch, self.config.future_actions_per_node, remaining - 1, node)
                value = max(self.action_value(item, branch, a, depth - 1, remaining - 1) for a in future)
            row["backed_up_value"] = value
            values.append(value)
        # Repeated/refusing reactions retain full probability mass. A terminal
        # branch contributes 1/K even when another branch expands into K leaves.
        return sum(values) / len(values)

    def plan(self, item: dict[str, Any], turns: list[dict[str, Any]], *, turn_index: int,
             max_real_turns: int, root_actions: list[dict[str, str]] | None = None) -> dict[str, Any]:
        self.trace = []
        node = f"turn-{turn_index}/root"
        remaining = max_real_turns - turn_index + 1
        depth = min(self.config.max_depth, remaining)
        if depth < 1:
            raise ValueError("No remaining real turns")
        actions = root_actions or self.propose(item, turns, self.config.root_actions, remaining, node)
        # Initial progress is identical across actions; logging it gives delta
        # values without changing ranking. No fictitious initial reply is added.
        initial = None
        failure = None
        values: list[float | None] = []
        try:
            initial = self.score(item, turns, node + "/initial")["progress_level"] / 4 if turns else 0.0
            values = [self.action_value(item, turns, a, depth, remaining) for a in actions]
            selected = max(range(len(actions)), key=lambda i: values[i])
        except Exception as exc:
            # Discard the partial ranking, select first VALID root action only.
            # Every failed request remains in the ledger with its cost/error.
            failure = f"{type(exc).__name__}: {exc}"
            values = [None] * len(actions)
            selected = 0
        return {"depth": depth, "candidate_actions": [{**a, "value": v,
                    "progress_delta": v - initial if v is not None and initial is not None else None}
                    for a, v in zip(actions, values)],
                "selected_action_id": actions[selected]["action_id"], "selected_text": actions[selected]["text"],
                "trace": self.trace, "fallback_used": failure is not None, "planning_failure": failure,
                "search_summary": {"root_actions": len(actions), "reaction_samples_per_action": self.config.reaction_samples_per_action,
                    "depth": depth, "world_model_prediction_count": sum(t["kind"] == "world_model_prediction" for t in self.trace),
                    "progress_score_count": sum(t["kind"] == "progress_score" for t in self.trace)}}
