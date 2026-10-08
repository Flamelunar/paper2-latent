"""LatentRATT wrapper for a Hugging Face causal LM.

The attention layer is the upstream implementation in upstream_R3/src/latent/layers.py.
One continuous thought is inserted between the chat prompt and public utterance.
"""
from __future__ import annotations

import sys
import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM = Path(__file__).resolve().parent / "upstream_R3/src/latent"
sys.path.insert(0, str(UPSTREAM))
from layers import SelfAttentionLayer  # noqa: E402


class LatentPersuader(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.attention = SelfAttentionLayer(backbone.config.hidden_size)
        embedding = backbone.get_input_embeddings().weight
        self.attention.to(device=embedding.device, dtype=embedding.dtype)

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def thought(self, prompt_ids: torch.Tensor, *, freeze_backbone: bool = False):
        mask = torch.ones_like(prompt_ids)
        with torch.set_grad_enabled(not freeze_backbone):
            output = self.backbone(input_ids=prompt_ids, attention_mask=mask,
                                   output_hidden_states=True, use_cache=False)
            hidden = output.hidden_states[-1]
        if freeze_backbone:
            hidden = hidden.detach()
        positions = torch.full((prompt_ids.shape[0],), prompt_ids.shape[1],
                               device=prompt_ids.device, dtype=torch.long)
        return self.attention(hidden, mask, positions).unsqueeze(1)

    def target_logps(self, prompt_ids: torch.Tensor, target_ids: torch.Tensor,
                     thought: torch.Tensor):
        embeddings = self.backbone.get_input_embeddings()
        full = torch.cat((embeddings(prompt_ids), thought, embeddings(target_ids)), dim=1)
        output = self.backbone(inputs_embeds=full, use_cache=False)
        start = prompt_ids.shape[1]
        logits = output.logits[:, start:start + target_ids.shape[1], :]
        return F.log_softmax(logits.float(), dim=-1).gather(
            -1, target_ids.unsqueeze(-1)).squeeze(-1)

    def cached_prompt_state(self, prompt_ids: torch.Tensor):
        """Cache the frozen prompt pass for repeated LR-GRPO latent proposals."""
        mask = torch.ones_like(prompt_ids)
        with torch.no_grad():
            output = self.backbone(
                input_ids=prompt_ids, attention_mask=mask, output_hidden_states=True,
                use_cache=True, logits_to_keep=1)
        hidden = output.hidden_states[-1].detach()
        positions = torch.full((prompt_ids.shape[0],), prompt_ids.shape[1],
                               device=prompt_ids.device, dtype=torch.long)
        return hidden, mask, positions, output.past_key_values

    def latent_from_hidden(self, hidden: torch.Tensor, mask: torch.Tensor,
                           positions: torch.Tensor):
        return self.attention(hidden, mask, positions).unsqueeze(1)

    def _fork_prefix_cache(self, cache):
        # Transformers' DynamicCache is mutated by each forward call. Create a
        # fresh view of its prefix tensors for each proposal; new tokens are
        # appended into new tensors, leaving the shared prompt prefix intact.
        if hasattr(cache, "layers"):
            cache_data = []
            for layer in cache.layers:
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if keys is None or values is None:
                    cache_data.append((keys, values))
                    continue
                entry = [keys, values]
                sliding_window = getattr(layer, "sliding_window", None)
                if sliding_window is not None:
                    entry.append(sliding_window)
                cache_data.append(tuple(entry))
            try:
                return cache.__class__(ddp_cache_data=cache_data,
                                       config=self.backbone.config)
            except TypeError:
                pass
        if isinstance(cache, tuple):
            return tuple(tuple(t for t in layer) for layer in cache)
        return copy.deepcopy(cache)

    def target_logps_from_cache(self, target_ids: torch.Tensor, thought: torch.Tensor,
                                prompt_length: int, prefix_cache):
        """Score targets from a cached prompt while retaining gradients to thought."""
        target_embeds = self.backbone.get_input_embeddings()(target_ids)
        continuation = torch.cat((thought, target_embeds), dim=1)
        attention_mask = torch.ones(
            (target_ids.shape[0], prompt_length + continuation.shape[1]),
            device=target_ids.device, dtype=torch.long)
        output = self.backbone(
            inputs_embeds=continuation, attention_mask=attention_mask,
            past_key_values=self._fork_prefix_cache(prefix_cache), use_cache=False,
            logits_to_keep=target_ids.shape[1] + 1)
        logits = output.logits[:, :target_ids.shape[1], :]
        return F.log_softmax(logits.float(), dim=-1).gather(
            -1, target_ids.unsqueeze(-1)).squeeze(-1)

    def sft_loss(self, prompt_ids: torch.Tensor, target_ids: torch.Tensor):
        thought = self.thought(prompt_ids)
        return -self.target_logps(prompt_ids, target_ids, thought).mean()

    @torch.no_grad()
    def generate_text(self, prompt_ids: torch.Tensor, *, max_new_tokens: int,
                      eos_token_id: int, temperature: float = 0.95,
                      top_p: float = 0.7, top_k: int = 50):
        thought = self.thought(prompt_ids, freeze_backbone=True)
        embeds = torch.cat((self.backbone.get_input_embeddings()(prompt_ids), thought), dim=1)
        result = []
        output = self.backbone(inputs_embeds=embeds, use_cache=True)
        past = output.past_key_values
        logits = output.logits[:, -1, :]
        for _ in range(max_new_tokens):
            if temperature <= 0:
                token = logits.argmax(-1, keepdim=True)
            else:
                scores = logits.float() / temperature
                if top_k > 0:
                    kth = scores.topk(min(top_k, scores.shape[-1]), dim=-1).values[:, -1:]
                    scores = scores.masked_fill(scores < kth, float("-inf"))
                if 0 < top_p < 1:
                    sorted_scores, sorted_indices = scores.sort(descending=True)
                    cumulative = sorted_scores.softmax(-1).cumsum(-1)
                    remove = cumulative - sorted_scores.softmax(-1) > top_p
                    sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
                    scores = torch.full_like(scores, float("-inf")).scatter(-1, sorted_indices, sorted_scores)
                token = torch.multinomial(scores.softmax(-1), 1)
            value = token.item()
            if value == eos_token_id:
                break
            result.append(value)
            output = self.backbone(input_ids=token, past_key_values=past, use_cache=True)
            past = output.past_key_values
            logits = output.logits[:, -1, :]
        return result


def lr_grpo_loss(model: LatentPersuader, examples: list[tuple[torch.Tensor, torch.Tensor]],
                 *, group_size: int = 4, sigma: float = 1.5, clip_epsilon: float = 0.2,
                 backward_per_proposal: bool = False, gradient_scale: float = 1.0):
    """Negative-PPL rewards and batch-relative advantages from LatentR3.

    The backbone is frozen in this phase. Group members share one prompt and
    target; the unperturbed member supplies each group's baseline reward.
    """
    if len(examples) < 2:
        raise ValueError("LR-GRPO needs at least two examples for batch baseline")
    groups = []
    with torch.no_grad():
        for prompt, target in examples:
            hidden, mask, positions, prefix_cache = model.cached_prompt_state(prompt)
            center = model.latent_from_hidden(hidden, mask, positions)
            group = []
            for sample in range(group_size):
                noise = torch.zeros_like(center) if sample == 0 else torch.randn_like(center) * sigma
                latent = center + noise
                logps = model.target_logps_from_cache(
                    target, latent, prompt.shape[1], prefix_cache)
                reward = -torch.exp(-logps.mean()).detach()
                group.append((noise, logps.detach(), reward))
            groups.append((target, hidden, mask, positions, prefix_cache, group))
    proposals = [proposal for *_, group in groups for proposal in group]
    rewards = torch.stack([proposal[2] for proposal in proposals]).flatten()
    baseline = rewards[::group_size].mean()
    advantages = (rewards - baseline) / torch.linalg.vector_norm(rewards - baseline).clamp_min(1e-6)
    detached_loss = torch.zeros((), device=rewards.device)
    losses = []
    proposal_index = 0
    for target, hidden, mask, positions, prefix_cache, group in groups:
        center = model.latent_from_hidden(hidden, mask, positions)
        center_gradient = None
        for noise, old_logps, _ in group:
            advantage = advantages[proposal_index]
            proposal_index += 1
            logps = model.target_logps_from_cache(
                target, center + noise, hidden.shape[1], prefix_cache)
            ratio = (logps - old_logps).clamp(-20, 20).exp()
            clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
            proposal_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
            detached_loss += proposal_loss.detach()
            if backward_per_proposal:
                # Every proposal in a group differentiates through the same
                # latent center. Keep that small attention graph alive until
                # all proposal gradients have been accumulated below.
                gradient = torch.autograd.grad(
                    proposal_loss * gradient_scale / len(proposals), center, retain_graph=True)[0]
                center_gradient = gradient if center_gradient is None else center_gradient + gradient
            else:
                losses.append(proposal_loss)
        if backward_per_proposal:
            center.backward(center_gradient)
    if backward_per_proposal:
        return detached_loss / len(proposals), rewards.mean().item()
    return torch.stack(losses).mean(), rewards.mean().item()
