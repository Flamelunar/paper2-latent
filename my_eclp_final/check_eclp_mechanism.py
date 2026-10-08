#!/usr/bin/env python3
"""Small reproducible checks; no pretrained models, data edits or training runs."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import torch
from torch.nn import functional as F


def load_module(path, name="eclp_mechanism_checked"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, default=Path(__file__).with_name("eclp_elpo.py"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    m = load_module(args.implementation)
    device = torch.device(args.device)
    if device.type == "cuda":
        m.cuda_bf16(args.device)
    report = {"implementation_sha256": m.IMPLEMENTATION_SHA256, "version": m.VERSION,
              "environment": m.environment_versions(), "device": str(device),
              "not_a_task_effectiveness_experiment": True, "checks": []}
    for mode in ("eclp", "no_error", "no_correction", "stateless"):
        for seed in (41, 42, 43):
            torch.manual_seed(seed)
            cfg = m.Config(width=32, feature_dim=16, attention_heads=4,
                           action_dim=8, effect_rank=4, state_mode=mode)
            cfg.validate()
            state = m.EvidenceState(cfg).to(device=device, dtype=m.BF16).eval()
            generator = torch.Generator().manual_seed(seed + 100)

            def sample(*shape):
                return torch.randn(shape, generator=generator).to(device=device, dtype=m.BF16)

            B = state.initial(sample(1, 5, cfg.feature_dim))
            policy_B = B.detach()
            first_correction = None
            observations = []
            for step in range(4):
                spoken, tokens, observed = sample(1, 16), sample(1, 4, 16), sample(1, 16)
                prior, predicted = state.prior(B, spoken)
                pprior, ppredicted = state.policy_prior(policy_B, spoken)
                assert torch.equal(prior, pprior), (mode, seed, step, "prior values differ")
                assert torch.equal(predicted, ppredicted), (mode, seed, step, "prediction values differ")
                observations.append(F.mse_loss(predicted.float(), observed.float()))
                B = state.correct(prior, predicted, tokens, observed)
                policy_B = state.policy_correct(pprior, ppredicted, tokens, observed)
                assert torch.equal(B, policy_B), (mode, seed, step, "posterior values differ")
                if step == 0:
                    first_correction = policy_B
            weights = sample(*policy_B.shape).float()
            public = (policy_B.float() * weights).sum()
            named = list(state.named_parameters())
            params = [parameter for _, parameter in named]
            gradients = (torch.autograd.grad(public, params, allow_unused=True, retain_graph=True)
                         if public.requires_grad else [None] * len(params))
            norms = {name: 0.0 if grad is None else float(grad.float().norm())
                     for (name, _), grad in zip(named, gradients)}
            observation_grads = torch.autograd.grad(sum(observations), params, allow_unused=True)
            observation_norm = max(0.0 if g is None else float(g.float().norm()) for g in observation_grads)
            isolated = max(norms[n] for n in norms if n == "slots" or n.startswith(
                ("read.", "to_memory.", "transition.", "prior_norm.", "predict.")))
            correction = max(norms[n] for n in norms if n.startswith(("correction.", "gate.", "posterior_norm.")))
            error = norms["error_projection.weight"] + norms["error_projection.bias"]
            assert isolated == 0, (mode, seed, "predictor leak", isolated)
            assert observation_norm > 0, (mode, seed, "observation has no gradient")
            if mode in ("eclp", "no_error"):
                assert correction > 0, (mode, seed, "missing correction gradient")
                history_grad = torch.autograd.grad(public, first_correction, retain_graph=True)[0]
                assert torch.isfinite(history_grad).all() and float(history_grad.float().norm()) > 0
            else:
                assert correction == 0, (mode, seed, "unexpected correction gradient")
            assert (error > 0) if mode == "eclp" else (error == 0), (mode, seed, error)
            report["checks"].append({"mode": mode, "seed": seed, "cycles": 4,
                "state_values_bitwise_equal": True, "public_predictor_gradient": isolated,
                "public_correction_gradient": correction, "public_error_gradient": error,
                "observation_gradient_max": observation_norm})

    # The policy prior retains the true derivative w.r.t. input state; its
    # weights, unlike observation prior weights, receive no public gradient.
    cfg = m.Config(width=16, feature_dim=8, attention_heads=4, action_dim=8, effect_rank=4)
    module = m.EvidenceState(cfg).double().cpu()
    original = torch.randn(1, 2, 16, dtype=torch.float64, requires_grad=True)
    frozen = original.detach().clone().requires_grad_(True)
    spoken = torch.randn(1, 8, dtype=torch.float64)
    probe = torch.randn_like(original)
    normal, _ = module.prior(original, spoken)
    policy, _ = module.policy_prior(frozen, spoken)
    actual = torch.autograd.grad((normal * probe).sum(), original)[0]
    checked = torch.autograd.grad((policy * probe).sum(), frozen)[0]
    assert torch.equal(normal, policy)
    torch.testing.assert_close(actual, checked, rtol=1e-12, atol=1e-12)
    report["prior_input_jacobian_matches_actual_prior"] = True

    # Replacing a file cannot alter a running module's provenance. An old
    # version is refused even when its recorded SHA has been mislabeled.
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "snapshot.py"
        source.write_bytes(args.implementation.read_bytes())
        loaded = load_module(source, "eclp_hash_snapshot_checked")
        bound = loaded.IMPLEMENTATION_SHA256
        source.write_bytes(source.read_bytes() + b"\n# disk edit after import\n")
        assert loaded.sha256(source) != bound and loaded.IMPLEMENTATION_SHA256 == bound
        checkpoint = Path(temp) / "checkpoint"
        checkpoint.mkdir()
        loaded.atomic_json(checkpoint / "checkpoint.json", {"version": "eclp-elpo-bf16-v2.0",
            "lock_id": "test", "implementation_sha256": bound})
        try:
            loaded.verify_checkpoint(checkpoint, {"lock_id": "test"})
        except ValueError as error:
            assert "version" in str(error)
        else:
            raise AssertionError("old checkpoint version accepted")
    report["startup_hash_immutable_and_old_version_rejected"] = True
    # The actor checkpoint defaults to 1.05. Verify that embeds-only generation
    # retains the ordinary HF penalty on the real prompt IDs, once per token.
    from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
    prompt = torch.tensor([[1, 3, 3]], device=device)
    generated = torch.tensor([[2, 3]], device=device)
    scores = torch.tensor([[0.0, 2.0, -2.0, 4.0, 5.0]], device=device)
    expected = RepetitionPenaltyLogitsProcessor(1.05)(torch.cat([prompt, generated], dim=1), scores.clone())
    actual = m.PromptRepetitionPenalty(prompt, 1.05)(generated, scores.clone())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[0, 4] == scores[0, 4]
    report["public_prompt_repetition_penalty_matches_hf"] = True
    report["status"] = "PASS_MECHANISM_ONLY"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        m.atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
