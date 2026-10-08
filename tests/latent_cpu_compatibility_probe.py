"""Exercise both real causal-LM classes at tiny sizes; never load full weights.

This checks CPU API/cache/gradient compatibility, not the required full-model
GPU smoke test, which the supervisor runs before each family's full training.
"""

import json
import math
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline5_latentRATT"))
from latent_model import LatentPersuader, lr_grpo_loss
from run_latent_ratt import LORA_TARGETS


def probe(model_path):
    source = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config = type(source)(vocab_size=128, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=128, use_cache=False,
                          bos_token_id=1, eos_token_id=2, pad_token_id=0)
    backbone = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    assert next(backbone.parameters()).device.type == "cpu"
    names = {name.rsplit(".", 1)[-1] for name, _ in backbone.named_modules()}
    assert not set(LORA_TARGETS) - names
    backbone = get_peft_model(backbone, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0, target_modules=LORA_TARGETS,
        bias="none", task_type="CAUSAL_LM"))
    model = LatentPersuader(backbone)
    prompt = torch.tensor([[1, 4, 5, 6, 7]])
    target = torch.tensor([[8, 9, 2]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model.sft_loss(prompt, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.backbone.named_parameters() if ".lora_" in name)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.attention.parameters())
    model.zero_grad(set_to_none=True)
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    model.eval()
    examples = [(prompt, target), (torch.tensor([[1, 10, 11, 12]]), torch.tensor([[13, 14, 2]]))]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        hidden, mask, positions, cache = model.cached_prompt_state(prompt)
        original_length = cache.get_seq_length()
        thought = model.latent_from_hidden(hidden, mask, positions)
        model.target_logps_from_cache(target, thought, prompt.shape[1], cache)
        assert cache.get_seq_length() == original_length
        rl_loss, reward = lr_grpo_loss(model, examples, group_size=4,
                                       sigma=1.5, backward_per_proposal=True)
    assert torch.isfinite(rl_loss) and math.isfinite(reward)
    gradients = [p.grad for p in model.attention.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        tokens = model.generate_text(prompt, max_new_tokens=3, eos_token_id=2,
                                     temperature=.95, top_p=.7, top_k=50)
    assert len(tokens) <= 3 and all(isinstance(token, int) for token in tokens)
    assert not torch.cuda.is_initialized()
    print(json.dumps({"model_path": model_path, "source_model_type": source.model_type,
                      "tested_class": type(backbone.base_model.model).__name__,
                      "test_hidden_size": 64, "device": "cpu", "compute_dtype": "bfloat16",
                      "sft_loss": loss.item(), "rl_loss": rl_loss.item(), "reward": reward,
                      "sft_backward": "passed", "rl_backward": "passed",
                      "prefix_cache_immutable": True, "generation_tokens": len(tokens),
                      "full_model_gpu_smoke": "still_required"}), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.manual_seed(42)
    for path in ("/app/hzj/models/Meta-Llama-3.1-8B-Instruct",
                 "/app/hzj/models/DeepSeek-R1-Distill-Qwen-14B"):
        probe(path)
