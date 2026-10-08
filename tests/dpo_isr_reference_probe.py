"""CPU BF16 probe of actual TRL's frozen ISR reference adapter behavior."""
import json
import tempfile
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

torch.set_num_threads(2)
torch.manual_seed(42)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'exp_code_ljj'))
from repro.train_backend import _tokenizer, preference_messages
dataset = Dataset.from_list([preference_messages({'instruction': 'A voluntary request.',
              'system': 'Respect the other speaker.', 'history': [],
              'output': ['Please consider it.', 'No thanks.']}) for _ in range(2)])
for base in ['/app/hzj/models/Meta-Llama-3.1-8B-Instruct',
             '/app/hzj/models/DeepSeek-R1-Distill-Qwen-14B']:
    source = AutoConfig.from_pretrained(base, local_files_only=True)
    tokenizer = _tokenizer(base)
    config = type(source)(vocab_size=source.vocab_size, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=256, bos_token_id=tokenizer.bos_token_id,
                          eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=False)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    policy = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0,
                           target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if '.lora_B.' in name:
                parameter.fill_(0.03)
    initial = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()
               if '.default.' in name}
    with tempfile.TemporaryDirectory(prefix='ljj-dpo-cpu-probe-') as directory:
        args = DPOConfig(output_dir=directory, use_cpu=True, bf16=True, max_length=128,
                         per_device_train_batch_size=1, gradient_accumulation_steps=1,
                         max_steps=1, learning_rate=1e-3, gradient_checkpointing=False,
                         logging_strategy='no', save_strategy='no', report_to=[],
                         disable_dropout=True, beta=0.1)
        trainer = DPOTrainer(model=policy, ref_model=None, args=args,
                             train_dataset=dataset, processing_class=tokenizer)
        assert trainer.ref_model is None and 'ref' in policy.peft_config
        for name, tensor in initial.items():
            reference = policy.get_parameter(name.replace('.default.', '.ref.'))
            assert not reference.requires_grad
            assert torch.equal(reference.detach(), tensor)
        result = trainer.train()
        assert torch.isfinite(torch.tensor(result.training_loss))
        for name, tensor in initial.items():
            assert torch.equal(policy.get_parameter(name.replace('.default.', '.ref.')).detach(), tensor)
        assert any(not torch.equal(policy.get_parameter(name).detach(), tensor) for name, tensor in initial.items())
        assert not torch.cuda.is_initialized()
        print(json.dumps({'model': base, 'tested_class': type(model).__name__, 'device': 'cpu',
                          'compute_dtype': 'bfloat16', 'reference_is_initial_isr_adapter': True,
                          'reference_stays_frozen_after_dpo_step': True, 'policy_updates': True,
                          'actual_family_tokenizer_and_training_chat_template': True,
                          'train_loss': result.training_loss}), flush=True)
