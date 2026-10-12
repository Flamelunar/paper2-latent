"""Matched public-only SFT; no ECLP heads or private targets are optimized."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--resume', type=Path)
    args = ap.parse_args()
    sys.path.insert(0, str(args.root / 'my_eclp_final'))
    sys.path.insert(0, str(args.root / 'exp_code_ljj'))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    import peft.tuners.lora.model as lora_model
    from eclp_elpo import Config, FP32MasterAdamW, capture_rng, restore_rng, seed_all, public_history
    from repro.persuadee_simulation import dialogue_prompt
    from experiment_io import atomic_json, append_jsonl, immutable_json, writer_lock
    cfg = Config.load(args.root / 'my_eclp_final/config_gpu4.json')
    seed_all(cfg.seed)
    source = args.root / 'eclp_runs_gpu4/prepared/sft.jsonl'
    rows = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    if len(rows) != 59868 or any(r['turns'][-1]['role'] != 'persuader' for r in rows):
        raise ValueError('Unexpected joint-SFT public target selection')
    manifest = {'variant': 'ordinary_public_sft', 'base_model': cfg.base_model,
        'data_path': str(source), 'data_sha256': digest(source), 'sessions': len(rows),
        'epochs': 1, 'seed': cfg.seed, 'rank': cfg.lora_rank, 'alpha': cfg.lora_alpha,
        'dropout': cfg.lora_dropout, 'lr': 1e-4, 'accumulation': 8, 'weight_decay': 0.01,
        'max_prompt_tokens': cfg.max_prompt_tokens, 'max_target_tokens': cfg.max_target_tokens,
        'precision': 'bfloat16', 'quantization': 'none', 'supervision': 'last public utterance only',
        'prompt_source_sha256': digest(args.root / 'exp_code_ljj/repro/persuadee_simulation.py'),
        'source_sha256': digest(__file__), 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'optimizer': 'ECLP FP32MasterAdamW', 'private_targets_used': False}
    args.output.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output / '.training.lock'):
        immutable_json(args.output / 'manifest.json', manifest)
        if (args.output / 'train.jsonl').exists() and not args.resume:
            raise FileExistsError('Existing run requires explicit resume')
        load_started = time.monotonic()
        lora_model.is_bnb_available = lambda: False
        lora_model.is_bnb_4bit_available = lambda: False
        tok = AutoTokenizer.from_pretrained(cfg.base_model, local_files_only=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16,
            local_files_only=True, attn_implementation='sdpa').to('cuda:0')
        base.requires_grad_(False)
        if args.resume:
            model = PeftModel.from_pretrained(base, str(args.resume / 'adapter'),
                is_trainable=True, autocast_adapter_dtype=False)
        else:
            model = get_peft_model(base, LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout, bias='none', task_type='CAUSAL_LM',
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']),
                autocast_adapter_dtype=False)
        model.to(dtype=torch.bfloat16)
        model.config.use_cache = False
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        params = [p for p in model.parameters() if p.requires_grad]
        if any('lora_' not in n for n, p in model.named_parameters() if p.requires_grad):
            raise ValueError('Public SFT must update only LoRA')
        optimizer = FP32MasterAdamW(params, lr=1e-4, weight_decay=0.01)
        position = step = 0
        previous_seconds = 0.0
        if args.resume:
            state = torch.load(args.resume / 'trainer.pt', map_location='cpu', weights_only=False)
            if state['manifest'] != manifest:
                raise ValueError('Resume manifest mismatch')
            optimizer.load_state_dict(state['optimizer'])
            restore_rng(state['rng'])
            position, step, previous_seconds = state['position'], state['step'], state['training_seconds']
            log = args.output / 'train.jsonl'
            if log.exists():
                kept = [json.loads(s) for s in log.read_text().splitlines() if s.strip()]
                log.write_text(''.join(json.dumps(r) + '\n' for r in kept if r['step'] <= step))
        load_seconds = time.monotonic() - load_started
        order = list(range(len(rows)))
        random.Random(cfg.seed).shuffle(order)
        torch.cuda.synchronize()
        started = time.monotonic()

        def save():
            torch.cuda.synchronize()
            elapsed = previous_seconds + time.monotonic() - started
            path = args.output / f'checkpoint-{step:06d}'
            if path.exists():
                raise FileExistsError(path)
            path.mkdir()
            model.save_pretrained(path / 'adapter')
            tok.save_pretrained(path / 'adapter')
            torch.save({'manifest': manifest, 'optimizer': optimizer.state_dict(), 'rng': capture_rng(),
                'position': position, 'step': step, 'training_seconds': elapsed}, path / 'trainer.pt')
            atomic_json(args.output / 'last.json', {'checkpoint': str(path), 'step': step,
                'position': position, 'training_seconds': elapsed})
            return path

        while position < len(order):
            batch = order[position:position + 8]
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            tokens = 0
            for idx in batch:
                row = rows[idx]
                history = public_history(row['turns'][:-1])
                while True:
                    prompt = dialogue_prompt({'scenario': row['scenario']}, history)
                    ids = tok.apply_chat_template([{'role': 'system', 'content': cfg.actor_system_prompt},
                        {'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True)
                    if hasattr(ids, 'input_ids'):
                        ids = ids.input_ids
                    if isinstance(ids, torch.Tensor):
                        ids = ids.tolist()
                    if ids and isinstance(ids[0], list):
                        if len(ids) != 1:
                            raise ValueError('Unexpected batched prompt encoding')
                        ids = ids[0]
                    if len(ids) <= cfg.max_prompt_tokens:
                        break
                    if len(history) <= 2:
                        raise ValueError('Prompt over budget with latest feedback retained')
                    history = history[2:]
                target = tok.encode(row['turns'][-1]['content'], add_special_tokens=False)[:cfg.max_target_tokens - 1]
                if not target:
                    raise ValueError('Empty public target')
                target.append(tok.eos_token_id)
                x = torch.tensor([ids + target], device='cuda:0')
                labels = torch.tensor([[-100] * len(ids) + target], device='cuda:0')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(input_ids=x, attention_mask=torch.ones_like(x), labels=labels,
                        use_cache=False).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite public SFT loss')
                (loss / len(batch)).backward()
                total += float(loss.detach()) / len(batch)
                tokens += len(target)
            grad = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            position += len(batch)
            step += 1
            torch.cuda.synchronize()
            record = {'step': step, 'position': position, 'loss': total, 'grad_norm': float(grad),
                'supervised_tokens': tokens, 'training_seconds': previous_seconds + time.monotonic() - started}
            append_jsonl(args.output / 'train.jsonl', record)
            atomic_json(args.output / 'progress.json', record)
            print(json.dumps(record), flush=True)
            if step % 500 == 0:
                save()
        final = save()
        torch.cuda.synchronize()
        elapsed = previous_seconds + time.monotonic() - started
        atomic_json(args.output / 'summary.json', {'complete': True, 'epochs': 1, 'sessions': position,
            'steps': step, 'checkpoint': str(final), 'training_seconds': elapsed,
            'training_minutes': elapsed / 60, 'load_seconds_this_invocation': load_seconds,
            'timing_scope': 'training data processing, optimization and intermediate/final saves; excludes model load'})


if __name__ == '__main__':
    main()
