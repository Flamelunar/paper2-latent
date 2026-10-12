"""Public-text GRPO with grouped advantages and a fixed ordinary-SFT reference.

The token clipped objective is conventional GRPO. The ECLP comparison retains
its optimizer and behavior-KL guard, whose text version uses a sampled K3
estimate per completion rather than an analytic Gaussian KL.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--sft', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--resume', type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'my_eclp_final'))
    sys.path.insert(0, str(root / 'exp_code_ljj'))
    # Freeze this protocol alongside the comparison source before launch.
    sys.path.insert(0, str(Path(__file__).parent))
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import peft.tuners.lora.model as lora_model
    from eclp_elpo import Config, FP32MasterAdamW, capture_rng, restore_rng, seed_all, public_history, scenario_seed, points_for_split, verified_prepared
    from experiment_io import read_json, atomic_json, immutable_json, append_jsonl, digest, object_hash, writer_lock
    from repro.persuadee_simulation import dialogue_prompt
    from repro.simulation import LocalTransformersEndpoint
    from repro.gpu_capacity import wait_for_bf16_capacity
    from protocol import SchemaFactory, speech_boundary, narration
    from grouped_rollouts import sample_group, reward_group

    cfg = Config.load(root / 'my_eclp_final/config_gpu4.json')
    plan = read_json(root / 'my_eclp_final/offline_training_plan.json')
    prepared = root / 'eclp_runs_gpu4/prepared'
    verified_prepared(prepared, read_json(root / 'eclp_runs_gpu4/protocol.lock.json'))
    points = points_for_split(prepared, 'rl_train')
    if len(points) != 1800 or len({p['scenario_id'] for p in points}) != 1800:
        raise ValueError('Expected the same 1800 distinct training contexts')
    order = list(range(len(points)))
    random.Random(cfg.seed).shuffle(order)
    points = [points[i] for i in order]
    public_points = [{'scenario_id': p['scenario_id'], 'scenario': {k: str(p['scenario'][k]) if k in ('persuader', 'persuadee') else p['scenario'][k] for k in ['background', 'goal', 'persuader', 'persuadee']},
                      'history': public_history(p['history'])} for p in points]
    if any(len(p['history']) % 2 for p in public_points):
        raise ValueError('Training decision histories must end after a persuadee reply')
    sft_manifest = read_json(args.sft.parent / 'manifest.json')
    sft_summary = read_json(args.sft.parent / 'summary.json')
    if not sft_summary['complete'] or Path(sft_summary['checkpoint']).resolve() != args.sft.resolve():
        raise ValueError('Use the complete, one-epoch ordinary SFT checkpoint')
    if sft_manifest['variant'] != 'ordinary_public_sft' or sft_manifest['private_targets_used']:
        raise ValueError('Ordinary GRPO cannot start from ECLP joint-SFT weights')
    group_size = plan['candidates']
    context_batch = plan['contexts_per_update']
    update_batch = plan['policy_update_batch_size']
    update_epochs = plan['policy_update_epochs']
    if (group_size, context_batch, update_batch, update_epochs, plan['rl_epochs']) != (4, 64, 64, 2, 1):
        raise ValueError('Canonical ECLP optimization budget changed; reconcile before launch')
    lr = plan['policy_learning_rate']
    manifest = {'variant': 'ordinary_sft_text_grpo', 'algorithm': 'token-level clipped GRPO',
        'base_model': cfg.base_model, 'sft_checkpoint': str(args.sft),
        'sft_adapter_sha256': digest(args.sft / 'adapter/adapter_model.safetensors'),
        'rl_data_sha256': digest(prepared / 'rl_train.jsonl'), 'scenario_ids': [p['scenario_id'] for p in points],
        'epochs': 1, 'contexts': len(points), 'group_size': group_size, 'contexts_per_block': context_batch,
        'policy_update_batch_size': update_batch, 'policy_update_epochs': update_epochs,
        'lr': lr, 'beta': cfg.kl_beta, 'clip_epsilon': cfg.clip_epsilon, 'behavior_kl_bound': cfg.max_kl,
        'behavior_kl_estimator': 'max per-completion mean K3(old||current) on sampled tokens; rollback/backoff',
        'reference_kl_estimator': 'GRPO K3(current||fixed-SFT), sampled completion tokens',
        'advantage': 'per-context population mean/std, floor=0.001, clip=10',
        'train_reward': 'current persuadee stance: none=0, engagement=1/3, conditional=2/3, accepted=1',
        'simulator': cfg.persuadee_model, 'train_judge': cfg.judge_model,
        'reward_protocol_sha256': digest(Path(__file__).parent / 'protocol.py'),
        'decode': {'temperature': cfg.temperature, 'top_p': cfg.top_p, 'top_k': cfg.top_k, 'max_new_tokens': cfg.max_new_tokens},
        'rollout_batch_size': 4, 'rollout_source_sha256': digest(Path(__file__).with_name('grouped_rollouts.py')),
        'sampling': 'independent per-replica CUDA generators with historical RowSampler; BF16 batch rounding may differ from singletons',
        'precision': 'bfloat16', 'quantization': 'none', 'seed': cfg.seed,
        'source_sha256': digest(Path(__file__)), 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'optimizer': 'FP32MasterAdamW', 'weight_decay': 0, 'test_reference_used': False,
        'reward_parse_failure': 'three fixed-rubric attempts with current-reply excerpt-schema recovery, then abort uncommitted block; no fake reward'}
    migration_path = args.output / 'role_name_migration.json'
    if migration_path.exists():
        migration = read_json(migration_path)
        if migration['new_source_sha256'] != digest(Path(__file__)):
            raise ValueError('Role-name migration source mismatch')
        original_manifest = read_json(args.output / 'manifest.json')
        expected_manifest = dict(manifest, source_sha256=migration['old_source_sha256'])
        if original_manifest != expected_manifest:
            raise ValueError('Migration changed settings beyond role-name normalization')
        manifest = original_manifest
    args.output.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output / '.training.lock'):
        immutable_json(args.output / 'manifest.json', manifest)
        if (args.output / 'last.json').exists() and not args.resume:
            raise FileExistsError('Committed run requires resume')
        seed_all(cfg.seed)
        wait_for_bf16_capacity([cfg.base_model, cfg.persuadee_model, cfg.judge_model], [0],
            reserve_gib=12, description='Qwen2.5 public-text GRPO on physical GPU4')
        load_started = time.monotonic()
        lora_model.is_bnb_available = lambda: False
        lora_model.is_bnb_4bit_available = lambda: False
        tok = AutoTokenizer.from_pretrained(cfg.base_model, local_files_only=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=torch.bfloat16,
            local_files_only=True, attn_implementation='sdpa').to('cuda:0')
        base.requires_grad_(False)
        policy_path = (args.resume or args.sft) / 'adapter'
        model = PeftModel.from_pretrained(base, str(policy_path), is_trainable=True, autocast_adapter_dtype=False)
        model.load_adapter(str(args.sft / 'adapter'), adapter_name='ref', is_trainable=False, autocast_adapter_dtype=False)
        model.to(dtype=torch.bfloat16)
        model.config.use_cache = False

        def activate(name):
            model.set_adapter(name)
            for n, p in model.named_parameters():
                p.requires_grad_(name == 'default' and 'lora_' in n and '.default.' in n)

        activate('default')
        params = [p for p in model.parameters() if p.requires_grad]
        if any('lora_' not in n or '.default.' not in n for n, p in model.named_parameters() if p.requires_grad):
            raise ValueError('Optimizer must update only the default policy LoRA')
        from eclp_elpo import tensor_digest
        def reference_hash():
            reference = {n: p for n, p in model.named_parameters() if 'lora_' in n and '.ref.' in n}
            if not reference:
                raise ValueError('Fixed SFT reference adapter is missing')
            return tensor_digest(reference)
        reference_before = reference_hash()
        if not args.resume:
            named = dict(model.named_parameters())
            for name, p in named.items():
                if 'lora_' in name and '.default.' in name:
                    ref_name = name.replace('.default.', '.ref.')
                    if ref_name not in named or not torch.equal(p, named[ref_name]):
                        raise ValueError('Initial text policy differs from the fixed ordinary-SFT reference')
        print(json.dumps({'stage': 'initialized', 'trainable_parameter_count': sum(p.numel() for p in params),
            'trainable_dtypes': sorted({str(p.dtype) for p in params}), 'reference_hash': reference_before,
            'reference_matches_initial_sft': not bool(args.resume), 'manifest_hash': object_hash(manifest)}), flush=True)
        optimizer = FP32MasterAdamW(params, lr=lr, weight_decay=0)
        cursor = updates = 0
        previous_seconds = 0.0
        if args.resume:
            committed = read_json(args.output / 'last.json')
            if Path(committed['checkpoint']).resolve() != args.resume.resolve():
                raise ValueError('Resume only the latest committed checkpoint')
            state = torch.load(args.resume / 'trainer.pt', map_location='cpu', weights_only=False)
            if state['manifest'] != manifest:
                raise ValueError('Resume contract mismatch')
            optimizer.load_state_dict(state['optimizer'])
            restore_rng(state['rng'])
            cursor, updates, previous_seconds = state['cursor'], state['updates'], state['training_seconds']
            previous_seconds = committed['training_seconds']
        model.eval()
        user = LocalTransformersEndpoint(model=cfg.persuadee_model, device='cuda:0', quantization='none')
        judge = LocalTransformersEndpoint(model=cfg.judge_model, device='cuda:0', quantization='none')
        factory = SchemaFactory(judge.tokenizer)
        torch.cuda.synchronize()
        load_seconds = time.monotonic() - load_started
        started = time.monotonic()

        def encode(point):
            history = list(point['history'])
            while True:
                ids = tok.apply_chat_template([{'role': 'system', 'content': cfg.actor_system_prompt},
                    {'role': 'user', 'content': dialogue_prompt(point, history)}], tokenize=True, add_generation_prompt=True)
                if hasattr(ids, 'input_ids'):
                    ids = ids.input_ids
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                if len(ids) <= cfg.max_prompt_tokens:
                    return ids
                if len(history) <= 2:
                    raise ValueError('Public training prompt over budget')
                history = history[2:]

        def logps(sample):
            sequence = sample['prompt_ids'] + sample['completion_ids']
            x = torch.tensor([sequence], device='cuda:0')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
            offset = len(sample['prompt_ids']) - 1
            logits = result.logits[0, offset:-1].float() / cfg.temperature
            y = torch.tensor(sample['completion_ids'], device='cuda:0')
            values = F.log_softmax(logits, dim=-1).gather(1, y[:, None]).squeeze(1)
            if not torch.isfinite(values).all():
                raise FloatingPointError('Nonfinite token log probability')
            return values

        def k3(delta):
            value = delta.exp() - delta - 1
            if not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite sampled KL')
            return value

        while cursor < len(points):
            end = min(cursor + context_batch, len(points))
            block = args.output / 'blocks' / f'block-{cursor:06d}-{end:06d}'
            block.mkdir(parents=True, exist_ok=True)
            batch_file = block / 'batch.pt'
            if batch_file.exists():
                batch_meta = read_json(block / 'batch_manifest.json')
                if batch_meta['manifest_hash'] != object_hash(manifest) or batch_meta['cursor'] != cursor:
                    raise ValueError('Cached rollouts belong to a different run')
                if digest(batch_file) != batch_meta['batch_sha256']:
                    raise ValueError('Cached rollout tensor contents changed')
                batch = torch.load(batch_file, map_location='cpu', weights_only=False)
            else:
                batch = []
                activate('default')
                model.eval()
                model.gradient_checkpointing_disable()
                for point in public_points[cursor:end]:
                    group = []
                    ids = encode(point)
                    seeds = [scenario_seed(cfg.seed, point['scenario_id'], 'standard-grpo-text', replica) for replica in range(group_size)]
                    generated = sample_group(model, tok, ids, seeds, cfg)
                    if len(generated) != group_size:
                        raise ValueError('Candidate group does not contain the configured four actions')
                    utterances = []
                    invalid_traces = {}
                    for replica, (_, raw) in enumerate(generated):
                        try:
                            spoken = speech_boundary(raw, point, 'persuader')
                            if spoken['thinking_filtered'] or raw.lstrip().startswith(('{', '<think>', 'Psychology:', 'Analysis:', 'Strategy:')):
                                raise ValueError('Direct output contained explicit analysis')
                            if narration(spoken['content'], point):
                                raise ValueError('Analysis instead of spoken persuasion')
                            utterances.append(spoken['content'])
                        except ValueError as error:
                            # Invalid actor actions are explicitly penalized, not replaced by golden text.
                            utterances.append(None)
                            invalid_traces[replica] = {'invalid_actor': str(error)}
                    outcomes = reward_group(point, utterances, seeds, user, judge, factory, cfg)
                    for replica, (completion_ids, raw) in enumerate(generated):
                        score, trace = outcomes[replica] if replica in outcomes else (0.0, invalid_traces[replica])
                        sample = {'scenario_id': point['scenario_id'], 'replica': replica,
                            'prompt_ids': ids, 'completion_ids': completion_ids, 'reward': score,
                            'raw_completion': raw, 'reward_trace': trace, 'seed': seeds[replica]}
                        if not completion_ids:
                            raise ValueError('Empty token action')
                        with torch.no_grad():
                            sample['old_logps'] = logps(sample).detach().cpu()
                            activate('ref')
                            sample['ref_logps'] = logps(sample).detach().cpu()
                            activate('default')
                        group.append(sample)
                        append_jsonl(block / 'rollouts.jsonl', {k: v for k, v in sample.items() if k not in ['old_logps', 'ref_logps']})
                    rewards = torch.tensor([s['reward'] for s in group])
                    advantages = ((rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(0.001)).clamp(-10, 10)
                    for sample, advantage in zip(group, advantages):
                        sample['advantage'] = float(advantage)
                    batch.extend(group)
                    sampling_info = {'stage': 'sampling', 'committed_contexts': cursor,
                        'sampled_contexts_in_block': len(batch) // group_size, 'total_contexts': len(points),
                        'scenario_id': point['scenario_id'], 'rewards': rewards.tolist(),
                        'advantages': advantages.tolist(), 'pid': os.getpid(),
                        'training_seconds': previous_seconds + time.monotonic() - started}
                    atomic_json(args.output / 'sampling_progress.json', sampling_info)
                    print(json.dumps(sampling_info), flush=True)
                temporary = block / 'batch.tmp.pt'
                torch.save(batch, temporary)
                temporary.replace(batch_file)
                atomic_json(block / 'batch_manifest.json', {'manifest_hash': object_hash(manifest),
                    'cursor': cursor, 'end': end, 'batch_sha256': digest(batch_file)})
            activate('default')
            model.eval()
            with torch.no_grad():
                replay_error = max(float((logps(s).cpu() - s['old_logps']).abs().max()) for s in batch)
            if replay_error > 1e-4:
                raise ValueError(f'Behavior log-probability replay mismatch: {replay_error}')
            model.train()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            block_updates = []
            stop_block = False
            for epoch in range(update_epochs):
                order = list(range(len(batch)))
                random.Random(cfg.seed + cursor + epoch).shuffle(order)
                for start in range(0, len(order), update_batch):
                    samples = [batch[i] for i in order[start:start + update_batch]]
                    optimizer.zero_grad(set_to_none=True)
                    max_pre = loss_total = 0.0
                    for sample in samples:
                        current = logps(sample)
                        old = sample['old_logps'].to('cuda:0')
                        ref = sample['ref_logps'].to('cuda:0')
                        ratio = (current - old).exp()
                        advantage = sample['advantage']
                        objective = torch.minimum(ratio * advantage,
                            ratio.clamp(1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * advantage)
                        loss = (-objective + cfg.kl_beta * k3(ref - current)).mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite GRPO loss')
                        (loss / len(samples)).backward()
                        loss_total += float(loss.detach()) / len(samples)
                        max_pre = max(max_pre, float(k3(current.detach() - old).mean()))
                    if max_pre > cfg.max_kl:
                        optimizer.zero_grad(set_to_none=True)
                        block_updates.append({'accepted': False, 'reason': 'pre_behavior_kl', 'max_kl': max_pre})
                        stop_block = True
                        break
                    weights = [p.detach().clone() for p in params]
                    optimizer_state = copy.deepcopy(optimizer.state_dict())
                    grad = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                    initial_lr = optimizer.param_groups[0]['lr']
                    attempts = []
                    for attempt in range(plan['max_lr_backoffs'] + 1):
                        optimizer.step()
                        with torch.no_grad():
                            max_post = max(float(k3(logps(s) - s['old_logps'].to('cuda:0')).mean()) for s in samples)
                        accepted = max_post <= cfg.max_kl
                        attempts.append({'lr': optimizer.param_groups[0]['lr'], 'post_behavior_kl_max': max_post,
                                         'accepted': accepted})
                        if accepted:
                            break
                        with torch.no_grad():
                            for p, value in zip(params, weights):
                                p.copy_(value)
                        optimizer.load_state_dict(optimizer_state)
                        if attempt == plan['max_lr_backoffs'] or attempts[-1]['lr'] <= plan['min_policy_learning_rate']:
                            break
                        for group in optimizer.param_groups:
                            group['lr'] = max(initial_lr / (2 ** (attempt + 1)), plan['min_policy_learning_rate'])
                    if accepted:
                        updates += 1
                    else:
                        stop_block = True
                    block_updates.append({'accepted': accepted, 'loss': loss_total, 'grad_norm': float(grad),
                        'pre_behavior_kl_max': max_pre, 'post_behavior_kl_max': max_post,
                        'backoff_attempts': attempts, 'reason': 'accepted' if accepted else 'post_behavior_kl_rollback'})
                    optimizer.zero_grad(set_to_none=True)
                    del weights, optimizer_state
                    if stop_block:
                        break
                if stop_block:
                    break
            cursor = end
            torch.cuda.synchronize()
            elapsed = previous_seconds + time.monotonic() - started
            checkpoint = args.output / f'checkpoint-{cursor:06d}'
            checkpoint.mkdir()
            model.save_pretrained(checkpoint / 'adapter', selected_adapters=['default'])
            tok.save_pretrained(checkpoint / 'adapter')
            torch.save({'manifest': manifest, 'optimizer': optimizer.state_dict(), 'rng': capture_rng(),
                'cursor': cursor, 'updates': updates, 'training_seconds': elapsed}, checkpoint / 'trainer.pt')
            elapsed = previous_seconds + time.monotonic() - started
            info = {'contexts': cursor, 'optimizer_updates': updates, 'checkpoint': str(checkpoint),
                'block_batch_sha256': digest(batch_file), 'block_updates': block_updates,
                'training_seconds': elapsed, 'reward_mean': sum(s['reward'] for s in batch) / len(batch),
                'nonzero_advantages': sum(s['advantage'] != 0 for s in batch), 'candidate_count': len(batch),
                'behavior_logp_replay_error': replay_error, 'reference_unchanged': reference_hash() == reference_before}
            if not info['reference_unchanged']:
                raise ValueError('The fixed ordinary-SFT reference changed')
            atomic_json(block / 'commit.json', info)
            atomic_json(args.output / 'last.json', info)
            atomic_json(args.output / 'progress.json', info)
            print(json.dumps(info), flush=True)
            del batch
            gc.collect()
        torch.cuda.synchronize()
        elapsed = previous_seconds + time.monotonic() - started
        atomic_json(args.output / 'summary.json', {'complete': True, 'epochs': 1,
            'contexts': cursor, 'optimizer_updates': updates, 'checkpoint': str(checkpoint),
            'training_seconds': elapsed, 'training_minutes': elapsed / 60,
            'load_seconds_this_invocation': load_seconds, 'reference_unchanged': reference_hash() == reference_before,
            'timing_scope': 'actor sampling, simulator, reward judge, token log probabilities, optimizer and checkpoints; excludes initial model loads'})


if __name__ == '__main__':
    main()
