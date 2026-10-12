"""192 five-round dialogues with the historical Ours public prompt/protocol."""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import sys
import time

from batching import Request, drive
from protocol import DECODE, REVISION, PUBLIC_RULE, public_point, speech_boundary, narration

METHOD = 'ordinary_sft_grpo'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--physical-gpu', type=int, default=4)
    args = ap.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'my_eclp_final'))
    sys.path.insert(0, str(root / 'exp_code_ljj'))
    import torch
    from peft import PeftModel
    import peft.tuners.lora.model as lora_model
    from experiment_io import read_json, read_jsonl, digest, object_hash, immutable_json, atomic_json, append_jsonl, writer_lock
    from eclp_elpo import Config, scenario_seed, seed_all
    from repro.simulation import LocalTransformersEndpoint, user_prompt, transcript
    from repro.gpu_capacity import wait_for_bf16_capacity
    cfg = Config.load(root / 'my_eclp_final/config_gpu4.json')
    summary = read_json(args.checkpoint.parent / 'summary.json')
    assert summary['complete'] and summary['contexts'] == 1800
    assert Path(summary['checkpoint']).resolve() == args.checkpoint.resolve()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.physical_gpu)
    points = read_jsonl(root / 'exp_code_ljj/data/evaluation_points.jsonl')
    selection = read_json(root / 'exp_code_ljj/data/test_selection_192.json')
    assert selection['original_indexes'] == [i for i, p in enumerate(points) if p['target_turn'] == 1 and p['history'] == []]
    assert digest(root / 'exp_code_ljj/data/evaluation_points.jsonl') == selection['source_sha256']
    order = sorted(selection['original_indexes'], key=lambda i: hashlib.sha256(('matched-v2-pilot-42:' + points[i]['scenario_id']).encode()).hexdigest())
    models = {}
    for label, name in [('base', cfg.base_model), ('persuadee', cfg.persuadee_model)]:
        p = Path(name)
        models[label] = {'path': str(p.resolve()), 'files': [
            {'file': x.name, 'bytes': x.stat().st_size, 'mtime_ns': x.stat().st_mtime_ns}
            for x in sorted(p.iterdir()) if x.is_file() and x.suffix in ['.json', '.jinja', '.safetensors']],
            'config_sha256': digest(p / 'config.json'), 'generation_config_sha256': digest(p / 'generation_config.json')}
    contract = {'revision': REVISION, 'expected_indexes': order, 'method': METHOD, 'decode': DECODE,
        'seed': 42, 'physical_gpu': args.physical_gpu, 'compute_dtype': 'bfloat16', 'quantization': 'none',
        'checkpoint': str(args.checkpoint), 'checkpoint_id': digest(args.checkpoint / 'adapter/adapter_model.safetensors'),
        'simulator': cfg.persuadee_model, 'models': models, 'simulator_input': 'historical background, goal, public history',
        'actor_prompt': 'exact historical our_direct spoken-only public prompt; no latent modules',
        'source_sha256': digest(Path(__file__)), 'protocol_sha256': digest(Path(__file__).with_name('protocol.py')),
        'batching_sha256': digest(Path(__file__).with_name('batching.py')), 'batch_size': 2,
        'selection_sha256': digest(root / 'exp_code_ljj/data/test_selection_192.json'),
        'judge_used_during_generation': False}
    with writer_lock(args.output / '.generation.lock'):
        immutable_json(args.output / 'generation_run.json', contract)
        run_hash = object_hash(contract)
        path = args.output / 'generation' / f'{METHOD}.jsonl'
        saved = read_jsonl(path) if path.exists() else []
        existing = {r['index']: r for r in saved}
        assert len(existing) == len(saved)
        for i, row in existing.items():
            assert i in order and row['run_hash'] == run_hash and row['initial_history'] == []
            assert row['scenario_id'] == points[i]['scenario_id'] and len(row['trajectory']) == 10
        todo = [i for i in order if i not in existing]
        if not todo:
            return
        seed_all(42)
        wait_for_bf16_capacity([cfg.base_model, cfg.persuadee_model], [0], reserve_gib=10,
            description='Qwen2.5 SFT+GRPO generation on physical GPU4')
        lora_model.is_bnb_available = lambda: False
        lora_model.is_bnb_4bit_available = lambda: False
        actor = LocalTransformersEndpoint(model=cfg.base_model, device='cuda:0', quantization='none')
        actor.model = PeftModel.from_pretrained(actor.model, str(args.checkpoint / 'adapter'),
            is_trainable=False, autocast_adapter_dtype=False).to(dtype=torch.bfloat16)
        actor.model.eval()
        user = LocalTransformersEndpoint(model=cfg.persuadee_model, device='cuda:0', quantization='none')
        try:
            def scene(i):
                point = public_point(points[i])
                history, trajectory = [], []
                started = time.monotonic()
                seed = scenario_seed(42, point['scenario_id'], 'target_grounded_v4')
                for round_number in range(1, 6):
                    s = point['scenario']
                    prompt = (f"Background: {s['background']}\nExact persuasion goal: {s['goal']}\nPersuader: {s['persuader']}\nPersuadee: {s['persuadee']}"
                        + '\nObserved conversation:\n' + transcript(history) + '\n\n' + PUBLIC_RULE
                        + '\nPrivately consider the other person’s concerns and an appropriate persuasion strategy. Output ONLY one natural spoken persuasion utterance. Do not output analysis, thinking, JSON, speaker labels, or future replies.')
                    system = 'You are a skilled persuader. Output only a natural spoken persuasion utterance; keep all analysis private.'
                    attempts = []
                    for attempt in range(3):
                        raw, usage = yield Request(actor, system, prompt, seed + 100 * round_number + 1000 * attempt, point)
                        attempts.append({'raw_output': raw, **usage})
                        try:
                            spoken = speech_boundary(raw, point, 'persuader')
                            if spoken['thinking_filtered'] or raw.lstrip().startswith(('{', '<think>', 'Psychology:', 'Analysis:', 'Strategy:')):
                                raise ValueError('Direct output contained explicit analysis')
                            if narration(spoken['content'], point):
                                raise ValueError('Analysis instead of spoken persuasion')
                            break
                        except ValueError:
                            if attempt == 2:
                                raise
                    spoken.update(round=round_number, attempts=attempts, analysis=None, model_calls=attempt + 1, **usage)
                    trajectory.append(spoken)
                    history.append({'role': 'persuader', 'content': spoken['content']})
                    system, prompt = user_prompt(point, history)
                    attempts = []
                    for attempt in range(3):
                        raw, usage = yield Request(user, system, prompt, seed + 2 * round_number + 1000 * attempt, point)
                        attempts.append({'raw_output': raw, **usage})
                        try:
                            reply = speech_boundary(raw, point, 'persuadee')
                            break
                        except ValueError:
                            if attempt == 2:
                                raise
                    reply.update(round=round_number, attempts=attempts, **usage)
                    trajectory.append(reply)
                    history.append({'role': 'persuadee', 'content': reply['content']})
                return {'index': i, 'scenario_id': points[i]['scenario_id'], 'scenario': points[i]['scenario'],
                    'method': METHOD, 'initial_history': [], 'reference': points[i]['reference'],
                    'trajectory': trajectory, 'final_history': history, 'rounds': 5, 'run_hash': run_hash,
                    'checkpoint_id': contract['checkpoint_id'], 'persuader_model': str(args.checkpoint),
                    'persuadee_model': cfg.persuadee_model, 'compute_dtype': 'bfloat16', 'quantization': 'none',
                    'execution_seed': seed, 'elapsed_seconds': time.monotonic() - started, 'batch_size_requested': 2}

            def save(row):
                append_jsonl(path, row)
                existing[row['index']] = row
                atomic_json(args.output / f'{METHOD}_progress.json', {'saved': len(existing), 'expected': 192,
                    'last_index': row['index'], 'pid': os.getpid()})
                print(METHOD, len(existing), '/192', flush=True)

            def trace(request, response):
                raw, usage = response
                append_jsonl(args.output / 'generation' / f'{METHOD}_request_trace.jsonl', {
                    'scenario_id': request.point['scenario_id'], 'seed': request.seed,
                    'system': request.system, 'prompt': request.prompt, 'raw_output': raw, 'usage': usage, 'run_hash': run_hash})

            for offset in range(0, len(todo), 2):
                drive([scene(i) for i in todo[offset:offset + 2]], save, trace)
            atomic_json(path.with_name(f'{METHOD}_summary.json'), {'n': len(existing), 'complete': len(existing) == 192,
                'generation_sha256': digest(path), 'run_hash': run_hash})
        finally:
            actor.close()
            user.close()
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
