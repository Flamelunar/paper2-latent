"""Regenerate three audited Llama B2 scenes, never fabricate their histories."""

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path

EXP = Path(__file__).resolve().parent
ADAPTER_ROOT = Path('/app/hzj/liujianjian/persuader_atapter_save')
PERSUADEE = '/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged'
METHOD = 'baseline2_dialogue_sft'
FAMILY = 'meta_llama31'
ALLOWED_REPAIRS = {455, 478, 618}
PROTOCOL = 'full_scene_regeneration_for_history_mismatch_v1'


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def history_mismatches(rows, points):
    if len(rows) != 1000 or len(points) != 1000:
        raise ValueError('History repair requires all 1000 fixed scenes')
    bad = []
    for index, (row, point) in enumerate(zip(rows, points)):
        if row.get('index') != index or row.get('scenario_id') != point['scenario_id']:
            raise ValueError('Unexpected test order or scenario during history repair')
        wanted = [{'role': turn['role'], 'content': turn['content']}
                  for turn in point['history'] + row['trajectory']]
        actual = row.get('final_history')
        if not isinstance(actual, list) or any(not isinstance(turn, dict) for turn in actual):
            bad.append(index)
        elif [{'role': turn.get('role'), 'content': turn.get('content')} for turn in actual] != wanted:
            bad.append(index)
    if not set(bad).issubset(ALLOWED_REPAIRS):
        raise ValueError(f'Unaudited history mismatch; refusing broad regeneration: {bad}')
    return bad


def atomic_replace(path, rows, *, original):
    if path.is_symlink():
        raise ValueError('Refusing to replace a symlink generation')
    mode = stat.S_IMODE(path.stat().st_mode)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name + '.', delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if path.read_bytes() != original:
            raise RuntimeError('Generation changed during repair; refusing to overwrite')
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def repair_llama_baseline2(*, device='cuda:0', exp_root=EXP,
                          adapter_root=ADAPTER_ROOT, endpoint_factory=None,
                          simulator=None):
    from report_new_baselines_v18 import load_rows, validate_generations

    file = exp_root / 'artifacts/generation_5rounds' / FAMILY / (METHOD + '.jsonl')
    test = exp_root / 'data/evaluation_points.jsonl'
    manifest = json.loads((exp_root / 'data/manifest.json').read_text(encoding='utf-8'))
    test_digest = hashlib.sha256(test.read_bytes()).hexdigest()
    if test_digest != manifest['files']['evaluation_points']['sha256']:
        raise ValueError('History repair fixed test checksum differs from manifest')
    source = file.read_bytes()
    rows, points = load_rows(file), load_rows(test)
    bad = history_mismatches(rows, points)
    model = str(adapter_root / FAMILY / METHOD)
    if not bad:
        validate_generations(rows, points, method=METHOD, expected_model=model)
        return []

    # Check every other invariant before loading models. This diagnostic copy
    # checks the unaffected 997 records; it is never saved, used as training
    # data, or scored by a Judge. Only genuine new dialogue repairs the source.
    import copy
    diagnostic = copy.deepcopy(rows)
    for index in bad:
        diagnostic[index]['final_history'] = points[index]['history'] + [
            {'role': turn['role'], 'content': turn['content']} for turn in diagnostic[index]['trajectory']]
    validate_generations(diagnostic, points, method=METHOD, expected_model=model)

    out = file.parent / 'baseline2_history_repair'
    out.mkdir(exist_ok=True)
    config = {'protocol': PROTOCOL, 'indices': bad, 'test_sha256': test_digest,
              'source_sha256': hashlib.sha256(source).hexdigest(),
              'persuader_model': model, 'persuadee_model': PERSUADEE,
              'device': device, 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
              'compute_dtype': 'bfloat16', 'quantization': 'none', 'seed': 42,
              'rounds': 5, 'temperature': .95, 'top_p': .7, 'top_k': 50,
              'max_new_tokens': 512, 'thinking_policy': 'final_answer_only',
              'original_row_sha256': {str(index): row_hash(rows[index]) for index in bad}}
    config_path = out / 'run_config.json'
    if config_path.is_file():
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Repair provenance changed; re-audit before continuing')
    else:
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        (out / 'quarantined_original_scenes.jsonl').write_text(
            ''.join(json.dumps(rows[index], ensure_ascii=False) + '\n' for index in bad), encoding='utf-8')
    cache_file = out / 'regenerated_scenes.jsonl'
    cache = {}
    if cache_file.is_file():
        with cache_file.open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                index = row['index']
                if (index not in bad or index in cache
                        or row.get('generation_repair', {}).get('original_row_sha256') != row_hash(rows[index])):
                    raise ValueError('Unexpected regenerated-scene cache')
                cache[index] = row
    pending = [index for index in bad if index not in cache]
    if pending:
        if endpoint_factory is None:
            from repro.simulation import LocalTransformersEndpoint
            from repro.gpu_capacity import wait_for_bf16_capacity
            from transformers import set_seed
            set_seed(42)
            wait_for_bf16_capacity([model, PERSUADEE], LocalTransformersEndpoint._parse_devices(device),
                                  reserve_gib=6, description='Full-scene B2 history repair')
            endpoint_factory = LocalTransformersEndpoint
        if simulator is None:
            from repro.persuadee_simulation import PersuaderMethod, simulate_case
            simulator = simulate_case
            method = PersuaderMethod(METHOD, 'Baseline 2: Dialogue-SFT', model, False)
        else:
            method = {'slug': METHOD, 'model': model}
        persuader = None
        persuadee = None
        print(f'B2 history repair: regenerating complete scenes {pending}, BF16 on {device}', flush=True)
        try:
            persuader = endpoint_factory(model=model, device=device)
            persuadee = endpoint_factory(model=PERSUADEE, device=device)
            for index in pending:
                result = simulator(point=points[index], method=method, persuader=persuader,
                                   persuadee=persuadee, rounds=5, temperature=.95,
                                   max_new_tokens=512, top_p=.7, top_k=50)
                replacement = {**rows[index], **result,
                    'generation_repair': {'protocol': PROTOCOL, 'regenerated_full_rounds': 5,
                                          'original_row_sha256': row_hash(rows[index]),
                                          'seed': 42, 'device': device}}
                probe = list(rows)
                for known, value in cache.items():
                    probe[known] = value
                probe[index] = replacement
                if index in history_mismatches(probe, points):
                    raise ValueError('New dialogue still has mismatched public history')
                validation_probe = list(diagnostic)
                for known, value in cache.items():
                    validation_probe[known] = value
                validation_probe[index] = replacement
                validate_generations(validation_probe, points, method=METHOD, expected_model=model)
                with cache_file.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(replacement, ensure_ascii=False) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                cache[index] = replacement
                print(f'B2 history repair: regenerated index {index} ({len(cache)}/{len(bad)})', flush=True)
        finally:
            if persuadee is not None:
                persuadee.close()
            if persuader is not None:
                persuader.close()
    repaired = list(rows)
    for index, replacement in cache.items():
        repaired[index] = replacement
    validate_generations(repaired, points, method=METHOD, expected_model=model)
    if any(row != repaired[index] for index, row in enumerate(rows) if index not in bad):
        raise RuntimeError('Repair attempted to modify an unaffected scene')
    atomic_replace(file, repaired, original=source)
    print(f'B2 history repair: validated all 1000 scenes; regenerated only {bad}', flush=True)
    return bad
