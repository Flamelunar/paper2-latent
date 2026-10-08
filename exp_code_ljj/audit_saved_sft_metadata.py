"""Audit saved SFT evidence without loading any model or changing checkpoints.

Earlier central record basenames collided across families. A retrospective
audit is deliberately not presented as the missing original runtime config.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADAPTER_ROOT = Path('/app/hzj/liujianjian/persuader_atapter_save')
FAMILIES = {
    'meta_llama31': ('/app/hzj/models/Meta-Llama-3.1-8B-Instruct', 0),
    'deepseek_r1_qwen14b': ('/app/hzj/models/DeepSeek-R1-Distill-Qwen-14B', 1),
}
STAGES = {'baseline2_dialogue_sft': 'dialogue_sft', 'baseline3_isr_sft': 'isr_sft'}
ARGUMENTS = ['output_dir', 'num_train_epochs', 'per_device_train_batch_size',
             'gradient_accumulation_steps', 'learning_rate', 'bf16', 'fp16',
             'gradient_checkpointing', 'seed', 'max_length', 'save_steps',
             'warmup_ratio', 'weight_decay', 'max_grad_norm', 'packing',
             'assistant_only_loss', 'completion_only_loss']


def saved_evidence(path: Path, torch):
    content = path.read_bytes()
    evidence = {'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(),
                'bytes': len(content)}
    if path.suffix == '.json':
        return json.loads(content), evidence
    if path.name != 'training_args.bin':
        raise ValueError('Only this job\'s saved Trainer arguments may be unpickled')
    return torch.load(io.BytesIO(content), map_location='cpu', weights_only=False), evidence


def training_arguments(directory: Path, torch):
    path = directory / 'training_args.bin'
    if not path.is_file():
        candidates = []
        for checkpoint in directory.glob('checkpoint-*'):
            match = re.fullmatch(r'checkpoint-(\d+)', checkpoint.name)
            required = [checkpoint / name for name in
                        ['training_args.bin', 'trainer_state.json', 'adapter_config.json',
                         'optimizer.pt', 'scheduler.pt', 'rng_state.pth']]
            if match and all(item.is_file() and item.stat().st_size for item in required):
                candidates.append((int(match.group(1)), checkpoint))
        if not candidates:
            return None, None, None
        _, checkpoint = max(candidates)
        path = checkpoint / 'training_args.bin'
    arguments, evidence = saved_evidence(path, torch)
    state_file = path.parent / 'trainer_state.json'
    state = None
    if state_file.is_file():
        value, state_evidence = saved_evidence(state_file, torch)
        state = {'global_step': value.get('global_step'), 'max_steps': value.get('max_steps'),
                 'epoch': value.get('epoch'), 'source': state_evidence}
    return {name: getattr(arguments, name, None) for name in ARGUMENTS}, evidence, state


def audit_one(family: str, stage: str, split: dict, torch):
    model, requested_gpu = FAMILIES[family]
    directory = ADAPTER_ROOT / family / stage
    if not directory.is_dir():
        return None
    arguments, argument_evidence, trainer_state = training_arguments(directory, torch)
    if arguments is None:
        return None
    if Path(arguments['output_dir']).resolve() != directory.resolve():
        raise ValueError(f'Saved arguments point to another output: {directory}')
    if arguments['bf16'] is not True or arguments['fp16'] is True:
        raise ValueError(f'Unexpected saved precision: {directory}')
    if arguments['num_train_epochs'] != 1 or arguments['gradient_accumulation_steps'] != 16:
        raise ValueError(f'Unexpected saved full-run training settings: {directory}')
    adapter_file = directory / 'adapter_config.json'
    if not adapter_file.is_file():
        adapter_file = Path(argument_evidence['path']).parent / 'adapter_config.json'
    adapter, adapter_evidence = saved_evidence(adapter_file, torch)
    if Path(adapter['base_model_name_or_path']).resolve() != Path(model).resolve():
        raise ValueError(f'Adapter belongs to another family: {directory}')
    dataset = STAGES[stage]
    report = {
        'audit_schema': 'saved_sft_evidence_v1',
        'audited_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'provenance_status': 'retrospective_audit_not_original_runtime_record',
        'family': family, 'stage': stage, 'base_model': model,
        'adapter_directory': str(directory), 'requested_physical_gpu': requested_gpu,
        'saved_trainer_arguments': arguments, 'saved_arguments_source': argument_evidence,
        'adapter_configuration': adapter, 'adapter_configuration_source': adapter_evidence,
        'latest_saved_trainer_state': trainer_state,
        'current_fixed_data_file': str(ROOT / 'data' / (dataset + '.jsonl')),
        'current_data_sha256_verified_against_manifest': split['sha256'][dataset],
        'verified_fixed_sft_rows': split['sft_sessions'],
        'verified_fixed_sft_scenarios': split['sft_scenarios'],
        'limitations': [
            'This is a retrospective audit, not a reconstruction of a lost runtime config.',
            'The current data hash matches the fixed manifest; this is not a recovered original runtime hash.',
            'Requested physical GPU follows the plan; no historical CUDA mask is invented.',
        ],
    }
    metrics_file = directory / 'train_metrics.json'
    if metrics_file.is_file():
        metrics, evidence = saved_evidence(metrics_file, torch)
        report['saved_train_metrics'] = metrics
        report['saved_train_metrics_source'] = evidence
    return report


def write_audit(report):
    directory = Path(report['adapter_directory'])
    target = directory / 'training_metadata_audit.json'
    if target.is_symlink():
        raise ValueError(f'Refusing a symlink audit target: {target}')
    descriptor, temporary_name = tempfile.mkstemp(prefix='.training_metadata_audit.', dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Save clearly labelled audits, never runtime configs')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from repro.profile import configure_profile_from_argv
    sys.argv = ['audit_saved_sft_metadata.py', '--profile', 'paper10k']
    configure_profile_from_argv()
    from repro.baseline_data import validate_fixed_splits
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('The audit must not initialize CUDA')
    split = validate_fixed_splits()
    for family in FAMILIES:
        for stage in STAGES:
            report = audit_one(family, stage, split, torch)
            if report is None:
                print(json.dumps({'family': family, 'stage': stage,
                                  'status': 'saved evidence not available yet'}), flush=True)
                continue
            output = write_audit(report) if args.write else None
            print(json.dumps({'family': family, 'stage': stage,
                              'status': report['provenance_status'],
                              'bf16_saved_arguments': report['saved_trainer_arguments']['bf16'],
                              'data_sha256': report['current_data_sha256_verified_against_manifest'],
                              'train_metrics_present': 'saved_train_metrics' in report,
                              'output': str(output) if output else None}), flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('Unexpected CUDA initialization while reading saved metadata')
    print('Saved SFT metadata audit completed without loading models or initializing CUDA.', flush=True)


if __name__ == '__main__':
    main()
