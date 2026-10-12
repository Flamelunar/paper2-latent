"""Create an explicit dependency-lock migration with byte-identical SFT weights.

The original lock/checkpoint stay intact. This release records changed support
sources under a new lock; it never tells the old verifier those sources match.
"""
import copy
import json
from pathlib import Path
import sys

ROOT = Path('/app/hzj/liujianjian/ljj-2-paper')
RUN = ROOT / 'exp_code_ljj/artifacts/qwen25_standard_grpo_20261011'
RELEASE = RUN / 'eclp_release'
sys.path.insert(0, str(ROOT / 'my_eclp_final'))
from eclp_elpo import Config, verify_checkpoint, verify_lock, verified_prepared, sha256
from experiment_io import read_json, atomic_json, object_hash, immutable_json


def main():
    cfg = Config.load(ROOT / 'my_eclp_final/config_gpu4.json')
    cfg.repo_root = str(ROOT)
    old = read_json(ROOT / 'eclp_runs_gpu4/protocol.lock.json')
    assert object_hash({k: v for k, v in old.items() if k != 'lock_id'}) == old['lock_id']
    assert cfg.protocol() == old['protocol']
    for name, checksum in old['data'].items():
        assert sha256(ROOT / f'exp_code_ljj/data/{name}.jsonl') == checksum, name
    old_prepared = verified_prepared(ROOT / 'eclp_runs_gpu4/prepared', old)
    source = ROOT / 'eclp_runs_gpu4/sft_full/checkpoint-007484'
    original = verify_checkpoint(source, old)
    assert original['stage'] == 'sft' and original['step'] == 7484
    assert original['architecture'] == cfg.architecture()
    hashes = {name: sha256(ROOT / name) for name in old['sources']}
    changed = {name: {'previous': old['sources'][name], 'current': value}
               for name, value in hashes.items() if value != old['sources'][name]}
    new = copy.deepcopy(old)
    new['sources'] = hashes
    new['dependency_migration'] = {'previous_lock_id': old['lock_id'], 'changed_sources': changed,
        'method_implementation_changed': False, 'protocol_and_data_changed': False,
        'original_sft_checkpoint_id': original['checkpoint_id']}
    new.pop('lock_id')
    new['lock_id'] = object_hash(new)
    RELEASE.mkdir(parents=True, exist_ok=True)
    immutable_json(RELEASE / 'protocol.lock.json', new)
    cfg.work_dir = str(RELEASE.relative_to(ROOT))
    config = read_json(ROOT / 'my_eclp_final/config_gpu4.json')
    config['repo_root'] = str(ROOT)
    config['work_dir'] = cfg.work_dir
    immutable_json(RELEASE / 'config.json', config)
    prepared = RELEASE / 'prepared'
    prepared.mkdir(exist_ok=True)
    for rel in old_prepared['files']:
        path = prepared / rel
        if not path.exists():
            path.symlink_to((ROOT / 'eclp_runs_gpu4/prepared' / rel).resolve())
    prepared_manifest = copy.deepcopy(old_prepared)
    prepared_manifest['lock_id'] = new['lock_id']
    prepared_manifest['dependency_migration_original_lock_id'] = old['lock_id']
    immutable_json(prepared / 'manifest.json', prepared_manifest)
    destination = RELEASE / 'sft/checkpoint-007484'
    if not destination.exists():
        destination.mkdir(parents=True)
        for rel in original['files']:
            path = destination / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to((source / rel).resolve())
        migrated = copy.deepcopy(original)
        migrated['lock_id'] = new['lock_id']
        migrated['provenance']['dependency_lock_migration'] = {
            'original_checkpoint_id': original['checkpoint_id'], 'original_lock_id': old['lock_id'],
            'changed_sources': changed, 'weights_and_optimizer_files_changed': False}
        migrated.pop('checkpoint_id')
        migrated['checkpoint_id'] = object_hash(migrated)
        atomic_json(destination / 'checkpoint.json', migrated)
    verified = verify_checkpoint(destination, new)
    assert verified['files'] == original['files']
    for key in ['architecture', 'executor_hash', 'policy_hash', 'reference_policy_hash', 'implementation_sha256']:
        assert verified[key] == original[key], key
    verify_lock(cfg, RELEASE / 'protocol.lock.json')
    migrated_prepared = verified_prepared(prepared, new)
    assert migrated_prepared['files'] == old_prepared['files']
    report = {'release': str(RELEASE), 'config': str(RELEASE / 'config.json'), 'checkpoint': str(destination),
        'prepared': str(prepared), 'prepared_data_file_hashes_identical': True,
        'old_lock_id': old['lock_id'], 'new_lock_id': new['lock_id'], 'changed_sources': changed,
        'original_checkpoint_id': original['checkpoint_id'], 'migrated_checkpoint_id': verified['checkpoint_id'],
        'weight_file_hashes_identical': True, 'method_code_hash_identical': True,
        'source_recovery': 'Exact locked helper sources were not found in inspected backups. A new dependency lock is explicit; old hashes were not bypassed or overwritten.'}
    atomic_json(RELEASE / 'migration.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
