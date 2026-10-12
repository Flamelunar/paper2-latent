"""Read-only prerequisites for the queued clean ECLP LPO epoch."""
import json
from pathlib import Path
import sys
import time

ROOT = Path('/app/hzj/liujianjian/ljj-2-paper')
RUN = ROOT / 'exp_code_ljj/artifacts/qwen25_standard_grpo_20261011'
sys.path.insert(0, str(ROOT / 'my_eclp_final'))
from eclp_elpo import Config, verify_lock, verified_prepared, verify_checkpoint

RELEASE = RUN / 'eclp_release'
cfg = Config.load(RELEASE / 'config.json')
cfg.repo_root = str(ROOT)
lock = verify_lock(cfg, RELEASE / 'protocol.lock.json')
prepared = verified_prepared(RELEASE / 'prepared', lock)
checkpoint = RELEASE / 'sft/checkpoint-007484'
metadata = verify_checkpoint(checkpoint, lock)
assert metadata['stage'] == 'sft' and metadata['step'] == 7484
assert metadata['architecture'] == cfg.architecture()
result = {'observed_unix': time.time(), 'lock_id': lock['lock_id'],
    'checkpoint_id': metadata['checkpoint_id'], 'architecture_matches': True,
    'source_and_prepared_data_verified': True, 'base_weight_fingerprint_verified_here': False,
    'dependency_migration': json.loads((RELEASE / 'migration.json').read_text()),
    'note': 'Canonical trainer separately verifies the base model fingerprint before loading.'}
(RUN / 'eclp_compatibility_audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
