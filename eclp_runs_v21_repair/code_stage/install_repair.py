#!/usr/bin/env python3
"""Install only files whose exact implementation passed the recorded smoke."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

ROOT = Path('/app/hzj/liujianjian/ljj-2-paper')
WORK = ROOT / 'eclp_runs_v21_repair'
STAGE = WORK / 'code_stage'
TARGET = ROOT / 'my_eclp_final'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((STAGE / 'DEPLOY_MANIFEST.json').read_text(encoding='utf-8'))
smoke = json.loads((WORK / 'gpu_smoke_repaired.json').read_text(encoding='utf-8'))
mechanism = json.loads((WORK / 'mechanism_repaired.json').read_text(encoding='utf-8'))
expected = manifest['files']['eclp_elpo.py']['sha256']
assert smoke['status'] == 'PASS_GPU_SMOKE_NOT_AN_EXPERIMENT_RESULT'
assert mechanism['status'] == 'PASS_MECHANISM_ONLY'
assert smoke['implementation_sha256'] == expected == mechanism['implementation_sha256']
assert mechanism['public_prompt_repetition_penalty_matches_hf']
for source, item in manifest['files'].items():
    assert digest(STAGE / source) == item['sha256'], source
compile((STAGE / 'eclp_elpo.py').read_text(encoding='utf-8'), 'eclp_elpo.py', 'exec')

backup = WORK / 'before_final_install' / str(time.time_ns())
backup.mkdir(parents=True)
for source, item in manifest['files'].items():
    target = TARGET / item['destination']
    if target.exists():
        saved = backup / item['destination']
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, saved)

for source, item in manifest['files'].items():
    target = TARGET / item['destination']
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o644
    temp = target.with_name(target.name + '.repair-install-' + str(os.getpid()))
    with temp.open('wb') as stream:
        stream.write((STAGE / source).read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temp, mode)
    os.replace(temp, target)
    assert digest(target) == item['sha256']

report = {'status': 'INSTALLED_VERIFIED_CODE', 'installed_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
          'implementation_sha256': expected, 'files': manifest['files'],
          'backup': str(backup), 'gpu_smoke': str(WORK / 'gpu_smoke_repaired.json'),
          'persuasion_effectiveness_verified': False}
(WORK / 'INSTALL_REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
