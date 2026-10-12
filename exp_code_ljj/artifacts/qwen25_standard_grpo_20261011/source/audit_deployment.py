"""Verify deployed source identity and record authoritative live process state."""
import ast
import hashlib
import json
from pathlib import Path
import time

from supervise import ROOT, RUN

expected = json.loads((RUN / 'source_inventory.json').read_text())
actual = {}
for name, checksum in expected.items():
    path = RUN / 'source' / name
    data = path.read_bytes()
    actual[name] = hashlib.sha256(data).hexdigest()
    if actual[name] != checksum:
        raise ValueError(f'Deployment differs: {name}')
    if name.endswith('.py'):
        ast.parse(data.decode('utf-8'), filename=str(path))
old = (ROOT / 'my_eclp_final/seven_methods_192/evaluate.py').read_text()
new = (RUN / 'source/evaluate.py').read_text()
new = new.replace('# Method dispatch only: retain the historical prompts, parsing and aggregation.\n', '')
new = new.replace(", 'ordinary_sft_grpo'", '').replace(",'ordinary_sft_grpo'", '')
if new != old:
    raise ValueError('Evaluator has changes beyond the added method dispatch')
state = {'observed_unix': time.time(), 'source_hashes': actual, 'evaluator_dispatch_only': True, 'processes': {}}
for filename, key in [('supervisor_launch.json', 'pid'), ('eclp_supervisor_launch.json', 'pid'), ('supervisor_state.json', 'pid')]:
    record = json.loads((RUN / filename).read_text())
    path = Path('/proc') / str(record[key]) / 'status'
    if not path.exists():
        raise RuntimeError(f'Expected live process missing: {record[key]}')
    lines = path.read_text().splitlines()
    state['processes'][filename] = {'pid': record[key], 'status': [s for s in lines if s.startswith(('Name:', 'State:', 'PPid:'))]}
    if any(s.startswith('State:') and ('Z (' in s or 'X (' in s) for s in lines):
        raise RuntimeError(f'Expected live process terminal: {record[key]}')
summary_path = RUN / 'training/sft/summary.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    exit_record = json.loads((RUN / 'exit.json').read_text())
    if not summary['complete'] or summary['sessions'] != 59868 or exit_record['returncode'] != 0:
        raise RuntimeError('SFT completion evidence is inconsistent')
    state['sft_completion'] = summary
else:
    record = json.loads((RUN / 'process.json').read_text())
    path = Path('/proc') / str(record['pid']) / 'status'
    if not path.exists():
        raise RuntimeError(f'Unfinished SFT process missing: {record["pid"]}')
    state['processes']['process.json'] = {'pid': record['pid'], 'status': path.read_text().splitlines()[:3]}
state['current_stage'] = json.loads((RUN / 'supervisor_state.json').read_text())['stage']
state['sft_progress'] = json.loads((RUN / 'training/sft/progress.json').read_text())
(RUN / 'deployment_audit.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'verified_source_files': len(actual), 'processes': state['processes'], 'sft_progress': state['sft_progress']}))
