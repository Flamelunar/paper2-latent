"""Archive the verified pre-update failure; do not touch any trained run."""
import json
import os
from pathlib import Path

run = Path('/app/hzj/liujianjian/ljj-2-paper/exp_code_ljj/artifacts/qwen25_standard_grpo_20261011')
state = json.loads((run / 'exit.json').read_text())
if state['returncode'] == 0:
    raise RuntimeError('Refuse recovery of successful run')
process = json.loads((run / 'process.json').read_text())
try:
    os.kill(process['pid'], 0)
except ProcessLookupError:
    pass
else:
    raise RuntimeError('Training process is live')
sft = run / 'training/sft'
if (sft / 'train.jsonl').exists() or list(sft.glob('checkpoint-*')):
    raise RuntimeError('Run has training updates; use checkpoint resume')
dest = run / 'startup_failure_transformers5'
dest.mkdir()
for name in ['sft.log', 'launcher.log', 'process.json', 'exit.json']:
    (run / name).rename(dest / name)
(sft / 'manifest.json').rename(dest / 'manifest.json')
print('Archived verified pre-update failure')
