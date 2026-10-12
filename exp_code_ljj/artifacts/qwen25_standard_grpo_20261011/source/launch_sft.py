import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/app/hzj/liujianjian/ljj-2-paper')
RUN = ROOT / 'exp_code_ljj/artifacts/qwen25_standard_grpo_20261011'
PYTHON = '/app/hzj/anaconda3/envs/ljj/bin/python'

def main():
    RUN.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, '-u', str(RUN / 'source/train_public_sft.py'),
               '--root', str(ROOT), '--output', str(RUN / 'training/sft')]
    if '--worker' in sys.argv:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='4', TOKENIZERS_PARALLELISM='false')
        with (RUN / 'sft.log').open('ab') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            (RUN / 'process.json').write_text(json.dumps({'wrapper_pid': os.getpid(),
                'pid': child.pid, 'physical_gpu': 4, 'command': command, 'started_unix': time.time()}, indent=2))
            code = child.wait()
        (RUN / 'exit.json').write_text(json.dumps({'returncode': code, 'ended_unix': time.time()}))
        return
    if (RUN / 'process.json').exists() or (RUN / 'sft.log').exists():
        raise FileExistsError('Existing launch; inspect its process instead of restarting')
    raw = subprocess.check_output(['nvidia-smi', '-i', '4', '--query-gpu=memory.free',
                                  '--format=csv,noheader,nounits'], text=True).strip()
    if int(raw) < 40000:
        raise RuntimeError('GPU4 has insufficient free memory; preserve existing tasks')
    with (RUN / 'launcher.log').open('ab') as log:
        child = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), '--worker'],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'wrapper_pid': child.pid, 'run': str(RUN), 'physical_gpu': 4}))

if __name__ == '__main__':
    main()
