"""Serial continuation on GPU4 after the already-running public SFT."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/app/hzj/liujianjian/ljj-2-paper')
RUN = ROOT / 'exp_code_ljj/artifacts/qwen25_standard_grpo_20261011'
PYTHON = '/app/hzj/anaconda3/envs/ljj/bin/python'

def write(name, value):
    path = RUN / name
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)

def stage(name, command):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='4', TOKENIZERS_PARALLELISM='false')
    with (RUN / f'{name}.log').open('ab') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        write('supervisor_state.json', {'stage': name, 'pid': child.pid, 'supervisor_pid': os.getpid(),
            'physical_gpu': 4, 'command': command, 'started_unix': time.time()})
        code = child.wait()
    with (RUN / 'commands.jsonl').open('a') as f:
        f.write(json.dumps({'stage': name, 'command': command, 'returncode': code, 'ended_unix': time.time()}) + '\n')
    if code:
        raise RuntimeError(f'{name} exited {code}; inspect its log before resuming')

def worker():
    process = json.loads((RUN / 'process.json').read_text())
    write('supervisor_state.json', {'stage': 'waiting_for_sft', 'sft_pid': process['pid'],
        'supervisor_pid': os.getpid(), 'physical_gpu': 4})
    while not (RUN / 'exit.json').exists():
        try:
            os.kill(process['pid'], 0)
        except ProcessLookupError:
            # A wrapper may be committing exit.json after the child disappears.
            time.sleep(2)
            if not (RUN / 'exit.json').exists():
                raise RuntimeError('SFT process vanished without a terminal status')
        time.sleep(15)
    if json.loads((RUN / 'exit.json').read_text())['returncode']:
        raise RuntimeError('SFT failed; preserve logs and stop the chain')
    sft = json.loads((RUN / 'training/sft/summary.json').read_text())
    if not sft['complete'] or sft['sessions'] != 59868:
        raise RuntimeError('SFT did not complete one epoch')
    resume_args = []
    if (RUN / 'training/grpo/last.json').exists():
        resume_args = ['--resume', json.loads((RUN / 'training/grpo/last.json').read_text())['checkpoint']]
    grpo_summary_path = RUN / 'training/grpo/summary.json'
    completed = json.loads(grpo_summary_path.read_text()) if grpo_summary_path.exists() else {}
    if not (completed.get('complete') and completed.get('contexts') == 1800):
        stage('grpo', [PYTHON, '-u', str(RUN / 'source/train_text_grpo.py'), '--root', str(ROOT),
            '--sft', sft['checkpoint'], '--output', str(RUN / 'training/grpo')] + resume_args)
    grpo = json.loads((RUN / 'training/grpo/summary.json').read_text())
    if not grpo['complete'] or grpo['contexts'] != 1800:
        raise RuntimeError('GRPO did not complete one epoch')
    stage('generation', [PYTHON, '-u', str(RUN / 'source/generate_standard.py'), '--root', str(ROOT),
        '--checkpoint', grpo['checkpoint'], '--output', str(RUN), '--physical-gpu', '4'])
    stage('selene', [PYTHON, '-u', str(RUN / 'source/evaluate.py'), '--repo-root', str(ROOT),
        '--output', str(RUN), '--physical-gpu', '4', '--judge', 'selene8b',
        '--method', 'ordinary_sft_grpo', '--limit', '192', '--batch-size', '4'])
    write('supervisor_state.json', {'stage': 'evaluation_finished', 'supervisor_pid': os.getpid(),
        'needs_completion_audit': True, 'finished_unix': time.time()})

def main():
    if '--worker' in sys.argv:
        try:
            worker()
        except Exception as error:
            write('supervisor_error.json', {'error': str(error), 'unix': time.time()})
            raise
        return
    if (RUN / 'supervisor_launch.json').exists():
        raise FileExistsError('Existing supervisor; inspect its live PID before recovery')
    with (RUN / 'supervisor.log').open('ab') as log:
        child = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write('supervisor_launch.json', {'pid': child.pid, 'launched_unix': time.time()})
    print(json.dumps({'supervisor_pid': child.pid, 'run': str(RUN)}))

if __name__ == '__main__':
    main()
