#!/usr/bin/env python3
"""Replace only the supervisor; the live SFT child keeps running unchanged."""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT = Path('/app/hzj/liujianjian/ljj-2-paper')
CODE = ROOT / 'my_eclp_final'
WORK = ROOT / 'eclp_runs_gpu4'
PYTHON = '/app/hzj/anaconda3/envs/ljj/bin/python'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(pid):
    path = Path('/proc') / str(pid)
    try:
        raw = (path / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        if fields[0] in {'Z', 'X'}:
            return None
        return {'pid': int(pid), 'start_ticks': int(fields[19]),
                'command': [x.decode() for x in (path / 'cmdline').read_bytes().split(b'\0') if x]}
    except FileNotFoundError:
        return None


def write_json(path, data):
    temp = path.with_name(path.name + '.tmp-install')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def install():
    diagnostic = json.loads((WORK / 'offline_gpu_diagnostic.json').read_text())
    if (diagnostic.get('status') != 'PASS_OFFLINE_GPU_DIAGNOSTIC_NOT_AN_EXPERIMENT_RESULT'
            or diagnostic.get('trainer_sha256') != sha(CODE / 'offline_latent_rl.py')):
        raise RuntimeError('Current offline trainer has not passed its GPU mechanism diagnostic')
    old = json.loads((WORK / 'pipeline_status.json').read_text())
    if old.get('phase') != 'sft_full':
        raise RuntimeError('Expected the authorized, original SFT phase before handoff')
    parent = identity(old['pipeline_pid'])
    child = identity(old['child_pid'])
    if (parent is None or child is None
            or str(CODE / 'run_gpu4_pipeline.py') not in parent['command']
            or str(CODE / 'run_main_experiment.py') not in child['command']
            or 'sft' not in child['command']):
        raise RuntimeError('SFT process identities differ from the expected GPU4 run')
    env_entries = (Path('/proc') / str(child['pid']) / 'environ').read_bytes().split(b'\0')
    if b'CUDA_VISIBLE_DEVICES=4' not in env_entries:
        raise RuntimeError('Refusing to adopt a training process on another GPU')
    legacy = json.loads((WORK / 'pipeline_contract.json').read_text())
    if sha(CODE / 'eclp_elpo.py') != legacy['code_sha256']['eclp_elpo.py']:
        raise RuntimeError('The running SFT core changed; do not perform a silent migration')
    for filename in ['offline_latent_rl.py', 'run_gpu4_offline_pipeline.py', 'evaluate_our.offline.pending.py']:
        ast.parse((CODE / filename).read_text(), filename=filename)
    backup = WORK / 'offline_handoff_backup'
    backup.mkdir(exist_ok=True)
    evaluator = CODE / 'evaluate_our.py'
    launcher = CODE / 'run_gpu4_full_chain.sh'
    if (backup / 'evaluate_our.py').exists() or (backup / 'run_gpu4_full_chain.sh').exists():
        raise FileExistsError('Handoff backups already exist; inspect the earlier handoff before repeating')
    original_evaluator, original_launcher = evaluator.read_bytes(), launcher.read_bytes()
    (backup / 'evaluate_our.py').write_bytes(original_evaluator)
    (backup / 'run_gpu4_full_chain.sh').write_bytes(original_launcher)
    stopped = False
    supervisor = None
    success = False
    try:
        os.kill(parent['pid'], signal.SIGSTOP)
        stopped = True
        handoff = {
            'created_at': datetime.datetime.now().astimezone().isoformat(),
            'old_supervisor': parent, 'sft_child': child, 'previous_status': old,
            'sft_core_sha256_unchanged': sha(CODE / 'eclp_elpo.py'),
            'reason': 'User requested offline implicit RL without training dialogues or Judges',
            'online_training_enabled': False,
        }
        write_json(WORK / 'offline_handoff.json', handoff)
        evaluator.write_bytes((CODE / 'evaluate_our.offline.pending.py').read_bytes())
        launcher.write_text(
            '#!/usr/bin/env bash\nset -Eeuo pipefail\n'
            'ROOT=/app/hzj/liujianjian/ljj-2-paper\n'
            'export CUDA_DEVICE_ORDER=PCI_BUS_ID\nexport CUDA_VISIBLE_DEVICES=4\n'
            'export PYTHONUNBUFFERED=1\nexport DAILYPERSUASION_PROFILE=paper10k\n'
            'export ALLOW_JUDGE_EVAL=1\nexport TOKENIZERS_PARALLELISM=false\n'
            'exec /app/hzj/anaconda3/envs/ljj/bin/python -u "$ROOT/my_eclp_final/run_gpu4_offline_pipeline.py" "$@"\n',
            encoding='utf-8')
        os.chmod(launcher, 0o755)
        command = [PYTHON, '-u', str(CODE / 'run_gpu4_offline_pipeline.py'),
                   '--adopt-from', str(WORK / 'offline_handoff.json')]
        with (WORK / 'logs/offline_pipeline.log').open('a') as output:
            supervisor = subprocess.Popen(command, cwd=ROOT, stdout=output,
                                          stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                          start_new_session=True)
        ready = False
        for _ in range(100):
            if supervisor.poll() is not None:
                raise RuntimeError('New supervisor exited; inspect offline_pipeline.log')
            path = WORK / 'pipeline_status.json'
            status = json.loads(path.read_text())
            if (status.get('pipeline_pid') == supervisor.pid and status.get('phase') == 'sft_adopted'
                    and status.get('child_pid') == child['pid'] and status.get('online_training_enabled') is False):
                ready = True
                break
            time.sleep(0.2)
        if not ready or identity(child['pid']) != child or identity(parent['pid']) != parent:
            raise RuntimeError('New supervisor did not adopt the same live SFT process')
        os.kill(parent['pid'], signal.SIGKILL)
        stopped = False
        success = True
        handoff.update(new_supervisor_pid=supervisor.pid, old_supervisor_terminated=True,
                       sft_child_preserved=True, completed_at=datetime.datetime.now().astimezone().isoformat())
        write_json(WORK / 'offline_handoff.json', handoff)
        print(json.dumps({'handoff': handoff, 'status': status}, ensure_ascii=False, indent=2))
    finally:
        if not success:
            if supervisor is not None and supervisor.poll() is None:
                supervisor.terminate()
                try:
                    supervisor.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    supervisor.kill()
                    supervisor.wait()
            evaluator.write_bytes(original_evaluator)
            launcher.write_bytes(original_launcher)
            if stopped and identity(parent['pid']) == parent:
                os.kill(parent['pid'], signal.SIGCONT)


if __name__ == '__main__':
    install()
