"""Time a clean ECLP LPO epoch and evaluate that exact checkpoint, serially."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from supervise import ROOT, RUN, PYTHON, stage, write

ECLP = RUN / 'eclp_release/offline_rl'

def worker():
    write('eclp_supervisor_state.json', {'stage': 'waiting_for_standard_comparison', 'pid': os.getpid()})
    while True:
        if (RUN / 'supervisor_error.json').exists():
            raise RuntimeError('Standard comparison failed; inspect it before continuing')
        status = json.loads((RUN / 'supervisor_state.json').read_text())
        if status['stage'] == 'evaluation_finished':
            break
        launched = json.loads((RUN / 'supervisor_launch.json').read_text())
        os.kill(launched['pid'], 0)
        time.sleep(15)
    if ECLP.exists():
        raise FileExistsError('Clean timing run already exists; inspect before resume')
    release = json.loads((RUN / 'eclp_release/migration.json').read_text())
    stage('eclp_clean_lpo', [PYTHON, '-u', str(ROOT / 'my_eclp_final/offline_latent_rl.py'),
        '--config', release['config'],
        '--plan', str(ROOT / 'my_eclp_final/offline_training_plan.json'), 'train',
        '--prepared', release['prepared'],
        '--checkpoint', release['checkpoint'], '--output', str(ECLP)])
    summary = json.loads((ECLP / 'summary.json').read_text())
    if not summary['complete'] or summary['contexts'] != 1800 or summary['continued_from_run_hash'] is not None:
        raise ValueError('ECLP timing must cover a clean complete epoch, not a resumed suffix')
    output = RUN / 'eclp_comparison'
    stage('eclp_generation', [PYTHON, '-u', str(RUN / 'source/generate_direct.py'),
        '--repo-root', str(ROOT), '--checkpoint', summary['checkpoint'], '--output', str(output),
        '--lock', str(RUN / 'eclp_release/protocol.lock.json'),
        '--physical-gpu', '4', '--method', 'our_direct', '--limit', '192'])
    stage('eclp_selene', [PYTHON, '-u', str(RUN / 'source/evaluate.py'), '--repo-root', str(ROOT),
        '--output', str(output), '--physical-gpu', '4', '--judge', 'selene8b',
        '--method', 'our_direct', '--limit', '192', '--batch-size', '4'])
    stage('finalize', [PYTHON, str(RUN / 'source/finalize.py')])
    write('eclp_supervisor_state.json', {'stage': 'artifacts_finished', 'pid': os.getpid(),
        'needs_completion_audit': True, 'finished_unix': time.time()})

def main():
    if '--worker' in sys.argv:
        try:
            worker()
        except Exception as error:
            write('eclp_supervisor_error.json', {'error': str(error), 'unix': time.time()})
            raise
        return
    if (RUN / 'eclp_supervisor_launch.json').exists():
        raise FileExistsError('Existing continuation; inspect its process')
    with (RUN / 'eclp_supervisor.log').open('ab') as log:
        child = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write('eclp_supervisor_launch.json', {'pid': child.pid, 'launched_unix': time.time()})
    print(json.dumps({'eclp_supervisor_pid': child.pid}))

if __name__ == '__main__':
    main()
