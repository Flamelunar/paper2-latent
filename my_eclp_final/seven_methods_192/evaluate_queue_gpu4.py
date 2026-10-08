"""Run Qwen3 evaluations for archived methods sequentially on GPU 4."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path('/app/hzj/liujianjian/ljj-2-paper')
    code = root / 'my_eclp_final/seven_methods_192'
    work = root / 'exp_code_ljj/artifacts/seven_methods_192_20261006'
    logdir = work / 'judges/qwen3_14b/logs'
    logdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='4', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4')
    for method in ['dialogue_sft', 'persugpt']:
        log = (logdir / f'{method}_gpu4.log').open('a')
        command = [sys.executable, '-u', str(code / 'evaluate.py'), '--repo-root', str(root),
                   '--output', str(work), '--physical-gpu', '4', '--judge', 'qwen3_14b',
                   '--method', method, '--limit', '192']
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        (logdir / f'{method}_worker.json').write_text(json.dumps({
            'pid': process.pid, 'gpu': 4, 'started_unix': time.time(), 'command': command}))
        print('launch', method, process.pid, flush=True)
        return_code = process.wait()
        log.close()
        if return_code:
            print('failed', method, return_code, flush=True)
            raise SystemExit(return_code)


if __name__ == '__main__':
    main()
