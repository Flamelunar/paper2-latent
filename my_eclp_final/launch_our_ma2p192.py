#!/usr/bin/env python3
"""Launch the durable Our+MA2P pipeline without depending on an SSH session."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    work=root/'eclp_runs_gpu4/our_ma2p192'
    work.mkdir(parents=True,exist_ok=True)
    status=work/'pipeline_status.json'
    if status.exists():
        previous=json.loads(status.read_text(encoding='utf-8'))
        if previous.get('state')=='running':
            pid=previous.get('supervisor_pid')
            try:
                os.kill(pid,0)
            except (ProcessLookupError,TypeError):
                pass
            else:
                raise RuntimeError(f'Pipeline supervisor {pid} is already running')
    command=[sys.executable,'-u',str(root/'my_eclp_final/run_our_ma2p192.py'),
             '--repo-root',str(root),'--checkpoint-mode','full_eclp']
    env=dict(os.environ,PYTHONUNBUFFERED='1')
    with (work/'supervisor.log').open('a',encoding='utf-8') as log:
        child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'supervisor_pid':child.pid,'log':str(work/'supervisor.log'),
                      'status':str(status),'command':command}))


if __name__=='__main__':main()
