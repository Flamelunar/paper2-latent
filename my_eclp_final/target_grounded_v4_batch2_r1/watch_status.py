"""Report actual Linux process states; never send signals to an experiment."""
import datetime
import fcntl
import json
from pathlib import Path
import sys
import time
work=Path(sys.argv[1]);lock=(work/'.watch_status.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
previous=None;last_change=time.monotonic()
while True:
    state=json.loads((work/'supervisor_state.json').read_text())
    processes={}
    for key in ['supervisor_pid','child_pid']:
        path=Path(f"/proc/{state[key]}/status")
        processes[key]={'pid':state[key],'state':next((line.split(':',1)[1].strip() for line in path.read_text().splitlines() if line.startswith('State:')),'unknown') if path.exists() else 'exited'}
    files=list((work/'generation').glob('*request_trace.jsonl'))+[work/'generation_progress.json',work/'evaluation_progress.json']
    fingerprint=tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files if p.exists())
    if fingerprint!=previous:previous=fingerprint;last_change=time.monotonic()
    paused=any(p['state'].startswith('T') for p in processes.values())
    result={'checked_at':datetime.datetime.now().isoformat(),'stage':state['stage'],'target':state['target'],
            'declared_status':state['status'],'processes':processes,'actually_paused':paused,
            'seconds_without_output_change_observed':round(time.monotonic()-last_change,1),
            'note':'Observe only; no automatic resume. File mtime absolute clock may differ on shared mounts.'}
    tmp=work/'live_status.json.tmp';tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2));tmp.replace(work/'live_status.json')
    if state['status'] in ['complete','failed','superseded_by_batch2']:break
    time.sleep(15)
