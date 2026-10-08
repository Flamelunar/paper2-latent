"""Stop only the v4 serial pilot, snapshot its completed rows, keep other jobs intact."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time
root=Path('/app/hzj/liujianjian/ljj-2-paper')
work=root/'exp_code_ljj/artifacts/target_grounded_v4'
state=json.loads((work/'supervisor_state.json').read_text())
parent=state['supervisor_pid'];child=state['child_pid']
identities={}
for pid,expected in [(parent,'target_grounded_v4/supervise.py'),(child,'target_grounded_v4/generate.py')]:
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
    if expected not in cmd:raise RuntimeError(f'PID identity mismatch {pid}')
    identities[pid]=cmd
os.kill(parent,signal.SIGSTOP);os.kill(child,signal.SIGSTOP)
snapshot=work/'switch_batch2_snapshot';snapshot.mkdir(exist_ok=False)
counts={};files={}
for path in sorted((work/'generation').glob('*.jsonl')):
    data=path.read_bytes()
    rows=[json.loads(line) for line in data.decode().splitlines()]
    assert all(len(r['trajectory'])==10 for r in rows)
    (snapshot/path.name).write_bytes(data)
    counts[path.stem]=len(rows);files[path.name]=hashlib.sha256(data).hexdigest()
(snapshot/'generation_run.json').write_bytes((work/'generation_run.json').read_bytes())
record={'time':datetime.datetime.now().isoformat(),'identities':identities,'saved_rows':counts,'file_sha256':files,
        'scope':'only serial target_grounded_v4 supervisor and child; original paused training untouched',
        'successor':str(root/'exp_code_ljj/artifacts/target_grounded_v4_batch2')}
(snapshot/'switch_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
for pid in [child,parent]:
    os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
for _ in range(30):
    remaining=[]
    for pid in [parent,child]:
        status=Path(f'/proc/{pid}/status')
        if status.exists() and '\nState:\tZ' not in status.read_text():remaining.append(pid)
    if not remaining:break
    time.sleep(1)
else:raise RuntimeError(f'Old worker did not terminate cleanly: {remaining}')
state.update(status='superseded_by_batch2',replacement=record['successor'],saved_rows=counts)
(work/'supervisor_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)
