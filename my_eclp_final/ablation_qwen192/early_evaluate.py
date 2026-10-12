"""Score committed scenes on GPU4 while generation continues."""
import json,os,time,subprocess,sys
from pathlib import Path
R=Path('/app/hzj/liujianjian/ljj-2-paper');D=R/'my_eclp_final/ablation_qwen192';A=R/'exp_code_ljj/artifacts/qwen25_ablation_192_20261010';PY='/app/hzj/anaconda3/envs/ljj/bin/python';sys.path.insert(0,str(D))
from supervise import report,ENV
from experiment_io import atomic_json
O=A/'wo_g'
try:
 for judge in ['selene8b','qwen3_14b']:
  marker=O/'early_eval_worker.json';atomic_json(marker,{'pid':os.getpid(),'judge':judge,'physical_gpu':4})
  while True:
   p=O/'judges'/judge/'evaluation/our_direct/summary.json';summary=json.loads(p.read_text()) if p.exists() else {};done=summary.get('n',0)
   if done==192:break
   source=O/'generation/our_direct.jsonl';n=sum(1 for x in source.open() if x.strip()) if source.exists() else 0
   # One initial call verifies the changed partial-generation path; later use >=20 new scenes.
   if n==0 or (done>0 and n<192 and n-done<20):time.sleep(20);continue
   free=int(subprocess.check_output(['nvidia-smi','--id=4','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());need=28*1024 if judge=='selene8b' else 40*1024
   if free<need:time.sleep(20);continue
   cmd=[PY,'-u',str(D/'evaluate.py'),'--repo-root',str(R),'--output',str(O),'--judge',judge,'--method','our_direct','--physical-gpu','4','--limit','192','--batch-size','4','--memory-cap-gib','22' if judge=='selene8b' else '34','--available-only']
   with (O/f'{judge}_early.log').open('a') as log:
    worker=subprocess.Popen(cmd,cwd=R,env=ENV,stdout=log,stderr=subprocess.STDOUT)
    atomic_json(O/f'{judge}_early_worker.json',{'pid':worker.pid,'cmd':cmd,'physical_gpu':4})
    while worker.poll() is None:report();time.sleep(20)
    if worker.returncode:raise RuntimeError(f'Early {judge} failed {worker.returncode}')
   report()
  print(judge,'complete192',flush=True)
 marker.unlink(missing_ok=True)
except Exception as e:
 atomic_json(O/'early_eval_error.json',{'error':repr(e)});raise
