"""Evaluate completed baselines after the reusable-row queue, before long Our training finishes."""
import os,signal,time,subprocess
from pathlib import Path
import supervise_campaign as s
reuse_pid=2089915
while Path(f'/proc/{reuse_pid}').exists():
 state=Path(f'/proc/{reuse_pid}/stat').read_text().split()[2]
 if state=='Z':break
 time.sleep(15)
parent=2029374
paused=False
try:
 if Path(f'/proc/{parent}/cmdline').exists() and b'supervise_campaign.py' in Path(f'/proc/{parent}/cmdline').read_bytes():
  os.kill(parent,signal.SIGSTOP);paused=True
 # Allow an already admitted generator to finish and release its model.
 while True:
  active=[]
  for family in s.FAMILIES:
   for method in s.GENS:
    p=s.generation_path(family,method)/'worker.worker.json'
    if p.exists():
     import json
     pid=json.loads(p.read_text())['pid'];proc=Path(f'/proc/{pid}/stat')
     if proc.exists() and proc.read_text().split()[2]!='Z':active.append(pid)
  if not active:break
  print('waiting for active generation',active,flush=True);time.sleep(20)
 for family in s.FAMILIES:
  for method in s.GENS:
   if not s.completed_generation(family,method):continue
   s.publish_generation(family,method)
   for judge in ['selene8b','qwen3_14b']:
    if s.completed_evaluation(family,method,judge):continue
    # Reuse newly judged inherited rows only when every model/prompt/source contract matches.
    import json,sys
    sys.path.insert(0,str(s.CODE));sys.path.insert(0,str(s.ROOT/'exp_code_ljj'))
    from experiment_io import read_json,read_jsonl,object_hash,append_jsonl,atomic_json,digest
    partial=s.ROOT/'exp_code_ljj/artifacts/reusable_partial192_20261007'/family
    old=partial/'judges'/judge/'evaluation'/method/'judgments.jsonl'
    dest=s.WORK/family/'judges'/judge/'evaluation'/method/'judgments.jsonl'
    if old.exists() and not dest.exists():
     a=read_json(partial/'judges'/judge/'evaluation_run.json');b=read_json(s.WORK/family/'judges'/judge/'evaluation_run.json')
     for key in a:
      if key not in ['evaluator_sha256']:assert a[key]==b[key],key
     sources={r['index']:r for r in read_jsonl(s.WORK/family/'generation'/f'{method}.jsonl')}
     selected=read_jsonl(old)
     for row in selected:
      assert row['contract_hash']==object_hash(a)
      assert row['source_row_hash']==object_hash(sources[row['index']])
     atomic_json(dest.parent/'imported_judgments_provenance.json',{'source':str(old),'sha256':digest(old),'source_contract_hash':object_hash(a),'destination_contract_hash':object_hash(b),'n':len(selected),'policy':'identical generation rows and judge/model/prompt contracts; retain all raw decisions'})
     for row in selected:
      row['contract_hash']=object_hash(b);append_jsonl(dest,row)
     print('reused latest judgments',family,method,judge,len(selected),flush=True)
    print('full192 evaluate',family,method,judge,flush=True)
    s.launch_eval(family,method,judge,4)
    subprocess.run([s.PY,str(s.JOB/'report_family.py')],check=True)
finally:
 if paused and Path(f'/proc/{parent}').exists():os.kill(parent,signal.SIGCONT)
