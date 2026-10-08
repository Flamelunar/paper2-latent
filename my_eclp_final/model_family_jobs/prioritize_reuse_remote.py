import os,json,signal,subprocess,time
from pathlib import Path
R=Path('/app/hzj/liujianjian/ljj-2-paper'); C=R/'my_eclp_final'; PY='/app/hzj/anaconda3/envs/ljj/bin/python'
parent=2029374
cmd=Path(f'/proc/{parent}/cmdline').read_bytes()
assert b'supervise_campaign.py' in cmd and b'--train-only' not in cmd
os.kill(parent,signal.SIGSTOP)
print('campaign scheduler paused; training and current generator continue',flush=True)
try:
 while Path('/proc/2029466').exists():
  state=Path('/proc/2029466/stat').read_text().split()[2]
  if state=='Z':break
  time.sleep(15)
 audit=json.loads((C/'model_family_jobs/source_audit.json').read_text())
 queue=[]
 for family in ['meta_llama31','deepseek_r1_qwen14b']:
  source=R/'exp_code_ljj/artifacts/model_family_repaired192_20261007'/family
  out=R/'exp_code_ljj/artifacts/reusable_partial192_20261007'/family
  (out/'generation').mkdir(parents=True,exist_ok=True)
  contract=json.loads((source/'generation_run.json').read_text())
  (out/'generation_run.json').write_text(json.dumps(contract,indent=2))
  for method,slug in [('dialogue_sft','baseline2_dialogue_sft'),('persugpt','baseline3_isr_simdpo')]:
   old=R/'exp_code_ljj/artifacts/generation_from_start192'/family/(slug+'.jsonl')
   info=audit[family][slug]
   import hashlib
   assert hashlib.sha256(old.read_bytes()).hexdigest()==info['sha256']
   rows=[json.loads(s) for s in old.read_text().splitlines() if s.strip()]
   rows=[r for r in rows if r['index'] not in info['affected_indexes']]
   if not rows:continue
   (out/'generation'/f'{method}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
   (out/'generation'/f'{method}_provenance.json').write_text(json.dumps({'source':str(old),'source_sha256':info['sha256'],'n':len(rows),'expected':192,'partial_diagnostic':True},indent=2))
   queue.append((out,method,len(rows)))
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='4',TRITON_CACHE_DIR=f'/tmp/ljj2-paper-triton-{os.getuid()}')
 for judge,cap in [('selene8b',22),('qwen3_14b',32)]:
  for out,method,n in queue:
   print('START',out.name,method,judge,n,flush=True)
   subprocess.run([PY,'-u',str(C/'seven_methods_192/evaluate.py'),'--repo-root',str(R),'--output',str(out),'--limit','192','--physical-gpu','4','--judge',judge,'--method',method,'--available-only','--batch-size','4','--memory-cap-gib',str(cap)],env=env,check=True)
finally:
 os.kill(parent,signal.SIGCONT)
 print('campaign scheduler resumed',flush=True)
