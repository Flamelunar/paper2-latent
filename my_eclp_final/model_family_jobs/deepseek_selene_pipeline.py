"""Judge stable batches of 20 on GPU1 while GPU0 keeps generating DeepSeek."""
import fcntl,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/app/hzj/liujianjian/ljj-2-paper');JOB=ROOT/'my_eclp_final/model_family_jobs';PY='/app/hzj/anaconda3/envs/ljj/bin/python'
WORK=ROOT/'exp_code_ljj/artifacts/deepseek_goal_strategy_192_20261008/deepseek_r1_qwen14b'
VIEW=WORK/'selene_pipeline';METHODS=['zero_shot_repaired','dialogue_sft','persugpt','ma2p_repaired','our_direct','latentratt']
def read(p):return json.loads(p.read_text())
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.tmp-{os.getpid()}');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def state(stage,**kw):atomic(WORK/'selene_pipeline_status.json',dict(stage=stage,pid=os.getpid(),gpu=1,time=time.time(),**kw))
def main():
 VIEW.mkdir(parents=True,exist_ok=True)
 with (VIEW/'.controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  contract=WORK/'generation_run.json';(VIEW/'generation_run.json').write_bytes(contract.read_bytes())
  for method in METHODS:
   target=WORK/'judges/selene8b/evaluation'/method/'summary.json'
   if target.exists() and read(target).get('complete'):continue
   while True:
    src=WORK/'generators'/method/'generation'/f'{method}.jsonl'
    lines=src.read_bytes().splitlines(keepends=True) if src.exists() else []
    # A growing final chunk would change cached prompts: use complete groups of20.
    if lines and not lines[-1].endswith(b'\n'):lines.pop()
    n=len(lines) if len(lines)==192 else len(lines)//20*20
    summary=VIEW/'judges/selene8b/evaluation'/method/'summary.json'
    done=read(summary).get('n',0) if summary.exists() else 0
    if n<20 or (n<=done and n!=192):
     state('waiting_generated_batch',method=method,generated=len(lines),judged=done);time.sleep(20);continue
    dest=VIEW/'generation'/f'{method}.jsonl';dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.jsonl.tmp');tmp.write_bytes(b''.join(lines[:n]));tmp.replace(dest)
    while True:
     free=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
     if free>=26*1024:break
     state('waiting_memory',method=method,free_mib=free,judged=done);time.sleep(20)
    cmd=[PY,'-u',str(ROOT/'my_eclp_final/seven_methods_192/evaluate.py'),'--repo-root',str(ROOT),'--output',str(VIEW),'--physical-gpu','1','--judge','selene8b','--method',method,'--limit','192','--available-only','--batch-size','4','--memory-cap-gib','22']
    log=VIEW/f'{method}.log';env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',DAILYPERSUASION_PROFILE='paper10k',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    with log.open('a') as f:
     proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     state('evaluating',method=method,n=n,worker_pid=proc.pid,log=str(log));code=proc.wait()
    if code:raise RuntimeError(f'Selene pipeline {method} exited {code}: {log}')
    result=read(summary)
    if n==192:
     assert result.get('complete') and result['n']==192
     assert dest.read_bytes()==src.read_bytes(),'Final live snapshot differs from full generated dialogue'
     generation=WORK/'generation'/f'{method}.jsonl';generation.parent.mkdir(exist_ok=True)
     if not generation.exists():generation.symlink_to(src)
     source_contract=VIEW/'judges/selene8b/evaluation_run.json';official=WORK/'judges/selene8b/evaluation_run.json'
     official.parent.mkdir(parents=True,exist_ok=True)
     if official.exists():assert read(official)==read(source_contract)
     else:official.write_bytes(source_contract.read_bytes())
     final=target.parent
     if not final.exists():shutil.copytree(summary.parent,final)
     else:raise RuntimeError(f'Official evaluation already exists, refusing to replace: {final}')
     atomic(final/'pipeline_provenance.json',{'gpu':1,'source':str(summary.parent),'n':192,'generation_matches_exactly':True,'policy':'partial batches never populate final scores; final192 imported with raw judgments and matching contract'})
     subprocess.run([PY,str(JOB/'report_family.py'),'--family','deepseek_r1_qwen14b'],check=True)
     state('method_complete',method=method,n=192);break
  state('complete')
if __name__=='__main__':
 try:main()
 except Exception as e:state('failed',error=repr(e));raise
