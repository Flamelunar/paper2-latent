"""Own only this run's workers, serialize by GPU, and persist live progress."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from report import render,LABELS

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.repo_root;work=a.output;code=Path(__file__).resolve().parent
 sys.path[:0]=[str(root/'my_eclp_final'),str(root/'exp_code_ljj')]
 from experiment_io import atomic_json,writer_lock
 with writer_lock(work/'.supervisor.lock'):
  logs=work/'logs';logs.mkdir(exist_ok=True);active={};completed=set();failed={};attempts={};handles=[]
  generation=['our_analysis','our_direct','zero_shot_repaired','ma2p_repaired','latentratt']
  def finished(m):
   p=work/'generation'/f'{m}.jsonl'
   return p.exists() and sum(1 for _ in p.open())==192
  def scored(j,m):
   p=work/'judges'/j/'evaluation'/m/'summary.json'
   return p.exists() and json.loads(p.read_text()).get('complete') is True
  def launch(gpu,key,script,args):
   attempts[key]=attempts.get(key,0)+1
   log=logs/f'{key}.attempt{attempts[key]}.log';f=log.open('a');handles.append(f)
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
   cmd=[sys.executable,str(code/script),'--repo-root',str(root),'--output',str(work if script=='evaluate.py' else work/'generators'/key.removeprefix('gen_')),'--physical-gpu',str(gpu),*args]
   proc=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   active[gpu]={'process':proc,'key':key,'log':str(log),'cmd':cmd,'started':time.time()}
   print('launch',gpu,key,proc.pid,flush=True)
  render(root,work)
  while True:
   for gpu,v in list(active.items()):
    rc=v['process'].poll()
    if rc is None:continue
    del active[gpu]
    if rc==0:completed.add(v['key']);print('done',v['key'],flush=True)
    elif attempts[v['key']]>=3:failed[v['key']]={'exit_code':rc,'log':v['log']};print('FAILED',v['key'],rc,flush=True)
    else:print('retry later',v['key'],rc,flush=True)
   if 1 not in active:
    todo=[m for m in generation if not finished(m) and 'gen_'+m not in failed]
    if todo:
     m=todo[0];launch(1,'gen_'+m,'generate_latent.py' if m=='latentratt' else 'generate_direct.py' if m=='our_direct' else 'generate.py',[] if m=='latentratt' else ['--method',m,'--limit','192'])
    else:
     todo=[m for m in LABELS if finished(m) and not scored('qwen3_14b',m) and 'qwen3_14b_'+m not in failed]
     if todo:launch(1,'qwen3_14b_'+todo[0],'evaluate.py',['--judge','qwen3_14b','--method',todo[0],'--limit','192'])
   if 0 not in active:
    todo=[m for m in ['dialogue_sft','persugpt',*generation] if finished(m) and not scored('selene8b',m) and 'selene8b_'+m not in failed]
    if todo:launch(0,'selene8b_'+todo[0],'evaluate.py',['--judge','selene8b','--method',todo[0],'--limit','192'])
   progress={}
   for m in LABELS:
    p=work/'generation'/f'{m}.jsonl';n=sum(1 for _ in p.open()) if p.exists() else 0
    progress[m]={'generated':n,'target':192,'source':'historical' if m in ['dialogue_sft','persugpt'] else 'new'}
    for j in ['selene8b','qwen3_14b']:
     p=work/'judges'/j/'evaluation'/m/'judgments.jsonl';progress[m][j+'_scored']=sum(1 for _ in p.open()) if p.exists() else 0
   workers={}
   for gpu,v in active.items():
    status=Path(f'/proc/{v["process"].pid}/status');state=next((line.split(':',1)[1].strip() for line in status.read_text().splitlines() if line.startswith('State:')),'missing') if status.exists() else 'exited'
    workers[str(gpu)]={'pid':v['process'].pid,'task':v['key'],'state':state,'log':v['log'],'elapsed_seconds':time.time()-v['started']}
   all_done=all(scored(j,m) for j in ['selene8b','qwen3_14b'] for m in LABELS)
   atomic_json(work/'live_status.json',{'updated_unix':time.time(),'supervisor_pid':os.getpid(),'status':'complete' if all_done else 'running' if active else 'blocked','progress':progress,'workers':workers,'failed':failed})
   render(root,work)
   if all_done or (not active and failed):break
   time.sleep(15)
  for f in handles:f.close()
if __name__=='__main__':main()
