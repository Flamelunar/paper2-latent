"""Adopt this experiment's running workers and fill safe parallel GPU slots."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from report import LABELS,render
HISTORICAL={'dialogue_sft','persugpt','latentratt'}
GENERATE=['our_analysis','our_direct','zero_shot_repaired','ma2p_repaired']

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=a.repo_root;work=a.output;code=Path(__file__).resolve().parent
 sys.path[:0]=[str(root/'my_eclp_final'),str(root/'exp_code_ljj')]
 from experiment_io import atomic_json,writer_lock
 active={};failed={};attempts={};logs=work/'logs';logs.mkdir(exist_ok=True)
 def generated(method):
  path=work/'generation'/f'{method}.jsonl'
  return path.exists() and sum(1 for line in path.open() if line.strip())==192
 def scored(judge,method):
  path=work/'judges'/judge/'evaluation'/method/'summary.json'
  if not path.exists():return False
  summary=json.loads(path.read_text());return summary.get('complete') is True and summary.get('n')==192
 def complete(task):return generated(task['method']) if task['kind']=='generation' else scored(task['judge'],task['method'])
 def state(pid):
  path=Path(f'/proc/{pid}/status')
  return next((line.split(':',1)[1].strip() for line in path.read_text().splitlines() if line.startswith('State:')),'missing') if path.exists() else 'exited'
 def free_memory():
  result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True)
  return {int(line.split(',')[0]):int(line.split(',')[1]) for line in result.stdout.splitlines()}
 def start(kind,method,gpu,judge=None):
  key=f'gen_{method}' if kind=='generation' else f'{judge}_{method}'
  attempts[key]=attempts.get(key,0)+1
  logfile=logs/f'{key}.parallel{attempts[key]}.{int(time.time())}.log';handle=logfile.open('a')
  script='generate_direct.py' if method=='our_direct' else 'generate_ma2p_recovery.py' if method=='ma2p_repaired' else 'generate.py'
  output=work/'generators'/method
  args=['--method',method,'--limit','192']
  if kind=='evaluation':script='evaluate.py';output=work;args=['--judge',judge,'--method',method,'--limit','192']
  command=[sys.executable,'-u',str(code/script),'--repo-root',str(root),'--output',str(output),'--physical-gpu',str(gpu),*args]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
  process=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
  task={'key':key,'kind':kind,'method':method,'judge':judge,'gpu':gpu,'pid':process.pid,'started':time.time(),'log':str(logfile),'command':command}
  active[key]=dict(task,process=process,handle=handle)
  atomic_json(work/'workers'/f'{key}.json',task)
  print('launch',key,'GPU',gpu,'PID',process.pid,flush=True)
 with writer_lock(work/'.supervisor.lock'):
  adoption=work/'adopt_workers.json'
  if adoption.exists():
   for task in json.loads(adoption.read_text()):
    if state(task['pid']) not in ['exited','missing']:
     active[task['key']]=dict(task,process=None,handle=None);attempts[task['key']]=1
     print('adopt',task['key'],task['pid'],flush=True)
  while True:
   for key,task in list(active.items()):
    process=task['process'];st=state(task['pid']);rc=process.poll() if process is not None else (None if st not in ['exited','missing'] and not st.startswith('Z') else 0)
    if rc is None:continue
    del active[key]
    if task['handle'] is not None:task['handle'].close()
    if complete(task):print('done',key,flush=True)
    elif attempts.get(key,1)>=3:failed[key]={'exit_code':rc,'log':task['log']};print('failed',key,rc,flush=True)
    else:print('retry queued',key,rc,flush=True)
   free=free_memory();slots={gpu:sum(task['gpu']==gpu for task in active.values()) for gpu in [0,1,4]}
   for method in GENERATE:
    key='gen_'+method
    if generated(method) or key in active or key in failed:continue
    frozen=work/'generators'/method/'generation_run.json'
    fixed=json.loads(frozen.read_text())['physical_gpu'] if frozen.exists() and method!='ma2p_repaired' else None
    generation_gpus=[1,0,4] if method=='ma2p_repaired' else [1,4]
    choices=[gpu for gpu in generation_gpus if slots[gpu]<2 and free[gpu]>=37*1024 and (fixed is None or fixed==gpu)]
    if not choices:continue
    gpu=choices[0];start('generation',method,gpu);slots[gpu]+=1;free[gpu]-=38*1024
   for judge,gpu,min_free in [('selene8b',0,20*1024),('qwen3_14b',4,36*1024)]:
    if any(task['judge']==judge for task in active.values()):continue
    if slots[gpu]>=2 or free[gpu]<min_free:continue
    order=['dialogue_sft','persugpt','latentratt','zero_shot_repaired','ma2p_repaired','our_analysis','our_direct']
    todo=[method for method in order if generated(method) and not scored(judge,method) and f'{judge}_{method}' not in failed]
    if todo:start('evaluation',todo[0],gpu,judge);slots[gpu]+=1
   progress={}
   for method in LABELS:
    path=work/'generation'/f'{method}.jsonl';n=sum(1 for line in path.open() if line.strip()) if path.exists() else 0
    progress[method]={'generated':n,'target':192,'source':'historical' if method in HISTORICAL else 'new'}
    for judge in ['selene8b','qwen3_14b']:
     path=work/'judges'/judge/'evaluation'/method/'judgments.jsonl'
     progress[method][judge+'_scored']=sum(1 for line in path.open() if line.strip()) if path.exists() else 0
   workers={key:{k:v for k,v in task.items() if k not in ['process','handle']}|{'state':state(task['pid'])} for key,task in active.items()}
   all_done=all(scored(judge,method) for judge in ['selene8b','qwen3_14b'] for method in LABELS)
   atomic_json(work/'live_status.json',{'updated_unix':time.time(),'supervisor_pid':os.getpid(),'status':'complete' if all_done else 'running' if active else 'blocked','progress':progress,'workers':workers,'failed':failed,'free_memory_mib':free})
   render(root,work)
   if all_done or (failed and not active):break
   time.sleep(15)
if __name__=='__main__':main()
