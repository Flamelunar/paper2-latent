import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/app/hzj/liujianjian/ljj-2-paper');JOB=ROOT/'my_eclp_final/model_family_jobs';WORK=ROOT/'exp_code_ljj/artifacts/deepseek_goal_strategy_192_20261008/deepseek_r1_qwen14b';PY='/app/hzj/anaconda3/envs/ljj/bin/python'
pid=593252
entry=Path('/proc')/str(pid)
assert b'train_deepseek_latentratt.py' in (entry/'cmdline').read_bytes()
children=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  s=(p/'stat').read_text().rsplit(')',1)[1].split()
  if int(s[1])==pid and s[0]!='Z':children.append(p.name)
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
assert not children,children
status=json.loads((WORK/'latentratt_queue/status.json').read_text())
assert status['stage'].startswith('waiting_for_gpu'),status
free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=62*1024,free
os.kill(pid,signal.SIGTERM)
for attempt in range(50):
 if not entry.exists() or (entry/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
 time.sleep(.1)
else:raise RuntimeError('Old queue controller did not stop')
env=dict(os.environ,DEEPSEEK_TRAIN_GPU='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
with (WORK/'latentratt_training_controller_gpu0.log').open('a') as log:
 proc=subprocess.Popen([PY,'-u',str(JOB/'train_deepseek_latentratt.py')],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
audit={'time':time.time(),'old_idle_queue_pid':pid,'new_controller_pid':proc.pid,'physical_gpu':0,'training_budget_unchanged':True,'reason':'GPU0 freed after completed MA2P dual evaluation'}
(WORK/'latentratt_gpu0_acceleration.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
