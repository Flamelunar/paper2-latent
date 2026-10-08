from pathlib import Path
import os,signal,time,subprocess,json
R=Path('/app/hzj/liujianjian/ljj-2-paper');J=R/'my_eclp_final/model_family_jobs';W=R/'eclp_runs_deepseek_r1_qwen14b_20261007';PY='/app/hzj/anaconda3/envs/ljj/bin/python'
def live(pid):
 p=Path('/proc')/str(pid)
 return p.exists() and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
for pid,match in [(2911832,b'train_deepseek_our.py'),(2911869,b'torch.distributed.run')]:
 p=Path('/proc')/str(pid)
 if live(pid):
  assert match in (p/'cmdline').read_bytes()
  os.kill(pid,signal.SIGTERM)
  for _ in range(100):
   if not live(pid):break
   time.sleep(.1)
  else:raise RuntimeError(f'Cannot stop failed rendezvous owner {pid}')
# A failed rendezvous must not leave any replica worker running.
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:c=(p/'cmdline').read_bytes();state=(p/'stat').read_text().rsplit(')',1)[1].split()[0]
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if b'train_eclp_sft_distributed.py' in c and state!='Z':raise RuntimeError(f'Unexpected live replica {p.name}')
(J/'train_deepseek_our.py').write_bytes((J/'train_deepseek_our_distributed.py').read_bytes())
env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
with (W/'distributed_supervisor.log').open('a') as log:
 p=subprocess.Popen([PY,'-u',str(J/'train_deepseek_our.py')],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('Loopback rendezvous supervisor',p.pid)
(W/'distributed_rendezvous_repair.json').write_text(json.dumps({'time':time.time(),'supervisor_pid':p.pid,'rendezvous':'static; 127.0.0.1; dynamic free port','reason':'standalone rendezvous advertised an unreachable pd182 address','checkpoint_preserved':True},indent=2)+'\n')
