from pathlib import Path
import os,signal,subprocess,time,json
R=Path('/app/hzj/liujianjian/ljj-2-paper');J=R/'my_eclp_final/model_family_jobs';PY='/app/hzj/anaconda3/envs/ljj/bin/python'
pid=2911885;p=Path('/proc')/str(pid)
assert b'llama_latentratt_accelerated.py' in (p/'cmdline').read_bytes()
status=json.loads((R/'exp_code_ljj/artifacts/model_family_fresh192_nogoal_20261007/meta_llama31/latentratt_queue/status.json').read_text())
assert status['stage']=='waiting_for_our_dual_judges_before_training_only_lr_repair',status
free=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=28*1024,free
os.kill(pid,signal.SIGTERM)
for _ in range(100):
 if not p.exists() or (p/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
 time.sleep(.1)
else:raise RuntimeError('Original idle waiter did not stop')
(J/'llama_latentratt_accelerated.py').write_bytes((J/'llama_latentratt_lr1e7_gpu0.py').read_bytes())
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
with (R/'exp_code_ljj/artifacts/llama_latent_lr1e7_supervisor.log').open('a') as log:
 proc=subprocess.Popen([PY,'-u',str(J/'llama_latentratt_accelerated.py')],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v={'time':time.time(),'pid':proc.pid,'training_gpu':0,'generation_gpu':4,'learning_rate':1e-7,'training_rows':59868,'rl_epochs':1,'diagnostic_basis':'matched TRAIN NLL degradation at 1e-6; no test scores used','training_budget_unchanged':True}
(R/'exp_code_ljj/artifacts/llama_latent_parallel_gpu0.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(v,indent=2))
