"""Verified project-only migration; preserve checkpoints/logs and all other jobs."""
import os,signal,time,json,subprocess,shutil,hashlib
from pathlib import Path
R=Path('/app/hzj/liujianjian/ljj-2-paper');J=R/'my_eclp_final/model_family_jobs';WORK=R/'exp_code_ljj/artifacts/deepseek_goal_strategy_192_20261008/deepseek_r1_qwen14b';PY='/app/hzj/anaconda3/envs/ljj/bin/python'
def command(pid):return (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ')
def live(pid):
 p=Path('/proc')/str(pid)
 return p.exists() and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
def stop(pid,expected):
 assert all(t in command(pid) for t in expected),(pid,command(pid))
 os.kill(pid,signal.SIGTERM)
 for _ in range(100):
  if not live(pid):return
  time.sleep(.1)
 raise RuntimeError(f'PID {pid} did not terminate; do not start duplicate jobs')
def spawn(script,env,log):
 with log.open('a') as stream:
  p=subprocess.Popen([PY,'-u',str(script)],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 return p.pid
# Inspect the concrete saved TRAIN-only diagnostic; no test metrics are used.
probe=json.loads((R/'exp_code_ljj/artifacts/llama_latent_lr1e6_matched_probe800_20261008/result.json').read_text())
assert probe['training_only'] and probe['complete']
sft=probe['checkpoints']['sft']['mean_nll'];rl=probe['checkpoints']['rl_fp32_accum16_lr1e6_20261008/checkpoint-800']['mean_nll']
assert rl>4*sft
# Stop only the failed Llama baseline controller and its verified RL worker.
stop(590444,[b'llama_latentratt_accelerated.py'])
stop(590492,[b'train_with_budget.py',b'--physical-gpu 4',b'rl_fp32_accum16_lr1e6_20261008'])
# Move the DeepSeek baseline from GPU0 to GPU4, using its committed checkpoint.
stop(2505468,[b'train_deepseek_latentratt.py'])
stop(2505576,[b'train_with_budget.py',b'--physical-gpu 0',b'DeepSeek-R1-Distill-Qwen-14B'])
# Find the actual single-GPU Our supervisor by walking the verified worker's ancestors.
worker=4005823;assert b'--entry eclp' in command(worker) and b'--physical-gpu 1' in command(worker)
parents=[];pid=worker
for _ in range(8):
 p=Path('/proc')/str(pid);parts=(p/'stat').read_text().rsplit(')',1)[1].split();pid=int(parts[1])
 if pid<=1:break
 c=command(pid)
 if b'train_deepseek_our.py' in c or (b'supervise_campaign.py' in c and b'--train-only deepseek_r1_qwen14b' in c):parents.append((pid,c))
assert parents, 'Cannot establish original Our training ownership'
for pid,c in reversed(parents):stop(pid,[c.strip()])
stop(worker,[b'--entry eclp',b'--physical-gpu 1'])
# Preserve the original supervisor source and keep its filename for adoption by the campaign.
old=J/'train_deepseek_our.py';backup=old.with_name('train_deepseek_our.py.before_distributed_20261008');assert not backup.exists();shutil.copy2(old,backup)
old.write_bytes((J/'train_deepseek_our_distributed.py').read_bytes())
# Admission based on observed 35 GiB baseline peak plus >=8 GiB margin on GPU4.
p=J/'train_deepseek_latentratt.py';s=p.read_text();backup=p.with_name('train_deepseek_latentratt.py.before_gpu4_migration_20261008');assert not backup.exists();shutil.copy2(p,backup)
s=s.replace('wait_for_gpu(62, ', 'wait_for_gpu(44 if GPU == 4 else 62, ');p.write_text(s)
# Route the Llama RL repair after the Our dual judges, with lower LR chosen on TRAIN diagnostics.
p=J/'llama_latentratt_accelerated.py';backup=p.with_name('llama_latentratt_accelerated.py.before_lr1e7_20261008');assert not backup.exists();shutil.copy2(p,backup);p.write_bytes((J/'llama_latentratt_lr1e7_queued.py').read_bytes())
env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',DAILYPERSUASION_PROFILE='paper10k')
# Start the Our supervisor first so the existing campaign adopts it immediately.
our=spawn(old,env,R/'eclp_runs_deepseek_r1_qwen14b_20261007/distributed_supervisor.log')
latent=spawn(J/'train_deepseek_latentratt.py',dict(env,DEEPSEEK_TRAIN_GPU='4'),WORK/'latentratt_training_controller_gpu4.log')
llama=spawn(J/'llama_latentratt_accelerated.py',dict(env,CUDA_VISIBLE_DEVICES='4'),R/'exp_code_ljj/artifacts/llama_latent_lr1e7_supervisor.log')
audit={'time':time.time(),'our_distributed_supervisor_pid':our,'deepseek_latent_gpu4_supervisor_pid':latent,'llama_latent_repair_waiter_pid':llama,
       'our_gpus':[0,1],'deepseek_latent_gpu':4,'global_accumulation':8,'sft_rows':59868,'sft_epochs':1,
       'llama_latent_lr_repair':{'from':1e-6,'to':1e-7,'sft_train_probe_nll':sft,'rl800_train_probe_nll':rl,'test_used':False,'failed_artifacts_preserved':True},
       'other_jobs':'preserved; GPUs2/3 untouched; Llama Our generation remains running',
       'original_our_supervisors_stopped':[pid for pid,_ in parents]}
(WORK/'distributed_gpu014_migration.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
