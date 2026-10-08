from pathlib import Path
import os,json,subprocess,time
r=Path('/app/hzj/liujianjian/ljj-2-paper');j=r/'my_eclp_final/model_family_jobs';w=r/'exp_code_ljj/artifacts/deepseek_goal_strategy_192_20261008/deepseek_r1_qwen14b'
p=j/'train_with_budget.py';s=p.read_text();old="a.physical_gpu not in {1,4}:raise ValueError('Training is supported only on physical GPU1 or GPU4')"
assert old in s
backup=p.with_name('train_with_budget.py.before_gpu0_latent_20261008')
assert not backup.exists()
backup.write_text(s)
s=s.replace(old,"(a.physical_gpu not in {0,1,4} or (a.physical_gpu==0 and a.entry!='latent')):raise ValueError('Training requires authorized GPU1/4, or GPU0 for the LatentRATT queue')")
p.write_text(s)
p=j/'train_deepseek_latentratt.py';s=p.read_text()
s=s.replace('stage="resuming_latentratt_sft_on_gpu4"','stage=f"resuming_latentratt_sft_on_gpu{GPU}"').replace('stage="training_latentratt_rl_on_gpu4"','stage=f"training_latentratt_rl_on_gpu{GPU}"')
p.write_text(s)
old_pid=2492916;entry=Path('/proc')/str(old_pid)
assert not entry.exists() or (entry/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
env=dict(os.environ,DEEPSEEK_TRAIN_GPU='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
with (w/'latentratt_training_controller_gpu0.log').open('a') as log:
 proc=subprocess.Popen(['/app/hzj/anaconda3/envs/ljj/bin/python','-u',str(j/'train_deepseek_latentratt.py')],cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('Restarted GPU0 training controller',proc.pid)
(w/'latentratt_gpu0_guard_fix.json').write_text(json.dumps({'time':time.time(),'new_controller_pid':proc.pid,'allowed_gpu0_entry':'latent','training_settings_changed':False},indent=2)+'\n')
