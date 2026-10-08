from pathlib import Path
import os,signal,time,json,subprocess
root=Path('/app/hzj/liujianjian/ljj-2-paper')
checks={331872:'llama_latentratt_accelerated.py',332100:'train_with_budget.py'}
for pid,needle in checks.items():
    p=Path(f'/proc/{pid}/cmdline')
    if not p.exists(): raise RuntimeError(f'PID {pid} absent; inspect first')
    cmd=p.read_bytes().decode().replace('\0',' ')
    if needle not in cmd: raise RuntimeError(f'PID mismatch {pid}: {cmd}')
for pid in checks: os.kill(pid,signal.SIGTERM)
time.sleep(3)
for pid in checks:
    if Path(f'/proc/{pid}/cmdline').exists():
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode()
        if checks[pid] in cmd: os.kill(pid,signal.SIGKILL)
p=root/'my_eclp_final/model_family_jobs/llama_latentratt_accelerated.py'
s=p.read_text()
backup=p.with_name(p.name+'.before_train_only_lr_adjustment')
if not backup.exists(): backup.write_text(s)
assert 'rl_fp32_accum16_20261008' in s and '"0.00001"' in s
s=s.replace('rl_fp32_accum16_20261008','rl_fp32_accum16_lr1e6_20261008').replace('"0.00001"','"0.000001"')
p.write_text(s)
audit={'time':time.time(),'reason':'Matched 12 training-only rows: SFT NLL 0.71697, accum16 RL100 NLL 2.49088; full LR remained unstable','stopped_verified_pids':checks,'old_artifacts_preserved':True,'new_learning_rate':1e-6,'new_start':'completed SFT; fresh optimizer','training_budget':'same 59868 rows, 1 epoch, batch2 accumulation16','unchanged':'upstream reward, advantage formula, sigma, group size, generation and judge protocol','test_outcomes_used':False}
(root/'exp_code_ljj/artifacts/llama_latentratt_train_only_lr_adjustment_20261008.json').write_text(json.dumps(audit,indent=2)+'\n')
with (root/'exp_code_ljj/artifacts/model_family_fresh192_nogoal_20261007/meta_llama31/latentratt_queue/accelerated_controller.log').open('ab') as log:
    child=subprocess.Popen(['/app/hzj/anaconda3/envs/ljj/bin/python','-u',str(p)],cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps({'controller_pid':child.pid,**audit}))
