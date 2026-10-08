from pathlib import Path
import os,signal,subprocess,time,json
root=Path('/app/hzj/liujianjian/ljj-2-paper');work=root/'exp_code_ljj/artifacts/deepseek_goal_strategy_192_20261008/deepseek_r1_qwen14b'
status=json.loads((work/'task_status.json').read_text())
assert status['stage']=='evaluation_qwen3_14b_dialogue_sft'
pids={status['pid']:'deepseek_goal_strategy_campaign.py',status['worker_pid']:'seven_methods_192/evaluate.py'}
for pid,needle in pids.items():
    assert needle in Path(f'/proc/{pid}/cmdline').read_bytes().decode()
for pid in pids:os.kill(pid,signal.SIGTERM)
time.sleep(3)
for pid,needle in pids.items():
    p=Path(f'/proc/{pid}/cmdline')
    if p.exists() and needle in p.read_bytes().decode():os.kill(pid,signal.SIGKILL)
p=root/'my_eclp_final/model_family_jobs/deepseek_goal_strategy_campaign.py';s=p.read_text()
backup=p.with_name(p.name+'.before_judge_batch8_20261008')
if not backup.exists():backup.write_text(s)
s=s.replace("[('qwen3_14b',36,32),('selene8b',24,22)]","[('qwen3_14b',44,40),('selene8b',24,22)]")
assert "'--batch-size','4'" in s
s=s.replace("'--batch-size','4'","'--batch-size',('8' if judge=='qwen3_14b' else '4')")
p.write_text(s)
with (work/'controller.log').open('ab') as log:
    child=subprocess.Popen(['/app/hzj/anaconda3/envs/ljj/bin/python','-u',str(p)],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
audit={'time':time.time(),'verified_stopped_pids':pids,'new_controller_pid':child.pid,'qwen_judge_batch':8,'qwen_memory_cap_gib':40,'unchanged':'generation, model BF16, prompts, greedy judging, indices, metrics; cached requests resumed','selene_batch':4}
(work/'judge_batch8_acceleration_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
