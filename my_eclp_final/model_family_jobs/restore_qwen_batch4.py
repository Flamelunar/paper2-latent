from pathlib import Path
import subprocess,json,time
root=Path('/app/hzj/liujianjian/ljj-2-paper');work=root/'exp_code_ljj/artifacts/deepseek_goal_strategy_192_20261008/deepseek_r1_qwen14b'
state=json.loads((work/'task_status.json').read_text());assert state['stage']=='failed' and state['pid']==653910
p=root/'my_eclp_final/model_family_jobs/deepseek_goal_strategy_campaign.py'
backup=p.with_name(p.name+'.before_judge_batch8_20261008')
p.write_text(backup.read_text())
with (work/'controller.log').open('ab') as log:
    proc=subprocess.Popen(['/app/hzj/anaconda3/envs/ljj/bin/python','-u',str(p)],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
audit={'time':time.time(),'new_controller_pid':proc.pid,'batch':4,'reason':'batch8 not supported by existing evaluation CLI; restored original without altering evaluation source or cache contract'}
(work/'judge_batch4_restoration_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit))
