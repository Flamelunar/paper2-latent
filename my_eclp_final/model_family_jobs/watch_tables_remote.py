import subprocess,time,json
from pathlib import Path
R=Path('/app/hzj/liujianjian/ljj-2-paper');J=R/'my_eclp_final/model_family_jobs';P='/app/hzj/anaconda3/envs/ljj/bin/python'
failures=0
while True:
 result=subprocess.run([P,str(J/'report_family.py'),'--family','meta_llama31'],capture_output=True,text=True)
 print(time.time(),'table_refresh',result.returncode,result.stderr[-1500:],flush=True)
 if result.returncode:
  failures+=1
  if failures>=10:raise RuntimeError('Table refresh failed ten consecutive times')
  time.sleep(15)
  continue
 failures=0
 audit_path=R/'exp_code_ljj/artifacts/model_family_fresh192_nogoal_20261007/table_completion_audit.json'
 if audit_path.exists():
  audit=json.loads(audit_path.read_text())
  family=audit.get('families',{}).get('meta_llama31',{})
  required=[f'{judge}/{method}' for judge in ['selene8b','qwen3_14b']
            for method in ['zero_shot_repaired','dialogue_sft','persugpt','ma2p_repaired','our_direct']]
  if required and all(family.get(key,{}).get('complete') and family[key].get('n')==192 for key in required):break
 time.sleep(60)
