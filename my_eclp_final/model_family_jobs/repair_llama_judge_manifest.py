from pathlib import Path
import json,sys,subprocess,os,time,shutil
R=Path('/app/hzj/liujianjian/ljj-2-paper');J=R/'my_eclp_final/model_family_jobs';W=R/'exp_code_ljj/artifacts/model_family_fresh192_nogoal_20261007/meta_llama31';O=W/'generators/our_direct';PY='/app/hzj/anaconda3/envs/ljj/bin/python'
sys.path.insert(0,str(R/'my_eclp_final'))
from experiment_io import object_hash,atomic_json,digest
recovery=json.loads((O/'public_retry_recovery.json').read_text())
original=O/'recovery_before_public_retry_20261008/campaign_generation_run.json'
old=json.loads(original.read_text());current=json.loads((W/'generation_run.json').read_text())
assert old['generation_script_sha256']==recovery['previous_source_sha256']
assert current['generation_script_sha256']==recovery['runtime_source_sha256']
assert {k:v for k,v in old.items() if k!='generation_script_sha256'}=={k:v for k,v in current.items() if k!='generation_script_sha256'}
assert digest(J/'generate_family.py')==recovery['runtime_source_sha256']
for judge in ['selene8b','qwen3_14b']:
 contract=json.loads((W/'judges'/judge/'evaluation_run.json').read_text())
 assert contract['generation_run_hash']==object_hash(old)
# Keep the initial campaign identity immutable; per-method source/row hashes carry recovery provenance.
shutil.copy2(W/'generation_run.json',O/'recovery_before_public_retry_20261008/campaign_generation_run.post_recovery.json')
atomic_json(W/'generation_run.json',old)
p=J/'llama_fresh_campaign.py';s=p.read_text()
needle=" manifest['generation_script_sha256']=__import__('hashlib').sha256((J/'generate_family.py').read_bytes()).hexdigest()\n"
assert needle in s
shutil.copy2(p,p.with_name('llama_fresh_campaign.py.before_recovery_manifest_20261008'))
s=s.replace(needle,needle+" recovery=W/'generators/our_direct/public_retry_recovery.json'\n if recovery.exists():\n  record=json.loads(recovery.read_text())\n  assert manifest['generation_script_sha256']==record['runtime_source_sha256']\n  manifest['generation_script_sha256']=record['previous_source_sha256']\n")
p.write_text(s)
summary=json.loads((O/'generation/our_direct_summary.json').read_text());assert summary['complete'] and summary['n']==192
assert digest(O/'generation/our_direct.jsonl')==summary['generation_sha256']
with (W/'recovery_controller.log').open('a') as log:
 proc=subprocess.Popen([PY,'-u',str(p)],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
atomic_json(O/'judge_manifest_recovery.json',{'time':time.time(),'controller_pid':proc.pid,'campaign_hash_restored':object_hash(old),'generation_sha256':summary['generation_sha256'],'existing_baseline_scores_preserved':True,'judge_code_and_prompts_changed':False,'per_method_recovery_provenance':str(O/'public_retry_recovery.json')})
print('Restored immutable campaign identity; continued Our judging',proc.pid)
