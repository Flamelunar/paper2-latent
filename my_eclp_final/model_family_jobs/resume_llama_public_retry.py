import os,json,hashlib,shutil,sys,subprocess,time
from pathlib import Path
R=Path('/app/hzj/liujianjian/ljj-2-paper');J=R/'my_eclp_final/model_family_jobs';W=R/'exp_code_ljj/artifacts/model_family_fresh192_nogoal_20261007/meta_llama31';O=W/'generators/our_direct';PY='/app/hzj/anaconda3/envs/ljj/bin/python'
sys.path.insert(0,str(R/'my_eclp_final'))
from experiment_io import object_hash,atomic_json,writer_lock
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  c=(p/'cmdline').read_bytes();state=(p/'stat').read_text().rsplit(')',1)[1].split()[0]
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if state!='Z' and (b'llama_fresh_campaign.py' in c or (b'generate_family.py' in c and b'meta_llama31' in c)):
  raise RuntimeError(f'Existing campaign/generator must finish first: {p.name}')
with writer_lock(O/'.generation.lock'):
 source=J/'generate_family.py';old_sha=hashlib.sha256(source.read_bytes()).hexdigest();new_source=J/'generate_family_public_retry.py';new_sha=hashlib.sha256(new_source.read_bytes()).hexdigest()
 backup=O/'recovery_before_public_retry_20261008';assert not backup.exists();backup.mkdir()
 for p in [source,O/'generation_run.json',O/'generation/our_direct.jsonl',W/'generation_run.json']:shutil.copy2(p,backup/('campaign_generation_run.json' if p==W/'generation_run.json' else p.name))
 contract=json.loads((O/'generation_run.json').read_text());assert contract['source_sha256']==old_sha
 old_hash=object_hash(contract);rows=[json.loads(x) for x in (O/'generation/our_direct.jsonl').open() if x.strip()]
 assert len(rows)==73 and all(v['run_hash']==old_hash and len(v['trajectory'])==10 and v['initial_history']==[] for v in rows)
 contract['source_sha256']=new_sha
 contract['our_output_retry_policy']='first3_original; up_to12_total; after3_public_speech_format_repair_only; first_valid_output; no_judge_selection'
 new_hash=object_hash(contract)
 for v in rows:
  v['pre_recovery_row_hash']=object_hash(v);v['pre_recovery_run_hash']=old_hash;v['run_hash']=new_hash
 source.write_bytes(new_source.read_bytes())
 atomic_json(O/'generation_run.json',contract)
 tmp=O/'generation/our_direct.jsonl.public-retry-tmp';tmp.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in rows));tmp.replace(O/'generation/our_direct.jsonl')
 manifest=json.loads((W/'generation_run.json').read_text());assert manifest['generation_script_sha256']==old_sha
 manifest['generation_script_sha256']=new_sha;atomic_json(W/'generation_run.json',manifest)
 atomic_json(O/'public_retry_recovery.json',{'time':time.time(),'preserved_rows':len(rows),'previous_source_sha256':old_sha,'runtime_source_sha256':new_sha,'previous_run_hash':old_hash,'runtime_run_hash':new_hash,'change':'first three attempts unchanged; subsequent format-only repair; no evaluator used; strict public boundary unchanged','backup':str(backup)})
with (W/'recovery_controller.log').open('a') as log:
 p=subprocess.Popen([PY,'-u',str(J/'llama_fresh_campaign.py')],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('Resumed Llama Our generation controller',p.pid)
