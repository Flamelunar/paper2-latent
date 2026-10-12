"""Verify ablation artifacts against fixed IDs, exact prompts and real judgments."""
import csv,json,hashlib,statistics,sys,time
from pathlib import Path
R=Path('/app/hzj/liujianjian/ljj-2-paper');D=R/'my_eclp_final/ablation_qwen192';A=R/'exp_code_ljj/artifacts/qwen25_ablation_192_20261010'
sys.path[:0]=[str(D),str(R/'my_eclp_final'),str(R/'exp_code_ljj'),str(R/'my_eclp_final/our_ma2p_deps')]
from experiment_io import object_hash,digest,atomic_json
from protocol import goal_prompt
from matched_eval_protocol_v2 import CONFIG

def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def inspect():
 old=R/'my_eclp_final/seven_methods_192';assert digest(D/'protocol.py')==digest(old/'protocol.py');assert digest(D/'win_rules.json')==digest(old/'win_rules.json')
 assert digest(old/'generate_direct.py')=='0c513888e5f28dda939bc78b12ff92deaa6352f4ffce58323d329f6dadf2223c'
 order=load(R/'exp_code_ljj/artifacts/seven_methods_192_20261006/generation_run.json')['expected_indexes'];selected=set(order)
 assert selected==set(load(R/'exp_code_ljj/data/test_selection_192.json')['original_indexes']) and len(order)==192
 gold={r['index']:r for r in rows(R/'eclp_runs_gpu4/evaluation_v2/golden/golden.jsonl')}
 cp=load(A/'checkpoint_provenance.json');rules=load(D/'win_rules.json');diagnostic=load(A/'wo_r_kl_only/diagnostic.json')
 assert diagnostic['contexts']==1800 and diagnostic['optimizer_updates']==226 and diagnostic['equivalent_to_wo_g']
 assert diagnostic['max_kl']==diagnostic['max_gradient_norm']==diagnostic['max_parameter_change']==0
 assert diagnostic['policy_initial_hash']==diagnostic['policy_final_hash']==diagnostic['reference_hash']
 updates=rows(A/'wo_r_kl_only/updates.jsonl');assert len(updates)==226 and all(v['total_loss']==v['policy_gradient_norm']==v['parameter_max_difference']==0 and not v['task_reward_enabled'] for v in updates)
 result={'time':time.time(),'protocols_verified':True,'fixed192_verified':True,'kl_only_equivalence_verified':True,'generation':{},'judges':{},'complete':False}
 for variant in ['wo_g','full_ours']:
  out=A/variant;manifest=out/'generation_run.json'
  if not manifest.exists():result['generation'][variant]={'n':0,'pending':True};continue
  m=load(manifest);assert m['expected_indexes']==order and m['checkpoint_id']==cp[variant]['checkpoint_id'] and m['seed']==42 and m['physical_gpu']==4 and m['compute_dtype']=='bfloat16' and m['quantization']=='none'
  assert m['source_sha256']==digest(D/'generate.py') and m['batching_sha256']==digest(D/'batching.py')
  assert m['eclp_source_sha256']==digest(R/'my_eclp_final/eclp_elpo.py')
  assert m['points_sha256']==digest(R/'exp_code_ljj/data/evaluation_points.jsonl')
  assert m['selection_sha256']==digest(R/'exp_code_ljj/data/test_selection_192.json')
  assert cp['wo_g']['executor_hash']==cp['full_ours']['executor_hash']
  mechanism=[json.loads(line.split('MECHANISM ',1)[1]) for line in (out/'generation.log').read_text().splitlines() if line.startswith('MECHANISM ')]
  assert mechanism and all(v['checkpoint_policy_hash']==cp[variant]['policy_hash'] and v['action_mode']=='mean' for v in mechanism)
  architecture=cp[variant]['architecture'];hidden=load(Path(m['models']['base']['path'])/'config.json')['hidden_size']
  assert all(v['action_shape']==[1,architecture['action_dim']] and v['state_shape']==[1,architecture['state_slots'],architecture['width']] and v['strategy_token_shape']==[1,architecture['strategy_slots'],hidden] for v in mechanism)
  assert m['simulator_input']=='background and public history only; goal omitted per ablation instruction'
  assert m['decode']=={'temperature':.95,'top_p':.7,'top_k':50,'max_tokens':512}
  source=rows(out/'generation/our_direct.jsonl');byid={v['index']:v for v in source};assert len(byid)==len(source) and set(byid)<=selected
  for i,v in byid.items():
   assert v['initial_history']==gold[i]['initial_history']==[] and v['scenario']==gold[i]['scenario'] and v['reference']==gold[i]['golden_first']
   assert v['checkpoint_id']==m['checkpoint_id'] and v['run_hash']==object_hash(m)
   assert len(v['trajectory'])==10 or v.get('generation_failed')
   assert all(t['role']==('persuader' if j%2==0 else 'persuadee') and t['content'].strip() for j,t in enumerate(v['trajectory']))
  failures=sum(v.get('generation_failed',False) for v in source);result['generation'][variant]={'n':len(source),'complete_five_rounds_n':len(source)-failures,'failed_n':failures,'hash':digest(out/'generation/our_direct.jsonl') if source else None}
  for judge in ['selene8b','qwen3_14b']:
   work=out/'judges'/judge;records=rows(work/'evaluation/our_direct/judgments.jsonl');idset={v['index'] for v in records};assert len(idset)==len(records) and idset<=set(byid)
   if records:
    contract=load(work/'evaluation_run.json');assert contract['physical_gpu']==4 and contract['protocol_sha256']==digest(D/'protocol.py') and contract['win_rules_sha256']==digest(D/'win_rules.json') and contract['temperature']==0 and contract['compute_dtype']=='bfloat16' and contract['quantization']=='none'
    assert contract['generation_run_hash']==object_hash(m)
    historical=load(R/'exp_code_ljj/artifacts/seven_methods_192_20261006/judges'/judge/'evaluation_run.json')
    for key in ['judge_model','model_files','golden_sha256','protocol_sha256','win_rules_sha256','original_win_protocol_sha256','temperature','max_format_retries','compute_dtype','quantization','batch_size']:assert contract[key]==historical[key],key
    for v in records:
     assert v['contract_hash']==object_hash(contract) and v['source_row_hash']==object_hash(byid[v['index']]) and v['golden_row_hash']==object_hash(gold[v['index']])
     win=v['win1']
     for pos,o in [('A','ab'),('B','ba')]:
      verdict=win[o];expected=None if verdict is None else (.5 if verdict=='Tie' else float(verdict==pos));assert win[o+'_credit']==expected
     if win['credit'] is not None:assert win['credit']==(win['ab_credit']+win['ba_credit'])/2
    # Reconstruct every complete request cache using original goal and Win@1 prompts.
    for cache in (work/'evaluation/our_direct/request_chunks').glob('*.json'):
     if cache.name.endswith('.partial.json'):continue
     c=load(cache);expected=[]
     for request in c['responses']:
      i,key=request['index'],request['key'];v=byid[i]
      if key.startswith('goal_r'):prompt=goal_prompt(v,int(key[6:]))
      else:
       assert key in ['win1_ab','win1_ba'];left,right=v['trajectory'][0]['content'],gold[i]['golden_first']
       if key.endswith('_ba'):left,right=right,left
       rule=rules['win1_rule']+'\nDistinguish engagement or preliminary activities from acceptance of the exact goal. For multi-part goals, assess progress on all required parts. Do not assume the goal was achieved merely because the dialogue ended politely.'
       prompt=f'Background: {v["scenario"]["background"]}\nExact goal: {v["scenario"]["goal"]}\nCommon preceding dialogue: (empty)\nCandidate A:\n{left}\n\nCandidate B:\n{right}\n\n{rule}\n\n{CONFIG["win_contract"]}'
      expected.append(hashlib.sha256(prompt.encode()).hexdigest())
     assert expected==c['prompt_hashes'],str(cache)
   result['judges'][variant+'/'+judge]={'n':len(records),'fixed_ids':bool(records),'row_and_golden_hashes':bool(records),'prompt_hashes':bool(records)}
 all_done=all(result['generation'].get(v,{}).get('n')==192 and all(result['judges'].get(v+'/'+j,{}).get('n')==192 for j in ['selene8b','qwen3_14b']) for v in ['wo_g','full_ours'])
 if all_done:
  for variant,meta in cp.items():
   for name,sha in meta['files'].items():assert digest(Path(meta['path'])/name)==sha
  reuse=load(A/'wo_r_kl_only/reuse_manifest.json');assert reuse['diagnostic_sha256']==digest(A/'wo_r_kl_only/diagnostic.json')
  assert (A/'wo_r_kl_only/generation').resolve()==(A/'wo_g/generation').resolve() and (A/'wo_r_kl_only/judges').resolve()==(A/'wo_g/judges').resolve()
  entries=list(csv.DictReader((R/'exp_code_ljj/results/ablation_qwen25_192.csv').open()));assert len(entries)==6
  for entry in entries:
   variant=entry['variant'];source='wo_g' if variant=='wo_r_kl_only' else variant;judge=entry['judge'];records=rows(A/source/'judges'/judge/'evaluation/our_direct/judgments.jsonl');assert len(records)==192
   first=[next((i for i,g in enumerate(v['goal_results'],1) if g is not None and g['success']),None) for v in records]
   success=100*sum(v is not None for v in first)/192;turn=statistics.fmean(v if v is not None else 5 for v in first);wins=[v['win1']['credit'] for v in records if v['win1']['credit'] is not None]
   assert abs(float(entry['Success'])-success)<1e-9 and abs(float(entry['Turn'])-turn)<1e-9 and abs(float(entry['Win@1'])-100*statistics.fmean(wins))<1e-9
   assert int(entry['generation_failed_scenarios'])==sum(v.get('generation_failed',False) for v in records)
  result['complete']=True;result['all_artifacts_verified']=True;result['generation_failure_counts_disclosed']=True
 atomic_json(A/'completion_audit.json',result)
 return result
if __name__=='__main__':print(json.dumps(inspect(),ensure_ascii=False,indent=2))
