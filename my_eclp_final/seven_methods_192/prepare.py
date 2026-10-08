"""Freeze the seven-method contract and extract two archived generation subsets."""
import argparse,hashlib,json,sys
from pathlib import Path
METHODS=['zero_shot_repaired','dialogue_sft','persugpt','ma2p_repaired','latentratt','our_analysis','our_direct']
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 sys.path[:0]=[str(a.repo_root/'exp_code_ljj'),str(a.repo_root/'my_eclp_final')]
 from experiment_io import read_json,read_jsonl,digest,immutable_json,atomic_json
 from matched_eval_protocol_v2 import validate_trajectory
 from protocol import speech_boundary,narration
 selection=read_json(a.repo_root/'exp_code_ljj/data/test_selection_192.json');points=read_jsonl(a.repo_root/'exp_code_ljj/data/evaluation_points.jsonl')
 assert digest(a.repo_root/'exp_code_ljj/data/evaluation_points.jsonl')==selection['source_sha256']
 assert selection['original_indexes']==[i for i,p in enumerate(points) if p['target_turn']==1 and p['history']==[]]
 order=sorted(selection['original_indexes'],key=lambda i:hashlib.sha256(('matched-v2-pilot-42:'+points[i]['scenario_id']).encode()).hexdigest())
 provenance={};audit=[]
 (a.output/'generation').mkdir(parents=True,exist_ok=True)
 for label,old in [('dialogue_sft','baseline2_dialogue_sft'),('persugpt','baseline3_isr_simdpo')]:
  path=a.repo_root/'exp_code_ljj/artifacts/generation_5rounds'/f'{old}.jsonl'
  rows=read_jsonl(path);by={r['index']:r for r in rows};assert len(by)==len(rows)
  extracted=[]
  for i in order:
   row=by[i];point=points[i]
   assert row['scenario_id']==point['scenario_id'] and row['scenario']==point['scenario'] and row['initial_history']==[]
   assert row['reference']==point['reference'];validate_trajectory(row['trajectory'])
   for t in row['trajectory']:
    boundary=speech_boundary(t['content'],point,t['role'])
    if boundary['future_dialogue_truncated'] or boundary['thinking_filtered'] or narration(t['content'],point):
     audit.append({'method':label,'index':i,'round':t.get('round'),'role':t['role'],'issue':'historical_public_output_boundary','policy':'preserved verbatim; not silently repaired'})
   extracted.append(row)
  target=a.output/'generation'/f'{label}.jsonl'
  content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in extracted)
  if target.exists():assert target.read_text()==content
  else:target.write_text(content)
  provenance[label]={'source':str(path),'source_sha256':digest(path),'extracted_sha256':digest(target),'n':len(extracted),'generation':'historical verbatim','recorded_persuadee_models':sorted({str(r.get('persuadee_model')) for r in extracted})}
 for label in METHODS:
  if label in provenance:continue
  path=a.output/'generation'/f'{label}.jsonl';source=a.output/'generators'/label/'generation'/f'{label}.jsonl'
  source.parent.mkdir(parents=True,exist_ok=True)
  if not path.exists() and not path.is_symlink():path.symlink_to(source)
 immutable_json(a.output/'generation_run.json',{'revision':'seven_methods_192_target_grounded_20261006','expected_indexes':order,'methods':METHODS,'selection_sha256':digest(a.repo_root/'exp_code_ljj/data/test_selection_192.json'),'historical_provenance':provenance,'old_scores_reused':False})
 atomic_json(a.output/'historical_boundary_audit.json',{'issues':audit,'n':len(audit),'policy':'historical generation unchanged; report limitations'})
 print(json.dumps({'n':len(order),'historical_methods':provenance,'historical_boundary_issues':len(audit)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
