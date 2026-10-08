"""Apply the user-approved historical LatentRATT reuse without rewriting frozen contracts."""
import argparse,json,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 sys.path[:0]=[str(a.repo_root/'my_eclp_final'),str(a.repo_root/'exp_code_ljj')]
 from experiment_io import read_json,read_jsonl,digest,atomic_json,immutable_json
 from matched_eval_protocol_v2 import validate_trajectory
 from protocol import speech_boundary,narration
 c=read_json(a.repo_root/'eclp_runs_gpu4/evaluation_v2/selene8b/baseline5_latentratt/run.json');source=Path(c['generation_source']);assert digest(source)==c['generation_sha256']
 rows=read_jsonl(source);by={r['index']:r for r in rows};assert len(rows)==len(by)
 frozen=read_json(a.output/'generation_run.json');points=read_jsonl(a.repo_root/'exp_code_ljj/data/evaluation_points.jsonl')
 extracted=[];issues=[]
 for i in frozen['expected_indexes']:
  row=by[i];point=points[i]
  assert row['index']==i and row['scenario_id']==point['scenario_id'] and row['scenario']==point['scenario'] and row['initial_history']==[] and row['reference']==point['reference']
  validate_trajectory(row['trajectory'])
  for t in row['trajectory']:
   result=speech_boundary(t['content'],point,t['role'])
   if result['future_dialogue_truncated'] or result['thinking_filtered'] or narration(t['content'],point):issues.append({'index':i,'round':t.get('round'),'role':t['role'],'issue':'historical_public_output_boundary'})
  extracted.append(row)
 assert len(extracted)==192
 target=a.output/'generators/latentratt/generation/latentratt.jsonl';target.parent.mkdir(parents=True,exist_ok=True)
 content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in extracted)
 if target.exists():assert target.read_text()==content
 else:target.write_text(content)
 immutable_json(a.output/'historical_latentratt_provenance.json',{'method':'latentratt','source':str(source),'source_sha256':digest(source),'extracted_sha256':digest(target),'n':192,'generation':'historical verbatim','decision':'User requested regeneration only for Zero-shot, MA2P and two Our variants','frozen_generation_contract_sha256':digest(a.output/'generation_run.json'),'recorded_persuadee_models':sorted({str(r.get('persuadee_model')) for r in extracted}),'boundary_issues':issues,'boundary_issue_count':len(issues)})
 print(json.dumps({'imported':192,'boundary_issue_count':len(issues),'source':str(source)}))
if __name__=='__main__':main()
