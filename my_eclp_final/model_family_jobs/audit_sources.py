import sys,json,re
from pathlib import Path
ROOT=Path('/app/hzj/liujianjian/ljj-2-paper');sys.path[:0]=[str(ROOT/'my_eclp_final'),str(ROOT/'my_eclp_final/seven_methods_192'),str(ROOT/'exp_code_ljj')]
from protocol import speech_boundary,narration
from experiment_io import digest,atomic_json,read_jsonl
selection=json.loads((ROOT/'exp_code_ljj/data/test_selection_192.json').read_text())['original_indexes'];points=read_jsonl(ROOT/'exp_code_ljj/data/evaluation_points.jsonl');results={}
for family in ['meta_llama31','deepseek_r1_qwen14b']:
 results[family]={}
 for path in sorted((ROOT/'exp_code_ljj/artifacts/generation_from_start192'/family).glob('*.jsonl')):
  rows=read_jsonl(path);issues=[];indexes=[];models=set()
  for row in rows:
   i=row['index'];p=points[i];indexes.append(i);models.add(row['persuader_model'])
   assert i in selection and row['scenario']==p['scenario'] and row['scenario_id']==p['scenario_id'] and row['reference']==p['reference'] and row['initial_history']==[]
   assert len(row['trajectory'])==10
   for j,t in enumerate(row['trajectory']):
    assert t['role']==('persuader' if j%2==0 else 'persuadee')
    try:
     b=speech_boundary(t['content'],p,t['role'])
     if b['content']!=t['content'] or b['future_dialogue_truncated'] or b['thinking_filtered'] or (t['role']=='persuader' and narration(t['content'],p)):issues.append({'index':i,'turn':j+1,'role':t['role'],'reason':'public speech boundary'})
    except ValueError as e:issues.append({'index':i,'turn':j+1,'role':t['role'],'reason':str(e)})
  assert len(set(indexes))==len(indexes)
  results[family][path.stem]={'rows':len(rows),'complete':set(indexes)==set(selection),'sha256':digest(path),'models':sorted(models),'boundary_issues':issues,'affected_indexes':sorted({v['index'] for v in issues}),'policy':'regenerate complete affected scenarios; never trim old history and call it new generation'}
atomic_json(ROOT/'my_eclp_final/model_family_jobs/source_audit.json',results)
print(json.dumps({f:{m:{'n':v['rows'],'affected_scenarios':len(v['affected_indexes']),'models':v['models']} for m,v in methods.items()} for f,methods in results.items()},ensure_ascii=False,indent=2))
