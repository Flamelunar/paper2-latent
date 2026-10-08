import json,hashlib,time,subprocess
from pathlib import Path
root=Path('/app/hzj/liujianjian/ljj-2-paper')
work=root/'exp_code_ljj/artifacts/deepseek_goal_strategy_192_20261008/deepseek_r1_qwen14b'
p=work/'generators/persugpt/generation/persugpt.jsonl'
rows=[json.loads(line) for line in p.read_text().splitlines()]
indices=[r['index'] for r in rows]
selected=json.loads((root/'exp_code_ljj/data/test_selection_192.json').read_text())['original_indexes']
points=[json.loads(line) for line in (root/'exp_code_ljj/data/evaluation_points.jsonl').read_text().splitlines()]
assert len(rows)==192 and len(set(indices))==192 and set(indices)==set(selected)
assert all(points[i]['target_turn']==1 and points[i]['history']==[] for i in indices)
summary=json.loads(p.with_name('persugpt_summary.json').read_text())
sha=hashlib.sha256(p.read_bytes()).hexdigest()
assert summary['complete'] and summary['generation_sha256']==sha
result={'time':time.time(),'method':'persugpt','generation_complete':True,'n':192,'unique_indices':192,'fixed192_selection_matches':True,'original_empty_history_target_turn1':True,'generation_sha256':sha,'judges':{}}
for name in ['selene8b','qwen3_14b']:
    s=work/'judges'/name/'evaluation/persugpt/summary.json'
    if s.exists():
        v=json.loads(s.read_text())
        if v.get('complete'):
            assert v['n']==192 and v['source_generation_sha256']==sha
        result['judges'][name]=v
    else: result['judges'][name]={'complete':False}
(work/'persugpt_fixed192_audit.json').write_text(json.dumps(result,indent=2)+'\n')
subprocess.run(['/app/hzj/anaconda3/envs/ljj/bin/python',str(root/'my_eclp_final/model_family_jobs/report_family.py'),'--family','deepseek_r1_qwen14b'],check=True)
print(json.dumps(result,indent=2))
