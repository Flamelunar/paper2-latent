import json,os,shutil,sys,time
from pathlib import Path
ROOT=Path('/app/hzj/liujianjian/ljj-2-paper')
EARLY=ROOT/'exp_code_ljj/artifacts/deepseek_our_selene_early_gpu0_20261010/deepseek_r1_qwen14b'
CANON=ROOT/'exp_code_ljj/artifacts/deepseek_two_stage_eval_v3_192_20261009/deepseek_r1_qwen14b'
sys.path.insert(0,str(ROOT/'my_eclp_final'))
from experiment_io import object_hash,digest
worker=json.loads((EARLY/'qwen_worker.json').read_text())['pid']
while Path(f'/proc/{worker}').exists():
    status=Path(f'/proc/{worker}/status').read_text()
    if 'State:	Z' in status:break
    time.sleep(30)
p=EARLY/'judges/qwen3_14b/evaluation/our_direct/summary.json'
if not p.exists():raise RuntimeError('Early evaluator ended without scores; see selene_early.log')
summary=json.loads(p.read_text())
records=[json.loads(x) for x in p.with_name('judgments.jsonl').read_text().splitlines() if x.strip()]
assert len(records)==summary['n']
source=ROOT/'exp_code_ljj/artifacts/deepseek_goal_strategy_192_20261008/deepseek_r1_qwen14b/generators/our_direct/generation/our_direct.jsonl'
rows={r['index']:r for r in map(json.loads,source.read_text().splitlines())}
gold={r['index']:r for r in map(json.loads,(ROOT/'eclp_runs_gpu4/evaluation_v2/golden/golden.jsonl').read_text().splitlines())}
for r in records:
    assert r['source_row_hash']==object_hash(rows[r['index']])
    assert r['golden_row_hash']==object_hash(gold[r['index']])
contract=json.loads((EARLY/'judges/qwen3_14b/evaluation_run.json').read_text())
assert contract['generation_run_hash']==object_hash(json.loads((CANON/'generation_run.json').read_text()))
repair=ROOT/'my_eclp_final/model_family_jobs/llama_public_repair_v3'
assert contract['protocol_sha256']==digest(repair/'protocol.py')
assert contract['win_rules_sha256']==digest(repair/'win_rules.json')
destination=CANON/'judges/qwen3_14b';destination.parent.mkdir(exist_ok=True)
adopted=False
if not destination.exists():
    temp=destination.with_name('.qwen3_14b-early-adoption')
    assert not temp.exists()
    shutil.copytree(EARLY/'judges/qwen3_14b',temp,ignore=shutil.ignore_patterns('*.lock'))
    try:os.rename(temp,destination);adopted=True
    except FileExistsError:shutil.rmtree(temp)
report={'early_evaluated_n':summary['n'],'canonical_cache_adopted':adopted,'source_and_gold_row_hashes_verified':True,'protocol_hashes_verified':True,'time':time.time()}
(EARLY/'qwen_adoption.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
