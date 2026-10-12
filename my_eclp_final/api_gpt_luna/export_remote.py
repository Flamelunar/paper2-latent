"""Executed on remote host; exports public-only inputs and exact current rubrics."""
import hashlib
import json
import sys
from pathlib import Path

root = Path('/app/hzj/liujianjian/ljj-2-paper')
code = root / 'my_eclp_final'
repair = code / 'model_family_jobs/llama_public_repair_v3'
sys.path[:0] = [str(repair), str(code)]
from protocol import goal_prompt
from matched_eval_protocol_v2 import CONFIG

art = root / 'exp_code_ljj/artifacts'
methods = ['zero_shot_repaired', 'dialogue_sft', 'persugpt', 'ma2p_repaired', 'latentratt', 'our_direct']
families = {
    'qwen25': art / 'seven_methods_192_20261006',
    'llama31': art / 'llama_two_stage_pilot_20261009/meta_llama31',
    'deepseek14b': art / 'deepseek_goal_strategy_192_20261008/deepseek_r1_qwen14b',
}
order = json.loads((root/'exp_code_ljj/data/test_selection_192.json').read_text())['original_indexes']
gold_path = root/'eclp_runs_gpu4/evaluation_v2/golden/golden.jsonl'
gold = {r['index']:r for r in map(json.loads, gold_path.read_text().splitlines())}
rules = json.loads((repair/'win_rules.json').read_text())
manifest = {'system': CONFIG['system'], 'expected_indexes': order, 'sources': [],
            'protocol_sha256': hashlib.sha256((repair/'protocol.py').read_bytes()).hexdigest(),
            'win_rules_sha256': hashlib.sha256((repair/'win_rules.json').read_bytes()).hexdigest(),
            'golden_sha256': hashlib.sha256(gold_path.read_bytes()).hexdigest()}
datasets = []
for family, work in families.items():
    for method in methods[:4] if family == 'deepseek14b' else methods:
        path = work/'generation'/f'{method}.jsonl'
        if not path.exists():
            path = work/'generators'/method/'generation'/f'{method}.jsonl'
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        assert len(rows)==192 and {r['index'] for r in rows}==set(order), (family,method,len(rows))
        source = {r['index']:r for r in rows}
        manifest['sources'].append({'family':family,'method':method,'path':str(path.resolve()),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'n':len(rows)})
        for i in order:
            r, g = source[i], gold[i]
            assert r['initial_history']==g['initial_history']==[]
            assert r['scenario']==g['scenario'] and r['reference']==g['golden_first']
            assert len(r['trajectory'])==10
            requests = []
            for turn in range(1,6):
                requests.append({'key':f'goal_r{turn}','kind':'goal','prompt':goal_prompt(r,turn),
                                 'current_reply':r['trajectory'][2*turn-1]['content']})
            for swap in ('ab','ba'):
                a,b = r['trajectory'][0]['content'],g['golden_first']
                if swap=='ba':a,b=b,a
                rule = rules['win1_rule']+'\nDistinguish engagement or preliminary activities from acceptance of the exact goal. For multi-part goals, assess progress on all required parts. Do not assume the goal was achieved merely because the dialogue ended politely.'
                prompt=(f'Background: {r["scenario"]["background"]}\nExact goal: {r["scenario"]["goal"]}\nCommon preceding dialogue: (empty)\n'
                    f'Candidate A:\n{a}\n\nCandidate B:\n{b}\n\n{rule}\n\n{CONFIG["win_contract"]}')
                requests.append({'key':f'win1_{swap}','kind':'win','prompt':prompt})
            datasets.append({'family':family,'method':method,'index':i,'requests':requests})
print(json.dumps({'manifest':manifest,'scenes':datasets},ensure_ascii=False))
