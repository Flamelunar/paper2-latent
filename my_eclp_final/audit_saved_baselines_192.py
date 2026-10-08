"""Read-only audit of the actual fixed-192 generation and saved judgments."""
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / 'my_eclp_final'))
from experiment_io import digest, read_json, read_jsonl, atomic_json, object_hash
from evaluate_our_ma2p192 import validate_record

ids = read_json(root / 'exp_code_ljj/data/test_selection_192.json')['original_indexes']
gold_path = root / 'eclp_runs_gpu4/evaluation_v2/golden/golden.jsonl'
gold = {r['index']: r for r in read_jsonl(gold_path)}
markers = re.compile(r'(?im)(?:^|\n)[ \t]*(user|assistant|system|persuader|persuadee)[ \t]*(?::|\r?\n)')
methods = ['baseline1_zero_shot', 'baseline2_dialogue_sft', 'baseline4_ma2p',
           'baseline5_latentratt', 'our_eclp_offline_lpo', 'our_ma2p']
summary = {}
examples = {}
selected = {}
for method in methods:
    d = root / 'eclp_runs_gpu4/evaluation_v2/selene8b' / method
    c = read_json(d / 'run.json')
    source = Path(c['generation_source'])
    assert digest(source) == c['generation_sha256']
    assert digest(gold_path) == c['golden_sha256']
    a = {r['index']: r for r in read_jsonl(source)}
    judged = {r['index']: r for r in read_jsonl(d / 'judgments.jsonl')}
    cases, turns, user_turns, actor_turns = [], [], [], []
    affected_success, success_before_contamination = [], []
    samples = []
    selected[method] = {'sources': [a[i] for i in ids], 'judgments': [judged[i] for i in ids]}
    for i in ids:
        row, j = a[i], judged[i]
        assert row['initial_history'] == gold[i]['initial_history'] == []
        validate_record(j, row, gold[i], object_hash(c))
        first_bad = None
        first_success = next((r for r,v in enumerate(j['round_success'],1) if v), None)
        for t, turn in enumerate(row['trajectory']):
            found = [m for m in markers.finditer(turn['content'])
                     if m.start() > 0 or m.group(1).lower() != turn['role']]
            if found:
                if first_bad is None: first_bad = t
                turns.append((i,t))
                (user_turns if turn['role']=='persuadee' else actor_turns).append((i,t))
                if len(samples) < 5:
                    samples.append({'index':i,'round':t//2+1,'role':turn['role'],
                                    'markers':[m.group(1) for m in found], 'content':turn['content'],
                                    'round_success':j['round_success'], 'goal':row['scenario']['goal']})
        if first_bad is not None:
            cases.append(i)
            if first_success is not None and 2*first_success-1 >= first_bad:
                affected_success.append(i)
        if first_success is not None and (first_bad is None or 2*first_success-1 < first_bad):
            success_before_contamination.append(i)
    first_words = [len(a[i]['trajectory'][0]['content'].split()) for i in ids]
    actor_words = [len(t['content'].split()) for i in ids for t in a[i]['trajectory'] if t['role']=='persuader']
    tracekeys = sorted(set(k for i in ids for t in a[i]['trajectory'] for k in t))
    r0 = a[ids[0]]
    group = [judged[i] for i in ids]
    subset_metrics = {'success_within5_percent':100*sum(any(r['round_success']) for r in group)/192}
    for metric in ['win1','win_full']:
        subset_metrics[metric+'_percent'] = 100*statistics.fmean(r[metric]['credit'] for r in group)
        subset_metrics[metric+'_order_disagreement_count'] = sum(
            (0.5 if r[metric]['ab']=='Tie' else float(r[metric]['ab']=='A')) !=
            (0.5 if r[metric]['ba']=='Tie' else float(r[metric]['ba']=='B')) for r in group)
    summary[method] = {'n':192,'generation_source':str(source),'generation_sha256':digest(source),
        'metadata':{k:r0.get(k) for k in ['protocol','persuader_model','persuadee_model','temperature','top_p','top_k','max_new_tokens','compute_dtype','quantization']},
        'continuation_marker_cases':len(cases),'continuation_marker_turns':len(turns),
        'persuadee_continuation_turns':len(user_turns),'persuader_continuation_turns':len(actor_turns),
        'first_success_prefix_contains_continuation_cases':len(affected_success),
        'first_success_prefix_contains_continuation_indexes':affected_success,
        'success_before_any_continuation_cases':len(success_before_contamination),
        'continuation_indexes':cases,'first_reply_mean_words':statistics.fmean(first_words),
        'first_reply_median_words':statistics.median(first_words), 'all_actor_mean_words':statistics.fmean(actor_words),
        'first_reply_exact_reference_matches':sum(a[i]['trajectory'][0]['content'].strip()==a[i]['reference'].strip() for i in ids),
        'stored_user_raw_output_same_as_content':sum(t.get('raw_output')==t['content'] for i in ids for t in a[i]['trajectory'] if t['role']=='persuadee'),
        'persisted_user_truncation_flags':sum(bool(t.get('future_dialogue_truncated')) for i in ids for t in a[i]['trajectory'] if t['role']=='persuadee'),
        'private_think_marker_turns':sum(bool(re.search(r'</?think\b',t['content'],re.I)) for i in ids for t in a[i]['trajectory']),
        'turn_keys':tracekeys,
        'metrics':subset_metrics}
    examples[method] = samples
atomic_json(out/'statistics.json', summary)
atomic_json(out/'continuation_examples.json', examples)
atomic_json(out/'selected_sources_and_judgments.json', selected)
print(json.dumps({m:{k:v for k,v in r.items() if k not in ['generation_source','generation_sha256','metadata','continuation_indexes','first_success_prefix_contains_continuation_indexes','turn_keys']} for m,r in summary.items()},indent=2))
