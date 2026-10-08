"""Matched, output-independent 20-scenario pilot; never updates the main table."""
import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--physical-gpu', type=int, default=1)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root / 'my_eclp_final'))
    sys.path.insert(0, str(root / 'exp_code_ljj'))
    import matched_eval_protocol_v2 as protocol
    from experiment_io import atomic_json, digest, immutable_json, object_hash, read_json, read_jsonl, writer_lock
    from evaluate_our_ma2p192 import validate_record
    rules_path = Path(__file__).with_name('preference_pilot_v3_rules.json')
    rules = read_json(rules_path)
    work = root / 'eclp_runs_gpu4/evaluation_v2'
    selection = read_json(root / 'exp_code_ljj/data/test_selection_192.json')
    gold_path = work / 'golden/golden.jsonl'
    all_gold = {r['index']: r for r in read_jsonl(gold_path)}
    indexes = sorted(selection['original_indexes'], key=lambda i: hashlib.sha256(
        ('matched-v2-pilot-42:' + all_gold[i]['scenario_id']).encode()).hexdigest())[:args.n]
    assert len(indexes) == args.n and len(set(indexes)) == args.n
    methods = ['baseline1_zero_shot', 'baseline2_dialogue_sft', 'baseline3_isr_simdpo',
               'baseline4_ma2p', 'baseline5_latentratt', 'our_eclp_offline_lpo', 'our_ma2p']
    sources, old_records, provenance = {}, {}, {}
    for method in methods:
        d = work / 'selene8b' / method
        if (d / 'run.json').exists():
            c = read_json(d / 'run.json')
            path = Path(c['generation_source'])
            assert digest(path) == c['generation_sha256']
            assert c['prompt_sha256'] == digest(protocol.PROMPT_FILE)
            old = read_jsonl(d / 'judgments.jsonl')
            assert len({r['index'] for r in old}) == len(old)
            old_records[method] = {r['index']: r for r in old}
        else:
            assert method == 'baseline3_isr_simdpo'
            path = root / 'exp_code_ljj/artifacts/generation_5rounds/baseline3_isr_simdpo.jsonl'
            c = None
            old_records[method] = {}
        rows = read_jsonl(path)
        assert len({r['index'] for r in rows}) == len(rows)
        by_index = {r['index']: r for r in rows}
        sources[method] = {i: by_index[i] for i in indexes}
        for i in indexes:
            row, g = by_index[i], all_gold[i]
            assert row['initial_history'] == g['initial_history'] == []
            assert row['scenario_id'] == g['scenario_id'] and row['scenario'] == g['scenario']
            assert row['reference'] == g['golden_first']
            protocol.validate_trajectory(row['trajectory'])
            protocol.validate_available_dialogue(g['golden_trajectory'])
            if c:
                validate_record(old_records[method][i], row, g, object_hash(c))
        provenance[method] = {'generation_source': str(path), 'generation_sha256': digest(path),
                              'old_judgments_sha256': digest(d / 'judgments.jsonl') if c else None,
                              'persuadee_model_recorded': rows[0].get('persuadee_model'),
                              'old_scores_source': 'saved_v2_judgments' if c else 'fresh_v2_evaluation'}
    table = root / 'exp_code_ljj/results/main_experiment_table.md'
    contract = {'revision': rules['revision'], 'judge_model': '/app/hzj/models/Selene-1-Mini-Llama-3.1-8B',
                'expected_indexes': indexes, 'n': args.n, 'methods': methods,
                'selection_policy': 'first n of SHA256 matched-v2-pilot-42:scenario_id; output-independent',
                'selection_sha256': digest(root / 'exp_code_ljj/data/test_selection_192.json'),
                'golden_sha256': digest(gold_path), 'rules_sha256': digest(rules_path),
                'script_sha256': digest(Path(__file__)), 'protocol_sha256': digest(Path(protocol.__file__)),
                'old_prompt_sha256': digest(protocol.PROMPT_FILE), 'provenance': provenance,
                'physical_gpu': args.physical_gpu, 'batch_size': 4, 'memory_cap_gib': 22,
                'compute_dtype': 'bfloat16', 'quantization': 'none', 'temperature': 0,
                'system_prompt': protocol.CONFIG['system'], 'success_rule': 'unchanged_v2',
                'decoding': 'same finite JSON grammar as v2; silent assessment; no rationale generated',
                'table_sha256_before': digest(table)}
    if args.validate_only:
        print(json.dumps({'validated': True, 'indexes': indexes, 'methods': methods}, indent=2))
        return
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.physical_gpu)
    contract_hash = object_hash(contract)

    def new_prompt(row, g, metric, order):
        if metric == 'win1':
            ours, reference = row['trajectory'][0]['content'], g['golden_first']
            desc = 'two alternative first Persuader replies'
        else:
            ours, reference = protocol.transcript(row['trajectory']), protocol.transcript(g['golden_trajectory'])
            desc = 'two persuasion dialogues'
        a, b = (ours, reference) if order == 'ab' else (reference, ours)
        return (f'You are an impartial evaluator comparing {desc} for the same scenario.\n\n'
                f'Background: {row["scenario"]["background"]}\nExact persuasion goal: {row["scenario"]["goal"]}\n'
                f'Common preceding conversation: {protocol.transcript(row["initial_history"])}\n\n'
                f'Candidate A:\n{a}\n\nCandidate B:\n{b}\n\n{rules[metric + "_rule"]}\n\n'
                f'{protocol.CONFIG["win_contract"]}')

    with writer_lock(args.output / '.writer.lock'):
        immutable_json(args.output / 'run.json', contract)
        judge = protocol.StrictJSONJudge(contract['judge_model'], memory_cap_gib=22)
        results = []
        try:
            for method in methods:
                items = []
                for i in indexes:
                    row, g = sources[method][i], all_gold[i]
                    for metric in ['win1', 'win_full']:
                        for order in ['ab', 'ba']:
                            items.append({'index': i, 'version': 'new', 'metric': metric, 'order': order,
                                          'kind': 'win', 'prompt': new_prompt(row, g, metric, order)})
                            if i not in old_records[method]:
                                items.append({'index': i, 'version': 'old', 'metric': metric, 'order': order,
                                              'kind': 'win', 'prompt': protocol.win_prompt(row, g, metric, order)})
                    if i not in old_records[method]:
                        for r in range(1, 6):
                            items.append({'index': i, 'version': 'old', 'metric': 'success', 'round': r,
                                          'kind': 'success', 'prompt': protocol.success_prompt(row, r)})
                items.sort(key=lambda q: (q['kind'], len(q['prompt'])))
                responses = []
                for offset in range(0, len(items), 4):
                    batch = items[offset:offset + 4]
                    cache = args.output / 'request_batches' / f'{method}_{offset:04d}.json'
                    hashes = [protocol.text_hash(q['prompt']) for q in batch]
                    if cache.exists():
                        saved = read_json(cache)
                        assert saved['contract_hash'] == contract_hash and saved['prompt_hashes'] == hashes
                        reply = saved['responses']
                    else:
                        reply = judge.complete_batch([q['prompt'] for q in batch], [q['kind'] for q in batch])
                        atomic_json(cache, {'contract_hash': contract_hash, 'prompt_hashes': hashes, 'responses': reply})
                    assert len(reply) == len(batch)
                    for q, response in zip(batch, reply):
                        assert response['prompt_sha256'] == protocol.text_hash(q['prompt'])
                        assert protocol.parse_verdict(response['raw_output'], q['kind']) == response['verdict']
                        responses.append({**q, 'response': response})
                    if (offset + 4) % 20 == 0:
                        atomic_json(args.output / 'progress.json', {'method': method, 'requests_completed': min(offset+4,len(items)),
                                    'requests_expected': len(items), 'methods_completed': len(results), 'state': 'running'})
                        print(method, min(offset+4,len(items)), '/', len(items), flush=True)
                records = []
                for i in indexes:
                    current = [q for q in responses if q['index'] == i]
                    old = old_records[method].get(i)
                    rounds = old['round_success'] if old else [next(q['response']['verdict']['success'] for q in current
                             if q['metric'] == 'success' and q['round'] == r) for r in range(1,6)]
                    item = {'index': i, 'scenario_id': all_gold[i]['scenario_id'], 'round_success': rounds}
                    for version in ['old', 'new']:
                        item[version] = {}
                        for metric in ['win1', 'win_full']:
                            v = {o: old[metric][o] if version == 'old' and old else next(
                                q['response']['verdict']['winner'] for q in current if q['version'] == version
                                and q['metric'] == metric and q['order'] == o) for o in ['ab','ba']}
                            v['credit'] = statistics.fmean(protocol.order_credit(v[o], o) for o in ['ab','ba'])
                            item[version][metric] = v
                    records.append(item)
                atomic_json(args.output / 'methods' / f'{method}.json', {'records': records, 'contract_hash': contract_hash})
                s = {'method': method, 'n': len(records), 'success_percent': 100*sum(any(r['round_success']) for r in records)/len(records)}
                for version in ['old', 'new']:
                    s[version] = {}
                    for metric in ['win1','win_full']:
                        v = [r[version][metric] for r in records]
                        s[version][metric] = {'score_percent': 100*statistics.fmean(x['credit'] for x in v),
                            'order_conflict_percent': 100*sum(protocol.order_credit(x['ab'],'ab') != protocol.order_credit(x['ba'],'ba') for x in v)/len(v),
                            'explicit_tie_percent': 100*sum(x[o]=='Tie' for x in v for o in ['ab','ba'])/(2*len(v)),
                            'position_A_percent': 100*sum(x[o]=='A' for x in v for o in ['ab','ba'])/(2*len(v))}
                results.append(s)
                atomic_json(args.output / 'summary.json', {'complete': len(results)==len(methods), 'n': args.n, 'results': results, 'contract_hash': contract_hash})
                print(json.dumps(s), flush=True)
        finally:
            judge.close()
        assert digest(table) == contract['table_sha256_before'], 'Main table changed during pilot'
        atomic_json(args.output / 'progress.json', {'state': 'complete', 'methods_completed': len(results), 'n': args.n,
                    'format_unresolved': 0, 'main_table_unchanged': True})


if __name__ == '__main__':
    main()
