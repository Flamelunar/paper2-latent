"""Verify and render the additional Selene Our+MA2P row without legacy changes."""
from pathlib import Path
from experiment_io import digest,object_hash,read_json,read_jsonl


def verified_hybrid_summary(root):
    d=root/'eclp_runs_gpu4/evaluation_v2/selene8b/our_ma2p'
    if not (d/'summary.json').exists():return None
    saved=read_json(d/'summary.json')
    if not saved.get('complete'):return None
    from evaluate_our_ma2p192 import summarize,validate_record
    from matched_eval_protocol_v2 import CONFIG,PROMPT_FILE
    c=read_json(d/'run.json')
    selection=read_json(root/'exp_code_ljj/data/test_selection_192.json')
    for k,v in {'revision':CONFIG['revision'],'judge':'selene8b','method_slug':'our_ma2p',
                'expected_indexes':selection['original_indexes'],'prompt_sha256':digest(PROMPT_FILE),
                'protocol_sha256':digest(root/'my_eclp_final/matched_eval_protocol_v2.py'),
                'evaluator_sha256':digest(root/'my_eclp_final/evaluate_our_ma2p192.py'),
                'test_selection_sha256':digest(root/'exp_code_ljj/data/test_selection_192.json')}.items():
        if c.get(k)!=v:raise ValueError(f'Hybrid evaluation contract differs: {k}')
    source=Path(c['generation_source']);goldfile=Path(c['golden_source'])
    generation_contract=read_json(source.parent/'run.json')
    if (generation_contract.get('checkpoint_mode')!='full_eclp'
            or generation_contract.get('role_interfaces')!={'perception':'full_eclp_latent_json_schema',
                                                              'world_model':'full_eclp_latent_json_schema',
                                                              'persuader':'full_eclp_latent_actor'}
            or generation_contract.get('source_sha256')!=digest(root/'my_eclp_final/generate_our_ma2p192.py')):
        raise ValueError('Hybrid table result is not the frozen three-role full-ECLP run')
    if digest(source)!=c['generation_sha256'] or digest(goldfile)!=c['golden_sha256']:
        raise ValueError('Hybrid generation or golden changed')
    ss={r['index']:r for r in read_jsonl(source)};gg={r['index']:r for r in read_jsonl(goldfile)}
    rows=read_jsonl(d/'judgments.jsonl');ids=set(selection['original_indexes'])
    if len(rows)!=192 or len(ss)!=192 or {r['index'] for r in rows}!=ids or set(ss)!=ids:
        raise ValueError('Hybrid row requires all fixed 192 scenarios')
    for r in rows:
        i=r['index']
        if ss[i]['initial_history']!=[] or gg[i]['initial_history']!=[]:raise ValueError('Hybrid input history is not empty')
        validate_record(r,ss[i],gg[i],object_hash(c))
    actual=summarize(rows,{i:gg[i] for i in ids},c)
    if any(saved.get(k)!=v for k,v in actual.items()) or saved.get('judgments_sha256')!=digest(d/'judgments.jsonl'):
        raise ValueError('Hybrid summary differs from judgments')
    return saved


def hybrid_table_row(root,template_only=False):
    s=None if template_only else verified_hybrid_summary(root)
    values='待评估 / 192 | - | - | - | - | -' if s is None else (
        f"192/192 | {s['success_within5_percent']:.2f} ({s['success_count']}/192) | "
        f"{s['win1_percent']:.2f} | {s['win_full_percent']:.2f} | "
        f"{s['mean_rounds_or_cap5']:.3f} | {s['format_unresolved_completed_cases']}")
    return f'|  | Our + MA2P | Qwen2.5-7B-Instruct (Our checkpoint) | {values} |'
