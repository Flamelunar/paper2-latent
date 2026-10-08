"""Paired pilot under one new Judge; append-only evaluation of new 192 runs."""
import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from protocol import REVISION,SchemaFactory,GOAL_SCHEMA,goal_prompt,validate_goal,json_batch

NEW_METHODS=['zero_shot_repaired','ma2p_repaired','our_analysis']
OLD_MAP={'old_zero_shot':'baseline1_zero_shot','old_ma2p':'baseline4_ma2p','old_our':'our_eclp_offline_lpo'}
PROGRESS={'none':0,'engagement':1,'conditional':2,'accepted':3}


def summarize(records,expected):
    n=len(records);first=[next((i for i,x in enumerate(r['goal_results'],1) if x['success']),None) for r in records]
    result={'n':n,'expected':expected,'complete':n==expected,'success_count':sum(x is not None for x in first),
        'success_percent':100*sum(x is not None for x in first)/n if n else None,
        'mean_rounds_fail5':statistics.fmean(x or 5 for x in first) if n else None,
        'final_progress_0_to_3':statistics.fmean(PROGRESS[r['goal_results'][4]['progress']] for r in records) if n else None,
        'final_success_percent':100*sum(r['goal_results'][4]['success'] for r in records)/n if n else None,
        'format_unresolved':0}
    for metric in ['win1','win_full']:
        result[metric+'_percent']=100*statistics.fmean(r[metric]['credit'] for r in records) if n else None
        result[metric+'_conflict_percent']=100*sum(r[metric]['ab_credit']!=r[metric]['ba_credit'] for r in records)/n if n else None
        result[metric+'_tie_percent']=100*sum(r[metric][o]=='Tie' for r in records for o in ['ab','ba'])/(2*n) if n else None
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=20);ap.add_argument('--physical-gpu',type=int,default=0)
    args=ap.parse_args();root=args.repo_root.resolve();work=args.output
    sys.path.insert(0,str(root/'my_eclp_final'));sys.path.insert(0,str(root/'exp_code_ljj'))
    from experiment_io import atomic_json,digest,immutable_json,object_hash,read_json,read_jsonl,writer_lock,append_jsonl
    from matched_eval_protocol_v2 import CONFIG,StrictJSONJudge,order_credit,text_hash,transcript,validate_trajectory,validate_available_dialogue
    from evaluate_our_ma2p192 import validate_record
    gen_contract=read_json(work/'generation_run.json');order=gen_contract['expected_indexes']
    rules_path=Path(__file__).with_name('win_rules.json');rules=read_json(rules_path)
    gold_path=root/'eclp_runs_gpu4/evaluation_v2/golden/golden.jsonl';gold={r['index']:r for r in read_jsonl(gold_path)}
    old_sources={};old_provenance={}
    for label,original in OLD_MAP.items():
        d=root/'eclp_runs_gpu4/evaluation_v2/selene8b'/original;c=read_json(d/'run.json');p=Path(c['generation_source'])
        assert digest(p)==c['generation_sha256']
        old_sources[label]={r['index']:r for r in read_jsonl(p)}
        old_provenance[label]={'source':str(p),'sha256':digest(p)}
    model=Path('/app/hzj/models/Selene-1-Mini-Llama-3.1-8B')
    contract={'revision':REVISION,'judge':'selene8b','judge_model':str(model),'physical_gpu':args.physical_gpu,
        'model_files':[{'file':p.name,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted(model.iterdir()) if p.suffix in ['.json','.safetensors']],
        'generation_run_hash':object_hash(gen_contract),'new_expected_indexes':order,'old_pilot_indexes':order[:20],
        'old_provenance':old_provenance,'golden_sha256':digest(gold_path),'win_rules_sha256':digest(rules_path),
        'protocol_sha256':digest(Path(__file__).with_name('protocol.py')),'evaluator_sha256':digest(Path(__file__)),
        'original_win_protocol_sha256':digest(root/'my_eclp_final/matched_eval_protocol_v2.py'),
        'batch_size':8,'goal_schema':'evidence quote -> progress -> success, strict booleans and quoted current-reply evidence',
        'success_rule':'any accepted round within five; mean rounds failures=5','win_credit':'mean of two swapped requests; win1,tie0.5,loss0',
        'analysis_visibility':'Our private analysis is archived but excluded from Judge and simulated Persuadee',
        'compute_dtype':'bfloat16','quantization':'none','temperature':0,'max_format_retries':2,
        'new_sources':'append-only generation; every evaluated row frozen by row SHA256'}
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.physical_gpu)
    assert args.limit in [20,192]
    run_hash=object_hash(contract)

    def win_prompt(row,g,metric,order_name):
        a,b=(row['trajectory'][0]['content'],g['golden_first']) if metric=='win1' else (transcript(row['trajectory']),transcript(g['golden_trajectory']))
        if order_name=='ba':a,b=b,a
        rule=rules[metric+'_rule']
        rule+='\nDistinguish engagement or preliminary activities from acceptance of the exact goal. For multi-part goals, assess progress on all required parts. Do not assume the goal was achieved merely because the dialogue ended politely.'
        return (f'Background: {row["scenario"]["background"]}\nExact goal: {row["scenario"]["goal"]}\nCommon preceding dialogue: (empty)\n'
                f'Candidate A:\n{a}\n\nCandidate B:\n{b}\n\n{rule}\n\n{CONFIG["win_contract"]}')

    with writer_lock(work/'.evaluation.lock'):
        immutable_json(work/'evaluation_run.json',contract)
        judge=None;factory=None
        try:
            labels=NEW_METHODS+(list(OLD_MAP) if args.limit==20 else [])
            for label in labels:
                expected=192 if label in NEW_METHODS else 20
                indexes=order[:args.limit] if label in NEW_METHODS else order[:20]
                p=work/'generation'/f'{label}.jsonl'
                source=old_sources[label] if label in OLD_MAP else {r['index']:r for r in read_jsonl(p)}
                for i in indexes:
                    row,g=source[i],gold[i]
                    assert row['initial_history']==g['initial_history']==[]
                    assert row['scenario_id']==g['scenario_id'] and row['scenario']==g['scenario'] and row['reference']==g['golden_first']
                    validate_trajectory(row['trajectory']);validate_available_dialogue(g['golden_trajectory'])
                    if label in NEW_METHODS:assert row['run_hash']==object_hash(gen_contract)
                path=work/'evaluation'/label/'judgments.jsonl'
                saved=read_jsonl(path) if path.exists() else [];existing={r['index']:r for r in saved}
                assert len(existing)==len(saved)
                for i,r in existing.items():
                    assert r['contract_hash']==run_hash and r['source_row_hash']==object_hash(source[i]) and r['golden_row_hash']==object_hash(gold[i])
                for start in range(0,len(indexes),20):
                    chunk=indexes[start:start+20]
                    if all(i in existing for i in chunk):continue
                    cache=work/'evaluation'/label/'request_chunks'/f'{chunk[0]:04d}.json'
                    queries=[]
                    for i in chunk:
                        for r in range(1,6):queries.append({'index':i,'key':f'goal_r{r}','kind':'goal','prompt':goal_prompt(source[i],r),'round':r})
                        for metric in ['win1','win_full']:
                            for o in ['ab','ba']:queries.append({'index':i,'key':f'{metric}_{o}','kind':'win','prompt':win_prompt(source[i],gold[i],metric,o)})
                    queries.sort(key=lambda q:(q['kind'],len(q['prompt'])))
                    hashes=[text_hash(q['prompt']) for q in queries]
                    if cache.exists():
                        cached=read_json(cache);assert cached['contract_hash']==run_hash and cached['prompt_hashes']==hashes
                        replies=cached['responses']
                    else:
                        if judge is None:
                            judge=StrictJSONJudge(str(model),memory_cap_gib=30);factory=SchemaFactory(judge.tokenizer)
                        replies=[]
                        for offset in range(0,len(queries),8):
                            batch=queries[offset:offset+8];kind=batch[0]['kind']
                            # Never mix goal schema and finite winner schema in one batch.
                            split=next((j for j,q in enumerate(batch) if q['kind']!=kind),len(batch))
                            groups=[batch[:split]]+([batch[split:]] if split<len(batch) else [])
                            for group in groups:
                                kind=group[0]['kind']
                                try:
                                    raw_replies=(json_batch(judge.endpoint,factory,[q['prompt'] for q in group],GOAL_SCHEMA,CONFIG['system'])
                                                 if kind=='goal' else judge.complete_batch([q['prompt'] for q in group],['win']*len(group)))
                                except Exception as exc:
                                    raw_replies=[{'batch_error':str(exc)} for _ in group]
                                for q,reply in zip(group,raw_replies):
                                    attempts=[]
                                    for attempt in range(3):
                                        try:
                                            if 'batch_error' in reply:raise ValueError(reply['batch_error'])
                                            value=(validate_goal(reply['value'],source[q['index']]['trajectory'][2*q['round']-1]['content'])
                                                   if kind=='goal' else reply['verdict'])
                                        except (ValueError,TypeError,KeyError) as exc:
                                            attempts.append({'error':str(exc),'response':reply})
                                            if attempt==2:
                                                atomic_json(work/'evaluation'/label/'unresolved_error.json',{'index':q['index'],'key':q['key'],'attempts':attempts})
                                                raise
                                            correction=q['prompt']+'\nFormat reminder: quote an exact excerpt from the CURRENT reply; accepted if and only if success is true. Do not add keys.' if kind=='goal' else q['prompt']
                                            try:
                                                reply=(json_batch(judge.endpoint,factory,[correction],GOAL_SCHEMA,CONFIG['system'])[0]
                                                       if kind=='goal' else judge.complete(correction,'win'))
                                            except Exception as retry_exc:
                                                reply={'batch_error':str(retry_exc)}
                                            continue
                                        replies.append({'index':q['index'],'key':q['key'],'value':value,'raw_response':reply,'format_retries':len(attempts),'retry_records':attempts})
                                        break
                            atomic_json(work/'evaluation_progress.json',{'method':label,'chunk_first_index':chunk[0],
                                'requests_completed':len(replies),'requests_expected':len(queries),'target':args.limit,'pid':os.getpid()})
                            print(label,'requests',len(replies),'/',len(queries),flush=True)
                        atomic_json(cache,{'contract_hash':run_hash,'prompt_hashes':hashes,'responses':replies})
                    assert len(replies)==len(queries)
                    lookup={(q['index'],q['key']):q for q in replies}
                    for i in chunk:
                        if i in existing:continue
                        goals=[lookup[i,f'goal_r{r}']['value'] for r in range(1,6)]
                        for r,v in enumerate(goals,1):validate_goal(v,source[i]['trajectory'][2*r-1]['content'])
                        row={'index':i,'scenario_id':source[i]['scenario_id'],'contract_hash':run_hash,
                            'source_row_hash':object_hash(source[i]),'golden_row_hash':object_hash(gold[i]),'goal_results':goals}
                        for metric in ['win1','win_full']:
                            v={o:lookup[i,f'{metric}_{o}']['value']['winner'] for o in ['ab','ba']}
                            v.update(ab_credit=order_credit(v['ab'],'ab'),ba_credit=order_credit(v['ba'],'ba'))
                            v['credit']=statistics.fmean([v['ab_credit'],v['ba_credit']]);row[metric]=v
                        row['format_retries']=sum(lookup[i,q['key']]['format_retries'] for q in queries if q['index']==i)
                        append_jsonl(path,row);existing[i]=row
                    summary=summarize(list(existing.values()),expected);summary.update(method=label,contract_hash=run_hash,
                        judgments_sha256=digest(path),source_generation_sha256=digest(p) if label in NEW_METHODS else old_provenance[label]['sha256'])
                    atomic_json(work/'evaluation'/label/'summary.json',summary)
                    print(json.dumps(summary),flush=True)
        finally:
            if judge is not None:judge.close()


if __name__=='__main__':main()
