"""Paired pilot under one new Judge; append-only evaluation of new 192 runs."""
import argparse
import hashlib
import json
import os
import statistics
import copy
import re
import sys
from pathlib import Path
from protocol import REVISION,SchemaFactory,GOAL_SCHEMA,goal_prompt,validate_goal,json_batch

NEW_METHODS=['zero_shot_repaired','dialogue_sft','persugpt','ma2p_repaired','latentratt','our_analysis','our_direct']
OLD_MAP={'old_zero_shot':'baseline1_zero_shot','old_ma2p':'baseline4_ma2p','old_our':'our_eclp_offline_lpo'}
PROGRESS={'none':0,'engagement':1,'conditional':2,'accepted':3}


def summarize(records,expected):
    n=len(records);first=[]
    for record in records:
        goals=record['goal_results']
        success_round=next((i for i,value in enumerate(goals,1) if value is not None and value['success']),None)
        first.append(success_round if success_round is not None else (None if any(value is None for value in goals) else 5))
    known=[value for value in first if value is not None]
    success_count=sum(any(value is not None and value['success'] for value in r['goal_results']) for r in records)
    final=[r['goal_results'][4] for r in records if r['goal_results'][4] is not None]
    result={'n':n,'expected':expected,'complete':n==expected,'success_evaluable_n':len(known),
        'success_count':success_count,
        'success_percent':100*success_count/len(known) if known else None,
        'success_aggregation_revision':'any_accepted_round_including_5_v2',
        'mean_rounds_fail5':statistics.fmean(known) if known else None,
        'final_progress_evaluable_n':sum(r['goal_results'][4] is not None for r in records),
        'final_progress_0_to_3':statistics.fmean(PROGRESS[value['progress']] for value in final) if final else None,
        'final_success_percent':100*sum(value['success'] for value in final)/len(final) if final else None,
        'format_unresolved':sum(value is None for r in records for value in r['goal_results'])}
    for metric in ['win1','win_full']:
        valid=[r[metric] for r in records if r[metric]['credit'] is not None]
        result[metric+'_evaluable_n']=len(valid)
        result[metric+'_percent']=100*statistics.fmean(r['credit'] for r in valid) if valid else None
        result[metric+'_conflict_percent']=100*sum(r['ab_credit']!=r['ba_credit'] for r in valid)/len(valid) if valid else None
        result[metric+'_tie_percent']=100*sum(r[o]=='Tie' for r in valid for o in ['ab','ba'])/(2*len(valid)) if valid else None
        result[metric+'_unresolved']=n-len(valid)
    return result


def excerpt_schema(current_reply):
    """Format-only fallback: evidence must be a substring of current reply."""
    text=re.sub(r'\s+',' ',current_reply).strip()
    candidates={''}
    for sentence in re.split(r'(?<=[.!?])\s+',text):
        if 0<len(sentence)<=240:candidates.add(sentence)
    words=list(re.finditer(r'\S+',text))
    for start in range(0,len(words),4):
        first=words[start].start();end=first
        for word in words[start:start+30]:
            if word.end()-first>200:break
            end=word.end()
        if end>first:candidates.add(text[first:end])
    schema=copy.deepcopy(GOAL_SCHEMA)
    schema['properties']['evidence']['enum']=sorted(candidates)
    return schema


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=20);ap.add_argument('--physical-gpu',type=int,default=0)
    ap.add_argument('--judge',choices=['selene8b','qwen3_14b'],required=True)
    ap.add_argument('--method',choices=NEW_METHODS,required=True)
    ap.add_argument('--batch-size',type=int,choices=[2,4],default=4)
    ap.add_argument('--memory-cap-gib',type=float)
    ap.add_argument('--available-only',action='store_true',help='Partial diagnostic: evaluate present rows only; expected remains 192')
    args=ap.parse_args();root=args.repo_root.resolve();generation_work=args.output;work=args.output/'judges'/args.judge
    work.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'my_eclp_final'));sys.path.insert(0,str(root/'exp_code_ljj'));sys.path.insert(0,str(root/'my_eclp_final/seven_methods_192'))
    from experiment_io import atomic_json,digest,immutable_json,object_hash,read_json,read_jsonl,writer_lock,append_jsonl
    from matched_eval_protocol_v2 import CONFIG,StrictJSONJudge,order_credit,text_hash,transcript,validate_trajectory,validate_available_dialogue
    from evaluate_our_ma2p192 import validate_record
    gen_contract=read_json(generation_work/'generation_run.json');order=gen_contract['expected_indexes']
    rules_path=Path(__file__).with_name('win_rules.json');rules=read_json(rules_path)
    gold_path=root/'eclp_runs_gpu4/evaluation_v2/golden/golden.jsonl';gold={r['index']:r for r in read_jsonl(gold_path)}
    old_provenance={}
    model=Path('/app/hzj/models/Selene-1-Mini-Llama-3.1-8B' if args.judge=='selene8b' else '/app/hzj/models/Qwen3-14B')
    contract={'revision':REVISION,'judge':args.judge,'judge_model':str(model),'physical_gpu':args.physical_gpu,
        'model_files':[{'file':p.name,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted(model.iterdir()) if p.suffix in ['.json','.safetensors']],
        'generation_run_hash':object_hash(gen_contract),'new_expected_indexes':order,'old_pilot_indexes':order[:20],
        'old_provenance':old_provenance,'golden_sha256':digest(gold_path),'win_rules_sha256':digest(rules_path),
        'protocol_sha256':digest(Path(__file__).with_name('protocol.py')),'evaluator_sha256':digest(Path(__file__)),
        'original_win_protocol_sha256':digest(root/'my_eclp_final/matched_eval_protocol_v2.py'),
        'batch_size':4,'goal_schema':'evidence quote -> progress -> success, strict booleans and quoted current-reply evidence',
        'success_rule':'any accepted round within five; mean rounds failures=5','win_credit':'mean of two swapped requests; win1,tie0.5,loss0',
        'analysis_visibility':'Our private analysis is archived but excluded from Judge and simulated Persuadee',
        'compute_dtype':'bfloat16','quantization':'none','temperature':0,'max_format_retries':2,
        'new_sources':'append-only generation; every evaluated row frozen by row SHA256'}
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.physical_gpu)
    assert args.limit in [20,192]
    contract_path=work/'evaluation_run.json'
    prior_contract=read_json(contract_path) if contract_path.exists() else None
    if prior_contract:
        for key in ['revision','judge','judge_model','model_files','generation_run_hash',
                    'new_expected_indexes','golden_sha256','win_rules_sha256','protocol_sha256',
                    'original_win_protocol_sha256','batch_size','goal_schema','success_rule','win_credit',
                    'analysis_visibility','compute_dtype','quantization','temperature','max_format_retries']:
            assert contract[key]==prior_contract[key],key
        atomic_json(work/'evaluation'/args.method/'format_recovery_revision.json',{'run_contract_hash':object_hash(prior_contract),
            'previous_evaluator_sha256':prior_contract['evaluator_sha256'],'recovery_evaluator_sha256':digest(Path(__file__)),
            'policy':'preserve the original prompt and rubric; unresolved malformed responses remain null and are excluded from metric denominators'})
        contract=prior_contract
    run_hash=object_hash(contract)

    def win_prompt(row,g,metric,order_name):
        a,b=(row['trajectory'][0]['content'],g['golden_first']) if metric=='win1' else (transcript(row['trajectory']),transcript(g['golden_trajectory']))
        if order_name=='ba':a,b=b,a
        rule=rules[metric+'_rule']
        rule+='\nDistinguish engagement or preliminary activities from acceptance of the exact goal. For multi-part goals, assess progress on all required parts. Do not assume the goal was achieved merely because the dialogue ended politely.'
        return (f'Background: {row["scenario"]["background"]}\nExact goal: {row["scenario"]["goal"]}\nCommon preceding dialogue: (empty)\n'
                f'Candidate A:\n{a}\n\nCandidate B:\n{b}\n\n{rule}\n\n{CONFIG["win_contract"]}')

    with writer_lock(work/'evaluation'/args.method/'.evaluation.lock'):
        immutable_json(work/'evaluation_run.json',contract)
        memory_cap=args.memory_cap_gib if args.memory_cap_gib is not None else (22 if args.judge=='selene8b' else 40)
        append_jsonl(work/'evaluation'/args.method/'execution_launches.jsonl',{
            'contract_hash':run_hash,'runtime_physical_gpu':args.physical_gpu,
            'runtime_batch_size':args.batch_size,'memory_cap_gib':memory_cap,
            'evaluator_sha256':digest(Path(__file__)),'pid':os.getpid()})
        judge=None;factory=None
        try:
            labels=[args.method]
            for label in labels:
                expected=192 if label in NEW_METHODS else 20
                indexes=order[:args.limit] if label in NEW_METHODS else order[:20]
                p=generation_work/'generation'/f'{label}.jsonl'
                source={r['index']:r for r in read_jsonl(p)}
                if args.available_only:
                    indexes=[i for i in indexes if i in source]
                    assert indexes, 'No available rows'
                    print('PARTIAL diagnostic',label,len(indexes),'of 192',flush=True)
                for i in indexes:
                    row,g=source[i],gold[i]
                    assert row['initial_history']==g['initial_history']==[]
                    assert row['scenario_id']==g['scenario_id'] and row['scenario']==g['scenario'] and row['reference']==g['golden_first']
                    validate_trajectory(row['trajectory']);validate_available_dialogue(g['golden_trajectory'])
                    assert i in gen_contract['expected_indexes']
                path=work/'evaluation'/label/'judgments.jsonl'
                saved=read_jsonl(path) if path.exists() else [];existing={r['index']:r for r in saved}
                assert len(existing)==len(saved)
                for i,r in existing.items():
                    assert r['contract_hash']==run_hash and r['source_row_hash']==object_hash(source[i]) and r['golden_row_hash']==object_hash(gold[i])
                for start in range(0,len(indexes),20):
                    chunk=indexes[start:start+20]
                    if all(i in existing for i in chunk):continue
                    chunk=[i for i in chunk if i not in existing]
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
                            from repro.gpu_capacity import wait_for_bf16_capacity
                            wait_for_bf16_capacity([model],[0],reserve_gib=8 if args.judge=='qwen3_14b' else 6,description=f'{args.judge} evaluation on physical GPU{args.physical_gpu}')
                            judge=StrictJSONJudge(str(model),memory_cap_gib=memory_cap);factory=SchemaFactory(judge.tokenizer)
                        partial=cache.with_suffix('.partial.json')
                        replies=[]
                        if partial.exists():
                            data=read_json(partial);assert data['contract_hash']==run_hash and data['prompt_hashes']==hashes
                            replies=data['responses']
                        done={(q['index'],q['key']) for q in replies}
                        pending=[q for q in queries if (q['index'],q['key']) not in done]
                        for offset in range(0,len(pending),args.batch_size):
                            batch=pending[offset:offset+args.batch_size];kind=batch[0]['kind']
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
                                                replies.append({'index':q['index'],'key':q['key'],'value':None,
                                                    'raw_response':reply,'format_retries':len(attempts),
                                                    'retry_records':attempts,'unresolved':True})
                                                break
                                            correction=q['prompt']+'\nFormat reminder: quote an exact excerpt from the CURRENT reply; accepted if and only if success is true. Do not add keys.' if kind=='goal' else q['prompt']
                                            try:
                                                reply=(json_batch(judge.endpoint,factory,[correction],
                                                    excerpt_schema(source[q['index']]['trajectory'][2*q['round']-1]['content']) if attempt==1 else GOAL_SCHEMA,CONFIG['system'])[0]
                                                       if kind=='goal' else judge.complete(correction,'win'))
                                            except Exception as retry_exc:
                                                reply={'batch_error':str(retry_exc)}
                                            continue
                                        replies.append({'index':q['index'],'key':q['key'],'value':value,'raw_response':reply,'format_retries':len(attempts),'retry_records':attempts})
                                        break
                            atomic_json(partial,{'contract_hash':run_hash,'prompt_hashes':hashes,'responses':replies})
                            atomic_json(work/'evaluation'/label/'evaluation_progress.json',{'method':label,'chunk_first_index':chunk[0],
                                'requests_completed':len(replies),'requests_expected':len(queries),'target':args.limit,'pid':os.getpid()})
                            print(label,'requests',len(replies),'/',len(queries),flush=True)
                        atomic_json(cache,{'contract_hash':run_hash,'prompt_hashes':hashes,'responses':replies})
                    assert len(replies)==len(queries)
                    lookup={(q['index'],q['key']):q for q in replies}
                    for i in chunk:
                        if i in existing:continue
                        goals=[lookup[i,f'goal_r{r}']['value'] for r in range(1,6)]
                        for r,v in enumerate(goals,1):
                            if v is not None:validate_goal(v,source[i]['trajectory'][2*r-1]['content'])
                        row={'index':i,'scenario_id':source[i]['scenario_id'],'contract_hash':run_hash,
                            'source_row_hash':object_hash(source[i]),'golden_row_hash':object_hash(gold[i]),'goal_results':goals}
                        for metric in ['win1','win_full']:
                            verdicts={o:lookup[i,f'{metric}_{o}']['value'] for o in ['ab','ba']}
                            v={o:(value['winner'] if value is not None else None) for o,value in verdicts.items()}
                            if all(value is not None for value in v.values()):
                                v.update(ab_credit=order_credit(v['ab'],'ab'),ba_credit=order_credit(v['ba'],'ba'))
                                v['credit']=statistics.fmean([v['ab_credit'],v['ba_credit']])
                            else:v.update(ab_credit=None,ba_credit=None,credit=None)
                            row[metric]=v
                        row['format_retries']=sum(lookup[i,q['key']]['format_retries'] for q in queries if q['index']==i)
                        append_jsonl(path,row);existing[i]=row
                    summary=summarize(list(existing.values()),expected);summary.update(method=label,contract_hash=run_hash,
                        judgments_sha256=digest(path),source_generation_sha256=digest(p))
                    atomic_json(work/'evaluation'/label/'summary.json',summary)
                    print(json.dumps(summary),flush=True)
        finally:
            if judge is not None:judge.close()


if __name__=='__main__':main()
