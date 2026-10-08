"""Three methods, one checkpoint/environment, fixed subset and public boundary."""
import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path
ROOT_CODE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT_CODE/'seven_methods_192'),str(ROOT_CODE),str(ROOT_CODE.parent/'exp_code_ljj'),str(ROOT_CODE.parent/'baseline5_latentRATT')]
from batching import Request,drive
from protocol import REVISION,DECODE,PUBLIC_RULE,ANALYSIS_SCHEMA,SchemaFactory,public_point,speech_boundary,narration,strict_json

METHODS=['zero_shot_repaired','ma2p_repaired','our_direct','dialogue_sft','persugpt']
INPUT_POLICY='deepseek_goal_all_candidates_our_only_20261008'


def header(point):
    s=point['scenario']
    text=f"Background: {s['background']}\nExact persuasion goal: {s['goal']}\nPersuader: {s['persuader']}\nPersuadee: {s['persuadee']}"
    if 'our_candidate_strategies' in point:
        text+='\nCandidate persuasion strategies (options, not established facts): '+json.dumps(point['our_candidate_strategies'],ensure_ascii=False)
    return text


def goal_blind_persuadee_prompt(point,history):
    from repro.simulation import transcript
    scenario=point['scenario']
    system=(
        f"You are {scenario['persuadee']}, speaking with {scenario['persuader']} in a "
        "realistic conversation. You know only the background and what has been said "
        "in the dialogue; the persuader's private goal is not disclosed to you. Use "
        "the stated personal context and observable concerns, stay in character, and "
        "react naturally to the latest message. You may resist, hesitate, ask questions, "
        "make a qualified commitment, or accept when the conversation earns it. Do not "
        "accept merely to be helpful or to end the conversation. Never mention this "
        "simulation or invent personal facts. Output one natural next utterance only."
    )
    user=(f"Persuasion background: {scenario['background']}\n\n"
          f"Dialogue so far:\n{transcript(history)}\n\n"
          "Write only the next natural persuadee utterance. Do not add a speaker label.")
    return system,user


def sample_local(endpoint,system,prompt,seed,point,schema=None,factory=None):
    import torch
    from eclp_elpo import independent_rng
    encoded=endpoint._encode_messages([{'role':'system','content':system},{'role':'user','content':prompt}])
    if hasattr(encoded,'input_ids'):encoded=encoded.input_ids
    width=encoded.shape[1]
    if width+512>endpoint.context_limit:raise ValueError('Generation context overflow; do not truncate')
    ids=encoded.to(endpoint.input_device)
    kwargs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=512,
                do_sample=True,temperature=.95,top_p=.7,top_k=50,pad_token_id=endpoint.tokenizer.pad_token_id)
    if schema is not None:
        kwargs['prefix_allowed_tokens_fn']=factory.constraint(schema,width)
    else:
        kwargs['stop_strings']=['\nuser\n','\nassistant\n','\nPersuader:','\nPersuadee:']+[
            '\n'+point['scenario'][r]+':' for r in ['persuader','persuadee']]
        kwargs['tokenizer']=endpoint.tokenizer
    with independent_rng(seed,endpoint.devices),torch.inference_mode():
        generated=endpoint.model.generate(**kwargs)
    raw=endpoint.tokenizer.decode(generated[0,width:],skip_special_tokens=True).strip()
    tokens=int(generated.shape[1]-width)
    del generated,ids
    return raw,{'input_tokens':int(width),'output_tokens':tokens}


def base_turn(endpoint,factory,method,point,history,plans,round_number,seed):
    from repro.simulation import transcript
    common=header(point)+'\nObserved conversation:\n'+transcript(history)+'\n\n'+PUBLIC_RULE
    trace={}
    calls=0
    if method in ['dialogue_sft','persugpt']:
        from repro.persuadee_simulation import dialogue_prompt
        from repro.constants import SYSTEM_PERSUADER
        from repro.simulation import extract_utterance
        attempts=[]
        prompt=dialogue_prompt(point,history)+'\n'+PUBLIC_RULE
        for attempt in range(3):
            retry=prompt if attempt==0 else prompt+'\nOutput only the actual spoken message; exclude planning, speaker labels and future replies.'
            raw,usage=yield Request(endpoint,SYSTEM_PERSUADER,retry,seed+1000*attempt,point)
            attempts.append({'raw_output':raw,**usage})
            try:
                text=extract_utterance(raw) if method=='persugpt' else raw
                result=speech_boundary(text,point,'persuader')
                if narration(result['content'],point):raise ValueError('Private plan instead of public speech')
            except ValueError:continue
            result.update(raw_output=raw,attempts=attempts,model_calls=attempt+1,**usage)
            return result
        raise ValueError('Baseline public boundary retries exhausted')
    if method=='ma2p_repaired':
        specs=[('perception',{'preventive':None,'generative':None}),('world_model',{'likely_reaction':None,'strategy':None})]
        for role,fields in specs:
            schema={'type':'object','properties':{k:{'type':'string','minLength':1,'maxLength':300} for k in fields},
                    'required':list(fields),'additionalProperties':False}
            if role=='perception':
                prompt=common+('\nInfer concerns and motivations ONLY from observable evidence; use unknown when unsupported. '
                               'Return JSON with preventive and generative string fields.')
                system="You are MA2P's Perception agent. Infer cautiously from observable evidence."
            else:
                prompt=common+f"\nPerception: {json.dumps(trace['perception']['value'])}\nPrevious plans: {json.dumps(plans)}\nTurn {round_number} of 5."
                prompt+='\nPlan a specific, respectful next step toward the exact goal. K=0; no external strategies or knowledge base. Return JSON with likely_reaction and strategy string fields.'
                system="You are MA2P's World Model. Plan from public context and short-term memory only."
            for attempt in range(3):
                raw,usage=yield Request(endpoint,system,prompt,seed+calls+1000*attempt,point,schema,factory);calls+=1
                try:
                    value=strict_json(raw)
                    if set(value)!=set(fields) or any(not isinstance(v,str) or not v.strip() for v in value.values()):raise ValueError('Invalid MA2P analysis')
                except (ValueError,TypeError) as exc:
                    trace.setdefault('format_errors',[]).append(str(exc));continue
                trace[role]={'value':value,'raw_output':raw,**usage};break
            else:raise ValueError('MA2P analysis format retries exhausted; no silent fallback')
        plans.append(trace['world_model']['value'])
        prompt=common+f"\nInferred concern and motivation: {json.dumps(trace['perception']['value'])}\nWorld Model plan: {json.dumps(trace['world_model']['value'])}"
        system="You are a skilled persuader following the MA2P World Model's plan."
    else:
        # DeepSeek new contract: background, exact goal and public history; no annotations.
        prompt=common
        system='You are a skilled persuader.'
    prompt+='\nOutput only one natural Persuader utterance addressed to the other person. No analysis, speaker labels, or future replies.'
    attempts=[]
    for attempt in range(3):
        raw,usage=yield Request(endpoint,system,prompt,seed+calls+1000*attempt,point);calls+=1
        attempts.append({'raw_output':raw,**usage})
        try:
            result=speech_boundary(raw,point,'persuader')
            if narration(result['content'],point):raise ValueError('Analysis was emitted instead of spoken utterance')
        except ValueError:continue
        result.update(agent_trace=trace,attempts=attempts,model_calls=calls,**usage)
        return result
    raise ValueError('Public Persuader format retries exhausted')


class OurActor:
    def __init__(self,root,cfg,checkpoint,mode="our_analysis"):
        import torch
        from eclp_elpo import BaselineBridge,ECLPRunner,public_history
        class Bridge(BaselineBridge):
            role_prompt=None
            def actor_prompt(self,point,history):
                if self.role_prompt is None:raise RuntimeError('Missing structured actor prompt')
                return self.role_prompt
        class FullContextRunner(ECLPRunner):
            def prompt_ids(self,point,history):
                prompt=self.bridge.actor_prompt(point,public_history(history))
                encoded=self.tokenizer.apply_chat_template([{'role':'system','content':self.cfg.actor_system_prompt},
                    {'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True)
                if hasattr(encoded,'input_ids'):encoded=encoded.input_ids
                if len(encoded)>6144:raise ValueError('New actor full prompt exceeds 6144 tokens; no history truncation')
                return encoded
        self.bridge=Bridge(cfg);self.runner=FullContextRunner(cfg,self.bridge,checkpoint)
        self.factory=SchemaFactory(self.runner.tokenizer);self.cfg=cfg;self.mode=mode

    def turn(self,point,live,history,seed):
        import torch
        from eclp_elpo import BF16,PromptRepetitionPenalty,independent_rng
        from repro.simulation import transcript
        prompt=header(point)+'\nObserved conversation:\n'+transcript(history)+'\n\n'+PUBLIC_RULE
        prompt+=('\nYou may explain your psychological assessment and strategy before the final spoken answer. '
                 'The psychology and strategy are private analysis, based only on the supplied evidence. '
                 'Use unknown if unsupported. In utterance, speak naturally to the other person; do not describe a plan or their mental state. '
                 'Return only JSON in this order: {"psychology":"brief assessment","strategy":"brief plan","utterance":"actual spoken persuasion"}.')
        if self.mode=='our_direct':
            prompt=header(point)+'\nObserved conversation:\n'+transcript(history)+'\n\n'+PUBLIC_RULE
            prompt+='\nPrivately consider the other person’s concerns and an appropriate persuasion strategy. Output ONLY one natural spoken persuasion utterance. Do not output analysis, thinking, JSON, speaker labels, or future replies.'
        self.bridge.role_prompt=prompt
        self.cfg.actor_system_prompt=('You are a skilled persuader. Output only a natural spoken persuasion utterance; keep all analysis private.' if self.mode=='our_direct' else 'You are a skilled persuader. Separate private analysis from the final spoken utterance.')
        attempts=[]
        for attempt in range(3):
            self.bridge.role_prompt=prompt
            with torch.no_grad(),torch.autocast(device_type='cuda',dtype=BF16):
                prefix,ctx,_=self.runner.context(point,live)
                action=self.runner.heads.policy(ctx)
                emb=torch.cat([prefix,self.runner.heads.strategy_tokens(action)],1)
                prompt_ids=torch.tensor([self.runner.prompt_ids(point,live.history)],device=self.runner.device)
            raw,usage=yield Request(self.runner,'','',seed+1000*attempt,point,ANALYSIS_SCHEMA if self.mode=='our_analysis' else None,self.factory,
                                    embeddings=emb,prompt_ids=prompt_ids)
            attempts.append({'raw_output':raw,**usage})
            try:
                if self.mode=='our_analysis':
                    value=strict_json(raw)
                    if set(value)!=set(ANALYSIS_SCHEMA['properties']) or any(not isinstance(v,str) or not v.strip() for v in value.values()):raise ValueError('Incomplete structured analysis')
                    result=speech_boundary(value['utterance'],point,'persuader')
                else:
                    value=None
                    result=speech_boundary(raw,point,'persuader')
                    if result['thinking_filtered'] or raw.lstrip().startswith(('{','<think>','Psychology:','Analysis:','Strategy:')):
                        raise ValueError('Direct output contained explicit analysis')
                if narration(result['content'],point):raise ValueError('Utterance contains narration')
            except (ValueError,TypeError):continue
            result.update(raw_output=raw,analysis=value,attempts=attempts,model_calls=attempt+1,**usage)
            return result
        raise ValueError('Our structured output retries exhausted; do not accept a plan as speech')

    def close(self):
        self.runner.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=20);ap.add_argument('--physical-gpu',type=int,default=0)
    ap.add_argument('--validate-only',action='store_true');ap.add_argument('--inherit-dir',type=Path)
    ap.add_argument('--fresh-generation',action='store_true',help='Regenerate every selected scenario; forbid archived dialogue inheritance')
    ap.add_argument('--family',choices=['meta_llama31','deepseek_r1_qwen14b'],required=True)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--memory-cap-gib',type=float,default=54)
    ap.add_argument('--method',choices=METHODS,required=True)
    ap.add_argument('--persuadee-goal-blind',action='store_true',
                    help='Hide the exact persuasion goal from the simulated persuadee')
    ap.add_argument('--smoke-two',action='store_true');args=ap.parse_args()
    root=args.repo_root.resolve();sys.path.insert(0,str(root/'my_eclp_final'));sys.path.insert(0,str(root/'exp_code_ljj'))
    from experiment_io import atomic_json,immutable_json,digest,object_hash,read_json,read_jsonl,append_jsonl,writer_lock
    from eclp_elpo import Config,verify_checkpoint,scenario_seed,seed_all,independent_rng
    from repro.simulation import LocalTransformersEndpoint,user_prompt
    from repro.gpu_capacity import wait_for_bf16_capacity
    import hashlib,torch
    # This endpoint is BF16 without quantization; avoid optional bnb dispatch.
    import peft.tuners.lora.model as lora_backend
    lora_backend.is_bnb_available=lambda:False
    lora_backend.is_bnb_4bit_available=lambda:False
    selection=read_json(root/'exp_code_ljj/data/test_selection_192.json');points=read_jsonl(root/'exp_code_ljj/data/evaluation_points.jsonl')
    assert selection['original_indexes']==[i for i,p in enumerate(points) if p['target_turn']==1 and p['history']==[]]
    assert digest(root/'exp_code_ljj/data/evaluation_points.jsonl')==selection['source_sha256']
    order=sorted(selection['original_indexes'],key=lambda i:hashlib.sha256(('matched-v2-pilot-42:'+points[i]['scenario_id']).encode()).hexdigest())
    assert args.limit in [20,192]
    if args.family!='deepseek_r1_qwen14b' or args.physical_gpu not in (0,1,4) or args.persuadee_goal_blind:
        raise ValueError('New DeepSeek policy requires family=deepseek_r1_qwen14b, GPU0/1/4, and a goal-visible simulator')
    if not args.fresh_generation or args.inherit_dir is not None:
        raise ValueError('New policy requires all-new dialogues without inheritance')
    assert args.method=='ma2p_repaired' and args.physical_gpu==0
    cfg=Config.load(args.config);cfg.repo_root=str(root);cfg.actor_device='cuda:0';cfg.persuadee_device='cuda:1'
    expected_base={'meta_llama31':'/app/hzj/models/Meta-Llama-3.1-8B-Instruct','deepseek_r1_qwen14b':'/app/hzj/models/DeepSeek-R1-Distill-Qwen-14B'}[args.family]
    assert Path(cfg.base_model).resolve()==Path(expected_base).resolve()
    checkpoint=args.checkpoint;meta={'checkpoint_id':None};actor_model=cfg.base_model
    if args.method=='our_direct':
        from eclp_elpo import verify_lock
        if checkpoint is None:raise ValueError('Our requires the completed model-specific RL checkpoint')
        lock=verify_lock(cfg,cfg.work/'protocol.lock.json',('base_model',))
        meta=verify_checkpoint(checkpoint,lock)
        assert meta['stage']=='rl' and meta['step']==1800
    elif args.method in ['dialogue_sft','persugpt']:
        slug='baseline2_dialogue_sft' if args.method=='dialogue_sft' else 'baseline3_simdpo'
        actor_model=str(Path('/app/hzj/liujianjian/persuader_atapter_save')/args.family/slug)
        assert Path(read_json(Path(actor_model)/'adapter_config.json')['base_model_name_or_path']).resolve()==Path(cfg.base_model).resolve()
    if args.fresh_generation and args.inherit_dir is not None:raise ValueError('Fresh generation cannot inherit dialogues')
    inherited=[];source_provenance=None
    if args.method in ['dialogue_sft','persugpt'] and not args.fresh_generation:
        slug='baseline2_dialogue_sft' if args.method=='dialogue_sft' else 'baseline3_isr_simdpo'
        src=root/'exp_code_ljj/artifacts/generation_from_start192'/args.family/(slug+'.jsonl')
        audit=read_json(ROOT_CODE/'model_family_jobs/source_audit.json')[args.family][slug]
        assert digest(src)==audit['sha256']
        inherited=[r for r in read_jsonl(src) if r['index'] not in audit['affected_indexes']]
        source_provenance={'path':str(src),'sha256':digest(src),'inherited_clean_indexes':[r['index'] for r in inherited],'regenerated_indexes':[i for i in order if i not in {r['index'] for r in inherited}]}

    models={}
    for label,path in [('base',cfg.base_model),('persuadee',cfg.persuadee_model)]:
        p=Path(path);models[label]={'path':str(p.resolve()),'files':[{'file':x.name,'bytes':x.stat().st_size,'mtime_ns':x.stat().st_mtime_ns}
            for x in sorted(p.iterdir()) if x.is_file() and x.suffix in ['.json','.jinja','.safetensors']],
            'config_sha256':digest(p/'config.json'),'generation_config_sha256':digest(p/'generation_config.json')}
    contract={'revision':REVISION,'expected_indexes':order,'pilot_indexes':order[:20],'checkpoint_id':meta['checkpoint_id'],
        'checkpoint':str(checkpoint) if checkpoint else None,'family':args.family,'actor_model':actor_model,'clean_inheritance':source_provenance,'models':models,'decode':DECODE,'seed':42,'physical_gpu':args.physical_gpu,
        'runtime_placement':{'actor_physical_gpu':0,'simulator_physical_gpu':4},'compute_dtype':'bfloat16','quantization':'none','public_policy':'first utterance only, before history, latent feedback and Judge',
        'input_policy':INPUT_POLICY,'goal_visible_to_actor':True,'strategy_visibility':'candidate list to our_direct only; no gold selected strategy','candidate_field':'strategies','method':args.method,'fresh_generation':args.fresh_generation,'simulator_prompt_sha256':digest(root/'exp_code_ljj/repro/simulation.py'),'dialogue_prompt_sha256':digest(root/'exp_code_ljj/repro/persuadee_simulation.py'),'our_interface':'full ECLP checkpoint; analysis archived or direct speech, depending on method',
        'persuadee_goal_blind':args.persuadee_goal_blind,
        'persuadee_prompt_policy':('background and observed public dialogue only; exact goal withheld' if args.persuadee_goal_blind else 'background, exact goal and observed public dialogue; no strategy or ISR'),
        'our_max_prompt_tokens':6144,'our_analysis_visibility':'raw archive only; utterance enters public history',
        'ma2p':'Perception -> World Model -> Persuader, K=0, strict schemas, no silent fallback',
        'zero_shot_inputs':'background, exact goal, and observed public history; no strategy, psychology or golden reply',
        'source_sha256':digest(Path(__file__)),'protocol_sha256':digest(ROOT_CODE/'seven_methods_192/protocol.py'),
        'eclp_source_sha256':digest(root/'my_eclp_final/eclp_elpo.py'),'points_sha256':selection['source_sha256'],
        'selection_sha256':digest(root/'exp_code_ljj/data/test_selection_192.json'),'judge_used_during_generation':False,
        'batch_size':2,'batching_sha256':digest(ROOT_CODE/'seven_methods_192/batching.py'),
        'sampling_implementation':'per-row seeded RNG, HF temperature/top-k/top-p warpers; batch numeric rounding can differ',
        'inherited_generation':({'path':str(args.inherit_dir.resolve()),'contract_sha256':digest(args.inherit_dir/'generation_run.json'),
            'source_files_sha256':{p.name:digest(p) for p in sorted((args.inherit_dir/'generation').glob('*.jsonl'))}} if args.inherit_dir else None)}
    if args.validate_only:print(json.dumps({'validated':True,'pilot_indexes':order[:20],'checkpoint_id':meta['checkpoint_id']}));return
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,4'
    torch.cuda.set_per_process_memory_fraction(40*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
    torch.cuda.set_per_process_memory_fraction(24*1024**3/torch.cuda.get_device_properties(1).total_memory,1)
    with writer_lock(args.output/'.generation.lock'):
        immutable_json(args.output/'generation_run.json',contract);run_hash=object_hash(contract)
        actor=user=factory=None
        try:
            for method in [args.method]:
                path=args.output/'generation'/f'{method}.jsonl'
                saved=read_jsonl(path) if path.exists() else [];existing={r['index']:r for r in saved}
                assert len(existing)==len(saved) and set(existing).issubset(order)
                for original in inherited:
                    if original['index'] in existing:continue
                    row=dict(original,method=method,run_hash=run_hash,generation_origin='verified_clean_historical_scenario',inherited_source_row_hash=object_hash(original))
                    append_jsonl(path,row);existing[row['index']]=row
                if args.inherit_dir:
                    legacy_contract=read_json(args.inherit_dir/'generation_run.json')
                    legacy_hash=object_hash(legacy_contract)
                    assert legacy_contract['checkpoint_id']==meta['checkpoint_id'] and legacy_contract['expected_indexes']==order
                    assert legacy_contract['models']==models and legacy_contract['decode']==DECODE
                    legacy_path=args.inherit_dir/'generation'/f'{method}.jsonl'
                    legacy_rows=read_jsonl(legacy_path) if legacy_path.exists() else []
                    for row in legacy_rows:
                        assert row['run_hash']==legacy_hash and len(row['trajectory'])==10 and row['initial_history']==[]
                        if row['index'] in existing:continue
                        row=dict(row,inherited_run_hash=legacy_hash,inherited_source_row_hash=object_hash(row),
                                 inherited_source=str(legacy_path),run_hash=run_hash,batch_size_requested=1)
                        append_jsonl(path,row);existing[row['index']]=row
                    print('inherited',method,len(legacy_rows),'completed rows',flush=True)
                for i,r in existing.items():
                    assert r['run_hash']==run_hash and r['initial_history']==[] and r['scenario_id']==points[i]['scenario_id']
                    assert len(r['trajectory'])==10
                todo=[i for i in order[:(2 if args.smoke_two else args.limit)] if i not in existing]
                if not todo:continue
                if user is None:
                    wait_for_bf16_capacity([cfg.base_model],[0],reserve_gib=4,description='MA2P actor on physical GPU0')
                    wait_for_bf16_capacity([cfg.persuadee_model],[1],reserve_gib=4,description='MA2P simulator on physical GPU4')
                    seed_all(42);user=LocalTransformersEndpoint(model=cfg.persuadee_model,device='cuda:1')
                if method in ['our_analysis','our_direct']:
                    if actor is not None:actor.close();actor=None;factory=None;gc.collect();torch.cuda.empty_cache()
                    actor=OurActor(root,cfg,checkpoint,mode=method)
                elif actor is None:
                    actor=LocalTransformersEndpoint(model=actor_model,device='cuda:0')
                    if method=='ma2p_repaired':
                        from generate_ma2p_recovery import SafeSchemaFactory
                        factory=SafeSchemaFactory(actor.tokenizer)
                    else:factory=None
                    if args.family=='deepseek_r1_qwen14b':actor.disable_thinking=True
                def scene(i):
                    point=public_point(points[i]);history=[];plans=[];trajectory=[];started=time.time()
                    if method=='our_direct':
                        candidates=points[i].get('strategies')
                        if not isinstance(candidates,list) or not candidates or any(not isinstance(x,str) or not x.strip() for x in candidates):
                            raise ValueError(f'Invalid candidate strategy list at index {i}')
                        point['our_candidate_strategies']=list(candidates)
                    seed=scenario_seed(42,point['scenario_id'],'target_grounded_v4')
                    if method in ['our_analysis','our_direct']:
                        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):live=actor.runner.initialize(point)
                    for r in range(1,6):
                        if method in ['our_analysis','our_direct']:spoken=yield from actor.turn(point,live,history,seed+100*r)
                        else:spoken=yield from base_turn(actor,factory,method,point,history,plans,r,seed+100*r)
                        if method=='our_direct':spoken['actor_input_prompt']=actor.bridge.role_prompt
                        spoken['round']=r;trajectory.append(spoken);history.append({'role':'persuader','content':spoken['content']})
                        if method in ['our_analysis','our_direct']:
                            with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):actor.runner.consume(live,point,spoken)
                        prompt_builder=goal_blind_persuadee_prompt if args.persuadee_goal_blind else user_prompt
                        system,prompt=prompt_builder(point,history);attempts=[]
                        for attempt in range(3):
                            raw,usage=yield Request(user,system,prompt,seed+2*r+1000*attempt,point)
                            attempts.append({'raw_output':raw,**usage})
                            try:reply=speech_boundary(raw,point,'persuadee')
                            except ValueError:continue
                            reply.update(round=r,attempts=attempts,**usage);break
                        else:raise ValueError(f'Persuadee public boundary retries exhausted: index={i}, round={r}, attempts={attempts!r}')
                        trajectory.append(reply);history.append({'role':'persuadee','content':reply['content']})
                        if method in ['our_analysis','our_direct']:
                            with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):actor.runner.consume(live,point,reply)
                    return {'index':i,'scenario_id':points[i]['scenario_id'],'scenario':points[i]['scenario'],'method':method,
                        'initial_history':[],'reference':points[i]['reference'],'trajectory':trajectory,'final_history':history,
                        'rounds':5,'run_hash':run_hash,'checkpoint_id':meta['checkpoint_id'] if method in ['our_analysis','our_direct'] else None,
                        'persuader_model':str(checkpoint) if method in ['our_analysis','our_direct'] else actor_model,
                        'persuadee_model':cfg.persuadee_model,'compute_dtype':'bfloat16','quantization':'none',
                        'input_policy':INPUT_POLICY,'candidate_strategies_visible':method=='our_direct','candidate_strategies':point.get('our_candidate_strategies',[]),'generation_origin':'new_goal_visible_scenario','execution_seed':seed,'elapsed_seconds':time.time()-started,'batch_size_requested':2}

                def save(row):
                    append_jsonl(path,row);existing[row['index']]=row
                    atomic_json(args.output/f'{method}_progress.json',{'stage':'generation','method':method,'saved':len(existing),
                        'target':args.limit,'last_index':row['index'],'seconds_per_scenario':row['elapsed_seconds'],'pid':os.getpid(),
                        'batch_size_requested':2})
                    print(method,len(existing),'/',args.limit,'index',row['index'],round(row['elapsed_seconds'],1),'seconds; batch2',flush=True)

                for offset in range(0,len(todo),2):
                    indexes=todo[offset:offset+2]
                    print('batch start',method,indexes,flush=True)
                    def trace(request,response):
                        raw,usage=response
                        append_jsonl(args.output/'generation'/f'{method}_request_trace.jsonl',{
                            'scenario_id':request.point['scenario_id'],'seed':request.seed,'system':request.system,'prompt':request.prompt,
                            'raw_output':raw,'usage':usage,'run_hash':run_hash})
                    drive([scene(i) for i in indexes],save,trace)
                atomic_json(args.output/'generation'/f'{method}_summary.json',{'n':len(existing),'complete':len(existing)==192,
                    'generation_sha256':digest(path),'run_hash':run_hash})
        finally:
            if actor is not None:actor.close()
            if user is not None:user.close()
            gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
