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
REPAIR_CODE=ROOT_CODE/'model_family_jobs/llama_public_repair_v3'
sys.path.insert(0,str(REPAIR_CODE))
from simulator_rpc import RemoteSimulator,install_remote_batching
install_remote_batching()
from public_speech import actor_prompt as public_actor_prompt,SPEECH_SYSTEM,validate_public,retry_prompt
from batching import Request,drive
from protocol import REVISION,DECODE,PUBLIC_RULE,ANALYSIS_SCHEMA,SchemaFactory,public_point,speech_boundary,narration,strict_json

METHODS=['our_direct']
PLAN_SCHEMA={'type':'object','properties':{k:{'type':'string','minLength':1,'maxLength':180} for k in ['concern','reason','next_step']},'required':['concern','reason','next_step'],'additionalProperties':False}


def header(point):
    s=point['scenario']
    return f"Background: {s['background']}\nExact persuasion goal: {s['goal']}\nPersuader: {s['persuader']}\nPersuadee: {s['persuadee']}"


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
        attempts=[]
        prompt=public_actor_prompt(point,history)
        for attempt in range(12):
            retry=prompt if attempt<3 else retry_prompt(prompt,point)
            raw,usage=yield Request(endpoint,SPEECH_SYSTEM,retry,seed+1000*attempt,point)
            attempt_record={'raw_output':raw,**usage}
            attempts.append(attempt_record)
            try:
                result=validate_public(raw,point)
            except ValueError as exc:
                attempt_record['format_rejection']=str(exc)
                continue
            result.update(raw_output=raw,attempts=attempts,model_calls=attempt+1,actor_input_prompt=retry,**usage)
            return result
        raise ValueError('Public speech retries exhausted; preserve trace, never accept narration')
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
        # User-defined zero-shot: background and observed history only, no goal or annotations.
        rule=PUBLIC_RULE.replace('and advance the exact persuasion goal ', '')
        prompt=f"Background: {point['scenario']['background']}\nObserved conversation:\n{transcript(history)}\n\n{rule}"
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
        from eclp_elpo import BF16
        from repro.simulation import transcript
        assert self.mode=='our_direct'
        note_prompt=header(point)+'\nObserved public dialogue:\n'+transcript(history)
        note_prompt+=('\nPrepare a concise PRIVATE decision note for the next reply. '
                      'Identify the concern supported by the latest public reply; use unknown if unsupported. '
                      'Give one relevant reason based only on established facts and one feasible next step toward the goal. '
                      'Do not invent reactions, evidence, psychological labels or commitments. '
                      'Use at most 60 words total. Return only JSON: concern, reason, next_step. '
                      'This note is not spoken to the other person.')
        private_attempts=[]
        note=None
        for attempt in range(3):
            self.cfg.actor_system_prompt='Prepare a short private decision note from the supplied public facts.'
            self.bridge.role_prompt=note_prompt
            with torch.no_grad(),torch.autocast(device_type='cuda',dtype=BF16):
                prefix,ctx,_=self.runner.context(point,live)
                action=self.runner.heads.policy(ctx)
                emb=torch.cat([prefix,self.runner.heads.strategy_tokens(action)],1)
                prompt_ids=torch.tensor([self.runner.prompt_ids(point,live.history)],device=self.runner.device)
            raw,usage=yield Request(self.runner,'','',seed+500000+1000*attempt,point,
                                   PLAN_SCHEMA,self.factory,embeddings=emb,prompt_ids=prompt_ids)
            record={'raw_output':raw,**usage};private_attempts.append(record)
            try:
                value=strict_json(raw)
                if set(value)!=set(PLAN_SCHEMA['properties']):raise ValueError('Missing note fields')
                if any(not isinstance(v,str) or not v.strip() for v in value.values()):raise ValueError('Empty note')
                if sum(len(v.split()) for v in value.values())>80:raise ValueError('Private note too long')
                note=value;break
            except (ValueError,TypeError) as exc:record['format_rejection']=str(exc)
        if note is None:raise ValueError('Private-note format exhausted; no silent one-stage fallback')
        prompt=public_actor_prompt(point,history)
        prompt+='\n\nPrivate decision note (guidance only, never established evidence): '+json.dumps(note,ensure_ascii=False)
        prompt+=('\nUse the note only where it matches the actual dialogue. Now speak directly to the other person. '
                 'Include the relevant reason naturally in your spoken reply, and address their latest concern. '
                 'Output only the actual spoken reply; never quote or describe the private note.')
        self.cfg.actor_system_prompt=SPEECH_SYSTEM
        attempts=[]
        for attempt in range(12):
            retry=prompt if attempt<3 else retry_prompt(prompt,point)
            self.bridge.role_prompt=retry
            with torch.no_grad(),torch.autocast(device_type='cuda',dtype=BF16):
                prefix,ctx,_=self.runner.context(point,live)
                action=self.runner.heads.policy(ctx)
                emb=torch.cat([prefix,self.runner.heads.strategy_tokens(action)],1)
                prompt_ids=torch.tensor([self.runner.prompt_ids(point,live.history)],device=self.runner.device)
            raw,usage=yield Request(self.runner,'','',seed+1000*attempt,point,
                                   embeddings=emb,prompt_ids=prompt_ids)
            record={'raw_output':raw,**usage};attempts.append(record)
            try:result=validate_public(raw,point)
            except ValueError as exc:
                record['format_rejection']=str(exc);continue
            result.update(raw_output=raw,analysis=None,private_note=note,private_note_attempts=private_attempts,
                          private_note_visible_to_simulator_or_judge=False,attempts=attempts,
                          model_calls=len(private_attempts)+len(attempts),public_model_calls=len(attempts),
                          actor_input_prompt=retry,output_variant='two_stage_private_note_then_public_reply',**usage)
            return result
        raise ValueError('Two-stage public speech retries exhausted')

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
    ap.add_argument('--simulator-url',required=True)
    ap.add_argument('--stop-after',type=int,default=192)
    ap.add_argument('--smoke-two',action='store_true');args=ap.parse_args()
    root=args.repo_root.resolve();sys.path.insert(0,str(root/'my_eclp_final'));sys.path.insert(0,str(root/'exp_code_ljj'))
    from experiment_io import atomic_json,immutable_json,digest,object_hash,read_json,read_jsonl,append_jsonl,writer_lock
    from eclp_elpo import Config,verify_checkpoint,scenario_seed,seed_all,independent_rng
    import peft.tuners.lora.model as lora_dispatch
    lora_dispatch.is_bnb_available=lambda: False
    lora_dispatch.is_bnb_4bit_available=lambda: False
    from repro.simulation import LocalTransformersEndpoint,user_prompt
    from repro.gpu_capacity import wait_for_bf16_capacity
    import hashlib,torch
    selection=read_json(root/'exp_code_ljj/data/test_selection_192.json');points=read_jsonl(root/'exp_code_ljj/data/evaluation_points.jsonl')
    assert selection['original_indexes']==[i for i,p in enumerate(points) if p['target_turn']==1 and p['history']==[]]
    assert digest(root/'exp_code_ljj/data/evaluation_points.jsonl')==selection['source_sha256']
    order=sorted(selection['original_indexes'],key=lambda i:hashlib.sha256(('matched-v2-pilot-42:'+points[i]['scenario_id']).encode()).hexdigest())
    assert args.limit in [20,192]
    cfg=Config.load(args.config);cfg.repo_root=str(root);cfg.actor_device=cfg.persuadee_device='cuda:0'
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
    contract={'revision':REVISION,'output_variant':'two_stage_private_note_then_public_reply','planning_visibility':'private archive only; public history and Judge see only content','plan_schema':PLAN_SCHEMA,'extra_inference_cost':'one bounded private-note call per turn plus final public reply','expected_indexes':order,'pilot_indexes':order[:20],'checkpoint_id':meta['checkpoint_id'],
        'checkpoint':str(checkpoint) if checkpoint else None,'family':args.family,'actor_model':actor_model,'clean_inheritance':source_provenance,'models':models,'decode':DECODE,'seed':42,'physical_gpu':args.physical_gpu,
        'compute_dtype':'bfloat16','quantization':'none','public_policy':'first utterance only, before history, latent feedback and Judge',
        'method':args.method,'fresh_generation':args.fresh_generation,'simulator_prompt_sha256':digest(root/'exp_code_ljj/repro/simulation.py'),'dialogue_prompt_sha256':digest(root/'exp_code_ljj/repro/persuadee_simulation.py'),'our_interface':'full ECLP checkpoint; analysis archived or direct speech, depending on method',
        'persuadee_goal_blind':args.persuadee_goal_blind,
        'persuadee_prompt_policy':('background and observed public dialogue only; exact goal withheld' if args.persuadee_goal_blind else 'shared legacy simulator prompt'),
        'our_output_retry_policy':'private note maximum3; public maximum12; first_valid; no_judge_selection','our_max_prompt_tokens':6144,'our_analysis_visibility':'raw archive only; utterance enters public history',
        'ma2p':'Perception -> World Model -> Persuader, K=0, strict schemas, no silent fallback',
        'zero_shot_inputs':'background and observed public history only; no goal, strategy, psychology or golden reply',
        'source_sha256':digest(Path(__file__)),'protocol_sha256':digest(REPAIR_CODE/'protocol.py'),'public_speech_sha256':digest(REPAIR_CODE/'public_speech.py'),'shared_simulator_transport':'local Unix socket; same seeded sampler; runtime endpoint in launch records','shared_simulator_gpu':4,
        'eclp_source_sha256':digest(root/'my_eclp_final/eclp_elpo.py'),'points_sha256':selection['source_sha256'],
        'selection_sha256':digest(root/'exp_code_ljj/data/test_selection_192.json'),'judge_used_during_generation':False,
        'batch_size':2,'batching_sha256':digest(ROOT_CODE/'seven_methods_192/batching.py'),
        'sampling_implementation':'per-row seeded RNG, HF temperature/top-k/top-p warpers; batch numeric rounding can differ',
        'inherited_generation':({'path':str(args.inherit_dir.resolve()),'contract_sha256':digest(args.inherit_dir/'generation_run.json'),
            'source_files_sha256':{p.name:digest(p) for p in sorted((args.inherit_dir/'generation').glob('*.jsonl'))}} if args.inherit_dir else None)}
    if args.validate_only:print(json.dumps({'validated':True,'pilot_indexes':order[:20],'checkpoint_id':meta['checkpoint_id']}));return
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(args.physical_gpu)
    torch.cuda.set_per_process_memory_fraction(args.memory_cap_gib*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
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
                assert 1<=args.stop_after<=192
                todo=[i for i in order[:(2 if args.smoke_two else args.stop_after)] if i not in existing]
                if not todo:continue
                if user is None:
                    wait_for_bf16_capacity([cfg.base_model],[0],reserve_gib=4,description=f'{args.family} actor on physical GPU{args.physical_gpu}; shared simulator on GPU4')
                    seed_all(42);user=RemoteSimulator(args.simulator_url)
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
                    seed=scenario_seed(42,point['scenario_id'],'target_grounded_v4')
                    if method in ['our_analysis','our_direct']:
                        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):live=actor.runner.initialize(point)
                    for r in range(1,6):
                        if method in ['our_analysis','our_direct']:spoken=yield from actor.turn(point,live,history,seed+100*r)
                        else:spoken=yield from base_turn(actor,factory,method,point,history,plans,r,seed+100*r)
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
                        'generation_origin':'new_two_stage_private_note_scenario','output_variant':'two_stage_private_note_then_public_reply','execution_seed':seed,'elapsed_seconds':time.time()-started,'batch_size_requested':2}

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
