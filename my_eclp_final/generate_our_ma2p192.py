#!/usr/bin/env python3
"""Original MA2P orchestration with one trained Our backend for all three roles."""
from __future__ import annotations
import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from experiment_io import atomic_json,append_jsonl,digest,immutable_json,object_hash,read_json,read_jsonl,writer_lock

GENERATOR_SOURCE_SHA256=digest(Path(__file__))


def selected_points(root):
    manifest_path=root/'exp_code_ljj/data/test_selection_192.json'
    m=read_json(manifest_path)
    points=read_jsonl(root/'exp_code_ljj/data/evaluation_points.jsonl')
    indexes=[i for i,p in enumerate(points) if p['target_turn']==1 and p['history']==[]]
    if indexes!=m['original_indexes'] or len(indexes)!=192 or digest(root/'exp_code_ljj/data/evaluation_points.jsonl')!=m['source_sha256']:
        raise ValueError('The fixed 192-scenario selection changed')
    return [(i,points[i]) for i in indexes],m


def public_point(point):
    return {'scenario_id':point['scenario_id'], 'scenario':{k:point['scenario'][k] for k in ('background','goal','persuader','persuadee')}}


NEXT_ROLE = re.compile(r'(?im)\n[ \t]*(?:user|assistant|system|persuader|persuadee)[ \t]*(?:[:：]|\r?\n)')


def first_public_reply(raw):
    from repro.public_answer import strip_thinking
    visible,filtered=strip_thinking(raw)
    boundary=NEXT_ROLE.search(visible)
    first=visible[:boundary.start()].strip() if boundary else visible
    if not first:
        raise ValueError('Persuadee emitted no first public reply')
    return first,filtered,bool(boundary)


def is_explicit_plan(raw,point):
    persuader=re.escape(point['scenario']['persuader'])
    persuadee=re.escape(point['scenario']['persuadee'])
    if re.search(rf'(?i)(?:^|\W)(?:{persuader}|the persuader)\s+(?:will|should|may|might|can|plans? to)\b',raw):
        return True
    return bool(re.match(rf'(?i)^\s*(?:{persuadee}|the persuadee)\s+(?:is|may|might|seems|appears|has|needs|wants)\b',raw))


class TrainedECLPEndpoint:
    """Switch only the role prompt; keep actual dialogue state separate from plans."""
    def __init__(self,runner,bridge,cfg):
        self.runner,self.bridge,self.cfg=runner,bridge,cfg
        self.point=self.live=None
        self.seed=0;self.calls=[]
        self.tokenizer_data=None
    def analysis_constraint(self,system):
        # Restrict JSON syntax; the full ECLP path still generates every value.
        sys.path.insert(0,str(Path(__file__).with_name('our_ma2p_deps')))
        from lmformatenforcer import JsonSchemaParser,TokenEnforcer,TokenEnforcerTokenizerData
        if self.tokenizer_data is None:
            tokenizer=self.runner.tokenizer
            zero=tokenizer.encode('0')[-1]
            special=set(tokenizer.all_special_ids)
            regular=[]
            for idx in range(len(tokenizer)):
                if idx in special:
                    continue
                after_zero=tokenizer.decode([zero,idx])[1:]
                decoded=tokenizer.decode([idx])
                regular.append((idx,after_zero,len(after_zero)>len(decoded)))
            def decode(ids):
                return tokenizer.decode(ids).rstrip('\ufffd')
            self.tokenizer_data=TokenEnforcerTokenizerData(regular,decode,tokenizer.eos_token_id,False,len(tokenizer))
        keys=('preventive','generative') if "Perception agent" in system else ('likely_reaction','strategy')
        schema={'type':'object','properties':{k:{'type':'string','minLength':1,'maxLength':300} for k in keys},
                'required':list(keys),'additionalProperties':False}
        enforcer=TokenEnforcer(self.tokenizer_data,JsonSchemaParser(schema))
        return keys,lambda batch_id,sent:enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens
    def constrained_speak(self,seed,prefix_allowed_tokens_fn):
        # Mirror ECLPRunner.speak's trained state/action path without changing
        # the checkpoint-bound implementation file.
        from eclp_elpo import BF16,PromptRepetitionPenalty,independent_rng
        import torch
        runner=self.runner
        prefix,ctx,_=runner.context(self.point,self.live)
        action=runner.heads.policy(ctx)
        embeddings=torch.cat([prefix,runner.heads.strategy_tokens(action)],1)
        stop=runner.llm.generation_config.eos_token_id or runner.tokenizer.eos_token_id
        penalty=float(runner.llm.generation_config.repetition_penalty)
        prompt_ids=torch.tensor([runner.prompt_ids(self.point,self.live.history)],device=runner.device)
        processors=[PromptRepetitionPenalty(prompt_ids,penalty)] if penalty!=1.0 else []
        with independent_rng(seed,[runner.device.index]),torch.autocast(device_type='cuda',dtype=BF16):
            generated=runner.llm.generate(
                inputs_embeds=embeddings,attention_mask=torch.ones(embeddings.shape[:2],device=runner.device,dtype=torch.long),
                max_new_tokens=self.cfg.max_new_tokens,do_sample=self.cfg.temperature>0,
                temperature=self.cfg.temperature,top_p=self.cfg.top_p,top_k=self.cfg.top_k,
                repetition_penalty=1.0,logits_processor=processors,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                eos_token_id=stop,pad_token_id=runner.tokenizer.pad_token_id,
                use_cache=True,return_dict_in_generate=True)
        ids=generated.sequences[0]
        if len(ids)>self.cfg.max_new_tokens:
            raise RuntimeError('Transformers inputs_embeds generation contract changed')
        return {'raw_output':runner.tokenizer.decode(ids,skip_special_tokens=True).strip(),
                'input_tokens_including_latents':embeddings.shape[1],'public_tokens':len(ids)}

    def complete(self,*,system,user,**decode):
        if any(decode.get(k)!=v for k,v in [('temperature',0.95),('top_p',0.7),('top_k',50),('max_tokens',512)]):
            raise ValueError('All role calls must keep the original MA2P decoding settings')
        import torch
        is_analysis=system in ("You are MA2P's Perception agent. Infer cautiously from observable evidence.",
                              "You are MA2P's World Model. Plan from public context and short-term memory only.")
        keys,prefix=self.analysis_constraint(system) if is_analysis else (None,None)
        call={'system':system,'user':user,'decode':decode,
              'checkpoint_interface':'full_eclp_latent_json_schema' if is_analysis else 'full_eclp_latent_actor'}
        self.calls.append(call)
        self.bridge.role_prompt=user
        original_system=self.cfg.actor_system_prompt
        self.cfg.actor_system_prompt=system
        try:
            last_raw=''
            last_public=''
            for attempt in range(3):
                with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    if is_analysis:
                        turn=self.constrained_speak(self.seed+len(self.calls)+1000*attempt,prefix)
                    else:
                        turn,_,_=self.runner.speak(self.point,self.live,self.seed+len(self.calls)+1000*attempt)
                raw=turn['raw_output']
                last_raw=raw
                call['input_tokens_including_latents']=turn['input_tokens_including_latents']
                call['generated_tokens']=turn['public_tokens']
                if not is_analysis:
                    last_public=turn['content']
                    if is_explicit_plan(turn['content'],self.point):
                        call.setdefault('rejected_plan_outputs',[]).append(raw)
                        continue
                    call['output_retries']=attempt
                    call['accepted_raw_output']=raw
                    call['accepted_public_content']=turn['content']
                    return raw
                try:
                    obj=json.loads(raw)
                except json.JSONDecodeError:
                    call.setdefault('invalid_json_outputs',[]).append(raw)
                    continue
                if set(obj)==set(keys) and all(isinstance(obj[k],str) and obj[k].strip() for k in keys):
                    call['format_retries']=attempt
                    return raw
                call.setdefault('invalid_json_outputs',[]).append(raw)
            call['retry_exhausted']='analysis_format' if is_analysis else 'plan_narration'
            if not is_analysis:
                call['output_retries']=2
                call['accepted_raw_output']=last_raw
                call['accepted_public_content']=last_public
            else:
                call['format_retries']=2
            return last_raw
        finally:
            self.cfg.actor_system_prompt=original_system
            self.bridge.role_prompt=None


def generate(args):
    root=args.repo_root.resolve()
    sys.path.insert(0,str(root/'exp_code_ljj'))
    from baseline4_ma2p import ma2p_turn,DECODE
    from repro.simulation import LocalTransformersEndpoint,user_prompt
    from repro.public_answer import public_turn
    from repro.gpu_capacity import wait_for_bf16_capacity
    from eclp_elpo import Config,BaselineBridge,ECLPRunner,verify_checkpoint,scenario_seed,seed_all,independent_rng
    import torch
    points,selection=selected_points(root)
    cfg=Config.load(root/'my_eclp_final/config_gpu4.json')
    cfg.repo_root=str(root);cfg.actor_device=args.actor_device;cfg.persuadee_device=args.persuadee_device
    lock=read_json(root/'eclp_runs_gpu4/protocol.lock.json')
    if object_hash({k:v for k,v in lock.items() if k!='lock_id'})!=lock['lock_id']:
        raise ValueError('Corrupt training checkpoint lock')
    meta=verify_checkpoint(args.checkpoint,lock)
    contract={'method':'our_ma2p','label':'Our + MA2P','selection_revision':selection['revision'],
              'original_indexes':[i for i,_ in points], 'checkpoint':str(args.checkpoint.resolve()),
              'checkpoint_id':meta['checkpoint_id'],'checkpoint_mode':args.checkpoint_mode,
              'all_agents_backend':'same_trained_our_checkpoint',
              'role_interfaces':{'perception':'full_eclp_latent_json_schema' if args.checkpoint_mode=='full_eclp' else 'trained_lora_standard_chat',
                                 'world_model':'full_eclp_latent_json_schema' if args.checkpoint_mode=='full_eclp' else 'trained_lora_standard_chat',
                                 'persuader':'full_eclp_latent_actor' if args.checkpoint_mode=='full_eclp' else 'trained_lora_standard_chat'},
              'analysis_format_constraint':'JSON schema only; free model-generated string values; lm-format-enforcer 0.11.3 core token enforcer',
              'ma2p_configurator_k':0,
              'persuadee_public_reply_policy':'first completed reply only; archive raw continuation and truncation flag',
              'persuader_plan_output_policy':'up to three same-prompt samples; if all narrate, retain final full-ECLP output and flag exhaustion',
              'online_update':False,'judge_used_during_generation':False,
              'compute_dtype':'bfloat16','quantization':'none','seed':cfg.seed,
              'decoding':DECODE,'rounds':5,'persuadee_model':cfg.persuadee_model,
              'actor_device':args.actor_device,'persuadee_device':args.persuadee_device,
              'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'source_sha256':GENERATOR_SOURCE_SHA256,
              'ma2p_source_sha256':digest(root/'exp_code_ljj/baseline4_ma2p.py'),
              'eclp_source_sha256':digest(root/'my_eclp_final/eclp_elpo.py'),
              'simulation_source_sha256':digest(root/'exp_code_ljj/repro/simulation.py'),
              'selection_sha256':digest(root/'exp_code_ljj/data/test_selection_192.json')}
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    with writer_lock(output/'.writer.lock'):
        immutable_json(output/'run.json',contract)
        generation=output/'generation.jsonl'
        saved=read_jsonl(generation) if generation.exists() else []
        existing={r['index']:r for r in saved}
        if len(existing)!=len(saved) or not set(existing).issubset(contract['original_indexes']):
            raise ValueError('Invalid saved generation indexes')
        for i,row in existing.items():
            if row['run_hash']!=object_hash(contract) or row['initial_history']!=[] or row['checkpoint_id']!=meta['checkpoint_id']:
                raise ValueError('Saved hybrid generation contract changed')
        todo=[(i,p) for i,p in points if i not in existing]
        if args.max_new_scenarios:todo=todo[:args.max_new_scenarios]
        if not todo:
            atomic_json(output/'generation_summary.json',{'complete':len(existing)==192,'n':len(existing),'generation_sha256':digest(generation),'run_hash':object_hash(contract)})
            return
        actor_devices=LocalTransformersEndpoint._parse_devices(args.actor_device)
        user_devices=LocalTransformersEndpoint._parse_devices(args.persuadee_device)
        if actor_devices==user_devices:
            wait_for_bf16_capacity([cfg.base_model,cfg.persuadee_model],actor_devices,
                                   reserve_gib=9,description='Our+MA2P actor and fixed Persuadee')
        else:
            wait_for_bf16_capacity([cfg.base_model],actor_devices,reserve_gib=7,description='Our+MA2P trained backend')
            wait_for_bf16_capacity([cfg.persuadee_model],user_devices,reserve_gib=5,description='Our+MA2P fixed Persuadee')
        seed_all(cfg.seed)
        class RoleBridge(BaselineBridge):
            role_prompt=None
            def actor_prompt(self,point,history):
                if self.role_prompt is None:raise RuntimeError('No active MA2P role prompt')
                return self.role_prompt
        bridge=RoleBridge(cfg) if args.checkpoint_mode=='full_eclp' else None
        runner=ECLPRunner(cfg,bridge,args.checkpoint) if bridge else None
        actor=TrainedECLPEndpoint(runner,bridge,cfg) if runner else LocalTransformersEndpoint(model=str(args.checkpoint/'adapter'),device=args.actor_device)
        persuadee=LocalTransformersEndpoint(model=cfg.persuadee_model,device=args.persuadee_device)
        started=time.time();completed_this_run=0
        try:
            for i,point in todo:
                case_start=time.time();public=public_point(point);history=[];plans=[];trajectory=[]
                seed=scenario_seed(cfg.seed,point['scenario_id'],'generation')
                if runner:
                    with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):live=runner.initialize(public)
                for r in range(1,6):
                    if runner:
                        actor.point,actor.live,actor.seed=public,live,seed+100*r
                        actor.calls=[]
                    with independent_rng(seed+100*r,actor_devices):spoken,trace=ma2p_turn(public,history,plans,actor,r)
                    plans.append(trace['world_model'])
                    turn={'round':r,'role':'persuader','content':spoken,'raw_output':spoken,'agent_trace':trace,
                          'prompt_type':'our_ma2p','backend_checkpoint_id':meta['checkpoint_id']}
                    if runner:
                        atomic_json(output/'role_requests'/f'{i:04d}_round{r}.json',{'index':i,'round':r,'calls':actor.calls,'checkpoint_id':meta['checkpoint_id']})
                    trajectory.append(turn);history.append({'role':'persuader','content':spoken})
                    if runner:
                        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):runner.consume(live,public,turn)
                    system,prompt=user_prompt(public,history)
                    with independent_rng(seed+2*(r-1)+1,user_devices):raw=persuadee.complete(system=system,user=prompt,**DECODE)
                    first,thinking_filtered,truncated=first_public_reply(raw)
                    visible=public_turn(first,seed=public,role='persuadee',endpoint=persuadee)
                    if NEXT_ROLE.search(visible['content']):
                        raise ValueError('Persuadee public reply still contains another dialogue turn')
                    visible['thinking_filtered']=visible['thinking_filtered'] or thinking_filtered
                    reply={'round':r,**visible,'raw_output':raw,'future_dialogue_truncated':truncated}
                    trajectory.append(reply);history.append({'role':'persuadee','content':reply['content']})
                    if runner:
                        with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):runner.consume(live,public,reply)
                row={'index':i,'method':'our_ma2p','scenario_id':point['scenario_id'],'source_id':point['source_id'],
                     'session_id':point['session_id'],'target_turn':1,'scenario':point['scenario'],
                     'reference':point['reference'],'initial_history':[],'trajectory':trajectory,'final_history':history,
                     'rounds':5,'persuader_model':str(args.checkpoint.resolve()),'checkpoint_id':meta['checkpoint_id'],
                     'persuadee_model':cfg.persuadee_model,'compute_dtype':'bfloat16','quantization':'none',
                     'execution_seed':seed,'run_hash':object_hash(contract),'elapsed_seconds':time.time()-case_start}
                append_jsonl(generation,row);existing[i]=row;completed_this_run+=1
                elapsed=time.time()-started
                atomic_json(output/'progress.json',{'stage':'generation','saved':len(existing),'expected':192,
                            'last_index':i,'elapsed_seconds':elapsed,'seconds_per_scenario':elapsed/completed_this_run,
                            'estimated_remaining_seconds':(192-len(existing))*elapsed/completed_this_run})
                print(f"Generated {len(existing)}/192; original index={i}; {row['elapsed_seconds']:.1f}s",flush=True)
            atomic_json(output/'generation_summary.json',{'complete':len(existing)==192,'n':len(existing),'generation_sha256':digest(generation),'run_hash':object_hash(contract)})
        finally:
            persuadee.close()
            if runner:runner.close();bridge.close()
            else:actor.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint-mode',choices=['full_eclp','lora_only'],default='full_eclp')
    p.add_argument('--actor-device',default='cuda:0');p.add_argument('--persuadee-device',default='cuda:1')
    p.add_argument('--max-new-scenarios',type=int,default=0)
    a=p.parse_args()
    if a.max_new_scenarios<0:p.error('max-new-scenarios must be nonnegative')
    generate(a)
if __name__=='__main__':main()
