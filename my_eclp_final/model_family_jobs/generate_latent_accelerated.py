"""Native LatentRATT token loop with a preserved public context boundary."""
import argparse,gc,os,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(CODE/'seven_methods_192'),str(CODE),str(CODE.parent/'baseline5_latentRATT'),str(CODE.parent/'exp_code_ljj')]
from protocol import PUBLIC_RULE,speech_boundary,narration,public_point,DECODE
from generate import sample_local
from model_family_jobs.llama_public_repair_v3.public_speech import narration

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--physical-gpu',type=int,required=True);p.add_argument('--smoke-two',action='store_true');p.add_argument('--family',choices=['meta_llama31','deepseek_r1_qwen14b'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--memory-cap-gib',type=float,default=54);p.add_argument('--persuadee-goal-blind',action='store_true');a=p.parse_args()
 root=a.repo_root;sys.path[:0]=[str(root/'baseline5_latentRATT'),str(root/'exp_code_ljj'),str(root/'my_eclp_final')]
 import torch
 from run_latent_ratt import load_model
 from direct_answer_prompt import direct_answer_ids
 from experiment_io import read_json,read_jsonl,digest,object_hash,immutable_json,atomic_json,append_jsonl,writer_lock
 from eclp_elpo import independent_rng,scenario_seed
 from repro.simulation import LocalTransformersEndpoint,user_prompt
 from repro.persuadee_simulation import dialogue_prompt
 from repro.gpu_capacity import wait_for_bf16_capacity
 assert os.environ['CUDA_VISIBLE_DEVICES']==str(a.physical_gpu)
 checkpoint=a.checkpoint
 base={'meta_llama31':'/app/hzj/models/Meta-Llama-3.1-8B-Instruct','deepseek_r1_qwen14b':'/app/hzj/models/DeepSeek-R1-Distill-Qwen-14B'}[a.family];sim='/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged'
 train=read_json(checkpoint/'run_config.json');assert train['stage']=='rl' and train['training_complete'] and train['model']==base and train['train_rows']==59868
 selection=read_json(root/'exp_code_ljj/data/test_selection_192.json');points=read_jsonl(root/'exp_code_ljj/data/evaluation_points.jsonl')
 assert digest(root/'exp_code_ljj/data/evaluation_points.jsonl')==selection['source_sha256']
 import hashlib
 order=sorted(selection['original_indexes'],key=lambda i:hashlib.sha256(('matched-v2-pilot-42:'+points[i]['scenario_id']).encode()).hexdigest())
 contract={'method':'latentratt','expected_indexes':order,'checkpoint':str(checkpoint),'adapter_sha256':digest(checkpoint/'adapter/adapter_model.safetensors'),'attention_sha256':digest(checkpoint/'attention.pt'),'training_config':train,'base':base,'persuadee':sim,'physical_gpu':a.physical_gpu,'persuadee_goal_blind':a.persuadee_goal_blind,'decode':DECODE,'seed':42,'max_prompt_tokens':1536,'history_policy':'drop oldest complete pairs only; preserve background, goal and latest pair','native_actor_batch_size':1,'source_sha256':digest(Path(__file__)),'protocol_sha256':digest(CODE/'seven_methods_192/protocol.py'),'native_source_sha256':digest(root/'baseline5_latentRATT/latent_model.py'),'public_boundary_checker_sha256':digest(CODE/'model_family_jobs/llama_public_repair_v3/public_speech.py')}
 with writer_lock(a.output/'.generation.lock'):
  immutable_json(a.output/'generation_run.json',contract);run_hash=object_hash(contract)
  path=a.output/'generation/latentratt.jsonl';saved=read_jsonl(path) if path.exists() else [];existing={r['index']:r for r in saved};assert len(existing)==len(saved)
  for i,r in existing.items():assert i in order and r['run_hash']==run_hash and r['initial_history']==[] and len(r['trajectory'])==10
  todo=[i for i in order[:2] if i not in existing] if a.smoke_two else [i for i in order if i not in existing]
  if not todo:return
  torch.cuda.set_per_process_memory_fraction(a.memory_cap_gib*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
  wait_for_bf16_capacity([base,sim],[0],reserve_gib=8,description='Seven methods LatentRATT')
  import peft.tuners.lora.model as lora
  lora.is_bnb_available=lambda:False;lora.is_bnb_4bit_available=lambda:False
  tokenizer,model=load_model(base,stage='eval',checkpoint=checkpoint,device='cuda:0');model.backbone.config.use_cache=True
  user=LocalTransformersEndpoint(model=sim,device='cuda:0')
  try:
   for i in todo:
    point=public_point(points[i]);assert points[i]['history']==[] and points[i]['target_turn']==1
    history=[];trajectory=[];seed=scenario_seed(42,point['scenario_id'],'target_grounded_v4');start=time.time()
    for r in range(1,6):
     visible=list(history);dropped=0
     while True:
      prompt=dialogue_prompt(point,visible)+'\n'+PUBLIC_RULE+'\nOutput only one spoken utterance, with no analysis or future replies.'
      ids=direct_answer_ids(tokenizer,[{'role':'system','content':'You are a skilled persuader.'},{'role':'user','content':prompt}],model_hint=base)
      if len(ids)<=1536:break
      if len(visible)<=2:raise ValueError(f'Native context cannot fit background/goal/latest pair: index={i}, round={r}')
      visible=visible[2:];dropped+=2
     attempts=[]
     for attempt in range(3):
      with independent_rng(seed+100*r+1000*attempt,[0]),torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
       tokens=model.generate_text(torch.tensor([ids],device=model.device),max_new_tokens=512,eos_token_id=tokenizer.eos_token_id,temperature=.95,top_p=.7,top_k=50)
      raw=tokenizer.decode(tokens,skip_special_tokens=True).strip();attempts.append({'raw_output':raw})
      try:
       spoken=speech_boundary(raw,point,'persuader')
       if narration(spoken['content'],point):raise ValueError('Native actor emitted analysis')
      except ValueError:continue
      break
     else:raise ValueError(f'LatentRATT public boundary failed: index={i}, round={r}')
     spoken.update(round=r,attempts=attempts,input_tokens=len(ids),dropped_history_turns=dropped)
     trajectory.append(spoken);history.append({'role':'persuader','content':spoken['content']})
     if a.persuadee_goal_blind:
      from generate_family import goal_blind_persuadee_prompt
      system,prompt=goal_blind_persuadee_prompt(point,history)
     else:system,prompt=user_prompt(point,history)
     attempts=[]
     for attempt in range(3):
      raw,usage=sample_local(user,system,prompt,seed+2*r+1000*attempt,point);attempts.append({'raw_output':raw,**usage})
      try:reply=speech_boundary(raw,point,'persuadee')
      except ValueError:continue
      reply.update(round=r,attempts=attempts,**usage);break
     else:raise ValueError(f'Persuadee boundary failed: index={i}, round={r}')
     trajectory.append(reply);history.append({'role':'persuadee','content':reply['content']})
    row={'index':i,'scenario_id':points[i]['scenario_id'],'scenario':points[i]['scenario'],'method':'latentratt','initial_history':[],'reference':points[i]['reference'],'trajectory':trajectory,'final_history':history,'rounds':5,'run_hash':run_hash,'persuader_model':str(checkpoint),'persuadee_model':sim,'compute_dtype':'bfloat16','quantization':'none','elapsed_seconds':time.time()-start}
    append_jsonl(path,row);existing[i]=row
    atomic_json(a.output/'latentratt_progress.json',{'saved':len(existing),'target':192,'last_index':i,'pid':os.getpid()})
    print('latentratt',len(existing),'/192',i,round(row['elapsed_seconds'],1),flush=True)
   atomic_json(a.output/'generation/latentratt_summary.json',{'n':len(existing),'complete':len(existing)==192,'generation_sha256':digest(path)})
  finally:user.close();del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
