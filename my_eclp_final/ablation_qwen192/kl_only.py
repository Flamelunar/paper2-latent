"""Exact KL-only ablation with verified, policy-independent cached contexts."""
import copy,json,random,sys,os
from pathlib import Path
import torch
R=Path('/app/hzj/liujianjian/ljj-2-paper');sys.path.insert(0,str(R/'my_eclp_final'))
from eclp_elpo import Config,StrategyPolicy,BF16,FP32MasterAdamW,equal_covariance_kl,load_tensors,tensor_digest,verify_checkpoint,seed_all
from experiment_io import read_json,digest,atomic_json,append_jsonl
A=R/'exp_code_ljj/artifacts/qwen25_ablation_192_20261010';O=A/'wo_r_kl_only';O.mkdir(exist_ok=True)
assert os.environ['CUDA_VISIBLE_DEVICES']=='4'
cfg=Config.load(R/'my_eclp_final/config_gpu4.json');seed_all(cfg.seed)
sft=R/'eclp_runs_gpu4/sft_full/checkpoint-007484';full=R/'eclp_runs_gpu4/offline_rl/checkpoint-001800';lock=read_json(R/'eclp_runs_gpu4/protocol.lock.json')
a=verify_checkpoint(sft,lock);b=verify_checkpoint(full,lock);assert a['stage']=='sft' and a['step']==7484;assert a['executor_hash']==b['executor_hash']
state=load_tensors(sft/'heads.pt');reference=load_tensors(sft/'reference_policy.pt');policy_state={k[len('policy.'):]:v for k,v in state.items() if k.startswith('policy.')};assert tensor_digest(policy_state)==tensor_digest(reference)
hidden=read_json(Path(cfg.base_model)/'config.json')['hidden_size'];policy=StrategyPolicy(cfg,hidden).to(device='cuda:0',dtype=BF16);policy.load_state_dict(policy_state,strict=True);policy.eval();ref=copy.deepcopy(policy).eval().requires_grad_(False)
initial={k:v.detach().clone() for k,v in policy.state_dict().items()};initial_hash=tensor_digest(initial)
optimizer=FP32MasterAdamW(policy.parameters(),lr=1e-5,weight_decay=0);assert not optimizer.state
plan=read_json(R/'eclp_runs_gpu4/offline_rl/continuation_manifest.json')['continuation_plan'];assert plan['policy_update_epochs']==2 and plan['policy_update_batch_size']==64
# Replay inputs depend exclusively on observed training history, frozen LoRA,
# state network and readouts; ECLPRunner.context never reads heads.policy.
# SFT and final executor hashes are identical, allowing reuse of these contexts.
blocks=[]
for p in (R/'eclp_runs_gpu4/offline_rl/blocks').glob('block-*/attempt-*/commit.json'):
 commit=read_json(p);run=read_json(p.parent/'run.json');blocks.append((run['cursor'],run['end'],p.parent,commit,run))
blocks.sort(key=lambda v:v[0]);cursor=0;updates=0;max_kl=max_grad=max_change=0.;seen=[];sources=[]
for start,end,path,commit,run in blocks:
 if start<cursor:continue
 assert start==cursor
 assert commit['parent_sft_checkpoint_id']==a['checkpoint_id'] and commit['executor_unchanged'] is True
 assert digest(path/'batch.pt')==commit['block_batch_sha256']
 data=load_tensors(path/'batch.pt');contexts=data['context'];chol=data['chol'];n=end-start
 assert len(contexts)==n*plan['candidates'];assert len(run['scenario_ids'])==n
 assert torch.equal(contexts.reshape(n,4,*contexts.shape[1:])[:,0],contexts.reshape(n,4,*contexts.shape[1:])[:,3])
 sources.append({'path':str(path/'batch.pt'),'sha256':commit['block_batch_sha256'],'cursor':start,'end':end});seen+=run['scenario_ids']
 for epoch in range(2):
  order=list(range(len(contexts)));random.Random(cfg.seed+start*7+epoch).shuffle(order)
  for offset in range(0,len(order),64):
   ids=order[offset:offset+64];ctx=contexts[ids].to('cuda:0');ch=chol[ids].to('cuda:0');before={k:v.detach().clone() for k,v in policy.state_dict().items()}
   optimizer.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=BF16):mean=policy(ctx);anchor=ref(ctx)
   kl=equal_covariance_kl(mean,anchor,ch);loss=cfg.kl_beta*kl.mean();loss.backward()
   grad=float(torch.nn.utils.clip_grad_norm_(list(policy.parameters()),1.0));optimizer.step()
   change=max(float((v-before[k]).abs().max()) for k,v in policy.state_dict().items());max_kl=max(max_kl,float(kl.max()));max_grad=max(max_grad,grad);max_change=max(max_change,change)
   assert float(kl.max())<=cfg.max_kl
   assert change==0 and grad==0 and float(loss)==0,'KL-only has nonzero update; do not reuse w/o G'
   append_jsonl(O/'updates.jsonl',{'update':updates,'context_cursor':start,'epoch':epoch,'kl':float(kl.mean()),'total_loss':float(loss),'policy_gradient_norm':grad,'parameter_max_difference':change,'task_reward_enabled':False,'nll_enabled':False,'correction_bonus_enabled':False,'policy_ratio_loss_enabled':False,'behavior_kl':0.,'reference_frozen':True})
   updates+=1
 cursor=end
 print('KL-only',cursor,'/1800; updates',updates,flush=True)
assert cursor==1800 and len(set(seen))==1800
assert tensor_digest(policy.state_dict())==initial_hash==tensor_digest(ref.state_dict())
for group in optimizer.param_groups:
 assert group['weight_decay']==0
for p,st in optimizer.state.items():assert torch.count_nonzero(st['exp_avg'])==0 and torch.count_nonzero(st['exp_avg_sq'])==0 and torch.equal(st['master'],p.float())
assert digest(sft/'heads.pt')==a['files']['heads.pt']
atomic_json(O/'diagnostic.json',{'complete':True,'contexts':cursor,'optimizer_updates':updates,'initial_sft_checkpoint':str(sft),'checkpoint_id':a['checkpoint_id'],'checkpoint_files':a['files'],'policy_initial_hash':initial_hash,'policy_final_hash':tensor_digest(policy.state_dict()),'reference_hash':tensor_digest(ref.state_dict()),'executor_hash':a['executor_hash'],'same_frozen_executor_as_context_source':True,'context_policy_independent':True,'context_code_sha256':digest(R/'my_eclp_final/eclp_elpo.py'),'cached_context_sources':sources,'training_plan':plan,'kl_estimator':'exact equal-covariance Gaussian KL(pi||SFT)','task_rewards_disabled':True,'loss':'beta * mean exact KL','max_kl':max_kl,'max_gradient_norm':max_grad,'max_parameter_change':max_change,'weight_decay':0,'optimizer_momentum_at_initialization':'fresh zero','dropout':0,'equivalent_to_wo_g':True,'training_generated_tokens':0,'simulator_calls':0,'judge_calls':0,'reuse_reason':'identical SFT parameters after full 1800-context KL-only budget; exact KL zero for every context and zero optimizer drift'})
