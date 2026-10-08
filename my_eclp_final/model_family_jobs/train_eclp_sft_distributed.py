"""Two GPU replicas; shard each original 8-session update, sum gradients once.

The ECLP architecture, loss, optimizer, shuffled order and checkpoint format
remain those of eclp_elpo.py. Only the execution of each update is parallel.
"""
import os
visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
local_rank=int(os.environ['LOCAL_RANK'])
assert visible==['0','1'] and local_rank in (0,1)
physical_gpu=int(visible[local_rank])
# Each fresh torchrun child exposes one physical GPU before importing Torch.
# This also preserves the single-visible-device RNG checkpoint contract.
os.environ['CUDA_VISIBLE_DEVICES']=str(physical_gpu)
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OMP_NUM_THREADS','4')
import sys,time,json,random,contextlib,hashlib
from pathlib import Path
from datetime import timedelta
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'my_eclp_final'),str(ROOT/'exp_code_ljj')]
import torch
import torch.distributed as dist
import peft.tuners.lora.model as lora
lora.is_bnb_available=lambda:False;lora.is_bnb_4bit_available=lambda:False
import eclp_elpo as core
RANK=int(os.environ['RANK']);WORLD=int(os.environ['WORLD_SIZE'])
assert WORLD==2 and RANK==local_rank
WRAPPER_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
torch.cuda.set_device(0)
assert torch.cuda.is_bf16_supported()
torch.cuda.set_per_process_memory_fraction(54*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
dist.init_process_group('nccl',timeout=timedelta(minutes=20),device_id=torch.device('cuda:0'))


def sync_gradients(params):
    present=torch.tensor([p.grad is not None for p in params],device='cuda:0',dtype=torch.int32)
    dist.all_reduce(present,op=dist.ReduceOp.SUM)
    any_grad=present.cpu().tolist()
    # FP32 communication avoids adding an extra BF16 reduction rounding stage.
    # Missing local gradients contribute zero; globally unused parameters stay None.
    flat=torch.cat([p.grad.detach().float().reshape(-1) if p.grad is not None
                    else torch.zeros(p.numel(),device=p.device,dtype=torch.float32) for p in params])
    dist.all_reduce(flat,op=dist.ReduceOp.SUM)
    core.finite(flat,'distributed summed gradient')
    offset=0
    for p,used in zip(params,any_grad):
        piece=flat[offset:offset+p.numel()].view_as(p);offset+=p.numel()
        if used:
            if p.grad is None:p.grad=torch.empty_like(p)
            p.grad.copy_(piece)
        else:p.grad=None
    del flat,present


def assert_replicas(params):
    digest=core.tensor_digest({str(i):p.detach() for i,p in enumerate(params)})
    values=[None]*WORLD;dist.all_gather_object(values,digest)
    if len(set(values))!=1:raise RuntimeError(f'Distributed parameter replicas diverged: {values}')
    return digest


def distributed_sft(args,cfg,lock):
    prepared=core.verified_prepared(args.prepared,lock)
    rows=list(core.jsonl(args.prepared/'sft.jsonl'))
    assert len(rows)==59868 and args.accumulation==8 and args.epochs==1 and args.max_steps==7484
    assert args.resume and args.checkpoint and not args.limit_scenarios
    meta=core.verify_checkpoint(args.checkpoint,lock);assert meta['stage']=='sft'
    run={'version':core.VERSION,'lock_id':lock['lock_id'],'stage':'sft','seed':cfg.seed,
         'architecture':cfg.architecture(),'data_sha256':prepared['files']['sft.jsonl'],
         'sessions':len(rows),'scenario_ids':sorted({r['scenario_id'] for r in rows}),
         'learning_rate':args.lr,'accumulation':args.accumulation}
    guard=core.exclusive_writer(args.output) if RANK==0 else contextlib.nullcontext()
    with guard:
        if RANK==0:
            assert core.read_json(args.output/'last.json')['checkpoint_id']==meta['checkpoint_id']
            core.immutable_run_manifest(args.output,run)
        dist.broadcast_object_list([run] if RANK==0 else [None],src=0)
        # immutable_run_manifest adds the established runtime fields only on rank0.
        # Read that exact persisted contract on every rank for optimizer resume.
        run=core.read_json(args.output/'run.json')
        core.seed_all(cfg.seed);bridge=core.BaselineBridge(cfg)
        runner=core.ECLPRunner(cfg,bridge,checkpoint=args.checkpoint,sft=True)
        params=core.trainable_parameters(runner)
        optimizer=core.FP32MasterAdamW(params,lr=args.lr,weight_decay=.01)
        state=core.load_tensors(args.checkpoint/'train_state.pt');tr=state['trainer']
        assert tr['run_hash']==core.object_hash(run)
        optimizer.load_state_dict(state['optimizer']);core.restore_rng(state['rng'])
        step,epoch,position=tr['step'],tr['epoch'],tr['next_position']
        assert args.max_steps>step
        if RANK==0:
            core.reconcile_sft_log(args.output/'train.jsonl',step)
            core.atomic_json(args.output/'distributed_runtime.json',{
                'wrapper_sha256':WRAPPER_SHA,'physical_gpus':[0,1],'world_size':2,
                'original_implementation_sha256':core.IMPLEMENTATION_SHA256,
                'global_sessions_per_update':8,'local_sessions_per_update':4,
                'source_checkpoint':str(args.checkpoint),'source_step':step,
                'sampling_order':'original shuffled eight-session update split by alternating index',
                'gradient_reduction':'SUM of local gradients already divided by global batch length; FP32 communication',
                'data_rows':59868,'epochs':1,'optimizer_target_steps':7484,
                'numerics':'parallel BF16 accumulation grouping can differ slightly; no objective or budget changes'})
        assert_replicas(params)
        start=time.time();last_saved=-1
        try:
            while epoch<args.epochs and step<args.max_steps:
                order=list(range(len(rows)));random.Random(cfg.seed+epoch).shuffle(order)
                while position<len(order) and step<args.max_steps:
                    batch=order[position:position+args.accumulation]
                    local_batch=batch[RANK::WORLD]
                    optimizer.zero_grad(set_to_none=True)
                    sums={k:0. for k in ('utterance','observation','strategy','reasoning')}
                    tokens=pairs=0
                    compute_start=time.time()
                    for idx in local_batch:
                        with torch.autocast(device_type='cuda',dtype=core.BF16):
                            losses,stats=runner.session_losses(rows[idx])
                            loss=(losses['utterance']+cfg.obs_weight*losses['observation']+
                                  cfg.strategy_weight*losses['strategy']+cfg.reasoning_weight*losses['reasoning'])
                        core.finite(loss,'SFT total').div(len(batch)).backward()
                        for k in sums:sums[k]+=float(losses[k].detach())/len(batch)
                        tokens+=stats['supervised_tokens'];pairs+=stats['feedback_pairs']
                    sync_gradients(params)
                    grad=torch.nn.utils.clip_grad_norm_(params,max_norm=1.,error_if_nonfinite=True)
                    optimizer.step();position+=len(batch);step+=1
                    agg=torch.tensor([*sums.values(),tokens,pairs],device='cuda:0',dtype=torch.float64)
                    dist.all_reduce(agg,op=dist.ReduceOp.SUM);numbers=agg.cpu().tolist()
                    duration=time.time()-compute_start
                    if RANK==0:
                        row={'step':step,'epoch':epoch,'next_position':position,
                             **dict(zip(sums,numbers[:4])), 'supervised_tokens':int(numbers[4]),
                             'feedback_pairs':int(numbers[5]),'grad_norm':float(grad),
                             'seconds_since_start':time.time()-start,'seconds_this_update':duration,
                             'max_memory_allocated':torch.cuda.max_memory_allocated(runner.device),
                             'world_size':2,'physical_gpus':[0,1],'global_sessions':len(batch)}
                        core.append_jsonl(args.output/'train.jsonl',row);print(json.dumps(row),flush=True)
                    if step%args.save_every==0:
                        replica_hash=assert_replicas(params)
                        if RANK==0:
                            tr={'step':step,'epoch':epoch,'next_position':position,'run_hash':core.object_hash(run)}
                            provenance=dict(run,parallel_runtime_sha256=WRAPPER_SHA,world_size=2,
                                            global_accumulation=args.accumulation,replica_parameter_hash=replica_hash)
                            core.save_checkpoint(runner,args.output,lock,'sft',step,optimizer,tr,provenance)
                        dist.barrier();last_saved=step
                if position>=len(order):epoch+=1;position=0
            if step!=last_saved:
                replica_hash=assert_replicas(params)
                if RANK==0:
                    tr={'step':step,'epoch':epoch,'next_position':position,'run_hash':core.object_hash(run)}
                    core.save_checkpoint(runner,args.output,lock,'sft',step,optimizer,tr,
                                         dict(run,parallel_runtime_sha256=WRAPPER_SHA,world_size=2,replica_parameter_hash=replica_hash))
                dist.barrier()
        finally:bridge.close()

core.command_sft=distributed_sft
try:
    core.main(sys.argv[1:])
finally:
    if dist.is_initialized():dist.destroy_process_group()
