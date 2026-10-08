"""Matched training-only diagnostic; never selects weights using test outcomes."""
import os, sys, json, shutil, random, time
from pathlib import Path
ROOT=Path('/app/hzj/liujianjian/ljj-2-paper')
BASE=Path('/app/hzj/liujianjian/persuader_atapter_save/meta_llama31/baseline5_latentratt')
OUT=ROOT/'exp_code_ljj/artifacts/llama_latent_lr1e6_matched_probe800_20261008'
# Await this verified training worker and complete checkpoint; no test data.
import subprocess
pending=BASE/'rl_fp32_accum16_lr1e6_20261008/checkpoint-800'
while not (pending/'training_state.pt').exists():
    proc=Path('/proc/590492')
    if not proc.exists(): raise RuntimeError('RL worker ended before checkpoint800; inspect training log')
    cmd=(proc/'cmdline').read_bytes()
    if b'rl_fp32_accum16_lr1e6_20261008' not in cmd: raise RuntimeError('Training PID no longer matches this run')
    print('waiting checkpoint800',flush=True)
    time.sleep(45)
while int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()) < 28*1024:
    print('waiting GPU1 diagnostic memory',flush=True)
    time.sleep(45)

OUT.mkdir(parents=True,exist_ok=True)
assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
sys.path[:0]=[str(ROOT/'baseline5_latentRATT'),str(ROOT/'exp_code_ljj')]
saved={}
for name in ['sft','rl_fp32_accum16_lr1e6_20261008/checkpoint-800']:
    src=BASE/name/'attention.pt'
    dest=OUT/(name.replace('/','_')+'_attention.pt')
    if not dest.exists(): shutil.copy2(src,dest)
    saved[name]=dest
import torch
torch.cuda.set_per_process_memory_fraction(24*1024**3/torch.cuda.get_device_properties(0).total_memory)
import peft.tuners.lora.model as lora
lora.is_bnb_available=lambda:False
lora.is_bnb_4bit_available=lambda:False
from run_latent_ratt import load_model, encode
from persuasion_data import iter_training_rows
rows=list(iter_training_rows(ROOT/'exp_code_ljj/data/isr_sft.jsonl'))
indices=random.Random(8291).sample(range(len(rows)),12)
tokenizer,model=load_model('/app/hzj/models/Meta-Llama-3.1-8B-Instruct',stage='rl',checkpoint=BASE/'sft',device='cuda:0')
result={'training_only':True,'indices':indices,'backbone':'frozen SFT adapter shared across checkpoints','checkpoints':{}}
with torch.inference_mode():
    examples=[encode(rows[i][0],rows[i][1],tokenizer,1536,256,model.device) for i in indices]
    for name,state in saved.items():
        model.attention.load_state_dict(torch.load(state,map_location='cpu',weights_only=True))
        losses=[]
        for prompt,target in examples:
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                h,m,pos,cache=model.cached_prompt_state(prompt)
                center=model.latent_from_hidden(h,m,pos)
                nll=-model.target_logps_from_cache(target,center,prompt.shape[1],cache).mean()
            losses.append(float(nll))
        result['checkpoints'][name]={'mean_nll':sum(losses)/len(losses),'per_example_nll':losses}
        print(name,result['checkpoints'][name],flush=True)
        (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
result['complete']=True
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
