"""Run unchanged training kernels with an explicit GPU memory budget."""
import argparse,os,runpy,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--entry',choices=['eclp','offline','latent'],required=True);p.add_argument('--physical-gpu',type=int,required=True);p.add_argument('--memory-cap-gib',type=float,required=True);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.physical_gpu) or (a.physical_gpu not in {0,1,4} or (a.physical_gpu==0 and a.entry!='latent')):raise ValueError('Training requires authorized GPU1/4, or GPU0 for the LatentRATT queue')
root=Path(__file__).resolve().parents[2];code=root/'my_eclp_final';sys.path[:0]=[str(code),str(root/'baseline5_latentRATT'),str(root/'exp_code_ljj')]
import torch
if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 is required')
torch.cuda.set_per_process_memory_fraction(a.memory_cap_gib*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
import peft.tuners.lora.model as lora
lora.is_bnb_available=lambda:False;lora.is_bnb_4bit_available=lambda:False
offline_entry=code/'model_family_jobs'/f'offline_latent_rl_gpu{a.physical_gpu}.py'
entry={'eclp':code/'eclp_elpo.py','offline':offline_entry,'latent':root/'baseline5_latentRATT/run_latent_ratt.py'}[a.entry]
sys.argv=[str(entry)]+(a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments)
runpy.run_path(str(entry),run_name='__main__')
