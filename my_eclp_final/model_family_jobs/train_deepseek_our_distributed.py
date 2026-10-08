"""Resume full SFT with 0+1 GPU gradient summation, then the original offline RL."""
import fcntl,os,json,subprocess,time,socket
from pathlib import Path
import supervise_campaign as s
ROOT=s.ROOT;JOB=s.JOB;WORK=ROOT/'eclp_runs_deepseek_r1_qwen14b_20261007';PY=s.PY

def free(gpu):return int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
def status(phase,**kw):s.atomic(WORK/'job_status.json',dict(state='running',phase=phase,gpus=[0,1],supervisor_pid=os.getpid(),time=time.time(),**kw))

def main():
 with (WORK/'.accelerated_training.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  out=WORK/'sft_full_retry_20261007'
  if not (out/'checkpoint-007484/checkpoint.json').exists():
   saved=json.loads((out/'last.json').read_text());checkpoint=Path(saved['checkpoint'])
   while min(free(0),free(1))<60*1024:
    status('waiting_distributed_memory',free_gpu0_mib=free(0),free_gpu1_mib=free(1));time.sleep(30)
   with socket.socket() as reservation:
    reservation.bind(('127.0.0.1',0));master_port=reservation.getsockname()[1]
   cmd=[PY,'-m','torch.distributed.run','--master_addr','127.0.0.1','--master_port',str(master_port),'--nnodes','1','--nproc_per_node','2',str(JOB/'train_eclp_sft_distributed.py'),
        '--config',str(WORK/'config.json'),'sft','--prepared',str(WORK/'prepared'),'--output',str(out),
        '--epochs','1','--max-steps','7484','--accumulation','8','--lr','0.0001','--save-every','50',
        '--checkpoint',str(checkpoint),'--resume']
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',CUDA_DEVICE_ORDER='PCI_BUS_ID',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',NCCL_SOCKET_IFNAME='lo',GLOO_SOCKET_IFNAME='lo',NCCL_IB_DISABLE='1',DAILYPERSUASION_PROFILE='paper10k')
   with (WORK/'sft_retry.log').open('a') as log:
    p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    status('distributed_sft',worker_pid=p.pid,command=cmd,source_checkpoint=str(checkpoint));code=p.wait()
   if code:raise RuntimeError(f'Distributed SFT exited {code}; see {WORK}/sft_retry.log; resume only committed checkpoint')
  assert (out/'checkpoint-007484/checkpoint.json').exists()
  # train_family skips the completed SFT and retains the existing offline-RL contract.
  s.TRAIN_GPU=1;s.train_family('deepseek_r1_qwen14b')
if __name__=='__main__':
 try:main()
 except Exception as e:
  s.atomic(WORK/'job_status.json',dict(state='failed',gpus=[0,1],supervisor_pid=os.getpid(),error=repr(e),time=time.time()))
  raise
