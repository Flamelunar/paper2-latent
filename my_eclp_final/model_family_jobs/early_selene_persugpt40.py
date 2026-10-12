import os,json,time,subprocess
from pathlib import Path
R=Path('/app/hzj/liujianjian/ljj-2-paper');W=R/'exp_code_ljj/artifacts/llama_two_stage_pilot_20261009/meta_llama31';PY='/app/hzj/anaconda3/envs/ljj/bin/python'
status=W/'early_selene_persugpt40.json'
def state(stage,**kw):
    data={'time':time.time(),'pid':os.getpid(),'stage':stage,**kw};tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(status)
while True:
    summary=W/'judges/selene8b/evaluation/persugpt/summary.json'
    if summary.exists() and json.loads(summary.read_text()).get('n',0)>=40:
        state('already_complete');raise SystemExit(0)
    generation=R/'exp_code_ljj/artifacts/llama_public_repair_v3_20261009/meta_llama31/generators/persugpt/generation/persugpt_summary.json'
    n=json.loads(generation.read_text()).get('n',0) if generation.exists() else 0
    free=int(subprocess.check_output(['nvidia-smi','--id=4','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if n>=40 and free>=26*1024:break
    state('waiting_committed40_and_capacity',generated=n,free_mib=free);time.sleep(10)
cmd=[PY,'-u',str(R/'my_eclp_final/model_family_jobs/llama_public_repair_v3/evaluate.py'),'--repo-root',str(R),'--output',str(W),'--judge','selene8b','--method','persugpt','--physical-gpu','4','--limit','192','--batch-size','4','--memory-cap-gib','22','--available-only']
log=W/'judges/selene8b/persugpt_early_gpu4.log'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',TOKENIZERS_PARALLELISM='false',TRITON_CACHE_DIR=str(W/'native_triton_cache'),PYTHONPATH=':'.join(str(R/p) for p in ['my_eclp_final/model_family_jobs/llama_public_repair_v3','my_eclp_final/seven_methods_192','my_eclp_final','exp_code_ljj','baseline5_latentRATT','my_eclp_final/our_ma2p_deps']))
with log.open('a') as stream:
    p=subprocess.Popen(cmd,env=env,cwd=R,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    state('evaluating',worker_pid=p.pid,gpu=4,method='persugpt',expected_available_n=40,command=cmd)
    code=p.wait()
if code:
    state('failed',exit_code=code);raise SystemExit(code)
summary=json.loads(summary.read_text());state('complete',evaluated_n=summary['n'],success_percent=summary['success_percent'])
