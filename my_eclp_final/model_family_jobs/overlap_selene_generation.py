"""Use freed GPU4 actor capacity for the unchanged PersuGPT Selene rubric."""
import json,os,subprocess,time,fcntl
from pathlib import Path
R=Path('/app/hzj/liujianjian/ljj-2-paper');W=R/'exp_code_ljj/artifacts/llama_two_stage_pilot_20261009/meta_llama31';PY='/app/hzj/anaconda3/envs/ljj/bin/python'
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def state(stage,**kw):
    p=W/'overlap_selene_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'time':time.time(),'pid':os.getpid(),'stage':stage,**kw},ensure_ascii=False,indent=2));tmp.replace(p)
def main():
    while True:
        campaign=read(W/'live_status.json');stage=campaign.get('stage')
        if stage=='complete_fixed192_generation_and_evaluation':state('complete');return
        if stage=='failed':raise RuntimeError('Main campaign failed')
        threshold=campaign.get('target_prefix_n',0)
        if stage!='generating_parallel' or threshold<60:
            state('waiting_generation_phase');time.sleep(15);continue
        ours=read(W/'generators/our_direct/generation/our_direct_summary.json').get('n',0)
        pers=read(W/'generators/persugpt/generation/persugpt_summary.json').get('n',0)
        scored=read(W/'judges/selene8b/evaluation/persugpt/summary.json').get('n',0)
        free=int(subprocess.check_output(['nvidia-smi','--id=4','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if not (pers>=threshold and scored<threshold and ours<=threshold-4 and free>=26*1024):
            state('waiting_committed_generation',target=threshold,our_n=ours,persugpt_n=pers,scored_n=scored,free_mib=free);time.sleep(15);continue
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',TOKENIZERS_PARALLELISM='false',TRITON_CACHE_DIR=str(W/'native_triton_cache'),PYTHONPATH=':'.join(str(R/p) for p in ['my_eclp_final/model_family_jobs/llama_public_repair_v3','my_eclp_final/seven_methods_192','my_eclp_final','exp_code_ljj','baseline5_latentRATT','my_eclp_final/our_ma2p_deps']))
        cmd=[PY,'-u',str(R/'my_eclp_final/model_family_jobs/llama_public_repair_v3/evaluate.py'),'--repo-root',str(R),'--output',str(W),'--judge','selene8b','--method','persugpt','--physical-gpu','4','--limit','192','--batch-size','4','--memory-cap-gib','22','--available-only']
        with (W/'judges/selene8b/persugpt_overlap_gpu4.log').open('a') as stream:
            worker=subprocess.Popen(cmd,env=env,cwd=R,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            state('evaluating_during_generation',worker_pid=worker.pid,target=threshold,gpu=4)
            code=worker.wait()
        if code:raise RuntimeError(f'Selene overlap evaluation exited {code}')
        state('prefix_complete',evaluated_n=read(W/'judges/selene8b/evaluation/persugpt/summary.json').get('n'))
if __name__=='__main__':
    try:
        with (W/'.overlap_selene.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
    except Exception as exc:state('failed',error=repr(exc));raise
