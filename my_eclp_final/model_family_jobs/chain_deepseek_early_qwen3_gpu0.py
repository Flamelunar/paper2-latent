import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/app/hzj/liujianjian/ljj-2-paper');JOB=ROOT/'my_eclp_final/model_family_jobs'
WORK=ROOT/'exp_code_ljj/artifacts/deepseek_our_selene_early_gpu0_20261010/deepseek_r1_qwen14b'
PY='/app/hzj/anaconda3/envs/ljj/bin/python'
selene=json.loads((WORK/'worker.json').read_text())['pid']
while Path(f'/proc/{selene}').exists():
    if 'State:	Z' in Path(f'/proc/{selene}/status').read_text():break
    time.sleep(30)
while int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<40*1024:time.sleep(30)
assert not (WORK/'qwen_worker.json').exists()
cmd=[PY,'-u',str(JOB/'llama_public_repair_v3/evaluate.py'),'--repo-root',str(ROOT),'--output',str(WORK),'--judge','qwen3_14b','--method','our_direct','--physical-gpu','0','--limit','192','--batch-size','4','--memory-cap-gib','34','--available-only']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
env['PYTHONPATH']=':'.join([str(JOB/'llama_public_repair_v3'),str(ROOT/'my_eclp_final/seven_methods_192'),str(ROOT/'my_eclp_final'),str(ROOT/'exp_code_ljj'),str(ROOT/'baseline5_latentRATT'),str(ROOT/'my_eclp_final/our_ma2p_deps')])
with (WORK/'qwen_early.log').open('a') as f:
    worker=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(WORK/'qwen_worker.json').write_text(json.dumps({'pid':worker.pid,'gpu':0,'command':cmd,'time':time.time()},indent=2))
print('Early Qwen3 worker:',worker.pid,flush=True)
with (WORK/'qwen_adoption.log').open('a') as f:
    adopter=subprocess.Popen([PY,'-u',str(JOB/'adopt_deepseek_early_qwen3.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('Early Qwen3 cache adopter:',adopter.pid,flush=True)
