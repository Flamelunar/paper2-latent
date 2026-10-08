import os,signal,subprocess,time,json
from pathlib import Path
ROOT=Path('/app/hzj/liujianjian/ljj-2-paper');JOB=ROOT/'my_eclp_final/model_family_jobs';WORK=ROOT/'exp_code_ljj/artifacts/deepseek_goal_strategy_192_20261008/deepseek_r1_qwen14b';PY='/app/hzj/anaconda3/envs/ljj/bin/python'
def complete(p):
 return p.exists() and (v:=json.loads(p.read_text())).get('complete') is True and v.get('n')==192
def main():
 controller=658815;worker=1151993
 cmdline=(Path('/proc')/str(controller)/'cmdline').read_bytes()
 assert b'deepseek_goal_strategy_campaign.py' in cmdline
 assert b'evaluate.py' in (Path('/proc')/str(worker)/'cmdline').read_bytes()
 out=WORK/'generators/ma2p_repaired';out.mkdir(parents=True,exist_ok=True)
 assert not (out/'generation/ma2p_repaired.jsonl').exists()
 os.kill(controller,signal.SIGTERM)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,4',CUDA_DEVICE_ORDER='PCI_BUS_ID',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',DAILYPERSUASION_PROFILE='paper10k')
 cmd=[PY,'-u',str(JOB/'generate_deepseek_goal_strategy_split.py'),'--repo-root',str(ROOT),'--output',str(out),'--physical-gpu','0','--family','deepseek_r1_qwen14b','--config',str(ROOT/'eclp_runs_deepseek_r1_qwen14b_20261007/config.json'),'--method','ma2p_repaired','--limit','192','--memory-cap-gib','40','--fresh-generation']
 with (out/'worker.log').open('a') as log:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (WORK/'split_ma2p_runtime.json').write_text(json.dumps({'worker_pid':proc.pid,'supervisor_pid':os.getpid(),'actor_gpu':0,'simulator_gpu':4,'preserved_qwen_worker':worker,'command':cmd},indent=2))
  print('Started split MA2P',proc.pid,'preserved Qwen judge',worker,flush=True)
  code=proc.wait()
  if code!=0:raise RuntimeError(f'Split MA2P failed {code}; see {out}/worker.log')
 assert complete(out/'generation/ma2p_repaired_summary.json')
 while not complete(WORK/'judges/qwen3_14b/evaluation/persugpt/summary.json'):
  p=Path('/proc')/str(worker)
  if not p.exists() or p.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z':raise RuntimeError('Qwen judge stopped before completion')
  time.sleep(20)
 with (WORK/'controller.log').open('a') as log:
  p=subprocess.Popen([PY,'-u',str(JOB/'deepseek_goal_strategy_campaign.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print('Resumed main controller',p.pid,flush=True)
if __name__=='__main__':main()
