import json, os, subprocess, sys, time, re
from pathlib import Path
ROOT=Path('/app/hzj/liujianjian/ljj-2-paper'); CODE=ROOT/'my_eclp_final'; HERE=CODE/'model_family_jobs'; PY=sys.executable
FAMILIES={'meta_llama31':'/app/hzj/models/Meta-Llama-3.1-8B-Instruct','deepseek_r1_qwen14b':'/app/hzj/models/DeepSeek-R1-Distill-Qwen-14B'}
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+'.tmp-'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)
def run(cmd,log,env):
 with log.open('a') as f:
  p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  save(log.with_suffix('.pid.json'),{'pid':p.pid,'command':cmd,'gpu':env['CUDA_VISIBLE_DEVICES']})
  if p.wait():raise RuntimeError('Failed: '+str(log))
def train(family):
 work=ROOT/('eclp_runs_'+family+'_20261007');work.mkdir(exist_ok=True)
 cfg=json.loads((CODE/'config_gpu4.json').read_text());cfg.update(repo_root=str(ROOT),work_dir=str(work),base_model=FAMILIES[family]);save(work/'config.json',cfg)
 plan=json.loads((CODE/'offline_training_plan.json').read_text());plan.update(physical_gpu=4,test_scenarios=192,plan_version='model_family_offline_lpo_20261007');save(work/'plan.json',plan)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',DAILYPERSUASION_PROFILE='paper10k',ALLOW_JUDGE_EVAL='1')
 base=[PY,'-u',str(CODE/'eclp_elpo.py'),'--config',str(work/'config.json')]
 try:
  for phase,cmd,done in [('lock',base+['lock'],work/'protocol.lock.json'),('prepare',base+['prepare','--output',str(work/'prepared')],work/'prepared/manifest.json'),('sft',base+['sft','--prepared',str(work/'prepared'),'--output',str(work/'sft_full'),'--epochs','1','--max-steps','7484','--accumulation','8','--lr','0.0001','--save-every','500'],work/'sft_full/checkpoint-007484/checkpoint.json')]:
   save(work/'job_status.json',{'phase':phase,'state':'running','gpu':1,'pid':os.getpid(),'time':time.time()})
   if not done.exists():
    if phase=='sft':
     while True:
      free=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
      need=44000 if family=='deepseek_r1_qwen14b' else 31000
      if free>=need:break
      save(work/'job_status.json',{'phase':'sft','state':'waiting_for_memory','gpu':1,'free_mib':free,'required_mib':need,'pid':os.getpid()});time.sleep(30)
    run(cmd,work/(phase+'.log'),env)
  save(work/'job_status.json',{'phase':'offline_rl','state':'running','gpu':1,'pid':os.getpid()})
  run([PY,'-u',str(HERE/'offline_latent_rl_gpu1.py'),'--config',str(work/'config.json'),'--plan',str(work/'plan.json'),'train','--prepared',str(work/'prepared'),'--checkpoint',str(work/'sft_full/checkpoint-007484'),'--output',str(work/'offline_rl')],work/'offline_rl.log',env)
  save(work/'job_status.json',{'phase':'training','state':'complete','next':'fixed192 generation and two judges','gpu':1})
 except Exception as e:
  save(work/'job_status.json',{'state':'failed','error':str(e),'gpu':1});raise

def evaluate(judge):
 from concurrent.futures import ThreadPoolExecutor
 selection=json.loads((ROOT/'exp_code_ljj/data/test_selection_192.json').read_text())['original_indexes']; gold={r['index']:r for r in map(json.loads,(ROOT/'eclp_runs_gpu4/evaluation_v2/golden/golden.jsonl').read_text().splitlines())}
 mapping={'baseline1_zero_shot':'zero_shot_repaired','baseline2_dialogue_sft':'dialogue_sft','baseline3_isr_simdpo':'persugpt','baseline4_ma2p':'ma2p_repaired'}
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
 for family in FAMILIES:
  work=ROOT/'exp_code_ljj/artifacts/model_family_latest192_20261007'/family;work.mkdir(parents=True,exist_ok=True)
  save(work/'generation_run.json',{'revision':'archived_sources_latest_judge_20261007','expected_indexes':selection,'family':family,'source_policy':'unaltered original rows; source contamination audited separately'})
  for source,method in mapping.items():
   src=ROOT/'exp_code_ljj/artifacts/generation_from_start192'/family/(source+'.jsonl')
   if not src.exists():continue
   rows=[json.loads(s) for s in src.read_text().splitlines()]; indices=[r['index'] for r in rows]
   if set(indices)!=set(selection) or len(rows)!=192:save(work/(source+'_audit.json'),{'status':'incomplete','n':len(rows)});continue
   bad=[]
   for r in rows:
    g=gold[r['index']]; assert r['scenario']==g['scenario'] and r['scenario_id']==g['scenario_id'] and r['initial_history']==[] and r['reference']==g['golden_first'];assert len(r['trajectory'])==10
    if any(re.search(r'\n(?:user|assistant)\n',t['content']) for t in r['trajectory']):bad.append(r['index'])
   save(work/(source+'_audit.json'),{'n':192,'identity':'passed','speaker_boundary_contamination_indexes':bad,'warning':'these are diagnostic scores of archived outputs, not repaired generation' if bad else ''})
   dst=work/'generation'/(method+'.jsonl');dst.parent.mkdir(exist_ok=True)
   if dst.exists():assert dst.read_bytes()==src.read_bytes()
   else:dst.write_bytes(src.read_bytes())
   run([PY,'-u',str(CODE/'seven_methods_192/evaluate.py'),'--repo-root',str(ROOT),'--output',str(work),'--physical-gpu','4','--judge',judge,'--method',method,'--limit','192','--batch-size','2','--memory-cap-gib','22' if judge=='selene8b' else '32'],work/(judge+'_'+method+'.log'),env)
if __name__=='__main__':
 if sys.argv[1]=='train':train(sys.argv[2])
 else:evaluate(sys.argv[2])
