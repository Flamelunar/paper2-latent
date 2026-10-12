"""Old Success + current Win, fixed192; preserve running training/generation."""
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path('/app/hzj/liujianjian/ljj-2-paper')
JOB=ROOT/'my_eclp_final/model_family_jobs'
PROTOCOL=JOB/'judge_oldsuccess_newwin_20261010'
ART=ROOT/'exp_code_ljj/artifacts'
OUT=ART/'oldsuccess_newwin_192_20261010'
PY='/app/hzj/anaconda3/envs/ljj/bin/python'
METHODS=['latentratt','our_direct']
SOURCES={'qwen25':ART/'seven_methods_192_20261006',
 'llama31':ART/'llama_two_stage_pilot_20261009/meta_llama31',
 'deepseek14b':ART/'deepseek_goal_strategy_192_20261008/deepseek_r1_qwen14b'}
TABLES={'qwen25':'main_experiment_table.md','llama31':'main_experiment_table_llama31.md','deepseek14b':'main_experiment_table_deepseek14b.md'}

def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(p)
def status(stage,**kw):atomic(OUT/'status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw))
def source(family,method):
 p=SOURCES[family]/'generation'/f'{method}.jsonl'
 if p.exists():return p
 return SOURCES[family]/'generators'/method/'generation'/f'{method}.jsonl'
def ready(family,method):
 p=source(family,method)
 if not p.exists():return False
 r=rows(p)
 return len(r)==192
def setup(family,method):
 work=OUT/family;(work/'generation').mkdir(parents=True,exist_ok=True)
 origin=SOURCES[family]/'generation_run.json'
 if family=='deepseek14b':origin=ART/'deepseek_two_stage_eval_v3_192_20261009/deepseek_r1_qwen14b/generation_run.json'
 manifest=read(origin);manifest.update(revision='old_success_new_win_20261010',
  evaluation_policy='old exact GOAL_RULE; latest position-neutral Win@1 and Win-Full; generation unchanged',
  protocol_sha256=sha(PROTOCOL/'protocol.py'),win_rules_sha256=sha(PROTOCOL/'win_rules.json'))
 path=work/'generation_run.json'
 if path.exists():assert read(path)==manifest
 else:atomic(path,manifest)
 p=work/'generation'/f'{method}.jsonl'
 if not p.exists() and not p.is_symlink():p.symlink_to(source(family,method))
 return work
def free(gpu):return int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
def publish(family):
 work=OUT/family
 lines=[f'# {family}: old Success + latest Win, fixed192','',
  'Success提示词精确复用seven_methods_192/protocol.py；Win@1及Win-Full使用当前位置中立规则。五轮任一轮成功；平均轮数失败计5；换序credit平均。仅重新评估既有公开对话，生成输入条件与方法接口不变。partial为临时结果。','',
  '| Judge | Method | N | Success (%) | Win@1 (%) | Win-Full (%) | Mean rounds (failure=5) | Success unresolved |',
  '|---|---|---:|---:|---:|---:|---:|---:|']
 for judge in ['selene8b','qwen3_14b']:
  for method in METHODS:
   p=work/'judges'/judge/'evaluation'/method/'summary.json'
   s=read(p) if p.exists() else {}
   vals=['-' if s.get(k) is None else f'{s[k]:.3f}' if k=='mean_rounds_fail5' else f'{s[k]:.2f}' for k in ['success_percent','win1_percent','win_full_percent','mean_rounds_fail5']]
   n=s.get('n',0);label=f'{n}/192'+(' (partial)' if 0<n<192 else '')
   lines.append(f'| {judge} | {method} | {label} | '+' | '.join(vals)+f" | {s.get('format_unresolved','-')} |")
 text='\n'.join(lines)+'\n'
 (work/'live_scores.md').write_text(text)
 result=ROOT/'exp_code_ljj/results'/f'{family}_oldsuccess_newwin_192.md';result.write_text(text)
 main=ROOT/'exp_code_ljj/results'/TABLES[family]
 if main.exists():
  original=main.read_text();start='<!-- hybrid-latent-only-start -->';end='<!-- hybrid-latent-only-end -->'
  if start in original:original=original[:original.index(start)]+original[original.index(end)+len(end):]
  main.write_text(original.rstrip()+'\n\n'+start+'\n'+text+'\n'+end+'\n')

def evaluate(family,method,judge,gpu):
 work=setup(family,method);p=work/'judges'/judge/'evaluation'/method/'summary.json'
 if p.exists() and read(p).get('complete'):return
 worker=work/'judges'/judge/f'{method}_worker.json'
 if worker.exists():
  pid=read(worker)['pid']
  while Path(f'/proc/{pid}/cmdline').exists():
   command=Path(f'/proc/{pid}/cmdline').read_bytes()
   if b'evaluate.py' not in command:break
   time.sleep(10)
  if p.exists() and read(p).get('complete'):return
 need=28*1024 if judge=='selene8b' else 40*1024
 while free(gpu)<need:time.sleep(30)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
 env['PYTHONPATH']=':'.join([str(PROTOCOL),str(ROOT/'my_eclp_final/seven_methods_192'),str(ROOT/'my_eclp_final'),str(ROOT/'exp_code_ljj'),str(ROOT/'baseline5_latentRATT'),str(ROOT/'my_eclp_final/our_ma2p_deps')])
 directory=work/'judges'/judge;directory.mkdir(parents=True,exist_ok=True)
 cmd=[PY,'-u',str(PROTOCOL/'evaluate.py'),'--repo-root',str(ROOT),'--output',str(work),'--judge',judge,'--method',method,'--physical-gpu',str(gpu),'--limit','192','--batch-size','4','--memory-cap-gib','22' if judge=='selene8b' else '34']
 with (directory/f'{method}.log').open('a') as log:
  child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 atomic(directory/f'{method}_worker.json',{'pid':child.pid,'gpu':gpu,'command':cmd,'time':time.time()})
 while child.poll() is None:time.sleep(10)
 if child.returncode:raise RuntimeError(f'{family}/{method}/{judge} exit {child.returncode}')
 assert read(p)['complete'] and read(p)['n']==192
def audit(family,method):
 import sys
 sys.path.insert(0,str(ROOT/'my_eclp_final'))
 from experiment_io import object_hash
 work=OUT/family;selection=read(ROOT/'exp_code_ljj/data/test_selection_192.json')['original_indexes']
 source_rows={r['index']:r for r in rows(source(family,method))};gold={r['index']:r for r in rows(ROOT/'eclp_runs_gpu4/evaluation_v2/golden/golden.jsonl')}
 assert len(source_rows)==192 and set(source_rows)==set(selection)
 for judge in ['selene8b','qwen3_14b']:
  directory=work/'judges'/judge;contract=read(directory/'evaluation_run.json')
  assert contract['protocol_sha256']==sha(PROTOCOL/'protocol.py') and contract['win_rules_sha256']==sha(PROTOCOL/'win_rules.json')
  records=rows(directory/'evaluation'/method/'judgments.jsonl');assert len(records)==192 and {r['index'] for r in records}==set(selection)
  for r in records:
   assert r['source_row_hash']==object_hash(source_rows[r['index']]) and r['golden_row_hash']==object_hash(gold[r['index']])
 atomic(work/f'{method}_audit.json',{'complete':True,'fixed192':True,'row_hashes_verified':True,'protocol_verified':True,'time':time.time()})
def generate_latent_if_ready():
 family='deepseek14b';method='latentratt'
 if ready(family,method):return
 base=Path('/app/hzj/liujianjian/persuader_atapter_save/deepseek_r1_qwen14b/baseline5_latentratt')
 candidates=[p.parent for p in base.glob('rl/**/run_config.json') if read(p).get('stage')=='rl' and read(p).get('training_complete') is True and read(p).get('train_rows')==59868]
 if not candidates:return
 cp=max(candidates,key=lambda p:(p/'run_config.json').stat().st_mtime)
 while free(1)<56*1024:time.sleep(30)
 dest=SOURCES[family]/'generators'/method;dest.mkdir(parents=True,exist_ok=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
 cmd=[PY,'-u',str(JOB/'generate_latent_family.py'),'--repo-root',str(ROOT),'--output',str(dest),'--physical-gpu','1','--family','deepseek_r1_qwen14b','--checkpoint',str(cp),'--memory-cap-gib','54']
 with (dest/'hybrid_priority_generation.log').open('a') as log:
  worker=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 status('generating_deepseek_latentratt',worker_pid=worker.pid)
 if worker.wait():raise RuntimeError('DeepSeek LatentRATT generation failed')
def run_method(family,method):
 work=setup(family,method);status('evaluating_hybrid',family=family,method=method)
 if free(1)>=40*1024:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   tasks=[pool.submit(evaluate,family,method,j,g) for j,g in [('selene8b',0),('qwen3_14b',1)]]
   while not all(f.done() for f in tasks):publish(family);time.sleep(10)
   for f in tasks:f.result()
 else:
  for judge in ['selene8b','qwen3_14b']:
   with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
    task=pool.submit(evaluate,family,method,judge,0)
    while not task.done():publish(family);time.sleep(10)
    task.result()
 audit(family,method);publish(family)
def main():
 OUT.mkdir(exist_ok=True)
 for family,method in [('llama31','latentratt'),('deepseek14b','our_direct'),('deepseek14b','latentratt')]:
  while not ready(family,method):
   if method=='latentratt' and family=='deepseek14b':generate_latent_if_ready()
   if ready(family,method):break
   status('waiting_priority_generation_or_training',family=family,method=method);time.sleep(30)
  run_method(family,method)
 status('priority_hybrid_evaluation_complete')
 for family in ['qwen25','llama31','deepseek14b']:
  for method in METHODS:
   if ready(family,method):run_method(family,method)
 status('all_hybrid_evaluation_complete')
if __name__=='__main__':
 try:main()
 except Exception as e:status('failed',error=repr(e));raise
