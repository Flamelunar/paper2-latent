"""GPU4-only Qwen ablations; no main experiment mutation."""
import concurrent.futures,csv,json,os,subprocess,time,sys,hashlib,statistics
from pathlib import Path
R=Path('/app/hzj/liujianjian/ljj-2-paper');D=R/'my_eclp_final/ablation_qwen192';A=R/'exp_code_ljj/artifacts/qwen25_ablation_192_20261010';PY='/app/hzj/anaconda3/envs/ljj/bin/python'
sys.path.insert(0,str(R/'my_eclp_final'))
from experiment_io import atomic_json,read_json,digest,object_hash
SFT=R/'eclp_runs_gpu4/sft_full/checkpoint-007484';FULL=R/'eclp_runs_gpu4/offline_rl/checkpoint-001800'
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=':'.join([str(D),str(R/'my_eclp_final'),str(R/'exp_code_ljj'),str(R/'my_eclp_final/our_ma2p_deps')]))
def status(stage,**kw):atomic_json(A/'status.json',{'stage':stage,'pid':os.getpid(),'time':time.time(),'physical_gpu':4,**kw})
def run(script,args,log):
 cmd=[PY,'-u',str(D/script),*map(str,args)];log.parent.mkdir(parents=True,exist_ok=True)
 with (A/'commands.jsonl').open('a') as f:f.write(json.dumps({'cmd':cmd,'env':{k:ENV[k] for k in ['CUDA_VISIBLE_DEVICES','PYTHONPATH']},'time':time.time()})+'\n')
 with log.open('a') as f:
  w=subprocess.Popen(cmd,cwd=R,env=ENV,stdout=f,stderr=subprocess.STDOUT)
  atomic_json(log.with_suffix('.worker.json'),{'pid':w.pid,'cmd':cmd,'physical_gpu':4})
  code=w.wait()
 if code:raise RuntimeError(f'{script} exit {code}; see {log}')
def generation(variant,cp,smoke=False):
 out=A/variant
 args=['--repo-root',R,'--output',out,'--physical-gpu','4','--method','our_direct','--limit','192','--checkpoint',cp,'--variant',variant]
 if smoke:args+=['--smoke-two']
 run('generate.py',args,out/'generation.log')
def audit_generation(variant):
 out=A/variant;rows=[json.loads(x) for x in (out/'generation/our_direct.jsonl').read_text().splitlines() if x.strip()];contract=read_json(out/'generation_run.json');order=contract['expected_indexes'];assert len(rows)==192 and {x['index'] for x in rows}==set(order)
 for row in rows:assert row['initial_history']==[] and (len(row['trajectory'])==10 or row.get('generation_failed') is True) and row['run_hash']==object_hash(contract) and row['checkpoint_id']==contract['checkpoint_id']
 return rows

def report():
 lines=['# Qwen2.5-7B 两项消融（固定192条）','','历史主表Ours的精确generate_direct.py接口；模拟器按本次要求仅读背景与公共历史。完整Ours重新生成，原主表不覆盖。旧主表Success/Win@1提示词原样保留；Turn缺失判定且无已确认成功时计5，并单独披露缺失数量。','', '| Variant | Judge | N | Success (%) | Win@1 (%) | Turn (all N; failure/missing=5) | Success unresolved rounds |','|---|---|---:|---:|---:|---:|---:|']
 entries=[];diag=A/'wo_r_kl_only/diagnostic.json';equivalent=diag.exists() and read_json(diag).get('equivalent_to_wo_g') is True
 for variant in ['full_ours','wo_g','wo_r_kl_only']:
  source='wo_g' if variant=='wo_r_kl_only' and equivalent else variant
  for judge in ['selene8b','qwen3_14b']:
   p=A/source/'judges'/judge/'evaluation/our_direct/judgments.jsonl'
   if not p.exists():continue
   records=[json.loads(x) for x in p.read_text().splitlines() if x.strip()];n=len(records);succ=0;turns=[];unknown=0;wins=[];unresolved=0
   for row in records:
    goals=row['goal_results'];first=next((i for i,g in enumerate(goals,1) if g is not None and g['success']),None);succ+=first is not None;turns.append(first if first is not None else 5);unknown+=first is None and any(g is None for g in goals);unresolved+=sum(g is None for g in goals[:row.get('generation_available_rounds',5)])
    if row['win1']['credit'] is not None:wins.append(row['win1']['credit'])
   e={'variant':variant,'backbone':'Qwen2.5-7B-Instruct','judge':judge,'N':n,'complete':n==192,'Success':100*succ/n,'Win@1':100*statistics.fmean(wins) if wins else None,'Turn':statistics.fmean(turns),'success_effective_n':n-unknown,'success_all_n':n,'success_count':succ,'win1_effective_n':len(wins),'turn_effective_n':n,'success_unresolved_rounds':unresolved,'generation_failed_scenarios':sum(x.get('generation_failed',False) for x in records),'generation_missing_goal_rounds':sum(5-x.get('generation_available_rounds',5) for x in records),'unknown_scenarios_counted_as_5':unknown,'win1_unresolved_scenarios':n-len(wins),'record_source':str(p),'record_sha256':digest(p),'reused_verified_kl_degeneracy':variant=='wo_r_kl_only'};entries.append(e)
   lines.append(f'| {variant} | {judge} | {n}/192'+(' (partial)' if n<192 else '')+f' | {e["Success"]:.2f} | '+('-' if e['Win@1'] is None else f'{e["Win@1"]:.2f}')+f' | {e["Turn"]:.3f} | {unresolved} |')
 lines += ['', 'w/o R（KL-only）经1800个context、226次实际更新验证，KL、梯度和参数变化均为0，复用w/o G记录；两行不是独立测量。', '', 'Success以全部已评场景为分母；Turn将无已确认成功的场景计为5。Win@1仅在两种次序均解析成功时计入有效分母。生成缺失回合与Judge解析失败分开报告：', '', '| Variant | Judge | Success有效N / 全部N | Win@1有效N | 生成失败场景 | 缺失回合 | 无确认成功且判定缺失场景 |', '|---|---|---:|---:|---:|---:|---:|']
 for e in entries:
  lines.append(f'| {e["variant"]} | {e["judge"]} | {e["success_effective_n"]}/{e["N"]} | {e["win1_effective_n"]} | {e["generation_failed_scenarios"]} | {e["generation_missing_goal_rounds"]} | {e["unknown_scenarios_counted_as_5"]} |')
 result=R/'exp_code_ljj/results/ablation_qwen25_192.md';result.write_text('\n'.join(lines)+'\n')
 if entries:
  with (R/'exp_code_ljj/results/ablation_qwen25_192.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(entries[0]));writer.writeheader();writer.writerows(entries)
 atomic_json(A/'scores.json',{'rows':entries,'kl_only_equivalence_verified':equivalent})
def evaluate(variant,judge):
 run('evaluate.py',['--repo-root',R,'--output',A/variant,'--judge',judge,'--method','our_direct','--physical-gpu','4','--limit','192','--batch-size','4','--memory-cap-gib','22' if judge=='selene8b' else '34'],A/variant/f'{judge}.log')
def main():
 A.mkdir(exist_ok=True);status('wo_g_mechanism_check');generation('wo_g',SFT,True)
 status('wo_g_generation');generation('wo_g',SFT);audit_generation('wo_g')
 status('kl_only_full_budget_verification')
 if not (A/'wo_r_kl_only/diagnostic.json').exists():run('kl_only.py',[],A/'wo_r_kl_only/diagnostic.log')
 assert read_json(A/'wo_r_kl_only/diagnostic.json')['equivalent_to_wo_g']
 status('full_ours_matched_generation');generation('full_ours',FULL);audit_generation('full_ours')
 for variant in ['wo_g','full_ours']:
  status('evaluating',variant=variant,judges=['selene8b','qwen3_14b'])
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   tasks=[pool.submit(evaluate,variant,j) for j in ['selene8b','qwen3_14b']]
   while not all(t.done() for t in tasks):report();time.sleep(20)
   for t in tasks:t.result()
  report()
 diag=read_json(A/'wo_r_kl_only/diagnostic.json')
 for part in ['generation','judges']:
  p=A/'wo_r_kl_only'/part
  if not p.exists():p.symlink_to(A/'wo_g'/part,target_is_directory=True)
 atomic_json(A/'wo_r_kl_only/reuse_manifest.json',{'diagnostic_sha256':digest(A/'wo_r_kl_only/diagnostic.json'),'same_checkpoint_id':diag['checkpoint_id'],'same_generation_config':True,'same_judges_prompts':True,'records_reused_from':str(A/'wo_g'),'basis':'verified exact-zero full-budget KL-only; parameters unchanged'})
 report();status('complete',variants=['full_ours','wo_g','wo_r_kl_only'])
if __name__=='__main__':
 try:main()
 except Exception as e:status('failed',error=repr(e));raise
