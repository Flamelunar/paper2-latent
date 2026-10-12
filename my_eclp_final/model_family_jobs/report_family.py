"""Render only provenance-checked current-generation scores into model tables."""
import argparse,hashlib,json,statistics,sys,time
from pathlib import Path
ROOT=Path('/app/hzj/liujianjian/ljj-2-paper');CODE=ROOT/'my_eclp_final';sys.path[:0]=[str(CODE),str(CODE/'seven_methods_192'),str(ROOT/'exp_code_ljj')]
from experiment_io import atomic_json,digest,object_hash,read_json,read_jsonl
from evaluate import summarize
from protocol import validate_goal,speech_boundary
WORK=ROOT/'exp_code_ljj/artifacts/model_family_repaired192_20261007'
FAMILIES={'meta_llama31':('Meta-Llama-3.1-8B-Instruct','main_experiment_table_llama31.md'),'deepseek_r1_qwen14b':('DeepSeek-R1-Distill-Qwen-14B','main_experiment_table_deepseek14b.md')}
METHODS={'zero_shot_repaired':'Baseline 1: Zero-shot','dialogue_sft':'Baseline 2: Dialogue-SFT','persugpt':'Baseline 3: ISR-SFT + SimDPO','ma2p_repaired':'Baseline 4: MA2P','latentratt':'Baseline 5: LatentRATT','our_direct':'Our: persuasion only'}
def render(selected_families=None):
 selection=read_json(ROOT/'exp_code_ljj/data/test_selection_192.json')['original_indexes'];gold={r['index']:r for r in read_jsonl(ROOT/'eclp_runs_gpu4/evaluation_v2/golden/golden.jsonl')};audits={};fresh_audits={}
 for family,(model,filename) in FAMILIES.items():
  if selected_families and family not in selected_families:continue
  # two_stage_report_guard: current campaign publishes its own verified table.
  if family=='meta_llama31' and (ROOT/'exp_code_ljj/artifacts/llama_two_stage_pilot_20261009/meta_llama31/generation_run.json').exists():
   atomic_json(ROOT/'exp_code_ljj/results/llama_legacy_report_deferred.json',{'time':time.time(),'reason':'two-stage campaign owns current table publication'})
   continue
  # deepseek_two_stage_report_guard
  if family=='deepseek_r1_qwen14b' and (ROOT/'exp_code_ljj/artifacts/deepseek_two_stage_eval_v3_192_20261009/deepseek_r1_qwen14b/PUBLISHED.json').exists():
   continue
  work=WORK/family
  fresh=ROOT/'exp_code_ljj/artifacts/model_family_fresh192_nogoal_20261007'/family
  new_deepseek=ROOT/'exp_code_ljj/artifacts/deepseek_goal_strategy_192_20261008'/family
  if family=='deepseek_r1_qwen14b' and (new_deepseek/'fresh_generation_policy.json').exists():fresh=new_deepseek
  is_fresh=(fresh/'fresh_generation_policy.json').exists()
  fresh_policy=read_json(fresh/'fresh_generation_policy.json') if is_fresh else {}
  if is_fresh:work=fresh
  latent_summaries=[work/'judges'/judge/'evaluation/latentratt/summary.json' for judge in ['selene8b','qwen3_14b']]
  latent_complete=all(path.is_file() and read_json(path).get('complete') is True and read_json(path).get('n')==192
                      for path in latent_summaries)
  latent_note='LatentRATT已完成双Judge固定192评估。' if latent_complete else 'LatentRATT待完整训练后再测。'
  table=ROOT/'exp_code_ljj/results'/filename;backup=table.with_name(table.name+'.before_repaired192_20261007')
  if table.exists() and not backup.exists():backup.write_bytes(table.read_bytes())
  lines=['## Main experiment: latest target-grounded evaluation (192 scenarios)','',f'Persuader: `{model}`. Persuadee: `/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged`. BF16, no quantization.','',
   '固定192条：原始 `target_turn == 1` 且 `history == []`，原始索引见 [test_selection_192.json](../data/test_selection_192.json)。初始历史为空，后续只保留本次公开生成历史。zero-shot仅输入背景与可观察历史，不输入目标、策略、心理标签或golden。Our使用对应骨干的完整ECLP checkpoint，公开只输出劝说语句。','',
   ((f'生成：zero-shot、SFT、PersuGPT、MA2P、Our均在固定192条上全新生成，不复用旧对话；所有方法共用不接收精确目标的模拟劝说对象（仅背景和公开对话），并统一使用当前解码设置与输出边界。zero-shot劝说者不接收目标；SFT、PersuGPT、MA2P、Our仍按各自方法使用目标条件。Our保留完整ECLP隐式状态/策略和最新persuasion-only接口。{latent_note}旧partial分数不进入此版本。' if fresh_policy.get('simulated_persuadee') else '生成：zero-shot保留已完成的修复后192条；SFT、PersuGPT、MA2P、Our全部重新生成192条，共用当前模拟器、采样参数及公开历史边界，不继承旧场景。Our保留完整ECLP隐式状态/策略和最新persuasion-only接口。LatentRATT待对应骨干正式训练完成后生成。旧partial分数不进入此版本。') if is_fresh else '生成：zero-shot、MA2P、Our重新生成；SFT与PersuGPT保持原模型及原方法，仅重生成角色串话污染的完整场景，其他场景经核验后复用。LatentRATT必须使用对应骨干的完整RL checkpoint。旧1000条评估及旧诊断分数不复用。'),'',
   '评估：复用最新 `seven_methods_192/protocol.py` 与 `win_rules.json`，两位Judge完全相同的192个场景、golden和指标。Success要求当前被劝说者接受准确目标，五轮内任一轮成功；Win@1比较首条劝说语句，Win-Full比较全部生成对话与golden全部可用对话。A/B换序用于控制位置偏差，两次credit取平均（胜1、平0.5、负0），不声称为两个独立样本。','',
   '一致性核查：当前成功提示词、Win规则、Judge模型文件及golden与Qwen2.5主表最新合同一致。已确认Selene在Llama SFT样本788第3轮将“认真考虑、稍后答复”误判为投资目标接受；保留原始判断，不单独修改该方法分数。旧复用场景的模拟器权重与当前权重是否完全相同尚未核实，高成功率暂不能直接视为模型提升。核查记录：[evaluation_consistency_audit_20261007.json](evaluation_consistency_audit_20261007.json)。','',
   '| Judge | Method | Persuader model | Parsed / selected N | Success within 5 rounds (%) | Win@1 (%) | Win-Full (%) | Mean rounds (failure=5) | Success format unresolved |',
   '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
  if fresh_policy.get('input_policy')=='deepseek_goal_all_candidates_our_only_20261008':
   lines[4]='固定192条：target_turn==1且原始history为空。所有方法的劝说方和模拟被劝说者均接收精确目标；仅Our额外接收场景候选策略列表，不提供gold已选策略、ISR或未来回复。基线可自行规划策略，MA2P保留三角色K=0流程。'
   lines[6]='本版本六种方法全部重新生成192条，不复用旧对话。生成使用GPU0，评估使用GPU0或GPU1，训练使用GPU1（2026-10-08用户新增授权）；BF16，训练数据不变。Qwen和Llama保持各自现有设置。Only-Our strategy access是本实验的额外输入条件，跨方法和跨模型对比须明确这一差异。'
   lines[10]='本表只读取DeepSeek 20261008新输入合同对应的对话与评估，不合并旧目标不可见或不含候选策略版本。Judge使用现有统一协议。'
  family_audit={}
  for judge,display in [('selene8b','Selene-1-Mini-Llama-3.1-8B (BF16)'),('qwen3_14b','Qwen3-14B (BF16)')]:
   for method,label in METHODS.items():
    eval_work=work
    gen=work/'generation'/f'{method}.jsonl';out=work/'judges'/judge/'evaluation'/method;summary=None;rows=[]
    partial=ROOT/'exp_code_ljj/artifacts/reusable_partial192_20261007'/family
    if not is_fresh and not (out/'judgments.jsonl').exists() and (partial/'judges'/judge/'evaluation'/method/'judgments.jsonl').exists():
     eval_work=partial;gen=partial/'generation'/f'{method}.jsonl';out=partial/'judges'/judge/'evaluation'/method
    if not gen.exists():
     gen=work/'generators'/method/'generation'/f'{method}.jsonl'
    if (out/'judgments.jsonl').exists():
     contract=read_json(eval_work/'judges'/judge/'evaluation_run.json');run_hash=object_hash(contract);sources={r['index']:r for r in read_jsonl(gen)};rows=read_jsonl(out/'judgments.jsonl')
     assert len({r['index'] for r in rows})==len(rows)
     assert contract['new_expected_indexes']==read_json(work/'generation_run.json')['expected_indexes']
     assert set(contract['new_expected_indexes'])==set(selection)
     assert contract['protocol_sha256']==digest(CODE/'seven_methods_192/protocol.py') and contract['win_rules_sha256']==digest(CODE/'seven_methods_192/win_rules.json')
     for r in rows:
      i=r['index'];assert i in selection and r['contract_hash']==run_hash and r['source_row_hash']==object_hash(sources[i]) and r['golden_row_hash']==object_hash(gold[i]);s=sources[i];assert s['scenario']==gold[i]['scenario'] and s['initial_history']==[] and len(s['trajectory'])==10
      for t in s['trajectory']:
       b=speech_boundary(t['content'],s,t['role']);assert b['content']==t['content'] and not b['thinking_filtered'] and not b['future_dialogue_truncated']
      for n,g in enumerate(r['goal_results'],1):
       if g is not None:validate_goal(g,s['trajectory'][2*n-1]['content'])
      for metric in ['win1','win_full']:
       v=r[metric]
       if v['credit'] is not None:assert v['credit']==statistics.fmean([v['ab_credit'],v['ba_credit']])
     summary=summarize(rows,192)
     assert summary['n']==len(rows)
     atomic_json(out/'verified_summary.json',dict(summary,judgments_sha256=digest(out/'judgments.jsonl'),generation_sha256=digest(gen),contract_hash=run_hash))
    if summary:
     n=summary['success_evaluable_n'];fmt=lambda key,digits=2: '-' if summary[key] is None else f'{summary[key]:.{digits}f}'
     vals=[f'{len(rows)}/192'+(' (partial)' if len(rows)<192 else ''),f'{fmt("success_percent")} ({summary["success_count"]}/{n} evaluable)',fmt('win1_percent'),fmt('win_full_percent'),fmt('mean_rounds_fail5',3),str(summary['format_unresolved'])]
     family_audit[judge+'/'+method]=dict(summary,generation_sha256=digest(gen),judgments_sha256=digest(out/'judgments.jsonl'))
    else:
     count=len(read_jsonl(gen)) if gen.exists() else 0
     status=f'待评估 / 192 (生成 {count}/192)'
     if count==0 and method=='our_direct':
      last=ROOT/f'eclp_runs_{family}_20261007/sft_full_retry_20261007/last.json'
      if last.exists():
       cp=Path(read_json(last)['checkpoint']);meta=read_json(cp/'checkpoint.json')
       status=f'待训练完成（SFT已保存第{meta["step"]}步；生成0/192）'
     if count==0 and method=='latentratt':
      native=Path('/app/hzj/liujianjian/persuader_atapter_save')/family/'baseline5_latentratt'
      if not (native/'rl/run_config.json').exists():
       configs=list((native/'sft').glob('checkpoint-*/run_config.json'))
       if configs:
        cfg=max((read_json(p) for p in configs),key=lambda v:v.get('completed_steps',0))
        status=f'待完整SFT与RL（SFT已保存第{cfg.get("completed_steps",0)}步；生成0/192）'
       else:status='待对应骨干正式训练；生成0/192'
     if count==0 and method=='latentratt':
      sft_final=native/'sft/run_config.json'
      if sft_final.exists() and read_json(sft_final).get('training_complete') is True:
       status='SFT完整1轮已完成；待RL及生成完成；生成0/192'
     vals=[status,'-','-','-','-','-'];family_audit[judge+'/'+method]={'complete':False,'generation_n':count,'evaluation_n':0,'status':status}
    lines.append('| '+' | '.join([display,label,model]+vals)+' |')
  lines+=['','平均轮数取首次成功轮，未成功且五轮判定完整的场景计5。若判定缺失且无已确认成功，Success及平均轮数分母排除该场景；Win两次换序均解析成功才计入各自分母。`Success format unresolved`为未解析的轮级成功判定数。完整192条才作为正式最终结果。标注partial的分数来自已审计可复用场景的最新评估，样本数不同，不能当作完整192条的横向比较。',
   '',f'结果目录：`{work}`。各行原始判断、提示词缓存、模型合同、生成来源哈希、换序冲突率及Win未解析数保存在对应Judge目录中；旧表备份为 `{backup.name}`。','']
  tmp=table.with_name(table.name+f'.tmp-{__import__("os").getpid()}');tmp.write_text('\n'.join(lines),encoding='utf-8');tmp.replace(table);audits[family]=family_audit
  if is_fresh:fresh_audits[family]=family_audit
 atomic_json(WORK/'table_completion_audit.json',{'time':time.time(),'families':audits,'complete':all(v.get('complete') and v.get('n')==192 for a in audits.values() for v in a.values())})
 if fresh_audits:
  atomic_json(ROOT/'exp_code_ljj/artifacts/model_family_fresh192_nogoal_20261007/table_completion_audit.json',{'time':time.time(),'families':fresh_audits,'complete':all(v.get('complete') and v.get('n')==192 for a in fresh_audits.values() for v in a.values())})
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--family',choices=FAMILIES.keys(),action='append')
 args=parser.parse_args()
 render(set(args.family) if args.family else None)
