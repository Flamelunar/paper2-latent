import json,time,re,os
from pathlib import Path
R=Path('/app/hzj/liujianjian/ljj-2-paper')
A=R/'exp_code_ljj/artifacts/oldsuccess_newwin_192_20261010/deepseek14b'
T=R/'exp_code_ljj/results/main_experiment_table_deepseek14b.md'
API=R/'exp_code_ljj/results/gpt_luna_deepseek_our_oldsuccess_newwin_192_20261010/summary.json'
def fmt(v,n=2):return '-' if v is None else f'{v:.{n}f}'
def update():
 text=T.read_text();lines=text.splitlines()
 for judge,label in [('selene8b','Selene-1-Mini-Llama-3.1-8B (BF16)'),('qwen3_14b','Qwen3-14B (BF16)')]:
  p=A/'judges'/judge/'evaluation/our_direct/summary.json'
  if not p.exists():continue
  d=json.loads(p.read_text());n=d['n'];valid=d.get('success_evaluable_n',n)
  count=f'{n}/192'+(' (partial)' if n<192 else '')
  success=fmt(d.get('success_percent'))+(f' ({d.get("success_count")}/{valid} evaluable)' if valid!=n else '')
  row=f'| {label} | Our: persuasion only | DeepSeek-R1-Distill-Qwen-14B | {count} | {success} | {fmt(d.get("win1_percent"))} | {fmt(d.get("win_full_percent"))} | {fmt(d.get("mean_rounds_fail5"),3)} | {d.get("format_unresolved",0)} |'
  for i,line in enumerate(lines):
   if line.startswith(f'| {label} | Our:'):lines[i]=row
 text='\n'.join(lines)+'\n'
 if API.exists() and 'GPT-6-luna：统一旧Success＋最新Win@1' not in text:
  ds=json.loads(API.read_text())['methods'];d=next((v for v in ds if v['family']=='deepseek14b' and v['method']=='our_direct'),None)
  if d:
   row=f'| Our | {d["n"]}/192'+(' (partial)' if d['n']<192 else '')+f' | {fmt(d.get("success_percent"))} | {fmt(d.get("win1_percent"))} | {fmt(d.get("mean_rounds_fail5"),3)} | {d.get("format_unresolved",0)} |'
   start=text.find('<!-- gpt-luna-results-start -->');end=text.find('<!-- gpt-luna-results-end -->')
   if start>=0 and end>start:
    section=text[start:end];section=re.sub(r'^\| Our \|.*$',lambda _:row,section,flags=re.M)
    note='\nOur本行使用旧Success＋最新Win@1，其他基线保留原v3结果；不同Success协议不作为严格统一协议比较。Our API结果目录：`gpt_luna_deepseek_our_oldsuccess_newwin_192_20261010/`。\n'
    if 'Our本行使用旧Success' not in section:section+=note
    text=text[:start]+section+text[end:]
 marker='<!-- deepseek-hybrid-protocol-note -->'
 if marker not in text:text+='\n'+marker+'\nSelene和Qwen3的Our行现更新为旧Success＋最新Win@1/Win-Full，其他基线分数保留。partial仅表示当前进度。Our生成192/192完成，私有理由不进入Judge。\n'
 tmp=T.with_name(T.name+'.hybrid-sync.tmp');tmp.write_text(text);tmp.replace(T)
while True:
 try:update()
 except Exception as e:print(repr(e),flush=True)
 time.sleep(30)
