"""Durable sequential pipeline; pilot results saved before 192 expansion."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def report(out,limit):
    labels=['zero_shot_repaired','ma2p_repaired','our_analysis']
    if limit==20:labels+=['old_zero_shot','old_ma2p','old_our']
    summaries={}
    rows=['# 修复版同场景对比', '',f'样本数：{limit}；Judge：Selene；版本：target_grounded_v4。',
          'Win 为与 golden 对比的换序平均得分（胜=1、平=0.5、负=0）；Win-Full 使用全部已记录对话。',
          '旧生成仅在20条中复评，原始输出未修复；新Zero-shot不输入目标。Our为完整checkpoint的分析＋话语变体，只有话语公开。', '',
          '| 方法 | 场景数 | 五轮目标成功率 (%) | Win@1 (%) | Win-Full (%) | 平均成功轮数（失败=5） | 格式未解决 |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for label in labels:
        p=out/'evaluation'/label/'summary.json';s=json.loads(p.read_text());assert s['n']==limit
        summaries[label]=s
        rows.append(f"| {label} | {s['n']} | {s['success_percent']:.2f} | {s['win1_percent']:.2f} | {s['win_full_percent']:.2f} | {s['mean_rounds_fail5']:.3f} | {s['format_unresolved']} |")
    write(out/f'comparison_{limit}.json',summaries)
    (out/f'comparison_{limit}.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print('\n'.join(rows),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['TOKENIZERS_PARALLELISM']='false'
    base=Path(__file__).resolve().parent
    state={'supervisor_pid':os.getpid(),'gpu':0,'started':datetime.datetime.now().isoformat()}
    try:
        for limit in [20,192]:
            for script in ['generate.py','evaluate.py']:
                state.update(stage=script,target=limit,status='running');write(out/'supervisor_state.json',state)
                cmd=[sys.executable,'-u',str(base/script),'--repo-root',str(a.repo_root),'--output',str(out),'--limit',str(limit),'--physical-gpu','0']
                with (out/f'{script[:-3]}_{limit}.log').open('a',encoding='utf-8') as log:
                    child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                    state['child_pid']=child.pid;write(out/'supervisor_state.json',state)
                    code=child.wait()
                if code:raise RuntimeError(f'{script} limit={limit} failed with exit={code}; see log; completed JSONL rows retained')
            report(out,limit)
            # Save pilot before any full-set summary is changed. Expand independent of scores.
            state.update(stage=f'report_{limit}',status='completed_stage');write(out/'supervisor_state.json',state)
        state.update(stage='all',status='complete',finished=datetime.datetime.now().isoformat());write(out/'supervisor_state.json',state)
    except Exception:
        state.update(status='failed',error=traceback.format_exc());write(out/'supervisor_state.json',state);raise


if __name__=='__main__':main()
