"""Insert a paired20 two-stage inference pilot; resume the existing campaign."""
import importlib.util
import json
import os
import signal
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT=Path('/app/hzj/liujianjian/ljj-2-paper')
CODE=ROOT/'my_eclp_final'
JOB=CODE/'model_family_jobs'
REPAIR=JOB/'llama_public_repair_v3'
BASE=ROOT/'exp_code_ljj/artifacts/llama_public_repair_v3_20261009/meta_llama31'
WORK=ROOT/'exp_code_ljj/artifacts/llama_two_stage_pilot_20261009/meta_llama31'
PY='/app/hzj/anaconda3/envs/ljj/bin/python'
sys.path[:0]=[str(REPAIR),str(CODE),str(ROOT/'exp_code_ljj')]
from experiment_io import atomic_json,read_json,read_jsonl,digest
from simulator_rpc import unix_request

def status(**values):
    atomic_json(WORK/'pilot_status.json',{'time':time.time(),'pid':os.getpid(),**values})

COMPARE_LOCK=threading.RLock()

def _compare():
    lines=['# Llama：同一前20场景，两阶段推理对照','',
           '训练权重固定；所有方法使用相同Judge规则和golden。两阶段的简短分析不进入公开对话或Judge；每轮额外增加一次分析生成。下列都是20条诊断分数，不代表完整192条结果。','',
           '| Judge | 版本 | 已生成 | 已评估 | Success (%) | Win@1 (%) | Win-Full (%) | 平均轮数 | Win@1换序冲突 (%) |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    report={'time':time.time(),'pilot_n':20,'scores':{},'selection_policy':'no checkpoint or reply selection using Judge scores'}
    for judge in ['selene8b','qwen3_14b']:
        for label,work,method in [('Our: speech only',BASE,'our_direct'),('Our: two-stage private note',WORK,'our_direct'),('PersuGPT',BASE,'persugpt')]:
            source=work/'generation'/f'{method}.jsonl'
            generated=len(read_jsonl(source)) if source.exists() else 0
            path=work/'judges'/judge/'evaluation'/method/'summary.json'
            summary=read_json(path) if path.exists() else {}
            def val(key,digits=2):
                v=summary.get(key)
                return '-' if v is None else f'{v:.{digits}f}'
            lines.append(f'| {judge} | {label} | {generated}/20 | {summary.get("n",0)}/20 | {val("success_percent")} | {val("win1_percent")} | {val("win_full_percent")} | {val("mean_rounds_fail5",3)} | {val("win1_conflict_percent")} |')
            report['scores'][judge+'/'+label]=summary
    atomic_json(WORK/'comparison.json',report)
    text='\n'.join(lines)+'\n'
    (WORK/'comparison.md').write_text(text,encoding='utf-8')
    (ROOT/'exp_code_ljj/results/llama_two_stage_pilot_comparison.md').write_text(text,encoding='utf-8')

def compare():
    with COMPARE_LOCK:
        _compare()

def main():
    WORK.mkdir(parents=True,exist_ok=True);(WORK/'generation').mkdir(exist_ok=True)
    handoff=read_json(BASE/'two_stage_handoff.json')
    old_pid=handoff['paused_supervisor_pid']
    # Let the already-running Qwen judge finish; it writes its own summary.
    status(stage='waiting_current_persugpt_judge',paused_campaign_pid=old_pid)
    while True:
        ready=all((BASE/'judges'/j/'evaluation'/m/'summary.json').exists() and
                  read_json(BASE/'judges'/j/'evaluation'/m/'summary.json').get('n')==20
                  for j in ['selene8b','qwen3_14b'] for m in ['our_direct','persugpt'])
        if ready:break
        compare();time.sleep(10)
    for j in ['selene8b','qwen3_14b']:
        worker=BASE/'workers'/('judge_'+j+'.json')
        if worker.exists():
            pid=read_json(worker)['pid']
            p=Path('/proc')/str(pid)/'stat'
            while p.exists() and p.read_text().rsplit(')',1)[1].split()[0]!='Z':
                time.sleep(2)
    spec=importlib.util.spec_from_file_location('live_campaign_tools',REPAIR/'supervise_live.py')
    tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)
    tools.WORK=WORK
    tools.update_scores=compare
    manifest=read_json(BASE/'generation_run.json')
    manifest.update(revision='llama_two_stage_private_note_pilot_20261009',output_variant='two_stage_private_note_then_public_reply',
                    generation_script_sha256=digest(JOB/'generate_llama_two_stage.py'),
                    pilot_n=20,private_note_visibility='raw trace only; no public/Judge visibility',
                    selection_policy='first format-valid note/reply; no Judge-based reply selection')
    atomic_json(WORK/'generation_run.json',manifest)
    source=WORK/'generators/our_direct/generation/our_direct.jsonl'
    link=WORK/'generation/our_direct.jsonl'
    if not link.exists() and not link.is_symlink():link.symlink_to(source)
    checkpoint=Path(read_json(ROOT/'eclp_runs_meta_llama31_20261007/offline_rl/summary.json')['checkpoint'])
    status(stage='simulator_loading',paused_campaign_pid=old_pid)
    socket_path=f'/tmp/llama-two-stage-{os.getpid()}.sock'
    simulator=tools.launch('simulator',4,[PY,'-u',str(REPAIR/'serve_simulator.py'),'--root',str(ROOT),
               '--socket',socket_path,'--status',str(WORK/'simulator_status.json')],WORK/'simulator.log')
    try:
        ready=False
        for _ in range(240):
            if simulator.poll() is not None:raise RuntimeError('Simulator startup failed')
            try:ready=unix_request(socket_path,'/health',timeout=2).get('ready') is True
            except Exception:pass
            if ready:break
            time.sleep(5)
        if not ready:raise RuntimeError('Simulator startup timed out')
        cmd=[PY,'-u',str(JOB/'generate_llama_two_stage.py'),'--repo-root',str(ROOT),'--output',str(WORK/'generators/our_direct'),
             '--family','meta_llama31','--config',str(ROOT/'eclp_runs_meta_llama31_20261007/config.json'),
             '--physical-gpu','1','--method','our_direct','--limit','192','--stop-after','20','--memory-cap-gib','22',
             '--fresh-generation','--persuadee-goal-blind','--simulator-url','unix:'+socket_path,'--checkpoint',str(checkpoint)]
        status(stage='generating_two_stage',target=20,actor_gpu=1,simulator_gpu=4)
        worker=tools.launch('our_two_stage',1,cmd,WORK/'generators/our_direct/worker.log')
        while worker.poll() is None:compare();time.sleep(10)
        if worker.returncode:raise RuntimeError(f'Two-stage generation failed: {worker.returncode}')
        tools.stop_owned('simulator')
        status(stage='evaluating_two_stage',target=20,judge_gpus={'selene8b':1,'qwen3_14b':4})
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(tools.evaluate_judge,j,g,['our_direct'],20) for j,g in [('selene8b',1),('qwen3_14b',4)]]
            for f in futures:f.result()
        compare();status(stage='pilot_complete',evaluated_n=20)
    finally:
        for name in list(tools.OWNED):tools.stop_owned(name)
        p=Path('/proc')/str(old_pid)/'cmdline'
        if p.exists() and b'llama_public_repair_v3/supervise_live.py' in p.read_bytes():
            os.kill(old_pid,signal.SIGCONT)
            atomic_json(WORK/'resumed_original_campaign.json',{'time':time.time(),'pid':old_pid,'no_score_based_variant_selection':True})

if __name__=='__main__':
    try:main()
    except Exception as exc:status(stage='failed',error=repr(exc));raise
