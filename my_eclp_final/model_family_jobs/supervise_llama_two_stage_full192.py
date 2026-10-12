"""Fixed192 incremental generation and paired evaluation; never touches training."""
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import threading
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT=Path('/app/hzj/liujianjian/ljj-2-paper')
CODE=ROOT/'my_eclp_final'
JOB=CODE/'model_family_jobs'
REPAIR=JOB/'llama_public_repair_v3'
WORK=ROOT/'exp_code_ljj/artifacts/llama_two_stage_pilot_20261009/meta_llama31'
BASE=ROOT/'exp_code_ljj/artifacts/llama_public_repair_v3_20261009/meta_llama31'
OLD=ROOT/'exp_code_ljj/artifacts/model_family_fresh192_nogoal_20261007/meta_llama31'
PY='/app/hzj/anaconda3/envs/ljj/bin/python'
sys.path[:0]=[str(CODE),str(ROOT/'exp_code_ljj')]
from experiment_io import atomic_json, read_json, read_jsonl, digest, object_hash
sys.path.insert(0,str(REPAIR))
from simulator_rpc import unix_request

METHODS=['our_direct','persugpt']
ALL_METHODS=['zero_shot_repaired','dialogue_sft','persugpt','ma2p_repaired','latentratt','our_direct']
OWNED={}
SCORES_LOCK=threading.RLock()

def state(**values):
    atomic_json(WORK/'live_status.json',dict(time=time.time(),supervisor_pid=os.getpid(),**values))

def launch(name,gpu,cmd,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
    env['PYTHONUNBUFFERED']='1';env['TOKENIZERS_PARALLELISM']='false'
    env['TRITON_CACHE_DIR']=str(WORK/'native_triton_cache')
    env['PYTHONPATH']=':'.join([str(REPAIR),str(CODE/'seven_methods_192'),str(CODE),str(ROOT/'exp_code_ljj'),str(ROOT/'baseline5_latentRATT'),str(CODE/'our_ma2p_deps')])
    with log.open('a') as stream:
        p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
    OWNED[name]=p
    atomic_json(WORK/'workers'/f'{name}.json',{'pid':p.pid,'gpu':gpu,'command':cmd,'log':str(log),'time':time.time()})
    return p

def stop_owned(name):
    p=OWNED.get(name)
    if p and p.poll() is None:
        p.terminate()
        try:p.wait(timeout=45)
        except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)

def _update_scores():
    result={'time':time.time(),'fixed_subset_n':192,'score_policy':'partial descriptive monitoring; no output or weight selection by scores','judges':{}}
    lines=['# Llama：Our / PersuGPT 实时进度','','固定192条；不足192时为临时分数。训练与权重不变。Success为五轮任一轮明确接受目标；Win含平局credit，换序两次取平均。','',
           '| Judge | Method | 已生成 | 已评估 | Success (%) | Win@1 (%) | Win-Full (%) | 平均轮数 | Win@1换序冲突 (%) |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for judge in ['selene8b','qwen3_14b']:
        result['judges'][judge]={}
        for method in METHODS:
            generated=len(read_jsonl(WORK/'generation'/f'{method}.jsonl')) if (WORK/'generation'/f'{method}.jsonl').exists() else 0
            path=WORK/'judges'/judge/'evaluation'/method/'summary.json'
            summary=read_json(path) if path.exists() else {}
            result['judges'][judge][method]=dict(summary,generated_n=generated)
            def val(key, digits=2):
                v=summary.get(key)
                return '-' if v is None else f'{v:.{digits}f}'
            lines.append(f'| {judge} | {method} | {generated}/192 | {summary.get("n",0)}/192 | {val("success_percent")} | {val("win1_percent")} | {val("win_full_percent")} | {val("mean_rounds_fail5",3)} | {val("win1_conflict_percent")} |')
        a=result['judges'][judge][METHODS[0]].get('n',0);b=result['judges'][judge][METHODS[1]].get('n',0)
        result['judges'][judge]['same_prefix_comparable_n']=min(a,b)
    atomic_json(WORK/'live_scores.json',result)
    (WORK/'live_scores.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'exp_code_ljj/results/llama_two_stage_full192_live.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def update_scores():
    with SCORES_LOCK:
        _update_scores()

def prepare():
    pilot=WORK/'pilot_status.json'
    while True:
        data=read_json(pilot) if pilot.exists() else {}
        if data.get('stage')=='failed':raise RuntimeError('Pilot failed: '+str(data))
        if data.get('stage')=='pilot_complete':break
        time.sleep(15)
    for method in ALL_METHODS:
        if method=='our_direct':continue
        source=BASE/'generators/persugpt' if method=='persugpt' else OLD/'generation'/f'{method}.jsonl'
        target=WORK/'generators/persugpt' if method=='persugpt' else WORK/'generation'/f'{method}.jsonl'
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists() and not target.is_symlink():target.symlink_to(source)
        if method=='persugpt':
            target=WORK/'generation/persugpt.jsonl'
            if not target.exists() and not target.is_symlink():target.symlink_to(BASE/'generation/persugpt.jsonl')
    atomic_json(WORK/'repair_plan.json',{'core_training_sha256':digest(CODE/'eclp_elpo.py'),'offline_trainer_sha256':digest(CODE/'offline_latent_rl.py'),'our_interface':'two_stage_private_note_then_public_reply','baseline_interfaces':'unchanged','fixed_n':192})
    return Path(read_json(ROOT/'eclp_runs_meta_llama31_20261007/offline_rl/summary.json')['checkpoint'])

def evaluate_judge(judge,gpu,methods,stage):
    for method in methods:
        source=WORK/'generation'/f'{method}.jsonl'
        if not source.exists():continue
        count=len(read_jsonl(source))
        if not count:continue
        summary=WORK/'judges'/judge/'evaluation'/method/'summary.json'
        if summary.exists() and read_json(summary).get('n')==count:continue
        cmd=[PY,'-u',str(REPAIR/'evaluate.py'),'--repo-root',str(ROOT),'--output',str(WORK),'--judge',judge,'--method',method,'--physical-gpu',str(gpu),'--limit','192','--batch-size','4','--memory-cap-gib','22' if judge=='selene8b' else '34','--available-only']
        p=launch('judge_'+judge,gpu,cmd,WORK/'judges'/judge/(method+'.v3.log'))
        while p.poll() is None:
            update_scores();time.sleep(10)
        if p.returncode:raise RuntimeError(f'{judge}/{method}: exit {p.returncode}')
        update_scores()

def publish_final():
    # Publish only complete, provenance-checked scores from the new evaluation.
    gold={r['index']:r for r in read_jsonl(ROOT/'eclp_runs_gpu4/evaluation_v2/golden/golden.jsonl')}
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('repair_evaluate',REPAIR/'evaluate.py');module=module_from_spec(spec);spec.loader.exec_module(module)
    lines=['## Llama-3.1-8B — two-stage Our + public repair v3, fixed192','',
           'Our 与 PersuGPT使用原训练权重全新生成192条，Our先生成私有简短理由、再输出公开劝说话语；PersuGPT公开只输出劝说话语；其余已完成基线复用上一版本的原始对话。各方法共用不接收精确目标的模拟被劝说者与相同解码设置。zero-shot劝说者仍只接收背景与公开历史。','',
           '所有方法统一使用llama_public_repair_v3中的成功判定与Win提示词。Success为五轮任一轮明确接受准确目标；Win@1对比首句、Win-Full对比全部可用对话，分别与同场景golden对比。换序credit取平均（胜1、平0.5、负0）；换序冲突率公开披露。格式缺失且无已确认成功的场景排除Success分母。','',
           f'结果与合同：`{WORK}`。本版本的Judge规则已修订，不能将旧Qwen2.5表直接作为相同评估版本的比较。','',
           '| Judge | Method | N | Success (%) | Win@1 (%) | Win-Full (%) | Mean rounds (failure=5) | Success unresolved | Win@1 order conflict (%) | Win-Full order conflict (%) |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for judge in ['selene8b','qwen3_14b']:
        for method in ALL_METHODS:
            p=WORK/'judges'/judge/'evaluation'/method/'summary.json'
            if not p.exists() or not read_json(p).get('complete'):
                lines.append(f'| {judge} | {method} | pending/192 | - | - | - | - | - | - | - |');continue
            source=WORK/'generation'/f'{method}.jsonl';rows={r['index']:r for r in read_jsonl(source)}
            records=read_jsonl(p.with_name('judgments.jsonl'));summary=module.summarize(records,192)
            assert summary['complete'] and len(rows)==192
            for r in records:
                assert r['source_row_hash']==object_hash(rows[r['index']]) and r['golden_row_hash']==object_hash(gold[r['index']])
            vals=[f'{summary[k]:.2f}' if summary[k] is not None else '-' for k in ['success_percent','win1_percent','win_full_percent','mean_rounds_fail5']]
            lines.append(f'| {judge} | {method} | 192/192 | '+ ' | '.join(vals)+f' | {summary["format_unresolved"]} | {summary["win1_conflict_percent"]:.2f} | {summary["win_full_conflict_percent"]:.2f} |')
    text='\n'.join(lines)+'\n'
    table=ROOT/'exp_code_ljj/results/main_experiment_table_llama31.md'
    backup=table.with_name(table.name+'.before_public_v3_20261009')
    if table.exists() and not backup.exists():backup.write_bytes(table.read_bytes())
    table.write_text(text)
    (WORK/'main_experiment_table_llama31.md').write_text(text)
    atomic_json(WORK/'PUBLISHED.json',{'time':time.time(),'table':str(table),'core_training_unchanged':digest(CODE/'eclp_elpo.py')==read_json(WORK/'repair_plan.json')['core_training_sha256']})

from contextlib import contextmanager
import fcntl
@contextmanager
def inference_slot():
    lock=ROOT/'exp_code_ljj/artifacts/.two_stage_inference_gpu14.lock'
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def main():
    checkpoint=prepare();update_scores()
    for threshold in list(range(20,192,20))+[192]:
        with inference_slot():
            counts={m:len(read_jsonl(WORK/'generation'/f'{m}.jsonl')) if (WORK/'generation'/f'{m}.jsonl').exists() else 0 for m in METHODS}
            if any(n<threshold for n in counts.values()):
                socket_path=f'/tmp/llama-public-v3-{os.getpid()}.sock'
                url='unix:'+socket_path
                state(stage='simulator_loading',target_prefix_n=threshold,generated=counts)
                simulator=launch('simulator',4,[PY,'-u',str(REPAIR/'serve_simulator.py'),'--root',str(ROOT),'--socket',socket_path,'--status',str(WORK/'simulator_status.json')],WORK/'simulator.log')
                ready=False
                for attempt in range(240):
                    if simulator.poll() is not None:raise RuntimeError('Shared simulator exited during loading')
                    try:
                        ready=unix_request(socket_path,'/health',timeout=2).get('ready') is True
                    except Exception:pass
                    if ready:break
                    time.sleep(5)
                if not ready:raise RuntimeError('Simulator startup timeout')
                state(stage='generating_parallel',target_prefix_n=threshold,generated=counts)
                workers=[]
                for method,gpu in [('our_direct',1),('persugpt',4)]:
                    if counts[method]>=threshold:continue
                    cmd=[PY,'-u',str(JOB/('generate_llama_two_stage.py' if method=='our_direct' else 'generate_llama_public_v3.py')),'--repo-root',str(ROOT),'--output',str((BASE if method=='persugpt' else WORK)/'generators'/method),'--family','meta_llama31','--config',str(ROOT/'eclp_runs_meta_llama31_20261007/config.json'),'--physical-gpu',str(gpu),'--method',method,'--limit','192','--stop-after',str(threshold),'--memory-cap-gib','22','--fresh-generation','--persuadee-goal-blind','--simulator-url',url]
                    if method=='our_direct':cmd+=['--checkpoint',str(checkpoint)]
                    workers.append(launch(method,gpu,cmd,WORK/'generators'/method/'worker.v3.log'))
                while any(p.poll() is None for p in workers):
                    if any(p.poll() not in [None,0] for p in workers):raise RuntimeError('A generation worker failed; see its log')
                    update_scores();time.sleep(10)
                if any(p.returncode for p in workers):raise RuntimeError('Generation worker failed')
                stop_owned('simulator')
            state(stage='evaluating_parallel',target_prefix_n=threshold)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(evaluate_judge,j,g,METHODS,threshold) for j,g in [('selene8b',1),('qwen3_14b',4)]]
                for f in futures:f.result()
            update_scores()
    with inference_slot():
        state(stage='reevaluating_unchanged_baselines')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(evaluate_judge,j,g,['zero_shot_repaired','dialogue_sft','ma2p_repaired','latentratt'],192) for j,g in [('selene8b',1),('qwen3_14b',4)]]
            for f in futures:f.result()
    latent=WORK/'generation/latentratt.jsonl'
    while not latent.exists() or len(read_jsonl(latent))!=192:
        state(stage='waiting_latentratt_generation',generated_n=len(read_jsonl(latent)) if latent.exists() else 0)
        update_scores();time.sleep(30)
    with inference_slot():
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(evaluate_judge,j,g,['latentratt'],192) for j,g in [('selene8b',1),('qwen3_14b',4)]]
            for f in futures:f.result()
    for j in ['selene8b','qwen3_14b']:
        for m in ALL_METHODS:
            final=read_json(WORK/'judges'/j/'evaluation'/m/'summary.json')
            assert final.get('complete') and final['n']==192,(j,m)
    plan=read_json(WORK/'repair_plan.json')
    assert digest(CODE/'eclp_elpo.py')==plan['core_training_sha256']
    assert digest(CODE/'offline_latent_rl.py')==plan['offline_trainer_sha256']
    publish_final();state(stage='complete_fixed192_generation_and_evaluation');update_scores()

if __name__=='__main__':
    try:main()
    except Exception as exc:
        for name in list(OWNED):stop_owned(name)
        state(stage='failed',error=repr(exc));update_scores();raise
