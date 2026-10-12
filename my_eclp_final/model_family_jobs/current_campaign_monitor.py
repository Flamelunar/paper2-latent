import json,time,subprocess
from pathlib import Path
root=Path('/app/hzj/liujianjian/ljj-2-paper')
work=root/'exp_code_ljj/artifacts/deepseek_goal_strategy_192_20261008/deepseek_r1_qwen14b'
needles=['train_eclp_sft_distributed.py','train_deepseek_our.py','generate_family.py','generate_deepseek_goal_strategy_split.py','launch_split_ma2p.py','probe_llama_latent_lr1e6_500.py','train_with_budget.py','deepseek_goal_strategy_campaign.py','deepseek_selene_pipeline.py','llama_latentratt_accelerated.py','train_deepseek_latentratt.py','seven_methods_192/evaluate.py','llama_fresh_campaign.py','generate_deepseek_goal_strategy.py','generate_latent_family.py','probe_llama_latent_lr1e6.py']
needles += ['generate_llama_two_stage.py','generate_qwen_two_stage.py','generate_deepseek_two_stage.py','generate_llama_public_v3.py','serve_simulator.py','llama_public_repair_v3/evaluate.py','supervise_llama_two_stage_full192.py','supervise_qwen_two_stage_full192.py','supervise_deepseek_uniform_two_stage_eval.py','overlap_selene_generation.py','sync_main_tables.py']
result={'time':time.time(),'live_processes':{},'deepseek_evaluations':{},'deepseek_generation':{}}
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():continue
    try:
        command=(entry/'cmdline').read_bytes().decode(errors='replace').replace('\0',' ')
        if not any(n in command for n in needles):continue
        if str(root) not in command and (entry/'cwd').resolve()!=root:continue
        stat=(entry/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':continue
        result['live_processes'][entry.name]={'state':stat[0],'cpu_ticks':int(stat[11])+int(stat[12]),'command':command}
    except (FileNotFoundError,ProcessLookupError,PermissionError):continue
for judge in ['selene8b','qwen3_14b']:
    for method in ['zero_shot_repaired','dialogue_sft','persugpt','ma2p_repaired','our_direct','latentratt']:
        d=work/'judges'/judge/'evaluation'/method
        summary=d/'summary.json';rows=d/'judgments.jsonl'
        count=sum(1 for line in rows.open() if line.strip()) if rows.exists() else 0
        info={'committed_scenes':count,'complete':False}
        if summary.exists():info.update(json.loads(summary.read_text()))
        caches=d/'request_chunks'
        if caches.exists():
            info['cached_partial_requests']=sum(len(json.loads(p.read_text()).get('responses',[])) for p in caches.glob('*.partial.json'))
        result['deepseek_evaluations'][judge+':'+method]=info
for method in ['zero_shot_repaired','dialogue_sft','persugpt','ma2p_repaired','our_direct','latentratt']:
    path=work/'generators'/method/'generation'/f'{method}.jsonl'
    count=sum(1 for line in path.open() if line.strip()) if path.exists() else 0
    result['deepseek_generation'][method]=count
result['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True)
result['current_two_stage_campaigns']={}
for family,relative in {
    'llama':'llama_two_stage_pilot_20261009/meta_llama31',
    'qwen':'qwen_two_stage_192_20261009/qwen25',
    'deepseek':'deepseek_two_stage_eval_v3_192_20261009/deepseek_r1_qwen14b',
}.items():
    directory=root/'exp_code_ljj/artifacts'/relative
    status=directory/'live_status.json'
    entry={'status':json.loads(status.read_text()) if status.exists() else {'stage':'queued'},'generation':{},'judges':{}}
    for method in ['zero_shot_repaired','dialogue_sft','persugpt','ma2p_repaired','our_direct','latentratt']:
        path=directory/'generation'/f'{method}.jsonl'
        entry['generation'][method]=sum(bool(line.strip()) for line in path.open()) if path.exists() else 0
        for judge in ['selene8b','qwen3_14b']:
            d=directory/'judges'/judge/'evaluation'/method
            summary=d/'summary.json';progress=d/'evaluation_progress.json'
            entry['judges'][judge+':'+method]={
                'summary':json.loads(summary.read_text()) if summary.exists() else {},
                'request_progress':json.loads(progress.read_text()) if progress.exists() else {},
            }
    result['current_two_stage_campaigns'][family]=entry
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
(root/'exp_code_ljj/artifacts/acceleration_gpu014_latest_monitor.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'time':result['time'],'pids':list(result['live_processes']),'generation':result['deepseek_generation'],'evaluations':{k:{key:v[key] for key in ['committed_scenes','complete','cached_partial_requests'] if key in v} for k,v in result['deepseek_evaluations'].items()},'two_stage':{f:{'stage':v['status'].get('stage'),'generation':v['generation'],'evaluated_n':{k:x['summary'].get('n',0) for k,x in v['judges'].items()}} for f,v in result['current_two_stage_campaigns'].items()},'gpu':result['gpu']},indent=2))
