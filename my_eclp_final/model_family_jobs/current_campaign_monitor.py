import json,time,subprocess
from pathlib import Path
root=Path('/app/hzj/liujianjian/ljj-2-paper')
work=root/'exp_code_ljj/artifacts/deepseek_goal_strategy_192_20261008/deepseek_r1_qwen14b'
needles=['train_eclp_sft_distributed.py','train_deepseek_our.py','generate_family.py','generate_deepseek_goal_strategy_split.py','launch_split_ma2p.py','probe_llama_latent_lr1e6_500.py','train_with_budget.py','deepseek_goal_strategy_campaign.py','deepseek_selene_pipeline.py','llama_latentratt_accelerated.py','train_deepseek_latentratt.py','seven_methods_192/evaluate.py','llama_fresh_campaign.py','generate_deepseek_goal_strategy.py','generate_latent_family.py','probe_llama_latent_lr1e6.py']
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
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
(root/'exp_code_ljj/artifacts/acceleration_gpu014_latest_monitor.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'time':result['time'],'pids':list(result['live_processes']),'generation':result['deepseek_generation'],'evaluations':{k:{key:v[key] for key in ['committed_scenes','complete','cached_partial_requests'] if key in v} for k,v in result['deepseek_evaluations'].items()},'gpu':result['gpu']},indent=2))
