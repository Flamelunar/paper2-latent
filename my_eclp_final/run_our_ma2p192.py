#!/usr/bin/env python3
"""Durable generation -> Selene evaluation -> verified table update."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from experiment_io import atomic_json,read_json,read_jsonl,writer_lock


def run(args):
    root=args.repo_root.resolve();work=root/'eclp_runs_gpu4/our_ma2p192';work.mkdir(parents=True,exist_ok=True)
    state=work/'pipeline_status.json'
    scripts=root/'my_eclp_final';python=sys.executable
    commands=[('generation',[python,'-u',str(scripts/'generate_our_ma2p192.py'),'--repo-root',str(root),
              '--checkpoint',str(root/'eclp_runs_gpu4/offline_rl/checkpoint-001800'),'--output',str(work/'generation'),
              '--checkpoint-mode',args.checkpoint_mode,'--actor-device','cuda:0','--persuadee-device','cuda:0'],
              dict(os.environ,CUDA_VISIBLE_DEVICES='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONUNBUFFERED='1')),
              ('selene_evaluation',[python,'-u',str(scripts/'evaluate_our_ma2p192.py'),'--repo-root',str(root),
              '--generation',str(work/'generation/generation.jsonl'),'--golden',str(root/'eclp_runs_gpu4/evaluation_v2/golden'),
              '--output',str(root/'eclp_runs_gpu4/evaluation_v2/selene8b/our_ma2p'),'--judge','selene8b',
              '--method','our_ma2p','--physical-gpu','4','--batch-size','4','--memory-cap-gib','22'],
              dict(os.environ,CUDA_VISIBLE_DEVICES='4',CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONUNBUFFERED='1'))]
    with writer_lock(work/'.supervisor.lock'):
        try:
            for stage,command,env in commands:
                log=work/(stage+'.log')
                with log.open('a') as handle:
                    child=subprocess.Popen(command,cwd=root,env=env,stdout=handle,stderr=subprocess.STDOUT)
                    atomic_json(state,{'state':'running','stage':stage,'supervisor_pid':os.getpid(),'child_pid':child.pid,
                                       'expected_scenarios':192,'checkpoint_mode':args.checkpoint_mode,'command':command,
                                       'log':str(log),'started_at':time.strftime('%FT%T%z')})
                    print(stage,'pid',child.pid,flush=True)
                    code=child.wait()
                if code:raise RuntimeError(f'{stage} exited with code {code}; inspect {log}')
                if stage=='generation':
                    done=read_json(work/'generation/generation_summary.json')
                    if not done['complete'] or done['n']!=192:raise RuntimeError('192-scenario generation is incomplete')
            from our_ma2p192_table import verified_hybrid_summary
            from update_matched_table_v2 import update_table
            summary=verified_hybrid_summary(root)
            if summary is None:raise RuntimeError('Selene summary is not complete')
            report=update_table(root/'exp_code_ljj/results/main_experiment_table.md',root/'eclp_runs_gpu4/evaluation_v2')
            atomic_json(work/'final_result.json',{'summary':summary,'table_update':report})
            atomic_json(state,{'state':'complete','stage':'table_updated','expected_scenarios':192,
                               'checkpoint_mode':args.checkpoint_mode,'summary':summary,'table_update':report,
                               'finished_at':time.strftime('%FT%T%z')})
            print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
        except Exception as exc:
            atomic_json(state,{'state':'failed','error':f'{type(exc).__name__}: {exc}','supervisor_pid':os.getpid(),
                               'checkpoint_mode':args.checkpoint_mode,'updated_at':time.strftime('%FT%T%z')})
            traceback.print_exc();raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--checkpoint-mode',choices=['full_eclp','lora_only'],default='full_eclp')
    run(p.parse_args())
if __name__=='__main__':main()
