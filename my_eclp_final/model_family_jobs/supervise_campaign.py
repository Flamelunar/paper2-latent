"""Resume the fixed-192 model-family campaign without taking GPU0."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/app/hzj/liujianjian/ljj-2-paper')
CODE = ROOT / 'my_eclp_final'
JOB = CODE / 'model_family_jobs'
WORK = ROOT / 'exp_code_ljj/artifacts/model_family_repaired192_20261007'
PY = '/app/hzj/anaconda3/envs/ljj/bin/python'
FAMILIES = {
    'meta_llama31': {'gpu': 1, 'model': '/app/hzj/models/Meta-Llama-3.1-8B-Instruct', 'cap': 42},
    'deepseek_r1_qwen14b': {'gpu': 1, 'model': '/app/hzj/models/DeepSeek-R1-Distill-Qwen-14B', 'cap': 54},
}
GENS = ['dialogue_sft', 'persugpt', 'ma2p_repaired']
ALL = ['zero_shot_repaired', *GENS, 'our_direct']
TRAIN_GPU = 4
DISPLAY = {'zero_shot_repaired':'Zero-shot', 'dialogue_sft':'Dialogue-SFT',
           'persugpt':'ISR-SFT + SimDPO', 'ma2p_repaired':'MA2P', 'our_direct':'Our'}


def atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def env_for(gpu: int):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                PYTHONUNBUFFERED='1', DAILYPERSUASION_PROFILE='paper10k',
                TRITON_CACHE_DIR=f'/tmp/ljj2-paper-triton-{os.getuid()}')


def launch(name: str, gpu: int, command: list[str], log: Path, extra=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = log.open('a', encoding='utf-8')
    proc = subprocess.Popen(command, cwd=ROOT, env=env_for(gpu), stdin=subprocess.DEVNULL,
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    atomic(log.with_suffix('.worker.json'), {'name':name,'pid':proc.pid,'gpu':gpu,
          'command':command,'started':time.time(),**(extra or {})})
    return proc


def generation_path(family, method):
    return WORK / family / 'generators' / method


def completed_generation(family, method):
    out = generation_path(family, method)
    path = out / 'generation' / f'{method}.jsonl'
    summary = out / 'generation' / f'{method}_summary.json'
    if not path.is_file() or not summary.is_file():
        return False
    data = json.loads(summary.read_text())
    return data.get('complete') is True and data.get('n') == 192


def completed_evaluation(family, method, judge):
    summary = WORK/family/'judges'/judge/'evaluation'/method/'summary.json'
    if not summary.is_file():
        return False
    data = json.loads(summary.read_text())
    return data.get('complete') is True and data.get('n') == 192


def completed_zero_pair(family):
    return all(completed_evaluation(family, 'zero_shot_repaired', judge)
               for judge in ['selene8b', 'qwen3_14b'])


def start_gen(family, method, gpu):
    out = generation_path(family, method)
    out.mkdir(parents=True, exist_ok=True)
    model = FAMILIES[family]['model']
    config = ROOT / f'eclp_runs_{family}_20261007/config.json'
    cap_gib = 54 if 'deepseek' in family else 44
    required_mib = int(cap_gib*1024)
    while True:
        free_mib = int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True).strip())
        if free_mib >= required_mib:
            break
        atomic(JOB/'campaign_status.json',{'state':'waiting_for_generator_memory','family':family,
            'method':method,'gpu':gpu,'free_mib':free_mib,'required_mib':required_mib,'gpu0':'unused'})
        time.sleep(30)
    command = [PY,'-u',str(JOB/'generate_family.py'),'--repo-root',str(ROOT),
       '--output',str(out),'--family',family,'--config',str(config),'--physical-gpu',str(gpu),
       '--method',method,'--limit','192','--memory-cap-gib',str(cap_gib)]
    if method == 'our_direct':
        command += ['--checkpoint',str(ROOT/f'eclp_runs_{family}_20261007/offline_rl/checkpoint-001800')]
    return launch(f'generate:{family}:{method}',gpu,command,out/'worker.log')


def seed_global_generation_manifest(family):
    out = WORK / family
    selection = json.loads((ROOT/'exp_code_ljj/data/test_selection_192.json').read_text())
    import hashlib
    points = [json.loads(line) for line in (ROOT/'exp_code_ljj/data/evaluation_points.jsonl').read_text().splitlines()]
    indexes = sorted(selection['original_indexes'], key=lambda i: hashlib.sha256(
        ('matched-v2-pilot-42:'+points[i]['scenario_id']).encode()).hexdigest())
    contract = {'revision':'model_family_repaired_generation_20261007','family':family,
       'expected_indexes':indexes,'selection_sha256':selection['source_sha256'],
       'methods':ALL,'source_policy':'immutable completed per-method run contracts; each row hash is verified by the judge',
       'zero_shot_inputs':'background and public history only; no goal, psychology, strategy or golden',
       'our_checkpoint':str(ROOT/f'eclp_runs_{family}_20261007/offline_rl/checkpoint-001800'),
       'compute_dtype':'bfloat16','quantization':'none'}
    path = out/'generation_run.json'
    if path.exists():
        assert json.loads(path.read_text()) == contract, 'Global generation manifest changed during campaign'
    else:
        atomic(path, contract)
    (out/'generation').mkdir(exist_ok=True)


def publish_generation(family, method):
    src = generation_path(family,method)/'generation'/f'{method}.jsonl'
    dst = WORK/family/'generation'/f'{method}.jsonl'
    if dst.is_symlink():
        assert dst.resolve() == src.resolve()
    elif dst.exists():
        assert dst.read_bytes() == src.read_bytes()
    else:
        dst.symlink_to(src)


def verify_old_gen(family, method):
    p = generation_path(family,method)/'worker.json'
    if not p.exists():
        return None
    info = json.loads(p.read_text())
    if live(info['pid']):
        return info['pid']
    if completed_generation(family,method):
        return None
    raise RuntimeError(f'Existing generator {family}/{method} exited before producing a complete fixed-192 file; inspect {p.parent}/worker.log')


def train_family(family):
    work = ROOT/f'eclp_runs_{family}_20261007'; config = work/'config.json'; prepared=work/'prepared'
    sft = work/'sft_full_retry_20261007'; cap=FAMILIES[family]['cap']; gpu=TRAIN_GPU
    status=work/'job_status.json'
    required_mib=int((cap+8)*1024)
    while True:
        free_mib=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free_mib>=required_mib:break
        atomic(status,{'state':'waiting_for_gpu_memory','phase':'sft','gpu':gpu,
                       'free_mib':free_mib,'required_mib':required_mib,'supervisor_pid':os.getpid()})
        time.sleep(30)
    atomic(status,{'state':'running','phase':'sft','gpu':gpu,'supervisor_pid':os.getpid()})
    target = sft/'checkpoint-007484/checkpoint.json'
    if not target.exists():
        if (sft/'last.json').exists():
            last=json.loads((sft/'last.json').read_text()); cp=Path(last['checkpoint'])
            step=int(json.loads((cp/'checkpoint.json').read_text())['step'])
            if step>=7484: raise RuntimeError(f'Unexpected SFT checkpoint step {step}')
        else:
            cp=None;step=0
        cmd=[PY,'-u',str(JOB/'train_with_budget.py'),'--entry','eclp','--physical-gpu',str(gpu),
             '--memory-cap-gib',str(cap),'--','--config',str(config),'sft','--prepared',str(prepared),
             '--output',str(sft),'--epochs','1','--max-steps','7484','--accumulation','8','--lr','0.0001','--save-every','50']
        if cp:cmd += ['--checkpoint',str(cp),'--resume']
        proc=launch(f'train:{family}:sft',gpu,cmd,work/'sft_retry.log',{'stage':'sft'})
        code=proc.wait()
        if code:
            atomic(status,{'state':'failed','phase':'sft','exit_code':code,'gpu':gpu,'log':str(work/'sft_retry.log')});raise RuntimeError(f'SFT failed: {work}/sft_retry.log')
    plan=work/'plan.json';atomic(status,{'state':'running','phase':'offline_rl','gpu':gpu,'supervisor_pid':os.getpid()})
    rl=work/'offline_rl/last.json'
    rl_summary=work/'offline_rl/summary.json'
    rl_complete=rl_summary.exists() and json.loads(rl_summary.read_text()).get('complete') is True
    if not rl_complete:
        cmd=[PY,'-u',str(JOB/'train_with_budget.py'),'--entry','offline','--physical-gpu',str(gpu),
             '--memory-cap-gib',str(cap),'--','--config',str(config),'--plan',str(plan),'train',
             '--prepared',str(prepared),'--checkpoint',str(sft/'checkpoint-007484'),
             '--output',str(work/'offline_rl')]
        proc=launch(f'train:{family}:offline_rl',gpu,cmd,work/'offline_rl_retry.log',{'stage':'offline_rl'})
        code=proc.wait()
        if code:
            atomic(status,{'state':'failed','phase':'offline_rl','exit_code':code,'gpu':gpu,'log':str(work/'offline_rl_retry.log')});raise RuntimeError(f'offline RL failed: {work}/offline_rl_retry.log')
    final=json.loads(rl.read_text());assert Path(final['checkpoint']).is_dir()
    completed=json.loads(rl_summary.read_text())
    assert completed['complete'] is True and completed['checkpoint_id']==final['checkpoint_id'], 'RL is not fully committed'
    atomic(status,{'state':'complete','phase':'offline_rl','gpu':gpu,'checkpoint':final['checkpoint'],'checkpoint_id':final['checkpoint_id']})


def launch_eval(family,method,judge,gpu=4):
    out=WORK/family
    model_name='Selene' if judge=='selene8b' else 'Qwen3'
    memory_cap=22 if judge=='selene8b' else 32
    required_mib=(memory_cap+4)*1024
    while True:
        free_mib=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free_mib>=required_mib:break
        atomic(JOB/'campaign_status.json',{'state':'waiting_for_judge_memory','family':family,
               'method':method,'judge':judge,'gpu':gpu,'free_mib':free_mib,
               'required_mib':required_mib,'gpu0':'unused'})
        time.sleep(30)
    env=env_for(gpu)
    cmd=[PY,'-u',str(CODE/'seven_methods_192/evaluate.py'),'--repo-root',str(ROOT),
         '--output',str(out),'--physical-gpu',str(gpu),'--judge',judge,'--method',method,
         '--limit','192','--batch-size','2','--memory-cap-gib',str(memory_cap)]
    log=out/'judges'/judge/f'{method}.campaign.log'
    proc=launch(f'eval:{family}:{method}:{judge}',gpu,cmd,log,{'judge':model_name})
    code=proc.wait()
    if code: raise RuntimeError(f'Evaluation failed: {log}')
    atomic(log.with_suffix('.complete.json'),{'exit_code':code,'completed':time.time()})


def evaluate_zero_pair(family,gpu):
    for judge in ['selene8b','qwen3_14b']:
        if not completed_evaluation(family,'zero_shot_repaired',judge):
            launch_eval(family,'zero_shot_repaired',judge,gpu)


def run():
    WORK.mkdir(parents=True,exist_ok=True)
    lock=JOB/'campaign.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
    except FileExistsError:
        owner=json.loads(lock.read_text())
        if live(owner['pid']): raise RuntimeError(f'Campaign supervisor already live at pid {owner["pid"]}')
        lock.unlink();fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
    os.write(fd,json.dumps({'pid':os.getpid(),'started':time.time()}).encode());os.close(fd)
    state=JOB/'campaign_status.json'
    try:
        for family in FAMILIES: seed_global_generation_manifest(family)
        initial=[('meta_llama31','zero_shot_repaired',TRAIN_GPU),('deepseek_r1_qwen14b','zero_shot_repaired',TRAIN_GPU)]
        for family,method,gpu in initial:
            if not completed_generation(family,method):
                pid=verify_old_gen(family,method)
                if not pid: raise RuntimeError(f'{family}/{method} missing worker identity')
        atomic(state,{'state':'waiting_for_fixed192_zero_shot','gpu0':'unused','workers':initial})
        zero_evaluations={}
        for f,m,gpu in initial:
            if completed_generation(f,m):
                publish_generation(f,m)
                if completed_zero_pair(f):
                    zero_evaluations[f]=None
                else:
                    zero_evaluations[f]=launch(f'evaluate-zero:{f}',gpu,
                        [PY,'-u',str(JOB/'supervise_campaign.py'),'--eval-zero',f,str(gpu)],
                        JOB/f'evaluate_zero_{f}.log')
        while not all(completed_generation(f,m) for f,m,_ in initial):
            for f,m,gpu in initial:
                if completed_generation(f,m) and f not in zero_evaluations:
                    publish_generation(f,m)
                    worker=None
                    if not completed_zero_pair(f):
                        worker=launch(f'evaluate-zero:{f}',gpu,
                            [PY,'-u',str(JOB/'supervise_campaign.py'),'--eval-zero',f,str(gpu)],
                            JOB/f'evaluate_zero_{f}.log')
                    zero_evaluations[f]=worker
                    atomic(state,{'state':'zero_shot_complete_evaluating','gpu0':'unused',
                                  'generation_complete':f,'evaluation_gpu':gpu,
                                  'evaluation_workers':{k:v.pid for k,v in zero_evaluations.items() if v is not None}})
                elif f not in zero_evaluations:
                    verify_old_gen(f,m)
            time.sleep(30)
        for family,method,_ in initial: publish_generation(family,method)
        for family,worker in zero_evaluations.items():
            if worker is not None and worker.wait():raise RuntimeError(f'Zero-shot evaluation failed for {family}; inspect {JOB}/evaluate_zero_{family}.log')
        subprocess.run([PY,str(JOB/'report_family.py')],cwd=ROOT,check=True)
        atomic(state,{'state':'zero_shot_and_two_judges_complete','time':time.time(),'gpu0':'unused'})

        # Our weights are trained only after both untrained zero-shot runs finish.
        # Train and regenerate on GPU4; memory admission prevents overlapping models from exceeding capacity.
        training=launch('train:meta_llama31',TRAIN_GPU,[PY,'-u',str(JOB/'supervise_campaign.py'),'--train-only','meta_llama31'],JOB/'nested_llama.log')
        eval_ready=[]
        for family in FAMILIES:
            for method in GENS:
                if completed_generation(family,method):
                    publish_generation(family,method);continue
                proc=start_gen(family,method,4)
                code=proc.wait()
                if code: raise RuntimeError(f'Generation failed: {generation_path(family,method)}/worker.log')
                if not completed_generation(family,method):raise RuntimeError(f'Incomplete generation: {family}/{method}')
                publish_generation(family,method)
                eval_ready.append((family,method))
        code=training.wait()
        if code: raise RuntimeError('Model-family training supervisor failed; inspect nested_llama.log')
        # Training the second backbone follows the first on GPU4. Preserve all checkpoints.
        training=launch('train:deepseek_r1_qwen14b',TRAIN_GPU,[PY,'-u',str(JOB/'supervise_campaign.py'),'--train-only','deepseek_r1_qwen14b'],JOB/'nested_deepseek.log')
        for family in FAMILIES:
            proc=start_gen(family,'ma2p_repaired',4) if not completed_generation(family,'ma2p_repaired') else None
            if proc and proc.wait():raise RuntimeError(f'MA2P generation failed for {family}')
            if completed_generation(family,'ma2p_repaired'):publish_generation(family,'ma2p_repaired')
        # Judge each available method as it becomes ready; both Judge models use GPU4 sequentially.
        for family in FAMILIES:
            for method in ['zero_shot_repaired',*GENS,'ma2p_repaired']:
                if not completed_generation(family,method):continue
                for judge in ['selene8b','qwen3_14b']:
                    summary=WORK/family/'judges'/judge/'evaluation'/method/'summary.json'
                    if summary.exists() and json.loads(summary.read_text()).get('complete') is True:continue
                    launch_eval(family,method,judge)
                    subprocess.run([PY,str(JOB/'report_family.py')],cwd=ROOT,check=True)
        code=training.wait()
        if code:raise RuntimeError('DeepSeek Our training failed; inspect nested_deepseek.log')
        for family in FAMILIES:
            proc=start_gen(family,'our_direct',TRAIN_GPU) if not completed_generation(family,'our_direct') else None
            if proc and proc.wait():raise RuntimeError(f'Our generation failed for {family}')
            if completed_generation(family,'our_direct'):publish_generation(family,'our_direct')
            for judge in ['selene8b','qwen3_14b']:
                launch_eval(family,'our_direct',judge)
                subprocess.run([PY,str(JOB/'report_family.py')],cwd=ROOT,check=True)
        subprocess.run([PY,str(JOB/'report_family.py')],cwd=ROOT,check=True)
        atomic(state,{'state':'complete_except_missing_latentratt_checkpoints','time':time.time(),'gpu0':'unused'})
    except Exception as exc:
        atomic(state,{'state':'failed','error':repr(exc),'time':time.time(),'gpu0':'unused'})
        raise


def train_only(family):
    train_family(family)


if __name__=='__main__':
    if sys.argv[1:2]==['--eval-zero']:
        evaluate_zero_pair(sys.argv[2],int(sys.argv[3]))
    elif sys.argv[1:2]==['--train-only']:
        train_only(sys.argv[2])
    else:
        run()
