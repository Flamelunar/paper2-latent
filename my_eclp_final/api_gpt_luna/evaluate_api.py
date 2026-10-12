"""Public-only fixed192 API evaluation, persistent cache, no GPU dependencies."""
import concurrent.futures as futures
import hashlib
import json
import random
import re
import sqlite3
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = 'gpt-6-luna'
URL = 'https://api.chatanywhere.tech/v1/chat/completions'
KEY = Path('/home/hezj/.config/paper2-luna/api_key').read_text().strip()

def validate(raw, request):
    def pairs(items):
        d = {}
        for k,v in items:
            if k in d: raise ValueError('Duplicate key')
            d[k] = v
        return d
    v = json.loads(raw, object_pairs_hook=pairs)
    if request['kind']=='win':
        if not isinstance(v,dict) or set(v)!={'winner'} or v['winner'] not in ('A','B','Tie'):
            raise ValueError('Invalid winner JSON')
    else:
        if not isinstance(v,dict) or set(v)!={'evidence','progress','success'}: raise ValueError('Invalid success fields')
        if type(v['success']) is not bool: raise ValueError('Success is not boolean')
        if v['progress'] not in ('none','engagement','conditional','accepted'): raise ValueError('Invalid progress')
        if v['success'] != (v['progress']=='accepted'): raise ValueError('Inconsistent progress')
        if not isinstance(v['evidence'],str) or len(v['evidence'])>240: raise ValueError('Invalid quote')
        norm = lambda s: re.sub(r'\s+',' ',s).strip()
        quote = norm(v['evidence'])
        if quote and quote not in norm(request['current_reply']): raise ValueError('Quote not in current reply')
        if v['progress']!='none' and not quote: raise ValueError('Missing quote')
    return v

def call(request, system):
    payload = {'model':MODEL, 'messages':[{'role':'system','content':system},
               {'role':'user','content':request['prompt']}],
               'max_completion_tokens':1024, 'response_format':{'type':'json_object'}}
    attempts=[]
    format_failures=0
    for attempt in range(7):
        try:
            req=urllib.request.Request(URL,data=json.dumps(payload).encode(),
                headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=120) as response:
                body=json.load(response)
            raw=body['choices'][0]['message']['content']
            record={'raw_output':raw,'model':body.get('model'),'usage':body.get('usage'),
                    'finish_reason':body['choices'][0].get('finish_reason')}
            attempts.append(record)
            try: value=validate(raw,request)
            except (ValueError,TypeError) as e:
                record['format_error']=str(e);format_failures+=1
                if format_failures>=3:return {'value':None,'attempts':attempts,'unresolved':'format'}
                payload['messages']=[{'role':'system','content':system},
                    {'role':'user','content':request['prompt']+'\n\nFormat-only correction: the evidence JSON string must be a literal excerpt of the CURRENT Persuadee reply. Do not add quotation marks inside the evidence string, do not paraphrase, and do not quote a previous round. Keep the original judgment rubric unchanged.'}]
                continue
            return {'value':value,'attempts':attempts}
        except urllib.error.HTTPError as e:
            detail=e.read().decode(errors='replace')[:1200].replace(KEY,'[redacted]')
            attempts.append({'http_status':e.code,'error':detail})
            if e.code in (400,401,402,403,404):
                raise RuntimeError(f'API configuration/billing failure HTTP {e.code}: {detail}')
        except (OSError,KeyError,ValueError,TypeError) as e:
            attempts.append({'error':str(e).replace(KEY,'[redacted]')[:500]})
        time.sleep(min(2**attempt,30)+random.random())
    return {'value':None,'attempts':attempts,'unresolved':'transport'}

def summarize(scenes, results):
    grouped={}
    for s in scenes:
        prefix=f"{s['family']}/{s['method']}/{s['index']}/"
        if not all(prefix+r['key'] in results for r in s['requests']):continue
        grouped.setdefault((s['family'],s['method']),[]).append((s,prefix))
    output=[]
    for family,method in sorted({(s['family'],s['method']) for s in scenes}):
        done=grouped.get((family,method),[]);rounds=[];wins=[];success=0;unresolved=0;conflicts=0
        for s,prefix in done:
            goals=[results[prefix+f'goal_r{i}']['value'] for i in range(1,6)]
            first=next((i for i,v in enumerate(goals,1) if v and v['success']),None)
            unresolved+=sum(v is None for v in goals)
            if first is not None: success+=1;rounds.append(first)
            elif all(v is not None for v in goals):rounds.append(5)
            ab=results[prefix+'win1_ab']['value'];ba=results[prefix+'win1_ba']['value']
            if ab and ba:
                credit=lambda v,pos: .5 if v['winner']=='Tie' else float(v['winner']==pos)
                a,b=credit(ab,'A'),credit(ba,'B');wins.append((a+b)/2);conflicts+=a!=b
        output.append({'family':family,'method':method,'n':len(done),'expected':192,'complete':len(done)==192,
            'success_evaluable_n':len(rounds),'success_count':success,
            'success_percent':100*success/len(rounds) if rounds else None,
            'win1_evaluable_n':len(wins),'win1_percent':100*statistics.fmean(wins) if wins else None,
            'mean_rounds_fail5':statistics.fmean(rounds) if rounds else None,'format_unresolved':unresolved,
            'win1_order_conflict_percent':100*conflicts/len(wins) if wins else None})
    atomic(ROOT/'summary.json',{'judge':MODEL,'updated':time.time(),'completed_requests':len(results),
        'total_requests':len(scenes)*7,'methods':output})
    lines=['# GPT-6-luna API evaluation — fixed192','',
        'Latest shared v3 goal and Win@1 rubrics; public dialogue only. Five rounds, first accepted round; failures=5. Swapped preference credits averaged. Partial scores are provisional. Generation inputs retain their existing family-specific conditions.','',
        '| Model | Method | N | Success (%) | Win@1 (%) | Mean rounds (failure=5) | Unresolved success rounds |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in output:
        fmt=lambda k: '-' if r[k] is None else f'{r[k]:.2f}'
        lines.append(f"| {r['family']} | {r['method']} | {r['n']}/192 | {fmt('success_percent')} | {fmt('win1_percent')} | {fmt('mean_rounds_fail5')} | {r['format_unresolved']} |")
    (ROOT/'main_experiment_table_gpt_luna.md').write_text('\n'.join(lines)+'\n')

def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)

def main():
    data=json.loads((ROOT/'inputs.json').read_text());scenes=data['scenes'];system=data['manifest']['system']
    # The shared system and win output contract must explicitly mention JSON.
    assert 'json' in system.lower() or all('json' in r['prompt'].lower() for s in scenes for r in s['requests'])
    contract=dict(data['manifest'],model=MODEL,url=URL,max_completion_tokens=1024,
                  response_format='json_object',concurrency=48,temperature='API default',
                  format_retry='original rubric plus literal-current-quote formatting reminder',
                  source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (ROOT/'run_contract.json').exists():
        prior=json.loads((ROOT/'run_contract.json').read_text())
        if prior!=contract:
            for key in data['manifest']: assert prior[key]==contract[key]
            for key in ['model','url','max_completion_tokens','response_format','temperature']:assert prior[key]==contract[key]
            revision=ROOT/'contract_before_format_retry.json'
            if not revision.exists():atomic(revision,prior)
            atomic(ROOT/'run_contract.json',contract)
    else:atomic(ROOT/'run_contract.json',contract)
    db=sqlite3.connect(ROOT/'requests.sqlite3');db.execute('create table if not exists responses (id text primary key, result text not null)');db.commit()
    results={k:json.loads(v) for k,v in db.execute('select id,result from responses')}
    for key,value in list(results.items()):
        if value['value'] is None:
            with (ROOT/'previous_unresolved.jsonl').open('a') as f:
                f.write(json.dumps({'id':key,'result':value},ensure_ascii=False)+'\n')
            db.execute('delete from responses where id=?',(key,));del results[key]
    db.commit()
    pending=[]
    for s in sorted(scenes,key=lambda s:(s['index'],s['family'],s['method'])):
        for r in s['requests']:
            key=f"{s['family']}/{s['method']}/{s['index']}/{r['key']}"
            if key not in results:pending.append((key,r))
    summarize(scenes,results)
    print(f'Start {MODEL}: {len(results)}/{len(scenes)*7} cached; concurrency 48',flush=True)
    iterator=iter(pending);active={};completed=0;last=time.monotonic()
    with futures.ThreadPoolExecutor(max_workers=48) as pool:
        def submit():
            item=next(iterator,None)
            if item:active[pool.submit(call,item[1],system)]=item
        for _ in range(96):submit()
        while active:
            done,_=futures.wait(active,timeout=10,return_when=futures.FIRST_COMPLETED)
            for f in done:
                key,r=active.pop(f)
                try:value=f.result()
                except Exception as e:
                    atomic(ROOT/'fatal_error.json',{'time':time.time(),'request':key,'error':str(e)})
                    for job in active:job.cancel()
                    raise
                value['prompt_sha256']=hashlib.sha256(r['prompt'].encode()).hexdigest()
                db.execute('insert into responses values (?,?)',(key,json.dumps(value,ensure_ascii=False)));db.commit()
                results[key]=value;completed+=1;submit()
            if time.monotonic()-last>=30:
                summarize(scenes,results);last=time.monotonic()
                print(f'Completed {len(results)}/{len(scenes)*7} requests',flush=True)
    summarize(scenes,results);db.close()
    print('API evaluation complete; inspect unresolved counts before treating scores as final.',flush=True)

if __name__=='__main__': main()
