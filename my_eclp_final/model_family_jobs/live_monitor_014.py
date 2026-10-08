import json,time,subprocess
from pathlib import Path
root=Path('/app/hzj/liujianjian/ljj-2-paper')
pids=[264531,263529,4005823,4038766,2029468,590444,590492,604231]
result={'time':time.time(),'processes':{}}
for pid in pids:
    proc=Path('/proc')/str(pid)
    try:
        cmd=(proc/'cmdline').read_bytes().decode().replace('\0',' ')
        stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
        result['processes'][str(pid)]={'cmd':cmd,'state':stat[0],'cpu_ticks':int(stat[11])+int(stat[12]),'ppid':int(stat[1])}
    except FileNotFoundError: result['processes'][str(pid)]={'missing':True}
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
path=root/'exp_code_ljj/artifacts/acceleration_gpu014_latest_monitor.json'
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
