"""Read-only minute-by-minute cluster activity; stop contacting a host on SSH failure."""
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]/'out/runpod'
PROBE = r'''
import json,pathlib,re,subprocess,time
root=pathlib.Path.home()/'kevin'
def cmd(args):
 try:
  r=subprocess.run(args,capture_output=True,text=True,timeout=15)
  return r.stdout.strip() if r.returncode==0 else r.stderr.strip()[-500:]
 except FileNotFoundError:return None
out={'time':time.time(),'gpus':cmd(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader']), 'gpu_processes':cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']), 'jobs':{}}
for directory in [root/'checkpoints/_ftlogs',root/'DreamVLA']:
 for p in directory.glob('af60v8f57*'):
  if p.suffix not in ('.log','.out') or not p.is_file():continue
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-24000));tail=f.read().decode('utf8','replace')
  progress=re.findall(r'([0-9]+)/([0-9]+)\s*\[([^\r\n]+)',tail)
  out['jobs'][p.name]={'mtime':p.stat().st_mtime,'progress':progress[-1:]}
out['slurm']=cmd(['squeue','-u','kp0374','--noheader']) if (root/'DreamVLA/cluster/adroit').exists() and str(root).startswith('/home/kp0374') else None
print(json.dumps(out))
'''
HOSTS = {
    'bluesclues': ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','sastrygrp-dvij@bluesclues.ist.berkeley.edu'],
    'biped': ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','dvij@biped.eecs.berkeley.edu'],
    'adroit': ['wsl.exe','-d','Ubuntu','-u','kevy0','--','ssh','-o','ControlPath=~/.ssh/adroit.sock','-o','ControlMaster=no','-o','BatchMode=yes','adroit'],
}


def read_host(item):
    name, args = item
    blocked = ROOT/('cluster_blocked_'+name)
    if blocked.exists():
        return name, {'blocked': blocked.read_text()}
    try:
        result = subprocess.run(args+['python3 -c '+shlex.quote(PROBE)], capture_output=True,
            text=True, timeout=45, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:
            raise RuntimeError(result.stderr[-1000:])
        return name, json.loads(result.stdout)
    except Exception as error:
        blocked.write_text(str(error))
        return name, {'blocked':str(error)}


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        while not (ROOT/'cluster_monitor.stop').exists():
            report={'time':time.time(),'hosts':dict(pool.map(read_host,HOSTS.items()))}
            report['alerts']=[{'host':h,'reason':v['blocked']} for h,v in report['hosts'].items() if 'blocked' in v]
            tmp=ROOT/'cluster_resources.tmp'
            tmp.write_text(json.dumps(report,indent=2));tmp.replace(ROOT/'cluster_resources.json')
            print(json.dumps(report),flush=True)
            time.sleep(60)


if __name__=='__main__':
    main()
