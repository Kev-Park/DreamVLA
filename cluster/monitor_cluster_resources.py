"""Read-only minute-by-minute cluster activity; stop contacting a host on SSH failure."""
import concurrent.futures
import json
import html
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
 except subprocess.TimeoutExpired:
  return 'PROBE_TIMEOUT: '+ ' '.join(args)
out={'time':time.time(),'gpus':cmd(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader']), 'gpu_processes':cmd(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']), 'jobs':{}}
for directory in [root/'checkpoints/_ftlogs',root/'DreamVLA']:
 for p in directory.glob('af60v8f57*'):
  if p.suffix not in ('.log','.out') or not p.is_file():continue
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-24000));tail=f.read().decode('utf8','replace')
  progress=re.findall(r'([0-9]+)/([0-9]+)\s*\[([^\r\n]+)',tail)
  out['jobs'][p.name]={'mtime':p.stat().st_mtime,'progress':progress[-1:]}
out['slurm']=cmd(['squeue','-u','kp0374','--noheader']) if (root/'DreamVLA/cluster/adroit').exists() and str(root).startswith('/home/kp0374') else None
out['pick_comparison']={}
for run,tag in [('pick10','pick10'),('pick50','pick50'),('af60v8f57_run01','af60v8f57')]:
 work=root/'dline'/tag
 status=work/'status'
 if not status.exists():continue
 text=status.read_text(errors='replace')
 matches=re.findall(re.escape(run)+r' GR00T eval: (\d+)/(\d+) = ([0-9.]+)%',text)
 item={'eval_done':(work/('ev_'+run+'.done')).exists(),'success':list(matches[-1]) if matches else None}
 metric=root/'eval_videos'/(run+'_mpjpe.json')
 if metric.exists():
  data=json.loads(metric.read_text())
  item.update(episodes=len(data.get('episodes',[])),missing=len(data.get('missing_metric_data',[])),
   world_cm=data.get('world_mpjpe_cm_episode_mean'),root_cm=data.get('root_aligned_mpjpe_cm_episode_mean'))
 out['pick_comparison'][run]=item
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


def write_comparison(report):
    blue=report['hosts'].get('bluesclues',{})
    rows=[]
    for run,label,demos in [('pick10','pick10',10),('pick50','pick50',50),('af60v8f57_run01','Full-base pick',342)]:
        data=blue.get('pick_comparison',{}).get(run,{})
        complete=(data.get('eval_done') and data.get('success') and
                  int(data['success'][1])==100 and data.get('episodes')==100 and
                  data.get('missing')==0 and data.get('world_cm') is not None and data.get('root_cm') is not None)
        values=[label,str(demos),'5.7',
                f"{data['success'][0]}/100 ({data['success'][2]}%)" if complete else 'Pending',
                f"{data['world_cm']:.2f}" if complete else 'Pending',
                f"{data['root_cm']:.2f}" if complete else 'Pending']
        rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
    timestamp=time.strftime('%Y-%m-%d %H:%M:%S %Z',time.localtime(report['time']))
    message='SSH access unavailable; results pending verification.' if blue.get('blocked') else '100 held-out references; episode-mean MPJPE in cm. Root alignment removes translation only.'
    page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>Pick comparison</title><style>body{font:16px system-ui;margin:40px}table{border-collapse:collapse}th,td{padding:12px;border:1px solid #bbb;text-align:left}</style><h1>Pick comparison</h1><p>Updated '+html.escape(timestamp)+'</p><table><tr><th>Run</th><th>Demos</th><th>Target effective epochs</th><th>Success</th><th>World MPJPE</th><th>Root-aligned MPJPE</th></tr>'+''.join(rows)+'</table><p>'+html.escape(message)+'</p>'
    destination=ROOT.parent/'pick_comparison.html'
    tmp=destination.with_suffix('.tmp');tmp.write_text(page,encoding='utf8');tmp.replace(destination)


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        while not (ROOT/'cluster_monitor.stop').exists():
            report={'time':time.time(),'hosts':dict(pool.map(read_host,HOSTS.items()))}
            report['alerts']=[{'host':h,'reason':v['blocked']} for h,v in report['hosts'].items() if 'blocked' in v]
            tmp=ROOT/'cluster_resources.tmp'
            tmp.write_text(json.dumps(report,indent=2));tmp.replace(ROOT/'cluster_resources.json')
            write_comparison(report)
            print(json.dumps(report),flush=True)
            time.sleep(60)


if __name__=='__main__':
    main()
