"""Independent, read-only pipeline audit; never retries a failed SSH connection.

Writes a heartbeat plus persistent alerts every minute. Does not restart workers,
change experiments, or interfere with active jobs.
"""
import html
import json
import math
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2] / 'out/runpod'
PROBE = r'''
import json, pathlib, subprocess, time
r=pathlib.Path.home()/'kevin'
p=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout.splitlines()
processes=[s for s in p if '/kevin/' in s and 'python3 -c' not in s]
out={'time':time.time(),'runs':{},'transfers':{},'lines':{},'workers':[]}
for s in processes:
 if not any(x in s for x in ('collect_sonic_adapter.py','convert_isaac_hdf5_to_lerobot.py','convert_isaac_incremental.py')):continue
 try:
  pid=int(s.split()[0]);log=pathlib.Path(f'/proc/{pid}/fd/1').resolve()
  if '/kevin/' not in str(log) or not log.is_file():continue
  st=log.stat();out['workers'].append({'pid':pid,'log':str(log),'size':st.st_size,'mtime':st.st_mtime})
 except (OSError,ValueError):pass
names=['pick1200_bf16_bs32','af60v8f57_run01']+[f'af60v8f57_dagger{i}_run01' for i in range(1,6)]
for name in names:
 tag='pick1200_bf16_bs32' if name.startswith('pick1200') else 'af60v8f57'
 w=r/'dline'/tag
 logs=list(w.glob('ev_'+name+'_*.log'))
 files=list((r/'eval_videos').glob(name+'_b*_traj.npz'))
 active=[s for s in processes if name in s and ('eval_vla_sonic.py' in s or 'runpod_receive.sh' in s or 'ckpt_transfer.py' in s)]
 item={'done':(w/('ev_'+name+'.done')).exists(),'episodes':len(files),'active':active,'logs':len(logs),
       'last_eval_output':max([f.stat().st_mtime for f in logs+files] or [0]),
       'checkpoint_ready':(r/'runpod_transport/received'/name).exists()}
 metric=r/'eval_videos'/(name+'_mpjpe.json')
 if metric.exists():
  d=json.loads(metric.read_text());item['metric']={'episodes':len(d.get('episodes',[])),
   'missing':len(d.get('missing_metric_data',[])),'world':d.get('world_mpjpe_cm_episode_mean'),
   'root':d.get('root_aligned_mpjpe_cm_episode_mean')}
 out['runs'][name]=item
 pack=r/'runpod_transport/packs'/name
 fs=list(pack.rglob('*')) if pack.exists() else []
 failed=r/'runpod_transport/received'/(name+'.failed')
 log=r/'runpod_transport'/(name+'.receive.log')
 tail=''
 if log.exists():
  with log.open('rb') as f:f.seek(max(0,log.stat().st_size-2500));tail=f.read().decode('utf8','replace')
 out['transfers'][name]={'failed':failed.read_text() if failed.exists() else None,
  'bytes':sum(f.stat().st_size for f in fs if f.is_file()),'active':bool(active),
  'delivered':item['checkpoint_ready'],'tail':tail}
for tag in ['af60v8f57','pick1200_bf16_bs32']:
 w=r/'dline'/tag;status=w/'status'
 out['lines'][tag]={'aborted':(w/'abort').exists(),'done':(w/'done').exists(),
 'mtime':status.stat().st_mtime if status.exists() else None}
print(json.dumps(out))
'''


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def assess(now, feeds, remote, history):
    alerts = []
    def alert(stage, reason):
        alerts.append({'stage': stage, 'reason': reason})
    for name, feed in feeds.items():
        if now-feed.get('time', 0) > 180:
            alert(name, 'Watcher heartbeat missing or older than three minutes')
    for warning in feeds.get('budget',{}).get('alerts',[]):
        if warning.get('reason')=='funding cutoff within 90 minutes':
            alert('budget','Funding cutoff within 90 minutes if current spending rate continues; verify planned Pod releases')
    for host, data in feeds.get('clusters',{}).get('hosts',{}).items():
        if host not in ('bluesclues','adroit'):continue
        if data.get('blocked'):alert(host,'Cluster access blocked; user troubleshooting required')
        for name, job in data.get('jobs',{}).items():
            progress=job.get('progress') or []
            if not progress or 'Terminated' in str(progress[-1]) or 'oom' in name:continue
            step,target,_=progress[-1]
            if int(step)>=int(target):continue
            key='cluster_training:'+host+':'+name
            prev=history.get(key,{'value':step,'since':now})
            if prev['value']!=step:prev={'value':step,'since':now}
            history[key]=prev
            if now-prev['since']>600:alert(name,'Cluster training has made no optimizer-step progress for ten minutes')
    status=feeds.get('status', {})
    for event in feeds.get('chain',{}).get('events',[]):
        if 'af60v8f57' in event or 'heartbeat stale' in event:
            alert('chain',event)
    for worker in remote.get('workers',[]):
        if now-worker['mtime']>900:
            alert(str(worker['pid']), 'Collection/conversion log stale for fifteen minutes: '+worker['log'])
    for pod in status.get('pods', []):
        if pod.get('failure_needs_review'):
            alert(pod['id'], 'Cloud training failed: '+pod['failure_needs_review'])
        current=pod.get('current') or {}
        if current.get('state')=='running':
            key='training:'+pod['id']; step=pod.get('training_step')
            prev=history.get(key, {'value':step,'since':now})
            if prev['value']!=step:prev={'value':step,'since':now}
            history[key]=prev
            if now-prev['since']>600:alert(current['run'], 'No optimizer-step progress for ten minutes')
    for tag, line in remote.get('lines',{}).items():
        if line['aborted']:alert(tag, 'Chain has an abort marker')
    for name, run in remote.get('runs',{}).items():
        if run['done']:
            m=run.get('metric',{})
            valid=(run['episodes']==100 and m.get('episodes')==100 and m.get('missing')==0
                   and all(isinstance(m.get(k),(int,float)) and math.isfinite(m[k]) for k in ('world','root')))
            if not valid:alert(name, 'Evaluation marked done without 100 episodes and complete finite MPJPE')
        elif run.get('last_eval_output'):
            age=now-run['last_eval_output']
            if age>300 and not run['active']:alert(name, 'Evaluation exited before producing a validated report')
            elif age>600:alert(name, 'Evaluation output stalled for ten minutes')
        elif run.get('checkpoint_ready'):
            key='eval_wait:'+name;history.setdefault(key, {'since':now})
            if now-history[key]['since']>300:alert(name, 'Delivered checkpoint has not started evaluation within five minutes')
    deliveries=feeds.get('deliveries',{}).get('deliveries',{})
    for name,state in deliveries.items():
        if state!='transferring':continue
        tr=remote.get('transfers',{}).get(name,{})
        if not tr or tr.get('delivered'):continue
        key='transfer:'+name;value=tr['bytes']
        prev=history.get(key,{'value':value,'since':now})
        if prev['value']!=value:prev={'value':value,'since':now}
        history[key]=prev
        if tr.get('failed'):alert(name,'Checkpoint delivery failed: '+tr['failed'].strip())
        elif not tr.get('active'):alert(name,'Checkpoint delivery says transferring, but no transfer/unpack worker exists')
        elif now-prev['since']>600:alert(name,'Checkpoint delivery has made no size progress for ten minutes; inspect packing/unpacking')
    return alerts


def main():
    history={}
    blocked=ROOT/'pipeline_health_ssh_blocked'
    while not (ROOT/'pipeline_health.stop').exists():
        now=time.time()
        feeds={n:read_json(ROOT/f) for n,f in [('status','status.json'),('budget','budget_guard_status.json'),('clusters','cluster_resources.json')]}
        feeds['chain']=read_json(ROOT.parent/'dagger_watch/status.json')
        # monitor_state has no timestamp; freshness is checked via status.json.
        deliveries=read_json(ROOT/'monitor_state.json')
        alerts=[];remote={}
        if blocked.exists():
            alerts.append({'stage':'SSH','reason':blocked.read_text()})
        else:
            try:
                p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
                    'sastrygrp-dvij@bluesclues.ist.berkeley.edu','python3 -c '+shlex.quote(PROBE)],
                    capture_output=True,text=True,encoding='utf8',errors='replace',timeout=45,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if p.returncode:
                    reason=f'Probe exit {p.returncode}: '+p.stderr[-1200:]
                    if p.returncode==255:blocked.write_text(reason)
                    raise RuntimeError(reason)
                remote=json.loads(p.stdout)
            except Exception as error:
                if isinstance(error,subprocess.TimeoutExpired):blocked.write_text('SSH probe timed out; user troubleshooting required')
                alerts.append({'stage':'pipeline_probe','reason':str(error)})
        transfer_feeds=dict(feeds,deliveries=dict(deliveries,time=now))
        alerts+=assess(now,transfer_feeds,remote,history)
        report={'time':now,'ok':not alerts,'alerts':alerts,'remote':remote}
        temp=ROOT/'pipeline_health.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(ROOT/'pipeline_health.json')
        if alerts:
            with (ROOT/'pipeline_alerts.jsonl').open('a') as f:f.write(json.dumps({'time':now,'alerts':alerts})+'\n')
        message='All checked stages healthy' if not alerts else '\n'.join(a['stage']+': '+a['reason'] for a in alerts)
        page='<meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Pipeline health</title><body style="font:18px system-ui;margin:32px"><h1>'+('Healthy' if not alerts else 'Action required')+'</h1><p>'+html.escape(time.strftime('%Y-%m-%d %H:%M:%S %Z'))+'</p><pre style="white-space:pre-wrap">'+html.escape(message)+'</pre>'
        (ROOT.parent/'pipeline_health.html').write_text(page,encoding='utf8')
        print(json.dumps({'time':now,'alerts':alerts}),flush=True)
        time.sleep(60)


if __name__=='__main__':
    main()
