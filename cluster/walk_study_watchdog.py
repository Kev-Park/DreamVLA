"""Read-only Windows watchdog for the walking study and its Adroit bridge.

Never restarts jobs or retries a host after a failed SSH request. Writes an
attention report independently of the workers being monitored.
"""
import argparse
import concurrent.futures
import ctypes
import json
import os
from pathlib import Path
import time

from walk_adroit_bridge import call, save


RESOURCE_HELPER = r'''
def account_resources():
 import csv,json,os,pathlib,subprocess,time,xml.etree.ElementTree as ET
 uid=os.getuid();hz=os.sysconf('SC_CLK_TCK');count=os.cpu_count()
 def snapshot():
  rows={}
  for p in pathlib.Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    if p.stat().st_uid!=uid:continue
    raw=(p/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
    rows[int(p.name)]=(int(fields[11])+int(fields[12]),fields[19])
   except (OSError,ValueError,IndexError):pass
  return rows
 before=snapshot();start=time.monotonic();time.sleep(1);after=snapshot();elapsed=time.monotonic()-start
 uptime=float(pathlib.Path('/proc/uptime').read_text().split()[0]);ticks=0
 for pid,(value,born) in after.items():
  if pid in before and before[pid][1]==born:ticks+=max(0,value-before[pid][0])
  elif int(born)/hz>=uptime-elapsed:ticks+=value
 result={'sampled_at':time.time(),'account_cpu_pct':ticks/hz/elapsed/count*100,
         'cpu_sample_seconds':elapsed,'logical_cpus':count}
 try:
  gpu_text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
  inventory=[]
  for fields in csv.reader(gpu_text.splitlines()):
   fields=[value.strip() for value in fields]
   inventory.append({'gpu':int(fields[0]),'uuid':fields[1],'name':fields[2],
                     'memory_used_mib':int(fields[3]),'memory_total_mib':int(fields[4]),'utilization_pct':int(fields[5])})
  mapping={row['uuid']:row['gpu'] for row in inventory};processes=[]
  xml=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=10))
  for device in xml.findall('gpu'):
   gpu=mapping[device.findtext('uuid')]
   for app in device.findall('processes/process_info'):
    pid=app.findtext('pid');p=pathlib.Path('/proc',pid)
    try:
     if p.stat().st_uid!=uid:continue
     raw=(p/'stat').read_text();state=raw[raw.rfind(')')+2:].split()[0]
     memory=app.findtext('used_memory','N/A').split()[0]
     processes.append({'gpu':gpu,'pid':int(pid),'memory_mib':int(memory) if memory.isdigit() else None,
                       'type':app.findtext('type'),'state':state})
    except (FileNotFoundError,ProcessLookupError):pass
  result.update(gpu_inventory=inventory,account_gpu_processes=processes)
  reservations=[];path=pathlib.Path.home()/'kevin/dline/gpu_slots.json'
  try:
   for row in json.loads(path.read_text()) if path.exists() else []:
    try:
     if pathlib.Path('/proc',str(row['owner'])).stat().st_uid==uid:reservations.append(row)
    except FileNotFoundError:pass
  except (OSError,ValueError,KeyError,TypeError) as exc:result['reservation_snapshot_error']=repr(exc)
  result.update(live_gpu_reservations=reservations,
                account_gpu_devices=sorted({row['gpu'] for row in processes+reservations}))
 except (OSError,subprocess.SubprocessError,ValueError,KeyError,IndexError,ET.ParseError) as exc:
  result['gpu_snapshot_error']=repr(exc)
 return result
'''

BLUE = RESOURCE_HELPER + r'''
import json,pathlib,shutil,time
b=pathlib.Path.home()/'kevin'
def read(relative):return json.loads((b/relative).read_text())
def process(pid):
 p=pathlib.Path('/proc')/str(pid)/'cmdline'
 return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''
collector=read('eval_videos/walk_v3_study/expert_collection/workflow.json')
health=read('eval_videos/walk_v3_study/gr00t/workflow.json')
for chain in health['chains']:
 for row in chain.get('rounds',[]):
  if 'evaluation' in row:
   evaluation=row['evaluation'];episodes=evaluation.pop('per_episode')
   evaluation['all_100_references_covered']=sorted(e['motion_id'] for e in episodes)==list(range(100))
collector.pop('completed_jobs',None)
collector['controller_live']=bool(process(collector['controller_pid']))
health['watcher_live']=bool(process(health['watcher_pid']))
failure=b/'eval_videos/walk_v3_study/gr00t/health_failure.json'
frozen=[r for r in read('eval_videos/walk_v3_study/sonic/results.json') if r['experiment']=='frozen_reference_grasp']
usage=shutil.disk_usage(b)
print(json.dumps({'observed_at':time.time(),'disk_free_gib':usage.free/2**30,'collector':collector,'gr00t':health,
 'health_failure':json.loads(failure.read_text()) if failure.exists() else None,'frozen':frozen,
 'account_resources':account_resources()}))
'''

BIPED = RESOURCE_HELPER + r'''
import json,pathlib,re,shutil,time
b=pathlib.Path.home()/'kevin';out=b/'eval_videos/walk_v3_study/biped_direct'
state=json.loads((out/'workflow.json').read_text())
def command(pid):
 p=pathlib.Path('/proc')/str(pid)/'cmdline'
 return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''
rows=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
for row in rows:
 log=out/('eval_'+str(row['iteration'])+'.log')
 with log.open('rb') as stream:
  stream.seek(max(0,log.stat().st_size-2000000));text=stream.read().decode(errors='replace')
 ranking=text.split('PER-CLIP OUTCOME RANKING (worst topple first)')[-1]
 counts=[(int(m[0]),int(m[1])) for m in re.findall(r'^\s+(\d+)\s+(\d+)\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+\s+',ranking,re.M)]
 row['all_100_references_covered']=(sorted(i for i,n in counts)==list(range(100))
  and sum(n for i,n in counts)==row['episodes'] and row['episodes']>=2000)
state['controller_command']=command(state['controller_pid'])
state['train_command']=command(state.get('train_pid',0))
process_status=pathlib.Path('/proc')/str(state.get('train_pid',0))/'status'
state['train_stopped']=bool(process_status.exists() and re.search(r'^State:\s+T\b',process_status.read_text(),re.M))
pause=out/'user_budget_pause.json'
state['user_pause']=json.loads(pause.read_text()) if pause.exists() else None
log=out/'train.log';state['train_log_age']=time.time()-log.stat().st_mtime
if state.get('active_eval'):
 log=out/('eval_'+str(state['active_eval']['iteration'])+'.log')
 state['eval_log_age']=time.time()-log.stat().st_mtime
confirmation=b/'eval_videos/walk_v3_study/direct_matched4500_seeds/workflow.json'
extra=json.loads(confirmation.read_text()) if confirmation.exists() else None
if extra:
 extra['controller_live']=bool(command(extra['controller_pid']))
 for job in extra.get('active_jobs',[]):
  job['command_live']=command(job['pid'])
  job['log_age']=time.time()-pathlib.Path(job['log']).stat().st_mtime
external=[]
for path in (b/'eval_videos/walk_v3_study/biped_eval').glob('*/workflow.json'):
 row=json.loads(path.read_text());row['controller_command']=command(row['controller_pid'])
 if row.get('active_job'):
  job=row['active_job'];job['command_live']=command(job['pid']);job['log_age']=time.time()-pathlib.Path(job['log']).stat().st_mtime
 external.append(row)
print(json.dumps({'observed_at':time.time(),'disk_free_gib':shutil.disk_usage(b).free/2**30,'workflow':state,'results':rows,'confirmation':extra,'gr00t_evaluations':external,'account_resources':account_resources()}))
'''


def pid_alive(pid):
    if os.name != 'nt':
        return Path('/proc', str(pid), 'cmdline').exists()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_uint32()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
    finally:
        kernel.CloseHandle(handle)


def assess(host, data, now, progress):
    alerts = []
    resources = data.get('account_resources', {})
    for field in ('gpu_snapshot_error', 'reservation_snapshot_error'):
        if resources.get(field):
            alerts.append('Account resource snapshot incomplete: '+resources[field])
    minimum_free = 150 if host == 'bluesclues' else 20
    if data['disk_free_gib'] < minimum_free:
        alerts.append(f"Only {data['disk_free_gib']:.1f} GiB free; review storage before further allocation")
    if host == 'bluesclues':
        health, collector = data['gr00t'], data['collector']
        if now-health['heartbeat'] > 120 or not health['watcher_live']:
            alerts.append('GR00T health watcher stopped or heartbeat stale')
        if data['health_failure']:
            alerts.append('GR00T health watcher recorded a failure')
        for chain in health['chains']:
            alerts.extend(chain['tag']+': '+a for a in chain.get('alerts', []))
            if chain['status'] in ('failed', 'stopped_without_completion'):
                alerts.append(chain['tag']+': '+chain['status'])
            for row in chain.get('rounds', []):
                if row.get('evaluated') and not row['evaluation']['all_100_references_covered']:
                    alerts.append(row['run']+': incomplete holdout coverage')
        if collector['status'] != 'complete' and (
                now-collector['heartbeat'] > 120 or not collector['controller_live']):
            alerts.append('Expert collector stopped or heartbeat stale')
        alerts.extend('Expert collection: '+a for a in collector.get('alerts', []))
        if collector.get('errors') or collector['status'] == 'failed':
            alerts.append('Expert collection recorded a failure')
        if len(data['frozen']) != 1 or data['frozen'][0]['episodes'] != 2000:
            alerts.append('Frozen SONIC baseline missing or incomplete')
    else:
        state = data['workflow']
        paused = state['status'] == 'paused_by_user'
        if paused:
            pause = state.get('user_pause') or {}
            if not (state.get('train_stopped') and state['train_command']
                    and pause.get('pid') == state.get('train_pid') and pause.get('action') == 'SIGSTOP'):
                alerts.append('Direct-token user pause is not backed by a stopped trainer and matching receipt')
            if now-state['heartbeat'] > 120 or not state['controller_command']:
                alerts.append('Paused direct-token controller stopped or heartbeat stale')
            if state.get('active_eval'):
                alerts.append('Direct-token controller has an unexpected active evaluation during user pause')
        elif state['status'] != 'complete':
            if now-state['heartbeat'] > 120 or not state['controller_command']:
                alerts.append('Direct-token controller stopped or heartbeat stale')
            if state['status'] == 'failed':
                alerts.append('Direct-token workflow failed: '+str(state.get('error')))
            if not state['train_command'] and state.get('latest_iteration', 0) < 8999:
                alerts.append('Direct-token training exited before the final update')
            iteration = state.get('latest_iteration')
            if progress.get('iteration') != iteration:
                progress.update(iteration=iteration, changed_at=now)
            if now-progress.setdefault('changed_at', now) > 900 and iteration != 8999:
                alerts.append('Direct-token training has not advanced for 15 minutes')
            if state['train_log_age'] > 900 and state.get('latest_iteration', 0) < 8999:
                alerts.append('Direct-token training log stalled')
            if state.get('eval_log_age', 0) > 900:
                alerts.append('Direct-token evaluation log stalled')
        alerts.extend(state.get('alerts', []))
        if state.get('errors'):
            alerts.append('Direct-token workflow recorded errors')
        for row in data['results']:
            if not row['all_100_references_covered']:
                alerts.append('Direct-token eval '+str(row['iteration'])+': incomplete holdout coverage')
        extra = data.get('confirmation')
        if extra:
            if extra['status'] == 'failed' or extra.get('errors'):
                alerts.append('Checkpoint-4500 confirmation failed: '+str(extra.get('error', extra.get('errors'))))
            elif extra['status'] != 'complete':
                if not extra['controller_live'] or now-extra['heartbeat'] > 120:
                    alerts.append('Checkpoint-4500 confirmation controller stopped or heartbeat stale')
                for job in extra.get('active_jobs', []):
                    if not job['command_live'] or job['log_age'] > 900:
                        alerts.append('Checkpoint-4500 confirmation worker stopped or log stalled: '+str(job['pid']))
        for evaluation in data.get('gr00t_evaluations', []):
            if evaluation['status'] == 'failed':
                alerts.append(evaluation['run']+': Biped evaluation failed: '+str(evaluation.get('error')))
            elif evaluation['status'] != 'complete':
                if not evaluation['controller_command'] or now-evaluation['heartbeat'] > 120:
                    alerts.append(evaluation['run']+': Biped evaluation controller stopped or heartbeat stale')
                job = evaluation.get('active_job')
                if job and (not job['command_live'] or job['log_age'] > 1800):
                    alerts.append(evaluation['run']+': Biped evaluation worker stopped or log stalled')
    return alerts


def review_adroit_bridge(bridge, state_dir):
    """Keep a documented idle submission hold visible without masking failures."""
    stopped = not pid_alive(bridge.get('pid', 0))
    if bridge.get('state') == 'operator_hold_walk200_handoff':
        try:
            receipt = json.loads((state_dir/'operator_hold_walk200_handoff.json').read_text())
        except (OSError, ValueError):
            receipt = {}
        valid = (stopped and bridge.get('outstanding') == 0
                 and isinstance(bridge.get('pid'), int) and bridge['pid'] > 0
                 and bool(bridge.get('hold_reason', '').strip())
                 and receipt.get('bridge_pid') == bridge.get('pid')
                 and receipt.get('purpose') == 'manual_walk200_finetune_handoff'
                 and receipt.get('outstanding') == 0
                 and isinstance(receipt.get('held_at'), (int, float)) and receipt['held_at'] > 0
                 and receipt.get('verified_process_stopped') is True)
        if valid:
            return [], [{'component': 'Adroit submission bridge',
                         'reason': bridge['hold_reason'], 'held_at': receipt['held_at'],
                         'bridge_pid': bridge['pid']}]
        return ['Adroit submission hold lacks a verified idle-stop receipt'], []
    if bridge.get('state') == 'stopped_for_review' or stopped:
        return ['Adroit transfer bridge stopped: '+str(bridge.get('error', 'process absent'))], []
    return [], []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--bridge-state-dir', type=Path, required=True)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir/'controller.lock').open('a+b')
    lock.seek(0)
    if os.name == 'nt':
        import msvcrt
        if not lock.read(1):
            lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous_file = args.state_dir/'workflow.json'
    previous = json.loads(previous_file.read_text()) if previous_file.exists() else {}
    # A monitor restart is not evidence that a failed SSH connection was restored.
    # The operator clears only the host explicitly restored by the user.
    disabled = previous.get('disabled_hosts', {})
    snapshots, progress = previous.get('hosts', {}), {}
    previous_alerts = previous_holds = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            futures = {executor.submit(call, host, script): host
                       for host, script in [('bluesclues', BLUE), ('biped', BIPED)] if host not in disabled}
            for future in concurrent.futures.as_completed(futures):
                host = futures[future]
                try:
                    snapshots[host] = future.result()
                except Exception as exc:
                    # Quarantine this host; a user must restore access before any new attempt.
                    disabled[host] = {'time': time.time(), 'error': repr(exc), 'requires_user_review': True}
            now = time.time()
            alerts = [host+': requests stopped; '+row['error'] for host, row in disabled.items()]
            for host, data in snapshots.items():
                if host not in disabled:
                    alerts.extend(host+': '+a for a in assess(host, data, now, progress))
            bridge_file = args.bridge_state_dir/'workflow.json'
            bridge = json.loads(bridge_file.read_text()) if bridge_file.exists() else {}
            bridge_alerts, operator_holds = review_adroit_bridge(bridge, args.bridge_state_dir)
            alerts.extend(bridge_alerts)
            for path in (args.state_dir.parent/'walk_biped_eval').glob('*/workflow.json'):
                remote_bridge = json.loads(path.read_text())
                if remote_bridge['state'] == 'stopped_for_review' or (
                        remote_bridge['state'] != 'complete' and not pid_alive(remote_bridge['pid'])):
                    alerts.append(remote_bridge['run']+': Biped evaluation bridge stopped: '+str(remote_bridge.get('error', 'process absent')))
            complete = not disabled and len(snapshots) == 2 and (
                snapshots['bluesclues']['gr00t']['status'] == 'complete'
                and snapshots['biped']['workflow']['status'] == 'complete')
            status = {'pid': os.getpid(), 'heartbeat': now, 'status': 'needs_attention' if alerts else 'complete' if complete else 'monitoring',
                      'alerts': alerts, 'operator_holds': operator_holds,
                      'disabled_hosts': disabled, 'hosts': snapshots, 'adroit_bridge': bridge}
            save(args.state_dir/'workflow.json', status)
            if alerts != previous_alerts or operator_holds != previous_holds:
                with (args.state_dir/'events.jsonl').open('a') as stream:
                    stream.write(json.dumps({'time': now, 'alerts': alerts, 'operator_holds': operator_holds})+'\n')
                previous_alerts, previous_holds = alerts, operator_holds
            if complete or len(disabled) == 2:
                return
            time.sleep(60)


if __name__ == '__main__':
    main()
