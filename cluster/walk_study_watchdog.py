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


BLUE = r'''
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
 'health_failure':json.loads(failure.read_text()) if failure.exists() else None,'frozen':frozen}))
'''

BIPED = r'''
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
print(json.dumps({'observed_at':time.time(),'disk_free_gib':shutil.disk_usage(b).free/2**30,'workflow':state,'results':rows,'confirmation':extra,'gr00t_evaluations':external}))
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
    disabled, snapshots, progress = {}, {}, {}
    previous_alerts = None
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
            if bridge.get('state') == 'stopped_for_review' or not pid_alive(bridge.get('pid', 0)):
                alerts.append('Adroit transfer bridge stopped: '+str(bridge.get('error', 'process absent')))
            for path in (args.state_dir.parent/'walk_biped_eval').glob('*/workflow.json'):
                remote_bridge = json.loads(path.read_text())
                if remote_bridge['state'] == 'stopped_for_review' or (
                        remote_bridge['state'] != 'complete' and not pid_alive(remote_bridge['pid'])):
                    alerts.append(remote_bridge['run']+': Biped evaluation bridge stopped: '+str(remote_bridge.get('error', 'process absent')))
            complete = not disabled and len(snapshots) == 2 and (
                snapshots['bluesclues']['gr00t']['status'] == 'complete'
                and snapshots['biped']['workflow']['status'] == 'complete')
            status = {'pid': os.getpid(), 'heartbeat': now, 'status': 'needs_attention' if alerts else 'complete' if complete else 'monitoring',
                      'alerts': alerts, 'disabled_hosts': disabled, 'hosts': snapshots, 'adroit_bridge': bridge}
            save(args.state_dir/'workflow.json', status)
            if alerts != previous_alerts:
                with (args.state_dir/'events.jsonl').open('a') as stream:
                    stream.write(json.dumps({'time': now, 'alerts': alerts})+'\n')
                previous_alerts = alerts
            if complete or len(disabled) == 2:
                return
            time.sleep(60)


if __name__ == '__main__':
    main()
