"""Windows relay and observer for the reviewed1-second walking experiment.

No local datasets, compression, package installation, job cancellation or retries.
Only this experiment's verified temporary copies may be removed after relocation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import walk_adroit_bridge as wire
from walk_adroit_bridge import RemoteCommandError,inventory,save,transfer
from walk_training_progress import watch_progress

_remote_call=wire.call
STATE_DIRECTORY=None


def call(host,script,**kwargs):
    hold=STATE_DIRECTORY/(host+'.connection_hold.json') if STATE_DIRECTORY else None
    if hold and hold.exists():raise RuntimeError('Connection held pending operator review: '+str(hold))
    try:return _remote_call(host,script,**kwargs)
    except (RuntimeError,subprocess.TimeoutExpired) as exc:
        if hold and not isinstance(exc,RemoteCommandError):
            save(hold,{'host':host,'at':time.time(),'error':repr(exc),'automatic_retry':False})
        raise


def configure(directory):
    global STATE_DIRECTORY
    STATE_DIRECTORY=Path(directory)
    wire.call=call

TAG='walk200_v3_late1s_20261024'
WORK='dline/'+TAG


def read_remote(host,relative):
    return call(host,"import json,pathlib\np=pathlib.Path.home()/'kevin'/REL\nprint(json.dumps(json.loads(p.read_text()) if p.exists() else None))".replace('REL',repr(relative)))


def collector_health():
    health=call('biped',r'''
import json,pathlib,time
w=pathlib.Path.home()/'kevin'/WORK;s=json.loads((w/'workflow.json').read_text())
keys=('status','controller_pid','completed_attempts','active_batch','error','heartbeat')
result={k:s[k] for k in keys if k in s};result['pid_live']=pathlib.Path('/proc',str(s['controller_pid'])).exists()
active=s.get('active_batch');heartbeat=w/(str(active)+'.heartbeat.json')
if s['status']=='collecting' and heartbeat.exists():
 h=json.loads(heartbeat.read_text());result['worker_log_age_seconds']=time.time()-h['log_mtime']
print(json.dumps(result))
'''.replace('WORK',repr(WORK)))
    assert health['status']!='failed',('Collector stopped;operator review required',health)
    assert health['status']=='collection_complete' or health['pid_live'],('Collector disappeared',health)
    assert health.get('worker_log_age_seconds',0)<1800,('Collector output stalled for30minutes;inspect its log',health)
    return health


def stage_observation(host,batch,conversion=False):
    return call(host,r'''
import json,os,pathlib,time
work=pathlib.Path.home()/'kevin'/WORK;batch=BATCH;conversion=CONVERSION
def load(p):return json.loads(p.read_text()) if p.exists() else None
status=load(work/(('convert_'+batch+'.workflow.json') if conversion else 'workflow.json'))
ready=load(work/(batch+('.converted.json' if conversion else '.complete.json')))
heartbeat=load(work/(batch+'.heartbeat.json')) if not conversion else None
live=None
if status:
 pid=status.get('pid',status.get('controller_pid'))
 live=bool(pid and pathlib.Path('/proc',str(pid)).exists())
 if not ready:
  assert status['status']!='failed',status
  if status['status'] not in ('complete','collection_complete'):assert live,('Controller disappeared',status)
 if heartbeat and status['status']=='collecting':
  assert not heartbeat.get('output_stalled_30min'),('No collector output for30minutes',heartbeat)
print(json.dumps({'workflow':status,'ready':ready,'pid_live':live,'heartbeat':heartbeat,'observed_at':time.time()}))
'''.replace('WORK',repr(WORK)).replace('BATCH',repr(batch)).replace('CONVERSION',repr(conversion)))


def write_remote(host,relative,value):
    return call(host,"""import json,pathlib
p=pathlib.Path.home()/'kevin'/REL;value=VALUE;p.parent.mkdir(parents=True,exist_ok=True)
if p.exists():assert json.loads(p.read_text())==value,'Metadata differs;reconcile before overwrite'
else:p.write_text(json.dumps(value,indent=2)+'\\n')
print(json.dumps({'written':str(p)}))
""".replace('REL',repr(relative)).replace('VALUE',repr(value)))


def checked_transfer(source,destination,relative,directory,label,inference=False):
    marker=directory/(label+'.transfer.json')
    if marker.exists():
        receipt=json.loads(marker.read_text())
        assert receipt.get('verified'),'Interrupted transfer needs explicit reconciliation: '+label
        expected=json.loads((directory/(label+'.source_inventory.json')).read_text())
        assert inventory(destination,relative,inference,single_cpu=destination=='bluesclues')==expected
        return
    if destination in ('bluesclues','biped'):
        size=call(source,"import pathlib,json\np=pathlib.Path.home()/'kevin'/REL\nskip={'optimizer.pt','scheduler.pt','rng_state.pth','training_args.bin'} if INFERENCE else set()\nprint(json.dumps(sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and f.name not in skip)))".replace('REL',repr(relative)).replace('INFERENCE',repr(inference)))
        free=call(destination,"import pathlib,shutil,json\nprint(json.dumps(shutil.disk_usage(pathlib.Path.home()/'kevin').free))")
        assert free>=2*size+8_000_000_000,'Insufficient reviewed conversion staging space;source retained'
    try:transfer(source,destination,relative,directory,label,inference=inference,
                 blue_single_cpu='bluesclues' in (source,destination))
    except BaseException:
        if marker.exists():
            failed=json.loads(marker.read_text())
            for host,key in ((source,'producer_code'),(destination,'consumer_code')):
                if failed.get(key)==255:
                    save(directory/(host+'.connection_hold.json'),{'host':host,'at':time.time(),'error':'SSH transfer failed: '+label,'automatic_retry':False})
        raise


def remove_relocated_copy(host,relative,expected,destination):
    # The manifest covers only the new, completed experiment batch. Existing
    # datasets, checkpoints and shared caches are outside this allowlist.
    script=r'''
import hashlib,json,os,pathlib
os.nice(19);os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
b=(pathlib.Path.home()/'kevin').resolve();p=b/__REL__;expected=EXPECTED
assert p.resolve().is_relative_to(b/'dline'/TAG) and not p.is_symlink()
pointer=p/'RELOCATED.json'
if pointer.exists():
 assert json.loads(pointer.read_text())['destination']==DEST
 print(json.dumps({'already_relocated':True}))
else:
 actual=sorted(str(x.relative_to(p)) for x in p.rglob('*') if x.is_file())
 assert actual==sorted(x['path'] for x in expected),'Temporary source inventory changed'
 for proc in pathlib.Path('/proc').iterdir():
  try:
   if not proc.name.isdigit() or proc.stat().st_uid!=os.getuid() or int(proc.name)==os.getpid():continue
   for fd in (proc/'fd').iterdir():
    try:target=os.readlink(fd)
    except OSError:continue
    assert not target.startswith(str(p)+'/'),'Temporary copy still open;retain it for review'
  except OSError:pass
 fingerprints={}
 for row in expected:
  f=p/row['path'];assert not f.is_symlink() and f.resolve().is_relative_to(p)
  st=f.stat();fingerprints[str(f)]=(st.st_size,st.st_mtime_ns,st.st_ino,st.st_dev)
  h=hashlib.sha256()
  with f.open('rb') as stream:
   for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
  assert f.stat().st_size==row['bytes'] and h.hexdigest()==row['sha256']
 for row in expected:
  f=p/row['path'];st=f.stat();assert fingerprints[str(f)]==(st.st_size,st.st_mtime_ns,st.st_ino,st.st_dev);f.unlink()
 for d in sorted((d for d in p.rglob('*') if d.is_dir()),key=lambda d:len(d.parts),reverse=True):d.rmdir()
 pointer.write_text(json.dumps({'destination':DEST,'verified_files':len(expected),'compression':False},indent=2)+'\n')
 print(json.dumps({'relocated_files':len(expected)}))
'''.replace('__REL__',repr(relative)).replace('EXPECTED',repr(expected)).replace('TAG',repr(TAG)).replace('DEST',repr(destination))
    return call(host,script,timeout=240)


def start_conversion(batch,cpu_cap=20):
    return call('bluesclues',r'''
import json,os,pathlib,shlex,subprocess
b=pathlib.Path.home()/'kevin';work=b/WORK;batch=BATCH
ready=work/(batch+'.converted.json');workflow=work/('convert_'+batch+'.workflow.json')
if ready.exists():print(json.dumps({'already_complete':True}))
elif workflow.exists():
 d=json.loads(workflow.read_text());assert d['status']=='running' and pathlib.Path('/proc',str(d['pid'])).exists(),d
 print(json.dumps({'already_running':True,'pid':d['pid']}))
else:
 name='walk_late1s_convert_kevin';assert subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode!=0,'Prior conversion session needs review'
 import sys
 sys.path.insert(0,str(b/'wt/locomotion/DreamVLA/cluster'))
 from account_cpu_usage import sample_account_cpu
 cpu=sample_account_cpu(2);assert cpu['cpu_measurement_complete'],cpu
 # Background account work is bursty; this small conversion fits one CPU.
 # Keep the existing four data shards and global normalization unchanged.
 if cpu['account_cpu_pct']+1/os.cpu_count()*100>CPU_CAP:
  print(json.dumps({'waiting_for_cpu_headroom':True,'account_cpu':cpu,'account_cap_pct':CPU_CAP}));sys.exit(0)
 review={'at':__import__('time').time(),'account_cpu':cpu,'allocated_cpu_mask':[0],'gpus':0,'account_cap_pct':CPU_CAP,'approval':'Existing2026-10-14 walking collection exception when30;otherwise normal20%'}
 (work/(batch+'.resource_review.json')).write_text(json.dumps(review,indent=2)+'\n')
 args=[str(b/'wt/locomotion/.venv/bin/python'),'-u',str(b/'wt/locomotion/DreamVLA/cluster/walk_late_takeover_process.py'),'convert','--tag',TAG,'--batch',batch]
 shell='export XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'
 shell+='exec taskset -c 0 '+shlex.join(args)+' > '+shlex.quote(str(work/(batch+'.conversion.log')))+' 2>&1'
 os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
 subprocess.run(['tmux','new-session','-d','-s',name,'bash','-lc',shell],check=True)
 print(json.dumps({'launched':name,'batch':batch,'cpu_affinity':[0],'gpus':0}))
'''.replace('WORK',repr(WORK)).replace('BATCH',repr(batch)).replace('TAG',repr(TAG)).replace('CPU_CAP',repr(cpu_cap)))


def start_assembly():
    return call('bluesclues',r'''
import json,os,pathlib,shlex,subprocess,sys,time
b=pathlib.Path.home()/'kevin';work=b/WORK;workflow=work/'assemble.workflow.json'
if (work/'dataset.ready.json').exists():print(json.dumps({'already_complete':True}))
elif workflow.exists():
 d=json.loads(workflow.read_text());assert d['status']=='running' and pathlib.Path('/proc',str(d['pid'])).exists(),d
 print(json.dumps({'already_running':True,'pid':d['pid']}))
else:
 sys.path.insert(0,str(b/'wt/locomotion/DreamVLA/cluster'))
 from account_cpu_usage import sample_account_cpu
 cpu=sample_account_cpu(2);assert cpu['cpu_measurement_complete'],cpu
 if cpu['account_cpu_pct']+1/os.cpu_count()*100>20:
  print(json.dumps({'waiting_for_cpu_headroom':True,'account_cpu':cpu,'account_cap_pct':20}));sys.exit(0)
 name='walk_late1s_assemble_kevin';assert subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode!=0
 args=[str(b/'GR00T-WholeBodyControl/.venv_data_collection/bin/python'),'-u',str(b/'wt/locomotion/DreamVLA/cluster/walk_late_takeover_process.py'),'assemble','--tag',TAG]
 shell='export XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n'
 shell+='exec taskset -c 0 '+shlex.join(args)+' > '+shlex.quote(str(work/'assembly.log'))+' 2>&1'
 (work/'assembly.resource_review.json').write_text(json.dumps({'at':time.time(),'account_cpu':cpu,'cpu_mask':[0],'gpus':0},indent=2)+'\n')
 os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
 subprocess.run(['tmux','new-session','-d','-s',name,'bash','-lc',shell],check=True)
 print(json.dumps({'launched':name,'cpu_mask':[0],'gpus':0}))
'''.replace('WORK',repr(WORK)).replace('TAG',repr(TAG)))


def submit(stage,ready=None):
    payload={'stage':stage,'ready':ready}
    return call('adroit',r'''
import json,pathlib,subprocess
b=pathlib.Path.home()/'kevin';work=b/WORK;p=PAYLOAD
receipt=work/(p['stage']+'.submission.json');logs=b/'eval_videos'/TAG;logs.mkdir(parents=True,exist_ok=True)
if receipt.exists():print(receipt.read_text())
else:
 repo=b/'wt/locomotion/DreamVLA';cmd=['sbatch','--parsable','--job-name='+TAG+'_'+p['stage'],'--output='+str(logs/(p['stage']+'-%j.log'))]
 assert p['stage']=='finetune'
 cmd += [str(repo/'cluster/adroit/walk_late_takeover_finetune.sbatch'),TAG,p['ready']['dataset'],str(p['ready']['steps'])]
 job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit()
 value={'job_id':job,'stage':p['stage'],'command':cmd};receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''.replace('WORK',repr(WORK)).replace('PAYLOAD',repr(payload)).replace('TAG',repr(TAG)))


def job_state(job):
    return call('adroit',r'''
import json,subprocess,sys
job=JOB
try:p=subprocess.run(['squeue','-j',job,'-h','-o','%T|%R'],capture_output=True,text=True,timeout=20)
except subprocess.TimeoutExpired:
 print(json.dumps({'state':'OBSERVATION_TIMEOUT','command':'squeue','job_id':job}));sys.exit(0)
if p.returncode and 'Invalid job id' not in p.stderr:
 print(json.dumps({'state':'OBSERVATION_ERROR','command':'squeue','job_id':job,'detail':p.stderr}));sys.exit(0)
if p.stdout.strip():
 result={'state':p.stdout.strip().split('|')[0],'detail':p.stdout.strip()}
 if result['state']=='RUNNING':
  try:
   import pathlib
   base=pathlib.Path.home()/'kevin';sys.path.insert(0,str(base/'wt/locomotion/DreamVLA/cluster'))
   from walk_training_progress import read_progress
   expected=json.loads((base/WORK_PATH/'dataset.ready.json').read_text())['steps']
   result['training_progress']=read_progress(base/'eval_videos'/EXPERIMENT/('finetune-'+job+'.log'),expected)
  except Exception as exc:result['training_progress']={'observation_error':repr(exc)}
 print(json.dumps(result))
else:
 try:accounting=subprocess.run(['sacct','-j',job,'-n','-X','-P','-o','State,ExitCode'],capture_output=True,text=True,timeout=20)
 except subprocess.TimeoutExpired:
  print(json.dumps({'state':'OBSERVATION_TIMEOUT','command':'sacct','job_id':job}));sys.exit(0)
 if accounting.returncode:
  print(json.dumps({'state':'OBSERVATION_ERROR','command':'sacct','job_id':job,'detail':accounting.stderr}));sys.exit(0)
 rows=accounting.stdout.strip().splitlines()
 if not rows:print(json.dumps({'state':'ACCOUNTING_PENDING'}))
 else:
  assert len(rows)==1,rows;state,exitcode=rows[0].split('|')[:2];print(json.dumps({'state':state,'exitcode':exitcode}))
'''.replace('JOB',repr(str(job))).replace('WORK_PATH',repr(WORK)).replace('EXPERIMENT',repr(TAG)))


def submitted_training_state(state):
    """Resume a known submission without recopying a dataset now used by GR00T."""
    training=state.get('training_submission')
    if not training:return None
    assert training['stage']=='finetune' and str(training['job_id']).isdigit(),training
    assert state.get('dataset') and len(state['completed_batches'])==20,state
    receipt=read_remote('adroit',WORK+'/finetune.submission.json')
    assert receipt==training,('Existing submission needs reconciliation;never resubmit',receipt,training)
    ready=read_remote('adroit',WORK+'/dataset.ready.json')
    assert ready==state['dataset'],('Submitted dataset identity changed',ready,state['dataset'])
    return ready,training


def start_biped_evaluation(checkpoint_relative,run_name=None):
    """Launch this experiment's control/final evaluation after collection releases GPU0."""
    run_name=run_name or TAG+'_r3'
    return call('biped',r'''
import fcntl,json,os,pathlib,shlex,subprocess,sys,time,xml.etree.ElementTree as ET
b=pathlib.Path.home()/'kevin';work=b/WORK;run=RUN_NAME;ready=work/(run+'.eval.json')
launch_lock=(work/'evaluation_launch.lock').open('a')
fcntl.flock(launch_lock,fcntl.LOCK_EX)
key='evaluate' if run==TAG+'_r3' else 'evaluate_'+run
workflow=work/(key+'.workflow.json');checkpoint=b/CHECKPOINT
collector=json.loads((work/'workflow.json').read_text())
assert collector['status']=='collection_complete' and collector['completed_attempts']==200
prerequisites=work/'evaluation_prerequisites.json'
if run==TAG+'_r3' and prerequisites.exists():
 controls=json.loads(prerequisites.read_text())['controls']
 assert all(pathlib.Path(name).name==name and not name.startswith('.') for name in controls)
 pending=[name for name in controls if not (work/(name+'.eval.json')).exists()]
 for name in controls:
  result_file=work/(name+'.eval.json')
  if result_file.exists():
   result=json.loads(result_file.read_text())
   assert result['episodes']==100 and sorted(row['motion_id'] for row in result['per_episode'])==list(range(100)),('Incomplete control',name)
 if pending:
  print(json.dumps({'waiting_for_control_evaluations':pending}));sys.exit(0)
if ready.exists():print(json.dumps({'already_complete':True}))
elif workflow.exists():
 d=json.loads(workflow.read_text());assert d['status']=='running' and pathlib.Path('/proc',str(d['pid'])).exists(),d
 print(json.dumps({'already_running':True,'pid':d['pid']}))
else:
 sys.path.insert(0,str(b/'wt/locomotion/DreamVLA/cluster'))
 from account_cpu_usage import sample_account_cpu
 cpu=sample_account_cpu(2);assert cpu['cpu_measurement_complete'],cpu
 mask=json.loads((b/'dline/cpu_share.json').read_text())['shared_mask'][:4]
 assert len(mask)==4
 if cpu['account_cpu_pct']+len(mask)/os.cpu_count()*100>20:
  print(json.dumps({'waiting_for_cpu_headroom':True,'account_cpu':cpu}));sys.exit(0)
 tree=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True))
 assert 'RTX' in tree.findall('gpu')[0].findtext('product_name'),'RTX rendering capability not verified'
 occupied=[]
 for device,gpu in enumerate(tree.findall('gpu')):
  for process in gpu.findall('processes/process_info'):
   proc=pathlib.Path('/proc',process.findtext('pid'))
   if proc.exists() and proc.stat().st_uid==os.getuid():occupied.append({'gpu':device,'pid':proc.name})
 if occupied:
  print(json.dumps({'waiting_for_gpu_headroom':True,'account_contexts':occupied}));sys.exit(0)
 name='walk_late1s_eval_kevin'
 if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:
  print(json.dumps({'waiting_for_gpu_headroom':True,'active_experiment_session':name}));sys.exit(0)
 assert (checkpoint/'model.safetensors.index.json').is_file()
 review={'at':time.time(),'account_cpu':cpu,'cpu_affinity':mask,'compute_gpu':0,
  'gpu1_graphics_exception':'Existing2026-10-24 approval,at most6MiB,only this experiment',
  'checkpoint':str(checkpoint),'episodes':100,'decoder':'previous full FSQ','run':run}
 (work/(key+'.resource_review.json')).write_text(json.dumps(review,indent=2)+'\n')
 args=[str(b/'wt/locomotion/.venv/bin/python'),'-u',str(b/'wt/locomotion/DreamVLA/cluster/walk_late_takeover_process.py'),
  'evaluate','--tag',TAG,'--checkpoint',str(checkpoint),'--gpu','0','--run-name',run]
 shell='export XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1\n'
 shell+='exec taskset -c '+','.join(map(str,mask))+' '+shlex.join(args)+' > '+shlex.quote(str(work/(key+'.controller.log')))+' 2>&1'
 os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
 subprocess.run(['tmux','new-session','-d','-s',name,'bash','-lc',shell],check=True)
 print(json.dumps({'launched':name,'review':review}))
'''.replace('WORK',repr(WORK)).replace('TAG',repr(TAG)).replace('CHECKPOINT',repr(checkpoint_relative)).replace('RUN_NAME',repr(run_name)))


def resources(host):
    return call(host,r'''
import json,pathlib,shutil,sys,time
b=pathlib.Path.home()/'kevin';sys.path.insert(0,str(b/'wt/locomotion/DreamVLA/cluster'))
from walk_study_watchdog import RESOURCE_HELPER
namespace={};exec(RESOURCE_HELPER,namespace);data=namespace['account_resources']()
for row in data.get('account_gpu_processes',[]):
 p=pathlib.Path('/proc',str(row['pid']),'cmdline')
 try:row['this_experiment']=EXPERIMENT in p.read_bytes().replace(b'\0',b' ').decode()
 except FileNotFoundError:row['this_experiment']=False
data.update(observed_at=time.time(),disk_free_bytes=shutil.disk_usage(b).free)
print(json.dumps(data))
'''.replace('EXPERIMENT',repr(TAG)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state-dir',required=True)
    a=p.parse_args();directory=Path(a.state_dir).resolve();directory.mkdir(parents=True,exist_ok=True)
    configure(directory)
    lock=(directory/'bridge.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_file=directory/'workflow.json'
    state=json.loads(state_file.read_text()) if state_file.exists() else {'tag':TAG,'started':time.time(),'completed_batches':[]}
    state.update(pid=os.getpid(),status='relaying_collection',heartbeat=time.time())
    state.pop('error',None)
    state['raw_archival']='Independent archive worker;never on the training dependency path'
    def record(**kwargs):
        state.update(heartbeat=time.time(),**kwargs);save(state_file,state)
    last_resource_review=0
    def observe_resources():
        nonlocal last_resource_review
        if time.time()-last_resource_review<300:return
        last_resource_review=time.time();snapshots={};alerts=[]
        health=collector_health();record(collector_health=health)
        for host in ('biped','bluesclues'):
            data=resources(host);snapshots[host]=data
            save(directory/(host+'.resources.latest.json'),data)
            with (directory/(host+'.resources.history.jsonl')).open('a') as stream:stream.write(json.dumps(data)+'\n')
            if not data.get('cpu_measurement_complete'):alerts.append(host+': CPU measurement incomplete')
            collecting=health['completed_attempts']<200
            cap=30 if host=='bluesclues' and collecting else 20
            if data['account_cpu_pct']>cap:alerts.append(host+': aggregate account CPU exceeds'+str(cap)+'%')
            if data.get('gpu_snapshot_error'):alerts.append(host+': GPU snapshot incomplete')
            devices=set(data.get('account_gpu_devices',[]))
            if host=='bluesclues' and len(devices)>5:alerts.append(host+': more than5 account GPUs')
            if host=='biped':
                extras=[p for p in data.get('account_gpu_processes',[]) if p['gpu']!=0]
                for proc in extras:
                    main=any(p['gpu']==0 and p['pid']==proc['pid'] for p in data['account_gpu_processes'])
                    if not (proc['gpu']==1 and proc['type']=='G' and proc['memory_mib'] is not None and proc['memory_mib']<=6 and main and proc['this_experiment']):
                        alerts.append(host+': GPU context outside the reviewed GPU0 plus6MiB graphics exception')
                if any(r['gpu']!=0 for r in data.get('live_gpu_reservations',[])):alerts.append(host+': reservation outside GPU0')
        save(directory/'resource_attention.json',{'at':time.time(),'alerts':alerts,'action':'read-only review;no account-wide enforcement'})
        record(latest_resource_review_at=time.time(),resource_alerts=alerts)
    def wait_for_blue_cpu(extra_cpus,purpose):
        while True:
            collector=collector_health();record(collector_health=collector)
            cap=30 if collector['completed_attempts']<200 else 20
            current=resources('bluesclues')
            assert current['cpu_measurement_complete'],current
            review={'purpose':purpose,'account_cpu_pct':current['account_cpu_pct'],'account_cap_pct':cap,'additional_cpu_upper_bound_pct':extra_cpus/current['logical_cpus']*100}
            if review['account_cpu_pct']+review['additional_cpu_upper_bound_pct']<=cap:
                record(status='relaying_collection',cpu_headroom_review=review);return cap
            record(status='waiting_for_cpu_headroom',cpu_headroom_review=review)
            observe_resources();time.sleep(30)
    try:
        resumed=submitted_training_state(state)
        if resumed:
            ready,training=resumed
            record(status='training',resumed_existing_submission=True)
        else:
            for index in range(20):
                batch=f'batch_{index:03d}'
                if batch in state['completed_batches']:continue
                record(active_batch=batch)
                report=None
                while report is None:
                    observe_resources()
                    observed=stage_observation('biped',batch)
                    report=observed['ready'];record(collector_observation=observed)
                    assert observed['workflow'] is not None,'Collector did not publish workflow; inspect its log'
                    if report is None:time.sleep(30)
                save(directory/(batch+'.complete.json'),report)
                raw=WORK+'/raw/'+batch
                has_raw=bool(report['episodes'])
                wait_for_blue_cpu(1,'Verified raw relay;dedicated transport and copy share one CPU')
                if has_raw:checked_transfer('biped','bluesclues',raw,directory,batch+'_raw_blue')
                write_remote('bluesclues',WORK+'/'+batch+'.complete.json',report)
                if has_raw:
                    expected=json.loads((directory/(batch+'_raw_blue.source_inventory.json')).read_text())
                    remove_relocated_copy('biped',raw,expected,{'host':'bluesclues','relative':raw})
                collector=observed['workflow']
                cpu_cap=30 if collector['completed_attempts']<200 else 20
                while True:
                    review=start_conversion(batch,cpu_cap=cpu_cap)
                    if not review.get('waiting_for_cpu_headroom'):break
                    record(status='waiting_for_cpu_headroom',conversion_launch_review=review)
                    observe_resources();time.sleep(30)
                    collector=collector_health();record(collector_health=collector)
                    cpu_cap=30 if collector['completed_attempts']<200 else 20
                record(status='relaying_collection')
                converted=None
                conversion_started=time.time()
                while converted is None:
                    observe_resources()
                    observed=stage_observation('bluesclues',batch,conversion=True)
                    converted=observed['ready'];status=observed['workflow']
                    assert status is not None or time.time()-conversion_started<90,'Converter did not publish workflow;inspect its log'
                    record(conversion_observation=observed)
                    if converted is None:time.sleep(30)
                save(directory/(batch+'.converted.json'),converted)
                requests=directory/'archive_requests';requests.mkdir(exist_ok=True)
                save(requests/(batch+'.json'),{'batch':batch,'raw':raw,'has_raw':has_raw,'report':report})
                state['completed_batches'].append(batch);record()
                print(json.dumps({'batch_relayed_and_converted':batch,'raw_archive_queued':True,'batches':len(state['completed_batches'])}),flush=True)
            record(status='assembling')
            while True:
                launch=start_assembly();record(assembly_launch=launch)
                if not launch.get('waiting_for_cpu_headroom'):break
                record(status='waiting_for_cpu_headroom',cpu_stage='assembly');observe_resources();time.sleep(30)
            record(status='assembling')
            started=time.time();ready=None
            while ready is None:
                observe_resources()
                ready=read_remote('bluesclues',WORK+'/dataset.ready.json')
                status=read_remote('bluesclues',WORK+'/assemble.workflow.json');record(assembly_state=status)
                assert status is not None or time.time()-started<90,'Assembly did not publish workflow;inspect its log'
                assert not status or status['status']!='failed',status
                if status and status['status']=='running':
                    live=call('bluesclues',"import json,pathlib\nprint(json.dumps(pathlib.Path('/proc',str(PID)).exists()))".replace('PID',repr(status['pid'])))
                    assert live,('Assembly process disappeared',status)
                if ready is None:time.sleep(30)
            assert ready['new_attempts']==200
            merged='datasets/'+TAG+'_r3/lerobot/ds'
            transferred=directory/'cumulative_dataset_adroit.transfer.json'
            # A verified recovery is already staged on Adroit. Its checksum-only
            # reconciliation there allocates no Bluesclues transfer worker.
            if not (transferred.exists() and json.loads(transferred.read_text()).get('verified')):
                wait_for_blue_cpu(1,'Cumulative training dataset transfer;dedicated transport and copy share one CPU')
            checked_transfer('bluesclues','adroit',merged,directory,'cumulative_dataset_adroit')
            target=call('adroit',"import json,pathlib\nprint(json.dumps(str(pathlib.Path.home()/'kevin'/REL)))".replace('REL',repr(merged)))
            ready=dict(ready,source_dataset=ready['dataset'],dataset=target)
            write_remote('adroit',WORK+'/dataset.ready.json',ready)
            # Preserve all batch provenance beside the single training dataset.
            for index in range(20):
                batch=f'batch_{index:03d}'
                write_remote('adroit',WORK+'/'+batch+'.complete.json',read_remote('bluesclues',WORK+'/'+batch+'.complete.json'))
                record_file=WORK+'/'+batch+'.converted.json'
                if read_remote('adroit',record_file) is None:
                    value=read_remote('bluesclues',record_file)
                    value=dict(value,source_dataset=value['dataset'],dataset_in_cumulative=target)
                    write_remote('adroit',record_file,value)
            record(status='training',dataset=ready)
            training=submit('finetune',ready);record(training_submission=training)
        while True:
            observe_resources()
            observed=job_state(training['job_id'])
            if observed['state'] in ('OBSERVATION_TIMEOUT','OBSERVATION_ERROR'):
                state['queue_observation_attention']={'since':state.get('queue_observation_attention',{}).get('since',time.time()),'latest':observed}
            else:state.pop('queue_observation_attention',None)
            if observed['state']=='RUNNING':
                state['training_progress_watch']=watch_progress(state.get('training_progress_watch'),observed.get('training_progress',{}),time.time())
            record(training_state=observed)
            if observed['state']=='COMPLETED':assert observed['exitcode']=='0:0';break
            assert not any(observed['state'].startswith(x) for x in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED')),observed
            time.sleep(30)
        trained=read_remote('adroit',WORK+'/finetune.ready.json')
        assert trained['steps']==ready['steps'],trained
        checkpoint=trained['checkpoint_relative']
        record(status='staging_final_checkpoint',trained=trained)
        while True:
            current=resources('biped');assert current['cpu_measurement_complete'],current
            if current['account_cpu_pct']+2/current['logical_cpus']*100<=20:break
            record(status='waiting_for_cpu_headroom',cpu_stage='final_checkpoint_copy');observe_resources();time.sleep(30)
        checked_transfer('adroit','biped',checkpoint,directory,'final_checkpoint_biped',inference=True)
        record(status='starting_evaluation')
        while True:
            launch=start_biped_evaluation(checkpoint);record(evaluation_launch=launch)
            if not any(launch.get(k) for k in ('waiting_for_cpu_headroom','waiting_for_gpu_headroom','waiting_for_control_evaluations')):break
            record(status='waiting_for_evaluation_resources');observe_resources();time.sleep(30)
        record(status='evaluating',evaluation_host='biped');started=time.time()
        while True:
            observe_resources()
            results=read_remote('biped',WORK+'/'+TAG+'_r3.eval.json')
            status=read_remote('biped',WORK+'/evaluate.workflow.json');record(evaluation_state=status)
            assert status is not None or time.time()-started<90,'Evaluator did not publish workflow;inspect its log'
            assert not status or status['status']!='failed',status
            if results is not None:break
            if status and status['status']=='running':
                live=call('biped',"import json,pathlib\nprint(json.dumps(pathlib.Path('/proc',str(PID)).exists()))".replace('PID',repr(status['pid'])))
                assert live,('Evaluation controller disappeared',status)
            time.sleep(30)
        assert results and results['episodes']==100
        assert sorted(row['motion_id'] for row in results['per_episode'])==list(range(100))
        save(directory/'results.json',results);record(status='complete',completed_at=time.time(),results=results)
    except BaseException as exc:
        record(status='stopped_for_review',error=repr(exc));raise
    finally:
        lock.close()


if __name__=='__main__':main()
