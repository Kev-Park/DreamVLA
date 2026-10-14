"""Stage one explicitly routed walking evaluation on Biped and return its results.

Streams uncompressed files, verifies SHA-256, never overwrites partial transfers,
and stops for review after any SSH failure. Only inference checkpoint files move.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time

from walk_adroit_bridge import call, save, ssh, FLAGS


def inventory(host, relative, inference=False):
    assert not PurePosixPath(relative).is_absolute() and '..' not in PurePosixPath(relative).parts
    return call(host, """import pathlib,hashlib,json
b=(pathlib.Path.home()/'kevin').resolve();p=b/REL
assert p.resolve().is_relative_to(b)
skip={'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','training_args.bin'} if INFERENCE else set()
rows=[]
for f in sorted(p.rglob('*')):
 if f.is_file() and f.name not in skip:
  assert not f.is_symlink(),str(f)
  h=hashlib.sha256()
  with f.open('rb') as stream:
   for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
  rows.append({'path':str(f.relative_to(p)),'bytes':f.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps(rows))
""".replace('REL', repr(relative)).replace('INFERENCE', repr(inference)), timeout=1800)


def reuse_inference_shards(destination, relative, expected):
    """Link exact immutable inference shards already cached for walking evals."""
    assert destination == 'biped'
    return call(destination, """import pathlib,hashlib,json,os,re
b=(pathlib.Path.home()/'kevin').resolve();target=b/REL
assert target.resolve().is_relative_to(b/'checkpoints')
target.mkdir(parents=True,exist_ok=True);reused=[]
for row in EXPECTED:
 if not re.fullmatch(r'model-\\d+-of-\\d+\\.safetensors',row['path']):continue
 output=target/row['path'];assert not output.exists()
 for candidate in sorted((b/'checkpoints').glob('walk*_v3_dagger57_20261013_r*/checkpoint-*/'+row['path'])):
  if candidate.parent==target or candidate.is_symlink():continue
  if not candidate.resolve().is_relative_to(b/'checkpoints') or candidate.stat().st_size!=row['bytes']:continue
  h=hashlib.sha256()
  with candidate.open('rb') as stream:
   for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
  if h.hexdigest()!=row['sha256']:continue
  os.link(candidate,output)
  reused.append(dict(row,source=str(candidate)));break
print(json.dumps(reused))
""".replace('REL', repr(relative)).replace('EXPECTED', repr(expected)), timeout=1800)


def transfer(source, destination, relative, work, label, inference=False):
    before = inventory(source, relative, inference)
    assert before and all(r['bytes'] > 0 for r in before)
    save(work/(label+'.source_inventory.json'), before)
    existing = inventory(destination, relative, inference)
    if existing:
        assert existing == before, 'Partial or different destination needs reconciliation: '+relative
        return
    needed = sum(r['bytes'] for r in before)
    call(destination, """import pathlib,json,shutil
b=(pathlib.Path.home()/'kevin').resolve();p=b/REL
assert p.resolve().is_relative_to(b)
assert shutil.disk_usage(b).free > NEEDED+20*2**30,'Less than 20 GiB headroom after transfer'
p.parent.mkdir(parents=True,exist_ok=True)
print(json.dumps({'ready':True}))
""".replace('REL', repr(relative)).replace('NEEDED', str(needed)))
    reused = reuse_inference_shards(destination, relative, before) if inference and destination == 'biped' else []
    save(work/(label+'.reused_shards.json'), reused)
    reused_names = {r['path'] for r in reused}
    paths = [str(PurePosixPath(relative)/r['path']) for r in before if r['path'] not in reused_names]
    with (work/(label+'.source.stderr')).open('ab') as a, (work/(label+'.destination.stderr')).open('ab') as b:
        producer = subprocess.Popen(ssh(source, 'tar -C ~/kevin -cf - -- '+' '.join(map(shlex.quote, paths))),
            stdout=subprocess.PIPE, stderr=a, creationflags=FLAGS)
        consumer = subprocess.Popen(ssh(destination, 'tar -C ~/kevin -xf -'), stdin=producer.stdout,
            stdout=subprocess.DEVNULL, stderr=b, creationflags=FLAGS)
        producer.stdout.close()
        consumer_code = consumer.wait()
        producer_code = producer.wait()
    assert producer_code == consumer_code == 0, 'Transfer failed; inspect '+label+' stderr files'
    assert inventory(destination, relative, inference) == before, 'Transfer checksum mismatch: '+relative
    save(work/(label+'.verified.json'), {'verified_at':time.time(),'bytes':needed,'files':len(before)})


def observe_and_return(run, work, state, request_relative):
    statusfile = work/'workflow.json'
    launched = state['launch']
    while True:
        observed = call('biped', """import pathlib,json
work=pathlib.Path(WORK);p=work/'workflow.json';process=pathlib.Path('/proc',str(PID),'cmdline')
assert p.exists() or (process.exists() and process.read_bytes()),'Evaluation controller exited before writing state: '+(work/'controller.log').read_text(errors='replace')[-2000:]
print(p.read_text() if p.exists() else 'null')
""".replace('WORK', repr(launched['work'])).replace('PID', str(launched['controller_pid'])))
        state.update(observation=observed,heartbeat=time.time());save(statusfile,state)
        if observed:
            assert observed['status']!='failed',observed
            if observed['status']=='complete':
                break
        time.sleep(30)
    state.update(state='returning_results',heartbeat=time.time());save(statusfile,state)
    transfer('biped','bluesclues','eval_videos/'+run,work,'results')
    reply_relative = request_relative.replace('.request.json','.reply.json')
    delivered = call('bluesclues', """import pathlib,json,math
b=pathlib.Path.home()/'kevin';run=RUN;p=b/REPLY;report=json.loads((b/'eval_videos'/run/'results.json').read_text())
assert report['pool']=='StitchMAX100v3_eval' and report['episodes']==100
assert sorted(r['motion_id'] for r in report['per_episode'])==list(range(100))
assert all(math.isfinite(report[k]) for k in ['success_pct','world_mpjpe_cm','root_aligned_mpjpe_cm','physical_held5_pct'])
assert not p.exists();data={'state':'completed','host':'biped','run':run};tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
print(json.dumps({'delivered':True,'success_pct':report['success_pct']}))
""".replace('RUN', repr(run)).replace('REPLY', repr(reply_relative)))
    state.update(state='complete',result=delivered,heartbeat=time.time(),completed_at=time.time());save(statusfile,state)


def main(run, work, resume_existing=False):
    assert re.fullmatch(r'walk(10|50|200)_v3_dagger57_20261013_r[0-3]', run)
    work.mkdir(parents=True, exist_ok=True)
    lock = (work/'controller.lock').open('a+b')
    import msvcrt
    lock.seek(0)
    if not lock.read(1):
        lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    statusfile = work/'workflow.json'
    budget = int(run.split('_')[0][4:])
    request_relative = f'dline/walk{budget}_v3_dagger57_20261013/external/evaluate_{run}.request.json'
    if resume_existing:
        state = json.loads(statusfile.read_text())
        assert state['run'] == run and state['state'] == 'stopped_for_review', state
        assert (work/'inference.verified.json').exists()
        assert state['launch'] == json.loads((work/'launch.json').read_text())
        assert state['launch']['controller_pid'] > 0
        with (work/'observation_recoveries.jsonl').open('a') as stream:
            stream.write(json.dumps(state)+'\n')
        state.update(pid=os.getpid(),state='resuming_existing_evaluation',heartbeat=time.time())
        state.pop('error',None)
        save(statusfile,state)
        try:
            observe_and_return(run,work,state,request_relative)
        except BaseException as exc:
            state.update(state='stopped_for_review',error=repr(exc),heartbeat=time.time())
            save(statusfile,state)
            raise
        return
    assert not statusfile.exists(), 'Reconcile the previous bridge before restarting'
    state = {'pid':os.getpid(),'run':run,'state':'waiting_for_request','started':time.time()}
    save(statusfile, state)
    try:
        while True:
            request = call('bluesclues', """import pathlib,json
b=(pathlib.Path.home()/'kevin').resolve();p=b/REL
print(json.dumps({'spec':json.loads(p.read_text()),'root':str(b)} if p.exists() else None))
""".replace('REL', repr(request_relative)))
            if request:
                break
            state.update(heartbeat=time.time());save(statusfile,state)
            time.sleep(30)
        spec = request['spec']
        assert spec['host']=='biped' and spec['phase']=='evaluate' and spec['run']==run
        assert spec['pool']=='StitchMAX100v3_eval' and spec['episodes']==100
        relative = str(PurePosixPath(spec['checkpoint']).relative_to(request['root']))
        assert re.fullmatch(r'checkpoints/'+run+r'/checkpoint-\d+', relative)
        state.update(state='staging_inference_weights',heartbeat=time.time());save(statusfile,state)
        transfer('bluesclues','biped',relative,work,'inference',inference=True)
        references = call('bluesclues', """import pathlib,glob,hashlib,json
b=pathlib.Path.home()/'kevin';rows=[]
for i,f in enumerate(glob.glob(str(b/'ref_motions/StitchMAX100v3_eval/*.pkl'))):
 p=pathlib.Path(f);rows.append({'canonical_id':i,'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
assert len(rows)==100
print(json.dumps(rows))
""")
        payload = {'run':run,'pool':spec['pool'],'episodes':100,'box':spec['box'],
                   'checkpoint_relative':relative,'canonical_references':references}
        launched = call('biped', """import pathlib,glob,hashlib,json,subprocess,os,shlex,time
p=PAYLOAD;b=pathlib.Path.home()/'kevin';repo=b/'wt/locomotion/DreamVLA'
native=[pathlib.Path(f) for f in glob.glob(str(b/'ref_motions/StitchMAX100v3_eval/*.pkl'))]
assert len(native)==100 and len({f.name for f in native})==100
lookup={f.name:(i,f) for i,f in enumerate(native)};mapping=[]
for row in p.pop('canonical_references'):
 i,f=lookup[row['name']];h=hashlib.sha256(f.read_bytes()).hexdigest();assert h==row['sha256'],f.name
 mapping.append({'source_id':i,'canonical_id':row['canonical_id'],'name':f.name,'source_sha256':h,'canonical_sha256':row['sha256']})
p['reference_mapping']={'pool':p['pool'],'source_host':'biped','canonical_host':'bluesclues','references':mapping}
work=b/'eval_videos/walk_v3_study/biped_eval'/p['run'];assert not (work/'request.json').exists();work.mkdir(parents=True,exist_ok=True)
(work/'request.json').write_text(json.dumps(p,indent=2));env=os.environ.copy();env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
cmd='cd '+shlex.quote(str(repo))+' && exec '+shlex.quote(str(b/'wt/locomotion/.venv/bin/python'))+' -u cluster/walk_biped_eval.py --work '+shlex.quote(str(work))+' >> '+shlex.quote(str(work/'controller.log'))+' 2>&1'
session='eval_'+p['run']+'_biped'
subprocess.run(['tmux','new-session','-d','-s',session,cmd],env=env,check=True)
pid=int(subprocess.check_output(['tmux','display-message','-p','-t',session,'#{pane_pid}'],text=True).strip())
print(json.dumps({'work':str(work),'session':session,'controller_pid':pid,'mapping':p['reference_mapping'],'launched_at':time.time()}))
""".replace('PAYLOAD', repr(payload)))
        save(work/'launch.json',launched)
        state.update(state='running',launch=launched,heartbeat=time.time());save(statusfile,state)
        observe_and_return(run, work, state, request_relative)
    except BaseException as exc:
        state.update(state='stopped_for_review',error=repr(exc),heartbeat=time.time());save(statusfile,state)
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--state-dir',type=Path,required=True)
    parser.add_argument('--resume-existing',action='store_true',help='Observe the verified existing remote evaluation after the user restores access; never launch a replacement.')
    args=parser.parse_args()
    main(args.run,args.state_dir,args.resume_existing)
