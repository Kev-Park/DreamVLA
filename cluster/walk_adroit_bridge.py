"""Windows control bridge: stage walking datasets, submit A100 fine-tunes, return checkpoints.

No retries after an SSH failure. No local dataset archives or compression: tar
streams between SSH channels, then SHA-256 inventories verify every file.
The bridge never cancels/restarts jobs because a queue observation times out.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time

BLUE = "sastrygrp-dvij@bluesclues.ist.berkeley.edu"
BIPED = "dvij@biped.eecs.berkeley.edu"
FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)
ADROIT_CONTROL_PATH = "~/.ssh/cm-adroit"


class RemoteCommandError(RuntimeError):
    """SSH reached the host, but the command there failed."""


def bounded_transport_command(command):
    """Budget one new, nonmultiplexed SSH transfer session to one CPU.

    Only the command and its own user-owned sshd parent are affected. Never use
    this wrapper with a shared control master or a privileged sshd process.
    The wrapper leaves stdin/stdout untouched for binary tar streams.
    """
    script = """import os,pathlib
parent=os.getppid();process=pathlib.Path('/proc',str(parent))
assert process.stat().st_uid==os.getuid(),'Refuse a privileged or foreign SSH parent'
assert (process/'comm').read_text().strip() in ('sshd','sshd-session'),'Dedicated SSH session not verified'
cpu=min(os.sched_getaffinity(0))
for pid in (parent,os.getpid()):
 for thread in pathlib.Path('/proc',str(pid),'task').iterdir():
  try:os.sched_setaffinity(int(thread.name),{cpu})
  except ProcessLookupError:pass
 assert os.sched_getaffinity(pid)=={cpu},'CPU allocation did not apply'
os.execvp('bash',['bash','-c',COMMAND])
""".replace('COMMAND', repr(command))
    return 'python3 -c ' + shlex.quote(script)


def ssh(host, command, single_cpu=False):
    if single_cpu:
        assert host == 'bluesclues', 'Never restrict a shared Adroit connection'
        command = bounded_transport_command(command)
    if host in ("bluesclues", "biped"):
        address = BLUE if host == "bluesclues" else BIPED
        options = ["-o", "ControlMaster=no", "-o", "ControlPath=none"] if single_cpu else []
        return ["ssh", *options, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", address, command]
    assert host == "adroit"
    return ["wsl.exe", "-d", "Ubuntu", "-u", "kevy0", "--", "bash", "-lc",
            f"ssh -o ControlPath={ADROIT_CONTROL_PATH} -o ControlMaster=no -o BatchMode=yes "
            "-o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 adroit "+shlex.quote(command)]


def call(host, script, timeout=90, single_cpu=False):
    # Small Adroit control reads need no streamed stdin. Keeping their source
    # in the remote command avoids an unnecessary WSL/SSH input stream. Bound
    # the encoded command size below Windows' command-line limit; larger data
    # payloads retain the existing stdin transport.
    inline = host == 'adroit' and len(script.encode('utf-8')) <= 6000
    # Encode source so WSL's additional command parsing cannot alter Python
    # backslashes, quotes or newlines. The remote Python still executes exactly
    # the original UTF-8 bytes; this is command transport, not data compression.
    encoded = base64.b64encode(script.encode('utf-8')).decode('ascii') if inline else None
    remote = ('python3 -c ' + shlex.quote("exec(__import__('base64').b64decode('" + encoded + "'))")) if inline else 'python3 -'
    command = ssh(host, remote, single_cpu=single_cpu)
    payload = {'stdin': subprocess.DEVNULL} if inline else {'input': script.encode()}
    p = subprocess.run(command, **payload, capture_output=True,
                       timeout=timeout, creationflags=FLAGS)
    if p.returncode:
        error = RuntimeError if p.returncode == 255 else RemoteCommandError
        raise error(f"{host} command failed; stopped for review: "+p.stderr.decode(errors="replace")[-1800:])
    return json.loads(p.stdout)


def save(path, data):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data, indent=2)+"\n")
    tmp.replace(path)


def inventory_script(relative, inference=False):
    return """import hashlib,json,pathlib
b=(pathlib.Path.home()/'kevin').resolve();p=b/REL
assert p.resolve().is_relative_to(b) and p.is_dir(),str(p)
rows=[]
skip={'optimizer.pt','scheduler.pt','rng_state.pth','training_args.bin'} if INFERENCE else set()
for f in sorted(p.rglob('*')):
 if f.is_file() and f.name not in skip:
  expected_digest=None
  if f.is_symlink():
   assert INFERENCE and f.suffix=='.safetensors','Unexpected inventory symlink: '+str(f)
   target=f.resolve(strict=True)
   assert target.is_relative_to(b/'checkpoints/_shared_weights/sha256') and target.suffix=='.safetensors','Weight link leaves the owned canonical store: '+str(f)
   expected_digest=target.stem
   assert len(expected_digest)==64 and all(c in '0123456789abcdef' for c in expected_digest),'Invalid canonical weight identity'
  h=hashlib.sha256()
  with f.open('rb') as stream:
   for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
  digest=h.hexdigest()
  assert expected_digest is None or digest==expected_digest,'Canonical weight content differs from its SHA256 identity: '+str(f)
  rows.append({'path':str(f.relative_to(p)),'bytes':f.stat().st_size,'sha256':digest})
assert rows
print(json.dumps(rows))
""".replace("REL", repr(relative)).replace("INFERENCE", repr(inference))


def inventory(host, relative, inference=False, single_cpu=False):
    script = inventory_script(relative, inference)
    return call(host, script, timeout=1800, **({'single_cpu': True} if single_cpu else {}))


def transfer(source, destination, relative, state_dir, label, inference=False, blue_single_cpu=False):
    # Both roots are ~/kevin; all streamed member paths are relative to that root.
    parts = PurePosixPath(relative).parts
    assert parts and ".." not in parts and not relative.startswith("/")
    before = inventory(source, relative, inference,
                       **({'single_cpu': True} if blue_single_cpu and source == 'bluesclues' else {}))
    save(state_dir/(label+".source_inventory.json"), before)
    call(destination, """import pathlib,json
b=(pathlib.Path.home()/'kevin').resolve();p=b/REL
assert p.resolve().is_relative_to(b)
assert not p.exists(),'Destination already exists; reconcile before transfer: '+str(p)
p.parent.mkdir(parents=True,exist_ok=True)
print(json.dumps({'ready':True}))
""".replace("REL", repr(relative)), **({'single_cpu': True} if blue_single_cpu and destination == 'bluesclues' else {}))
    state = {"source": source, "destination": destination, "relative": relative,
             "started": time.time(), "bytes": sum(r["bytes"] for r in before),
             "inference_and_trainer_metadata_only": inference, "blue_transport_single_cpu": blue_single_cpu}
    save(state_dir/(label+".transfer.json"), state)
    members = [str(PurePosixPath(relative)/r['path']) for r in before] if inference else [relative]
    archive = 'tar --dereference' if inference else 'tar'
    producer_cmd = ssh(source, archive+" -C ~/kevin -cf - -- "+' '.join(map(shlex.quote,members)),
                       **({'single_cpu': True} if blue_single_cpu and source == 'bluesclues' else {}))
    consumer_cmd = ssh(destination, "tar -C ~/kevin -xf -",
                       **({'single_cpu': True} if blue_single_cpu and destination == 'bluesclues' else {}))
    with (state_dir/(label+".source.stderr")).open("wb") as srcerr, (state_dir/(label+".destination.stderr")).open("wb") as dsterr:
        producer = subprocess.Popen(producer_cmd, stdout=subprocess.PIPE, stderr=srcerr, creationflags=FLAGS)
        consumer = subprocess.Popen(consumer_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                    stderr=dsterr, creationflags=FLAGS)
        transferred=0;last_progress=0
        try:
            while block := producer.stdout.read(1024*1024):
                consumer.stdin.write(block);transferred+=len(block)
                if time.time()-last_progress>=10:
                    state.update(stream_bytes=transferred,heartbeat=time.time())
                    save(state_dir/(label+'.transfer.json'),state);last_progress=time.time()
            consumer.stdin.close();producer.stdout.close()
            consumer_code=consumer.wait();producer_code=producer.wait()
        except BaseException:
            producer.stdout.close()
            try:consumer.stdin.close()
            except OSError:pass
            for process in (producer,consumer):
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.terminate();process.wait(timeout=15)
            state.update(failed=time.time(),producer_code=producer.returncode,consumer_code=consumer.returncode,stream_bytes=transferred)
            save(state_dir/(label+'.transfer.json'),state)
            raise
    state.update(producer_code=producer_code,consumer_code=consumer_code,stream_bytes=transferred,heartbeat=time.time())
    save(state_dir/(label+'.transfer.json'),state)
    assert producer_code == consumer_code == 0, f"Transfer failed: {label}; inspect stderr files"
    after = inventory(destination, relative, inference,
                      **({'single_cpu': True} if blue_single_cpu and destination == 'bluesclues' else {}))
    assert before == after, f"Checksum mismatch after transfer: {label}"
    state.update(completed=time.time(), verified=True, files=len(before))
    save(state_dir/(label+".transfer.json"), state)


def update_remote_code():
    return call("adroit", """import pathlib,subprocess,json
b=pathlib.Path.home()/'kevin';source=b/'DreamVLA';target=b/'wt/locomotion/DreamVLA'
if not target.exists():
 subprocess.run(['git','fetch','origin','locomotion'],cwd=source,check=True,stdout=subprocess.DEVNULL)
 target.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['git','worktree','add','--detach',str(target),'FETCH_HEAD'],cwd=source,check=True,stdout=subprocess.DEVNULL)
else:
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=target,text=True).strip()
 subprocess.run(['git','pull','--ff-only','origin','locomotion'],cwd=target,check=True,stdout=subprocess.DEVNULL)
print(json.dumps({'checkout':str(target),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,text=True).strip()}))
""")


def capacity_forecast():
    """Observe full-A100 queue availability without submitting or allocating jobs."""
    return call('adroit', """import pathlib,subprocess,json,time,concurrent.futures,re
b=pathlib.Path.home()/'kevin'
def predict(gpus):
 cmd=['sbatch','--test-only','--gres=gpu:'+str(gpus),'--cpus-per-task='+str(8*gpus),
      '--mem='+str(96*gpus)+'G',str(b/'wt/locomotion/DreamVLA/cluster/adroit/walk_gr00t_finetune.sbatch'),
      str(b/'datasets/walk_v3_study/base200/lerobot/ds'),'walk_capacity_preflight','44753']
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
 match=re.search(r'to start at (\\S+)',p.stderr)
 return {'gpus':gpus,'returncode':p.returncode,'predicted_start':match.group(1) if match else None,
         'stdout':p.stdout,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(2) as pool:rows=list(pool.map(predict,[1,2]))
assert all(r['returncode']==0 and r['predicted_start'] for r in rows),rows
print(json.dumps({'observed_at':time.time(),'cluster_timezone':list(time.tzname),
                  'predictions':rows,'test_only':True,'jobs_submitted':False}))
""", timeout=90)


def validate_finetune_request(spec):
    assert spec["phase"] == "finetune" and spec["host"] == "adroit"
    assert spec["global_batch_size"] == 16 and spec["passes"] == 5.7
    assert spec['budget'] in (10, 50, 200)
    assert re.fullmatch(f"walk{spec['budget']}_v3_dagger57_20261013_r[0-3]", spec['run'])
    assert spec['learning_rate'] == 0.0001 and spec['steps'] > 0


def completed_checkpoint(checkpoint, steps, slurm_state, exit_code):
    """Accept only a successful Slurm exit and a complete resumable checkpoint."""
    if slurm_state != 'COMPLETED' or exit_code != '0:0':
        return False
    try:
        required = ['trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth']
        if not all((checkpoint/name).is_file() and (checkpoint/name).stat().st_size > 8 for name in required):
            return False
        names = set(json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map'].values())
        return (bool(names) and all((checkpoint/n).is_file() and (checkpoint/n).stat().st_size > 8 for n in names)
                and json.loads((checkpoint/'trainer_state.json').read_text())['global_step'] == steps)
    except (OSError, ValueError, KeyError):
        return False


def process_request(request, state_dir):
    spec = request["spec"]
    validate_finetune_request(spec)
    run = spec["run"]
    directory = state_dir/run
    directory.mkdir(exist_ok=True)
    record = directory/"state.json"
    state = json.loads(record.read_text()) if record.exists() else {"state": "new", "request": request}
    if state["state"] == "new":
        state.update(state="staging", started=time.time())
        save(record, state)
        dataset = str(PurePosixPath(spec["dataset"]).relative_to(request["blue_root"]))
        transfer("bluesclues", "adroit", dataset, directory, "dataset")
        state.update(state="staged", dataset_relative=dataset)
        save(record, state)
    if state["state"] == "staging":
        raise RuntimeError("Interrupted dataset transfer needs reconciliation: "+run)
    if state["state"] == "staged":
        payload = {"dataset_relative": state["dataset_relative"], "run": run, "steps": spec["steps"]}
        code = """import pathlib,subprocess,json
p=PAYLOAD;b=pathlib.Path.home()/'kevin';out=b/'eval_videos/walk_v3_study/adroit';out.mkdir(parents=True,exist_ok=True)
receipt=out/(p['run']+'.submission.json')
if receipt.exists():print(receipt.read_text())
else:
 cmd=['sbatch','--parsable','--job-name='+p['run'],'--output='+str(out/(p['run']+'-%j.log')),
      str(b/'wt/locomotion/DreamVLA/cluster/adroit/walk_gr00t_finetune.sbatch'),str(b/p['dataset_relative']),p['run'],str(p['steps'])]
 result=subprocess.check_output(cmd,text=True).strip();job=result.split(';')[0];assert job.isdigit(),result
 d={'job_id':job,'run':p['run'],'command':cmd};receipt.write_text(json.dumps(d,indent=2));print(json.dumps(d))
""".replace("PAYLOAD", repr(payload))
        submitted = call("adroit", code)
        state.update(state="submitted", **submitted, submitted=time.time())
        save(record, state)
    if state["state"] == "submitted":
        payload = {"run": run, "steps": spec["steps"], "job_id": state["job_id"]}
        code = """import pathlib,subprocess,json,sys
p=PAYLOAD;b=pathlib.Path.home()/'kevin';ck=b/'checkpoints'/p['run']/('checkpoint-'+str(p['steps']))
sys.path.insert(0,str(b/'wt/locomotion/DreamVLA/cluster'))
from walk_adroit_bridge import completed_checkpoint
observed=subprocess.run(['squeue','-j',p['job_id'],'-h','-o','%T'],capture_output=True,text=True)
if observed.returncode and 'Invalid job id specified' not in observed.stderr:
 raise RuntimeError('Queue observation failed: '+observed.stderr)
q=observed.stdout.strip() if observed.returncode==0 else ''
d={'queue_state':q,'checkpoint':str(ck),'complete':False}
if not q:
 d['accounting']=subprocess.check_output(['sacct','-j',p['job_id'],'-n','-X','-P','-o','State,ExitCode'],text=True).strip()
 if not d['accounting']:
  d['queue_state']='ACCOUNTING_PENDING'
 else:
  rows=d['accounting'].splitlines();assert len(rows)==1,rows
  fields=rows[0].split('|');state,exit_code=fields[0].strip(),fields[1].strip()
  if state in ['PENDING','RUNNING','COMPLETING','CONFIGURING','SUSPENDED','RESIZING','REQUEUED','REQUEUE_FED','STAGE_OUT']:
   d['queue_state']='ACCOUNTING_'+state
  else:d['complete']=completed_checkpoint(ck,p['steps'],state,exit_code)
print(json.dumps(d))
""".replace("PAYLOAD", repr(payload))
        status = call("adroit", code)
        state.update(last_observation=status, heartbeat=time.time())
        save(record, state)
        if status["queue_state"]:
            return
        if not status["complete"]:
            raise RuntimeError("Slurm job exited without a verified final checkpoint: "+json.dumps(status))
        state["state"] = "returning_checkpoint"
        save(record, state)
        checkpoint = f"checkpoints/{run}/checkpoint-{spec['steps']}"
        # The full optimizer/RNG checkpoint stays on Adroit, where the successful
        # Slurm exit and resumability were verified above. Student collection and
        # independent subsequent fine-tunes need only inference files and steps.
        transfer("adroit", "bluesclues", checkpoint, directory, "checkpoint", inference=True)
        state['resumable_checkpoint_host'] = 'adroit'
        state['resumable_checkpoint'] = status['checkpoint']
        state.update(state="completed", completed=time.time())
        save(record, state)
    if state["state"] == "returning_checkpoint":
        raise RuntimeError("Interrupted checkpoint transfer needs reconciliation: "+run)
    if state["state"] == "completed":
        reply = str(PurePosixPath(request["relative"]).with_name(PurePosixPath(request["relative"]).name.replace(".request.json", ".reply.json")))
        code = """import pathlib,json
b=pathlib.Path.home()/'kevin';f=b/REL;d=DATA
if f.exists():assert json.loads(f.read_text())==d
else:
 t=f.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(f)
print(json.dumps({'delivered':True}))
""".replace("REL", repr(reply)).replace("DATA", repr({"state": "completed", "host": "adroit", "run": run,
            "resumable_checkpoint_host": state.get('resumable_checkpoint_host', 'adroit'),
            "resumable_checkpoint": state.get('resumable_checkpoint', state.get('last_observation', {}).get('checkpoint'))}))
        call("bluesclues", code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    statusfile = args.state_dir/"workflow.json"
    state = {"pid": os.getpid(), "started": time.time(), "state": "starting"}
    try:
        state["code"] = update_remote_code()
        last_capacity_check = 0
        while True:
            if time.time()-last_capacity_check >= 1800:
                forecast = capacity_forecast()
                save(args.state_dir/'capacity_latest.json', forecast)
                with (args.state_dir/'capacity_history.jsonl').open('a') as stream:
                    stream.write(json.dumps(forecast)+'\n')
                state['capacity_forecast'] = forecast
                last_capacity_check = time.time()
            requests = call("bluesclues", """import pathlib,json
b=(pathlib.Path.home()/'kevin').resolve();rows=[]
for budget in [10,50,200]:
 for p in sorted((b/f'dline/walk{budget}_v3_dagger57_20261013/external').glob('finetune_*.request.json')):
  if not p.with_name(p.name.replace('.request.json','.reply.json')).exists():
   spec=json.loads(p.read_text())
   if spec['host']=='adroit':rows.append({'relative':str(p.relative_to(b)),'blue_root':str(b),'spec':spec})
print(json.dumps(rows))
""")
            for request in requests:
                process_request(request, args.state_dir)
            state.update(state="monitoring", heartbeat=time.time(), outstanding=len(requests))
            save(statusfile, state)
            time.sleep(60)
    except BaseException as exc:
        state.update(state="stopped_for_review", heartbeat=time.time(), error=repr(exc))
        save(statusfile, state)
        raise


if __name__ == "__main__":
    main()
