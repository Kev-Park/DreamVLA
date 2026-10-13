"""Windows control bridge: stage walking datasets, submit A100 fine-tunes, return checkpoints.

No retries after an SSH failure. No local dataset archives or compression: tar
streams between SSH channels, then SHA-256 inventories verify every file.
The bridge never cancels/restarts jobs because a queue observation times out.
"""
import argparse
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


def ssh(host, command):
    if host in ("bluesclues", "biped"):
        address = BLUE if host == "bluesclues" else BIPED
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", address, command]
    assert host == "adroit"
    return ["wsl.exe", "-d", "Ubuntu", "-u", "kevy0", "--", "bash", "-lc",
            f"ssh -o ControlPath={ADROIT_CONTROL_PATH} -o ControlMaster=no -o BatchMode=yes "
            "-o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 adroit "+shlex.quote(command)]


def call(host, script, timeout=90):
    p = subprocess.run(ssh(host, "python3 -"), input=script.encode(), capture_output=True,
                       timeout=timeout, creationflags=FLAGS)
    if p.returncode:
        raise RuntimeError(f"{host} command failed; stopped for review: "+p.stderr.decode(errors="replace")[-1800:])
    return json.loads(p.stdout)


def save(path, data):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data, indent=2)+"\n")
    tmp.replace(path)


def inventory(host, relative):
    script = """import hashlib,json,pathlib
b=(pathlib.Path.home()/'kevin').resolve();p=b/REL
assert p.resolve().is_relative_to(b) and p.is_dir(),str(p)
rows=[]
for f in sorted(p.rglob('*')):
 if f.is_file():
  assert not f.is_symlink(),str(f)
  h=hashlib.sha256()
  with f.open('rb') as stream:
   for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
  rows.append({'path':str(f.relative_to(p)),'bytes':f.stat().st_size,'sha256':h.hexdigest()})
assert rows
print(json.dumps(rows))
""".replace("REL", repr(relative))
    return call(host, script, timeout=1800)


def transfer(source, destination, relative, state_dir, label):
    # Both roots are ~/kevin; all streamed member paths are relative to that root.
    parts = PurePosixPath(relative).parts
    assert parts and ".." not in parts and not relative.startswith("/")
    before = inventory(source, relative)
    save(state_dir/(label+".source_inventory.json"), before)
    call(destination, """import pathlib,json
b=(pathlib.Path.home()/'kevin').resolve();p=b/REL
assert p.resolve().is_relative_to(b)
assert not p.exists(),'Destination already exists; reconcile before transfer: '+str(p)
p.parent.mkdir(parents=True,exist_ok=True)
print(json.dumps({'ready':True}))
""".replace("REL", repr(relative)))
    state = {"source": source, "destination": destination, "relative": relative,
             "started": time.time(), "bytes": sum(r["bytes"] for r in before)}
    save(state_dir/(label+".transfer.json"), state)
    producer_cmd = ssh(source, "tar -C ~/kevin -cf - -- "+shlex.quote(relative))
    consumer_cmd = ssh(destination, "tar -C ~/kevin -xf -")
    with (state_dir/(label+".source.stderr")).open("wb") as srcerr, (state_dir/(label+".destination.stderr")).open("wb") as dsterr:
        producer = subprocess.Popen(producer_cmd, stdout=subprocess.PIPE, stderr=srcerr, creationflags=FLAGS)
        consumer = subprocess.Popen(consumer_cmd, stdin=producer.stdout, stdout=subprocess.DEVNULL,
                                    stderr=dsterr, creationflags=FLAGS)
        producer.stdout.close()
        consumer_code = consumer.wait()
        producer_code = producer.wait()
    assert producer_code == consumer_code == 0, f"Transfer failed: {label}; inspect stderr files"
    after = inventory(destination, relative)
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
        transfer("adroit", "bluesclues", checkpoint, directory, "checkpoint")
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
""".replace("REL", repr(reply)).replace("DATA", repr({"state": "completed", "host": "adroit", "run": run}))
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
        while True:
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
