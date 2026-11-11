"""Stage the existing R2 checkpoint, then evaluate it unchanged on Adroit."""
import argparse
import json
import os
from pathlib import Path
import time

import walk_late_takeover_bridge as bridge
from walk_adroit_bridge import save

TAG='walk200_r2_adroit_control_20261026'
CHECKPOINT='checkpoints/walk200_v3_dagger57_20261013_r2/checkpoint-92967'


def submit():
    return bridge.call('adroit',r'''
import hashlib,json,pathlib,subprocess,time
b=pathlib.Path.home()/'kevin';work=b/'dline'/TAG;out=b/'eval_videos'/TAG
work.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
checkpoint=b/CHECKPOINT;state=json.loads((checkpoint/'trainer_state.json').read_text());assert state['global_step']==92967
index=json.loads((checkpoint/'model.safetensors.index.json').read_text());assert all((checkpoint/n).is_file() for n in set(index['weight_map'].values()))
source=b/'dline/walk200_v3_late1s_20261024/inputs/box.json';target=work/'inputs/box.json';target.parent.mkdir(exist_ok=True)
if target.exists():assert target.read_bytes()==source.read_bytes()
else:target.write_bytes(source.read_bytes())
receipt=work/'submission.json'
if receipt.exists():print(receipt.read_text())
else:
 repo=b/'wt/locomotion/DreamVLA'
 command=['sbatch','--parsable','--job-name='+TAG,'--output='+str(out/'slurm-%j.log'),str(repo/'cluster/adroit/walk_reference_eval.sbatch'),TAG,str(checkpoint)]
 job=subprocess.check_output(command,text=True).strip().split(';')[0];assert job.isdigit()
 value={'at':time.time(),'job_id':job,'command':command,'model':'current R2 checkpoint92967','condition':'unchanged baseline','references':100,'pool':'StitchMAX100v3_eval','decoder':'previous full FSQ','fine_tune':False,'box_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
 receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''.replace('TAG',repr(TAG)).replace('CHECKPOINT',repr(CHECKPOINT)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir',required=True)
    parser.add_argument('--shared-state-dir',required=True)
    args=parser.parse_args();directory=Path(args.state_dir).resolve();directory.mkdir(exist_ok=True)
    bridge.configure(args.shared_state_dir)
    lock=(directory/'control.lock').open('a+b')
    if os.name=='nt':
        import ctypes,msvcrt
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        awake=ctypes.WinDLL('kernel32').SetThreadExecutionState;awake(0x80000001)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);awake=None
    file=directory/'workflow.json';state=json.loads(file.read_text()) if file.exists() else {'started':time.time(),'tag':TAG}
    state.update(pid=os.getpid(),status='staging_checkpoint',heartbeat=time.time())
    def record(**kwargs):state.update(heartbeat=time.time(),**kwargs);save(file,state)
    try:
        record()
        bridge.checked_transfer('biped','adroit',CHECKPOINT,directory,'r2_checkpoint')
        submission=submit();record(status='evaluating',submission=submission)
        while True:
            observed=bridge.job_state(submission['job_id']);record(slurm=observed)
            if observed['state']=='COMPLETED':assert observed['exitcode']=='0:0';break
            assert not any(observed['state'].startswith(s) for s in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED')),observed
            time.sleep(30)
        results=bridge.read_remote('adroit','eval_videos/'+TAG+'/results.json')
        assert results['episodes']==100 and sorted(e['motion_id'] for e in results['per_episode'])==list(range(100))
        save(directory/'results.json',results);record(status='complete',completed=time.time(),results=results)
    except BaseException as exc:
        record(status='stopped_for_review',error=repr(exc));raise
    finally:
        if awake:awake(0x80000000)


if __name__=='__main__':main()
