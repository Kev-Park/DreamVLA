"""Archive verified walking raw batches independently of training readiness."""
import argparse
import json
import os
from pathlib import Path
import time

import walk_late_takeover_bridge as bridge
from walk_adroit_bridge import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir',required=True)
    parser.add_argument('--lane',type=int,choices=(0,1),required=True)
    args=parser.parse_args();directory=Path(args.state_dir).resolve()
    bridge.configure(directory)
    lock=(directory/f'archive{args.lane}.lock').open('a+b')
    if os.name=='nt':
        import ctypes,msvcrt
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        awake=ctypes.WinDLL('kernel32').SetThreadExecutionState
        awake(0x80000001)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        awake=None
    path=directory/f'archive{args.lane}.workflow.json'
    state=json.loads(path.read_text()) if path.exists() else {'completed_batches':[],'started':time.time()}
    state.update(pid=os.getpid(),lane=args.lane,status='archiving',heartbeat=time.time())
    def record(**kwargs):
        state.update(heartbeat=time.time(),**kwargs);save(path,state)
    try:
        for index in range(args.lane,20,2):
            batch=f'batch_{index:03d}'
            if batch in state['completed_batches']:continue
            request=directory/'archive_requests'/(batch+'.json')
            receipt=directory/(batch+'_raw_adroit.transfer.json')
            main=json.loads((directory/'workflow.json').read_text())
            # These batches were fully archived and cleaned by the old relay.
            if not request.exists() and batch in main['completed_batches'] and receipt.exists() and json.loads(receipt.read_text()).get('verified'):
                state['completed_batches'].append(batch);record();continue
            while not request.exists():
                main=json.loads((directory/'workflow.json').read_text())
                record(status='waiting_for_converted_batch',active_batch=batch)
                if main['status']=='stopped_for_review':
                    record(status='waiting_for_upstream_review');return
                time.sleep(30)
            payload=json.loads(request.read_text())
            if payload['has_raw']:
                while True:
                    collector=bridge.read_remote('biped',bridge.WORK+'/workflow.json')
                    cap=30 if collector['completed_attempts']<200 else 20
                    resources=bridge.resources('bluesclues')
                    assert resources['cpu_measurement_complete'],resources
                    review={'account_cpu':resources['account_cpu_pct'],'account_cap_pct':cap,'extra_cpu_upper_bound':2/resources['logical_cpus']*100}
                    if review['account_cpu']+review['extra_cpu_upper_bound']<=cap:break
                    record(status='waiting_for_cpu_headroom',active_batch=batch,resource_review=review)
                    time.sleep(30)
                record(status='transferring',active_batch=batch,resource_review=review)
                bridge.checked_transfer('bluesclues','adroit',payload['raw'],directory,batch+'_raw_adroit')
                expected=json.loads((directory/(batch+'_raw_adroit.source_inventory.json')).read_text())
                record(status='releasing_verified_bluesclues_copy')
                bridge.remove_relocated_copy('bluesclues',payload['raw'],expected,{'host':'adroit','relative':payload['raw']})
            state['completed_batches'].append(batch);record(status='archiving')
            print(json.dumps({'raw_batch_archived':batch,'completed':len(state['completed_batches'])}),flush=True)
        record(status='complete',completed=time.time())
    except BaseException as exc:
        record(status='stopped_for_review',error=repr(exc));raise
    finally:
        if awake:awake(0x80000000)


if __name__=='__main__':main()
