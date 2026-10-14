"""Replace one idle pick controller after round-3 collection/conversion finish.

The existing exact-owner GPU reservation pause prevents training from starting
during the handoff. Active workers are never signalled. The replacement keeps
the original command, environment and working directory.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from pick_budget_chain import atomic, read
from watch_pick_budgets import observe


def ready(root, work, budget, health):
    if health['abort'] or not health['controller_live'] or health['active_workers']:
        return False
    receipt=work/'collection_r3.json'
    dataset=root/'datasets'/f'pick{budget}_dagger57_20261013_r3'
    if not receipt.exists() or not (dataset/'built.json').exists():
        return False
    collection=read(receipt);expected=read(work/'recipe.json')['motion_ids']
    assert collection['student_attempts']==budget
    assert collection['motion_ids']==expected and len(set(expected))==budget
    built=read(dataset/'built.json');info=read(dataset/'lerobot/ds/meta/info.json')
    assert built['episodes']==info['total_episodes'] and info['total_frames']>0
    assert built['new_attempts']==budget
    assert not (work/f'pick{budget}_dagger57_20261013_r3.trained.json').exists()
    return True


def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--budget',type=int,choices=[10,50],required=True)
    p.add_argument('--controller',type=int,required=True)
    a=p.parse_args();root=Path.home()/'kevin';work=root/'dline'/f'pick{a.budget}_dagger57_20261013'
    proc=Path('/proc')/str(a.controller)
    command=[os.fsdecode(x) for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    assert Path(command[1]).name=='pick_budget_chain.py' and command[-2:]==['--budget',str(a.budget)]
    start=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
    environment=dict(os.fsdecode(x).split('=',1) for x in (proc/'environ').read_bytes().split(b'\0') if x)
    cwd=os.readlink(proc/'cwd');state=work/'controller_handoff.json'
    def identity():
        assert int((work/'controller.lock').read_text())==a.controller
        assert (proc/'stat').read_text().rsplit(')',1)[1].split()[19]==start
    def own_pause(entry):
        return entry['owner']==a.controller and str(entry['owner_start'])==start and entry['count']==1 and entry['mb']==45000
    with (work/'controller_handoff.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            identity()
            assert any(own_pause(x) for x in read(root/'dline/gpu_reservation_pauses.json'))
            health=observe(work,time.time())
            assert not health['abort'],health['abort']
            atomic(state,{'phase':'waiting_for_idle_controller','time':time.time(),
                         'old_controller':a.controller,'handoff_pid':os.getpid(),'workers':health['active_workers']})
            if ready(root,work,a.budget,health):break
            time.sleep(15)
        identity()
        with (root/'dline/gpu_slots.lock').open('a+') as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX)
            assert not any(x['owner']==a.controller for x in read(root/'dline/gpu_slots.json'))
            assert not observe(work,time.time())['active_workers']
            atomic(state,{'phase':'stopping_idle_controller','time':time.time(),'old_controller':a.controller})
            os.kill(a.controller,signal.SIGTERM)
        deadline=time.monotonic()+30
        while True:
            try:
                if not (proc/'cmdline').read_bytes():break
            except FileNotFoundError:
                break
            if time.monotonic()>deadline:raise RuntimeError('Idle controller did not exit; no forced kill issued')
            time.sleep(0.25)
        with (root/'dline/gpu_slots.lock').open('a+') as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX)
            pause_file=root/'dline/gpu_reservation_pauses.json'
            atomic(pause_file,[x for x in read(pause_file) if not own_pause(x)])
        atomic(state,{'phase':'launching_replacement','time':time.time(),'old_controller':a.controller,
                     'new_controller':os.getpid(),'command':command,'environment_preserved':True})
        os.chdir(cwd)
        with (work/'controller_parallel_eval.log').open('x') as output:
            output.flush();os.dup2(output.fileno(),1);os.dup2(output.fileno(),2)
            os.execve(command[0],command,environment)


if __name__=='__main__':main()
