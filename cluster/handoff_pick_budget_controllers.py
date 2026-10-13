"""Replace base-stage supervisors after their live trainers finish naturally.

The old supervisor alone is stopped while its child trainer runs. No training
worker receives a signal. Only after final checkpoint and training completion
are verified is that stopped supervisor retired and the updated chain launched.
"""
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

from pick_budget_chain import Chain, ROOT, HERE, atomic, read, checkpoint_ok, steps_for

STOPPED={}


def proc(pid):
    p=Path('/proc')/str(pid)
    try:
        args=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        return {'args':args,'state':fields[0],'ppid':int(fields[1]),'start':fields[19]}
    except FileNotFoundError:
        return None


def main():
    pending=[]
    for budget in [10,50]:
        c=Chain(budget);run=c.tag+'_r0'
        if (c.work/f'{run}.trained.json').exists():
            print(json.dumps({'budget':budget,'state':'already_past_base_boundary','action':'left_original_controller_running'}),flush=True)
            continue
        parent=int((c.work/'controller.lock').read_text())
        parent_state=proc(parent)
        assert parent_state and 'pick_budget_chain.py' in parent_state['args'] and f'--budget {budget}' in parent_state['args']
        record=read(c.work/f'ft_{run}.process.json');trainer=record['pid'];state=proc(trainer)
        assert state and state['ppid']==parent and run in state['args'] and state['state'] not in ['Z','X']
        assert not (c.work/f'{run}.trained.json').exists()
        assert len(list(c.work.glob('*.process.json')))==1,'Only base-stage controllers can be handed off'
        os.kill(parent,signal.SIGSTOP)
        STOPPED[parent]=parent_state['start']
        item={'budget':budget,'parent':parent,'parent_start':parent_state['start'],
              'trainer':trainer,'trainer_start':state['start'],'state':'waiting_for_trainer',
              'time':time.time()}
        atomic(c.work/'controller_handoff.json',item);pending.append(item)
        print(json.dumps(item),flush=True)
    while pending:
        for item in pending[:]:
            c=Chain(item['budget']);run=c.tag+'_r0';state=proc(item['trainer'])
            if state and state['start']!=item['trainer_start']:raise RuntimeError('Trainer PID reused')
            if state and state['state'] not in ['Z','X']:
                atomic(c.work/'controller_handoff_heartbeat.json',dict(item,observed=time.time(),trainer_state=state['state']))
                continue
            data=ROOT/f'datasets/pick{item["budget"]}_base5p7/lerobot/ds'
            info=read(data/'meta/info.json');steps=steps_for(info['total_frames'])
            ck=ROOT/'checkpoints'/run/f'checkpoint-{steps}'
            log=(c.work/f'ft_{run}.log').read_text(errors='replace')
            assert checkpoint_ok(ck) and read(ck/'trainer_state.json')['global_step']==steps
            assert re.search(r"['\"]train_runtime['\"]\s*:",log),'No final training metrics'
            assert 'Traceback (most recent call last)' not in log,'Training log contains an exception'
            parent_state=proc(item['parent'])
            assert parent_state and parent_state['start']==item['parent_start'] and parent_state['state']=='T'
            # Only the already-stopped supervisor is retired; its trainer has exited.
            os.kill(item['parent'],signal.SIGKILL)
            STOPPED.pop(item['parent'],None)
            for _ in range(100):
                if not proc(item['parent']):break
                time.sleep(.1)
            assert not proc(item['parent'])
            record=read(c.work/f'ft_{run}.process.json')
            subprocess.run([os.sys.executable,str(HERE/'gpu_slots.py'),'release','--owner',str(item['parent']),
                            '--gpu',str(record['gpu'])],check=True)
            atomic(c.work/f'{run}.trained.json',{'run':run,'checkpoint':str(ck),'dataset':str(data),
                   'frames':info['total_frames'],'episodes':info['total_episodes'],'steps':steps,
                   'effective_passes':steps*16/info['total_frames'],'batch_size':16,'adopted_completed_base':True})
            c.log(f'EXIT ft_{run} code=0 verified by controller handoff')
            name=c.tag+'_v2_kevin'
            assert subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode
            command='export XLA_PYTHON_CLIENT_PREALLOCATE=false; source ~/miniconda3/etc/profile.d/conda.sh; conda activate dreamcontrol_51; exec python -u '+shlex.quote(str(HERE/'pick_budget_chain.py'))+' --budget '+str(item['budget'])+' > '+shlex.quote(str(c.work/'controller_v2.log'))+' 2>&1'
            subprocess.run(['tmux','new-session','-d','-s',name,command],check=True)
            item.update(state='handed_off',completed=time.time(),session=name)
            atomic(c.work/'controller_handoff.json',item);print(json.dumps(item),flush=True);pending.remove(item)
        if pending:time.sleep(15)


if __name__=='__main__':
    try:
        main()
    except BaseException:
        # If handoff validation fails, let the original controllers retain control.
        for pid,start in STOPPED.items():
            state=proc(pid)
            if state and state['start']==start and state['state']=='T':
                os.kill(pid,signal.SIGCONT)
        raise
