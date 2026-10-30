"""Collect one future batch with the exact command of the live reviewed run.

The ascending collector reuses its normal completed-batch marker. Its existing
stage-log guard prevents launching a second worker if this peer is unfinished.
No active controller, policy, seed, reference order, or success filter changes.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from gr00t_budget_chain import Chain, HERE, atomic, read, verify_collection_logs
from resume_walk_after_reboot import progress
from walk_late_takeover_collect import audit_batch
from watch_t20max_v3 import environment


def command_for_batch(command, work, name):
    result=list(command)
    assert Path(result[1]).name=='collect_sonic_adapter.py'
    for flag,value in (('--seed','9'),('--dagger-rescue-window-s','6.5'),
                       ('--dagger-chunk','8'),('--dagger-rollouts','1')):
        assert result[result.index(flag)+1]==value,(flag,result)
    changes={'--output-directory':str(work/'raw'/name),
             '--dagger-motion-ids':str(work/(name+'.ids.txt'))}
    for flag,value in changes.items():result[result.index(flag)+1]=value
    allowed={command.index(flag)+1 for flag in changes}
    assert all(a==b or i in allowed for i,(a,b) in enumerate(zip(command,result)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--batch',type=int,choices=(18,19),required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    assert args.tag=='walk200_v3_histwindow65_20261029'
    chain=Chain(200,tag=args.tag);work=chain.work;name=f'batch_{args.batch:03d}'
    lock=(work/(name+'.preassigned.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    primary=read(work/'workflow.json')
    assert primary['status']=='collecting' and primary['completed_attempts']<100
    proc=Path('/proc',str(primary['controller_pid']))
    assert proc.exists() and 'walk_late_takeover_collect.py' in (proc/'cmdline').read_text()
    assert args.batch>=int(primary['active_batch'].split('_')[1])+10
    assert shutil.disk_usage(work).free>=130_000_000_000
    assert not (work/(name+'.complete.json')).exists()
    assert not (work/(name+'.log')).exists(),'Existing attempt must never be repeated'
    (work/'raw'/name).mkdir(parents=True,exist_ok=True)
    selected=read(work/'inputs/recipe_source.json')['motion_ids'][args.batch*10:(args.batch+1)*10]
    assert selected==list(range(args.batch*10,(args.batch+1)*10))
    (work/(name+'.ids.txt')).write_text(' '.join(map(str,selected))+'\n')
    command=command_for_batch(read(work/'batch_000.process.json')['command'],work,name)
    chain.training_cwd=HERE.parent/'Training'
    chain.env=environment(0)
    chain.env.update(PYTHONPATH=str(chain.sim),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     OPENBLAS_NUM_THREADS='1',PXR_WORK_THREAD_LIMIT='4',TOKENIZERS_PARALLELISM='false')
    state={'status':'starting','pid':os.getpid(),'batch':name,'motion_ids':selected,
           'gpu':args.gpu,'started':time.time(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
           'command':command,'seed':9,'window_s':6.5,'primary_controller_unchanged':True}
    path=work/(name+'.preassigned.workflow.json');atomic(path,state)
    reserved=False
    try:
        devices=list(map(int,subprocess.check_output([os.sys.executable,str(HERE/'gpu_slots.py'),
            'reserve','--owner',str(os.getpid()),'--count','1','--mb','24000','--gpu',str(args.gpu)],text=True).split()))
        assert devices==[args.gpu],'Reviewed GPU unavailable; retain state for review'
        reserved=True;state['status']='collecting';atomic(path,state)
        chain.execute(command,name,args.gpu,chain.training_cwd)
        text=(work/(name+'.log')).read_text();report=verify_collection_logs([text],selected)
        observed=progress([text],selected);assert not observed['remaining_ids']
        out=work/'raw'/name;episodes=audit_batch(out,selected,6.5)
        assert sorted(r['motion_id'] for r in episodes)==observed['written_ids']
        report.update(batch=name,episodes=episodes,valid_skip_ids=observed['valid_skip_ids'],
                      raw_directory=str(out),completed_at=time.time())
        atomic(work/(name+'.complete.json'),report)
        state.update(status='complete',completed_at=time.time(),report=report);atomic(path,state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),failed_at=time.time());atomic(path,state);raise
    finally:
        if reserved:chain.release(args.gpu)


if __name__=='__main__':main()
