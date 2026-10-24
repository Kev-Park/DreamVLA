"""Collect the reviewed narrow-window walking ablation in bounded batches.

Runs inside the existing simulation environment and tmux. No automatic retries,
job termination, or policy changes. Remote relay is a separate operator workflow.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from gr00t_budget_chain import Chain, ROOT, HERE, atomic, read, verify_collection_logs
from watch_t20max_v3 import environment
from resume_walk_after_reboot import progress


def audit_batch(directory, expected, window_s):
    import h5py
    import numpy as np
    sys.path.insert(0, str(HERE.parent/'Training/scripts/reinforcement_learning/rsl_rl'))
    from vla_sonic.grasp_success import score
    box = read(directory.parent.parent/'inputs/box.json')
    episodes = []
    for path in sorted(directory.rglob('*.hdf5')):
        with h5py.File(path, 'r', locking=False) as f:
            g=f['data/demo_0'];md=json.loads(f.attrs['metadata_json']);rescue=md['rescue']
            n=len(g['actions']);motion=int(md['motion_id'])
            assert motion in expected and g['obs/motion_token'].shape == (n,64)
            assert g['obs/ego_view_image'].shape[0] == n
            assert np.isfinite(g['actions'][()]).all()
            result=score(g['obs/object_pos'][()],g['obs/object_quat'][()],
                g['obs/robot0_root_pos_w'][()],g['obs/robot0_root_quat_w'][()],
                g['teleop/right_wrist'][()],box_lo=np.array(box['lo']),box_hi=np.array(box['hi']))
            if rescue['pass']=='B':
                assert rescue['window_s']==window_s
                lo,hi=rescue['takeover_bounds'];k=rescue['takeover_at']
                assert lo<=k<hi and hi==rescue['grab_idx']
                assert lo==max(0,hi-round(window_s*50))
            grab=int(md.get('grab_idx',-1))
            if grab<0 and rescue['pass']=='B':grab=int(rescue['grab_idx'])
            student=np.ones(n,dtype=bool) if rescue['pass']=='A' else np.arange(n)<rescue['takeover_at']
            assert int(student.sum())==md['dagger']['vla_steps']
            near=np.abs(np.arange(n)-grab)<=25 if grab>=0 else np.zeros(n,dtype=bool)
            episodes.append({'path':str(path),'motion_id':motion,'frames':n,'bytes':path.stat().st_size,
                'four_rule_pass':bool(result['success']),'rescue':rescue,
                'reference_grab_step':grab,'student_frames':int(student.sum()),
                'near_grab_frames':int(near.sum()),'near_grab_student_frames':int((near&student).sum())})
    assert len({r['motion_id'] for r in episodes})==len(episodes)
    return episodes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--window-s',type=float,default=1.0)
    parser.add_argument('--batch-size',type=int,default=10)
    parser.add_argument('--attempts',type=int,default=200)
    parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    assert args.window_s>0 and 1<=args.batch_size<=10 and args.attempts==200
    chain=Chain(200,tag=args.tag)
    work=chain.work;inputs=work/'inputs';config=read(inputs/'config.json')
    lock=(work/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sim=HERE.parent/'Training/scripts/reinforcement_learning/rsl_rl'
    chain.sim=sim;chain.training_cwd=HERE.parent/'Training'
    chain.env=environment(0)
    chain.env.update(PYTHONPATH=str(sim),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',PXR_WORK_THREAD_LIMIT='4',TOKENIZERS_PARALLELISM='false')
    ids=read(inputs/'recipe_source.json')['motion_ids']
    assert ids==list(range(200)) and len(set(ids))==args.attempts
    checkpoint=ROOT/config['student_relative'];expert=ROOT/config['expert_relative']
    assert (checkpoint/'model.safetensors.index.json').is_file() and expert.is_file()
    state={'status':'collecting','controller_pid':os.getpid(),'started':time.time(),
        'tag':args.tag,'window_s':args.window_s,'expected_attempts':200,'completed_attempts':0,
        'student':str(checkpoint),'expert':str(expert),'seed':9,'batch_size':args.batch_size,
        'collection_decoder':'unchanged lattice snap','evaluation_decoder':'previous full FSQ',
        'cpu_affinity':sorted(os.sched_getaffinity(0)),'batches':[],
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()}
    atomic(work/'workflow.json',state)
    try:
        for start in range(0,len(ids),args.batch_size):
            batch=ids[start:start+args.batch_size];name=f'batch_{start//args.batch_size:03d}'
            out=work/'raw'/name;done=work/(name+'.complete.json')
            if done.exists():
                report=read(done);assert report['motion_ids']==batch
                state['batches'].append(report);state['completed_attempts']+=len(batch)
                continue
            # Preserve30GB plus a conservative5GB per1200-frame recording.
            # Nothing is deleted automatically by this controller.
            required=30_000_000_000+len(batch)*5_000_000_000
            while shutil.disk_usage(ROOT).free<required:
                state.update(status='waiting_for_verified_relay',heartbeat=time.time(),
                    free_bytes=shutil.disk_usage(ROOT).free,required_free_bytes=required)
                atomic(work/'workflow.json',state);time.sleep(30)
            out.mkdir(parents=True,exist_ok=True)
            idsfile=work/(name+'.ids.txt');idsfile.write_text(' '.join(map(str,batch))+'\n')
            reserved=subprocess.check_output([sys.executable,str(HERE/'gpu_slots.py'),'reserve',
                '--owner',str(os.getpid()),'--count','1','--mb','24000','--gpu',str(args.gpu)],text=True)
            gpus=list(map(int,reserved.split()))
            assert gpus==[args.gpu],'Reviewed GPU is unavailable; operator review required'
            gpu=gpus[0]
            try:
                state.update(status='collecting',active_batch=name,gpu=gpu,heartbeat=time.time())
                atomic(work/'workflow.json',state)
                cmd=[sys.executable,sim/'collect_sonic_adapter.py','--headless',
                    '--task','Isaac-Motion-Tracking-Pick-Cam-HOI-v0','--waist-dof','29',
                    '--sonic-pt',chain.pt,'--encoder-mode','g1','--residual-transform','additive_free',
                    '--residual-scale','0.1','--checkpoint-path',expert,'--ref-motions-path','StitchMAX200v3',
                    '--rollout-length','1200','--num-samples','100000','--seed','9',
                    '--output-directory',out,'--dagger-vla-checkpoint',checkpoint,
                    '--dagger-demo-root',inputs/'calibration','--dagger-motion-ids',idsfile,
                    '--dagger-chunk','8','--dagger-rescue','--dagger-rescue-box','file:'+str(inputs/'box.json'),
                    '--dagger-rescue-window-s',str(args.window_s),'--dagger-rescue-k-frac','1.0',
                    '--dagger-rescue-blend','0','--dagger-expert-base','reference','--dagger-rollouts','1',
                    '--kit_args=--/plugins/carb.tasking.plugin/threadCount=4']
                chain.execute(cmd,name,gpu,chain.training_cwd)
            finally:
                chain.release(gpu)
            text=(work/(name+'.log')).read_text()
            report=verify_collection_logs([text],batch)
            completed=progress([text],batch)
            assert completed['remaining_ids']==[]
            episodes=audit_batch(out,batch,args.window_s)
            assert sorted(r['motion_id'] for r in episodes)==completed['written_ids']
            report.update(batch=name,episodes=episodes,valid_skip_ids=completed['valid_skip_ids'],
                raw_directory=str(out),completed_at=time.time())
            atomic(done,report);state['batches'].append(report)
            state['completed_attempts']+=len(batch);state['heartbeat']=time.time()
            atomic(work/'workflow.json',state)
        state.update(status='collection_complete',completed_at=time.time())
        atomic(work/'workflow.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),failed_at=time.time())
        atomic(work/'workflow.json',state)
        raise


if __name__=='__main__':
    main()
