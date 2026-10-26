"""Convert, assemble and evaluate the approved walking takeover-window ablation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import gr00t_budget_chain as base
from watch_t20max_v3 import environment

ROOT,HERE=base.ROOT,base.HERE
base.REFS='StitchMAX200v3'
base.EVAL_REFS='StitchMAX100v3_eval'


def convert(tag,batch):
    import h5py
    import numpy as np
    sys.path.insert(0,str(HERE.parent/'Training/scripts/reinforcement_learning/rsl_rl'))
    from vla_sonic.grasp_success import score
    work=ROOT/'dline'/tag;report=base.read(work/(batch+'.complete.json'))
    done=work/(batch+'.converted.json')
    if done.exists():return base.read(done)
    inputs=work/'conversion'/batch/'hdf5';inputs.mkdir(parents=True,exist_ok=True)
    box=base.read(work/'inputs/box.json');raw=work/'raw'/batch;kept=[]
    expected={Path(r['path']).name:r for r in report['episodes']}
    files=sorted(raw.rglob('*.hdf5'));assert len(files)==len(expected)
    for p in files:
        original=expected[p.name]
        with h5py.File(p,'r',locking=False) as f:
            g=f['data/demo_0'];md=json.loads(f.attrs['metadata_json'])
            assert md['motion_id']==original['motion_id'] and len(g['actions'])==original['frames']
            verdict=score(g['obs/object_pos'][()],g['obs/object_quat'][()],
                g['obs/robot0_root_pos_w'][()],g['obs/robot0_root_quat_w'][()],
                g['teleop/right_wrist'][()],box_lo=np.array(box['lo']),box_hi=np.array(box['hi']))
            assert bool(verdict['success'])==original['four_rule_pass']
            if verdict['success']:kept.append(p)
    dataset=work/'converted'/batch/'ds'
    if kept:
        for i,p in enumerate(kept):
            link=inputs/f'{i:05d}__{p.name}'
            if not link.exists():link.symlink_to(p.resolve(strict=True))
            assert link.resolve(strict=True)==p.resolve(strict=True)
        subprocess.run(['bash',str(HERE/'convert_sharded.sh'),'--hdf5-root',str(inputs),
            '--recursive','--output-path',str(dataset.parent),'--dataset-name','ds',
            '--task-prompt','pick up the mustard bottle','--fps','50','--shards','4'],check=True)
        info=base.read(dataset/'meta/info.json');assert info['total_episodes']==len(kept)
    else:info={'total_episodes':0,'total_frames':0}
    result={'batch':batch,'attempts':len(report['motion_ids']),'episodes':len(kept),
        'frames':info['total_frames'],'dataset':str(dataset) if kept else None,
        'retained_motion_ids':[expected[p.name]['motion_id'] for p in kept],
        'completed_at':time.time(),'filter':'unchanged four-rule success'}
    base.atomic(done,result);return result


def assemble(tag):
    work=ROOT/'dline'/tag;prior=ROOT/'datasets/walk200_v3_dagger57_20261013_r2/lerobot/ds'
    prior_info=base.read(prior/'meta/info.json')
    assert prior_info['total_episodes']==416 and prior_info['total_frames']==260958
    batches=[base.read(work/f'batch_{i:03d}.converted.json') for i in range(20)]
    assert sum(r['attempts'] for r in batches)==200
    assert sum(r['episodes'] for r in batches)>0,'No passing additions; retain records and review before training'
    out=ROOT/'datasets'/(tag+'_r3')/'lerobot/ds';done=work/'dataset.ready.json'
    if done.exists():return base.read(done)
    assert not out.exists(),'Incomplete output needs explicit reconciliation'
    shards=[str(prior)]+[r['dataset'] for r in batches if r['dataset']]
    subprocess.run([sys.executable,str(HERE/'merge_lerobot.py'),
        '--shards',*shards,'--out',str(out)],check=True)
    info=base.read(out/'meta/info.json')
    assert info['total_episodes']==416+sum(r['episodes'] for r in batches)
    assert info['total_frames']==260958+sum(r['frames'] for r in batches)
    assert not any((out/'meta'/n).exists() for n in ('stats.json','relative_stats.json'))
    result={'dataset':str(out),'run':tag+'_r3','frames':info['total_frames'],
        'episodes':info['total_episodes'],'new_attempts':200,
        'new_retained_episodes':sum(r['episodes'] for r in batches),
        'steps':base.steps_for(info['total_frames']),'batch_size':16,'passes':5.7,
        'learning_rate':0.0001,'initialization':'independent nvidia/GR00T-N1.7-3B',
        'normalization':'regenerate globally on the single merged cumulative dataset',
        'completed_at':time.time(),'prefix_dataset':str(prior)}
    base.atomic(done,result);return result


def evaluate(tag,checkpoint,gpu,run_name=None):
    class Evaluation(base.Chain):
        def slots(self,count,mb):
            if os.environ.get('SLURM_JOB_ID'):
                return [0]*count
            assert count==1
            out=subprocess.check_output([sys.executable,str(HERE/'gpu_slots.py'),'reserve',
                '--owner',str(os.getpid()),'--count','1','--mb',str(mb),'--gpu',str(gpu)],text=True)
            ids=list(map(int,out.split()));assert ids==[gpu],'Reviewed GPU unavailable'
            return ids
        def release(self,device):
            if not os.environ.get('SLURM_JOB_ID'):super().release(device)
        def execute(self,args,label,gpu=None,cwd=None):
            if any(Path(str(x)).name=='eval_vla_sonic.py' for x in args):
                args=[*args,'--kit_args=--/plugins/carb.tasking.plugin/threadCount=2']
            return super().execute(args,label,None if os.environ.get('SLURM_JOB_ID') else gpu,cwd)
    chain=Evaluation(200,tag=tag);chain.env=environment(gpu)
    box=base.read(chain.work/'inputs/box.json')
    if (chain.work/'box.json').exists():assert base.read(chain.work/'box.json')==box
    else:base.atomic(chain.work/'box.json',box)
    chain.env.update(PYTHONPATH=str(HERE.parent/'Training/scripts/reinforcement_learning/rsl_rl'),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PXR_WORK_THREAD_LIMIT='4')
    chain.training_cwd=HERE.parent/'Training';chain.rollout_steps=1200
    if os.environ.get('SLURM_JOB_ID'):
        chain.env['CUDA_VISIBLE_DEVICES']=os.environ['CUDA_VISIBLE_DEVICES']
    chain.eval_workers=4 if os.environ.get('SLURM_JOB_ID') else 1
    chain.score_script=HERE/'walk_gr00t_score.py'
    run_name=run_name or tag+'_r3'
    chain.evaluate(run_name,Path(checkpoint))
    return base.read(chain.work/(run_name+'.eval.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['convert','assemble','evaluate'])
    p.add_argument('--tag',required=True);p.add_argument('--batch')
    p.add_argument('--checkpoint');p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args();work=ROOT/'dline'/a.tag;work.mkdir(parents=True,exist_ok=True)
    key=a.stage+('_'+a.batch if a.batch else '')
    lock=(work/(key+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'status':'running','stage':a.stage,'batch':a.batch,'pid':os.getpid(),'started':time.time()}
    base.atomic(work/(key+'.workflow.json'),state)
    try:
        if a.stage=='convert':assert a.batch;result=convert(a.tag,a.batch)
        elif a.stage=='assemble':result=assemble(a.tag)
        else:assert a.checkpoint;result=evaluate(a.tag,a.checkpoint,a.gpu)
        state.update(status='complete',completed=time.time(),result=result)
        base.atomic(work/(key+'.workflow.json'),state);print(json.dumps(state),flush=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),failed=time.time())
        base.atomic(work/(key+'.workflow.json'),state);raise


if __name__=='__main__':main()
