"""Reusable fixed-reference GR00T base plus three DAgger rounds.

Run inside dreamcontrol_51 on Bluesclues. Uses existing simulation/collection
code and whole-dataset normalization, with independent pretrained fine-tunes.
No implicit retries of failed collection: each reference has one student attempt.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

if os.name != 'nt':
    import fcntl

HERE = Path(__file__).resolve().parent
ROOT = Path.home() / 'kevin'
REFS = 'Holosoma_Pick_29_latband60max'
EVAL_REFS = REFS + '_eval'
PREFIX = '20261013'


def steps_for(frames):
    return math.ceil(frames * 57 / 160)


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def checkpoint_ok(path):
    try:
        names = set(read(path / 'model.safetensors.index.json')['weight_map'].values())
        return bool(names) and all((path / n).stat().st_size > 8 for n in names)
    except (OSError, ValueError, KeyError):
        return False


def shared_memory_pid_collision(text, pid, shm=Path('/dev/shm')):
    """Recognize Kit failing before startup on another user's recycled PID."""
    name = f'carb-RStringInternals-{pid}'
    lines = text.strip().splitlines()
    if len(lines) != 2 or name not in lines[0]:
        return False
    if 'Failed to create shared memory named '+name not in lines[0]:
        return False
    if lines[1] != 'terminate called without an active exception':
        return False
    try:
        return (shm/name).stat().st_uid != os.getuid()
    except OSError:
        return False


def verify_collection_logs(texts, expected):
    attempted, completed = [], []
    for text in texts:
        attempted += list(map(int, re.findall(r'\[INFO\] motion (\d+)_r0/\d+', text)))
        completed += list(map(int, re.findall(r'\[rescue\] motion (\d+): pass A 4-rule', text)))
        summaries = re.findall(r'Collection finished\..*errored=(\d+)', text)
        if summaries != ['0']:
            raise ValueError('Collection worker errored or did not finish')
    if sorted(attempted) != sorted(expected) or sorted(completed) != sorted(expected):
        raise ValueError(f'Student attempt coverage differs: attempts={attempted}, completed={completed}, expected={expected}')
    return {'student_attempts':len(attempted), 'motion_ids':sorted(attempted), 'errors':0}


class Chain:
    def __init__(self, budget, tag=None):
        self.budget = budget
        self.tag = tag or f'pick{budget}_dagger57_{PREFIX}'
        self.work = ROOT / 'dline' / self.tag
        self.work.mkdir(parents=True, exist_ok=True)
        self.sim = HERE.parent / 'Training/scripts/reinforcement_learning/rsl_rl'
        self.pt = ROOT / 'sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000'
        self.expert = ROOT / 'residuals/af60v8c_model_8998.pt'
        self.rollout_steps = 500
        self.training_cwd = ROOT/'DreamVLA/Training'
        self.score_script = HERE/'pick_budget_score.py'
        self.eval_no_ee_term = True
        self.env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='2',
                        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='2',
                        TOKENIZERS_PARALLELISM='false', HS_REWORK='1', HS_REWORK_OBJ_PC='1',
                        HS_REWORK_OBJ_GATE='1', HS_REWORK_CONTACT_LEAD='2',
                        PYTHONPATH=str(ROOT / 'DreamVLA/Training/scripts/reinforcement_learning/rsl_rl'))
        # All agents' workers inherit the same host-level CPU share.
        if (ROOT/'dline/cpu_share.json').is_file():
            os.sched_setaffinity(0, read(ROOT/'dline/cpu_share.json')['shared_mask'])

    def log(self, message):
        line = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ' + message
        print(line, flush=True)
        with (self.work/'status').open('a') as f:
            f.write(line + '\n')

    def execute(self, args, label, gpu=None, cwd=None):
        env = dict(self.env)
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        path = self.work/(label+'.log')
        # Failed stages require explicit diagnosis; never hide or overwrite their logs.
        if path.exists():
            raise RuntimeError(f'Existing stage log without verified completion: {path}')
        self.log(f'START {label} gpu={gpu}')
        for attempt in range(3):
            with path.open('w') as output:
                p = subprocess.Popen(list(map(str,args)), cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
                atomic(self.work/(label+'.process.json'), {'pid':p.pid, 'host':os.uname().nodename,
                       'started':time.time(), 'command':list(map(str,args)), 'gpu':gpu})
                while p.poll() is None:
                    atomic(self.work/(label+'.heartbeat.json'), {'time':time.time(), 'pid':p.pid,
                           'log_bytes':path.stat().st_size, 'log_mtime':path.stat().st_mtime,
                           'output_stalled_30min':time.time()-path.stat().st_mtime>1800})
                    try:
                        p.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
            if p.returncode != -6 or attempt == 2 or not shared_memory_pid_collision(path.read_text(), p.pid):
                break
            # Kit has not initialized or attempted an episode. Preserve the failed
            # launch, then obtain a new PID; never touch another user's shm files.
            archive = self.work/'startup_collisions'/f'{label}_{p.pid}'
            archive.mkdir(parents=True, exist_ok=False)
            for suffix in ('.log', '.process.json', '.heartbeat.json'):
                original = self.work/(label+suffix)
                if original.exists():
                    original.rename(archive/original.name)
            self.log(f'RETRY startup PID collision {label} old_pid={p.pid} archive={archive}')
        self.log(f'EXIT {label} code={p.returncode}')
        if p.returncode:
            raise RuntimeError(f'{label} exited {p.returncode}; see {path}')

    def slots(self, count, mb):
        while True:
            out = subprocess.check_output([sys.executable,str(HERE/'gpu_slots.py'),'reserve',
                   '--owner',str(os.getpid()),'--count',str(count),'--mb',str(mb)],text=True)
            ids = list(map(int,out.split()))
            if len(ids)==count:
                return ids
            for gpu in ids:
                self.release(gpu)
            self.log(f'WAIT GPU count={count} memory={mb}MiB')
            time.sleep(30)

    def release(self, gpu):
        subprocess.run([sys.executable,str(HERE/'gpu_slots.py'),'release','--owner',str(os.getpid()),
                        '--gpu',str(gpu)],check=True)

    def external(self, phase, run, payload):
        """An explicit routing decision delegates one stage, never a live worker.

        The bridge writes its job handle before launch and a terminal response only
        after transferring and verifying outputs. No timeout implies cancellation.
        """
        policy=self.work/'routing.json'
        host=read(policy).get(phase,{}).get(run) if policy.exists() else None
        if host is None or host=='bluesclues':return False
        assert host in ['biped','adroit'],host
        directory=self.work/'external';directory.mkdir(exist_ok=True)
        request=directory/f'{phase}_{run}.request.json'
        reply=directory/f'{phase}_{run}.reply.json'
        spec=dict(host=host,phase=phase,run=run,budget=self.budget,**payload)
        if request.exists():assert read(request)==spec
        else:atomic(request,spec)
        self.log(f'WAIT external {phase} {run} host={host}')
        while not reply.exists():
            atomic(directory/f'{phase}_{run}.waiting.json',{'time':time.time(),'controller_pid':os.getpid()})
            time.sleep(30)
        response=read(reply)
        assert response['state']=='completed',response
        assert response['host']==host and response['run']==run
        return True

    def prepare(self):
        manifest = read(ROOT/f'dline/pick{self.budget}/subset_manifest.json')
        sources = [Path(p) for p in manifest['episodes']]
        ids = [int(re.search(r'__motion_(\d+)\.hdf5$',p.name)[1]) for p in sources]
        assert len(ids)==len(set(ids))==self.budget and all(p.is_file() for p in sources)
        base = ROOT/f'datasets/pick{self.budget}_base5p7/lerobot/ds'
        info = read(base/'meta/info.json')
        assert info['total_episodes']==self.budget and not info.get('discarded_episode_indices')
        ordered = sorted((base.parent.parent/'hdf5').glob('*.hdf5'))
        assert [p.resolve() for p in ordered]==[p.resolve() for p in sources]
        assert len(list((ROOT/'ref_motions'/REFS).glob('pick_*.pkl')))==400
        assert len(list((ROOT/'ref_motions'/EVAL_REFS).glob('pick_*.pkl')))==100
        assert self.expert.is_file()
        original_box = ROOT/'dline/af60v8f57/box.json'
        (self.work/'box.json').write_text(original_box.read_text())
        self.ids = sorted(ids)
        self.sources = sources
        demo = self.work/'demo_root'; demo.mkdir(exist_ok=True)
        for p in sources:
            dest=demo/p.name
            if not dest.exists():dest.symlink_to(p)
            assert dest.resolve()==p.resolve()
        (self.work/'motion_ids.txt').write_text(' '.join(map(str,self.ids))+'\n')
        recipe = {'budget':self.budget, 'base_dataset':str(base), 'base_episodes':len(sources),
                  'base_frames':info['total_frames'], 'motion_ids':self.ids, 'source_episodes':list(map(str,sources)),
                  'rounds':3, 'student_attempts_per_round':self.budget, 'batch_size':16,
                  'passes':5.7, 'steps_rule':'ceil(5.7 * cumulative_frames / 16)',
                  'initialization':'nvidia/GR00T-N1.7-3B independently each round',
                  'expert':str(self.expert), 'eval_pool':EVAL_REFS, 'eval_episodes':100,
                  'collection':'existing rescue: student pass A; failed pass gets paired expert takeover; retain four-rule passes',
                  'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()}
        if (self.work/'recipe.json').exists():
            old=read(self.work/'recipe.json')
            assert all(old[k]==v for k,v in recipe.items() if k!='code_commit')
        atomic(self.work/'recipe.json',recipe)
        return base

    def finetune(self, dataset, round_id):
        info=read(dataset/'meta/info.json');steps=steps_for(info['total_frames'])
        run=f'{self.tag}_r{round_id}'
        ck=ROOT/'checkpoints'/run/f'checkpoint-{steps}'
        if (self.work/f'{run}.trained.json').exists():
            assert checkpoint_ok(ck)
            return run,ck
        if self.external('finetune',run,{'dataset':str(dataset),'checkpoint':str(ck),'steps':steps,
                         'frames':info['total_frames'],'episodes':info['total_episodes'],
                         'global_batch_size':16,'passes':5.7,'learning_rate':0.0001}):
            assert checkpoint_ok(ck) and read(ck/'trainer_state.json')['global_step']==steps
            atomic(self.work/f'{run}.trained.json',{'run':run,'checkpoint':str(ck),'dataset':str(dataset),
                   'frames':info['total_frames'],'episodes':info['total_episodes'],'steps':steps,
                   'effective_passes':steps*16/info['total_frames'],'batch_size':16,'external':True})
            return run,ck
        # The measured single-GPU trainer uses ~39 GB; reserve 40 GB plus the
        # allocator's 2 GB safety margin, accounting for existing GPU clients.
        gpu=self.slots(1,40000)[0]
        try:
            args=[ROOT/'Isaac-GR00T/.venv/bin/python','gr00t/experiment/launch_finetune.py',
                  '--base-model-path','nvidia/GR00T-N1.7-3B','--dataset-path',dataset,
                  '--embodiment-tag','unitree_g1_sonic','--num-gpus','1','--global-batch-size','16',
                  '--gradient-accumulation-steps','1','--learning-rate','0.0001',
                  '--dataloader-num-workers','4','--output-dir',ROOT/'checkpoints',
                  '--experiment-name',run,'--max-steps',str(steps),'--save-steps',str(steps),
                  '--save-total-limit','1']
            self.execute(args,'ft_'+run,gpu,cwd=ROOT/'Isaac-GR00T')
            assert checkpoint_ok(ck)
            state=read(ck/'trainer_state.json');assert state['global_step']==steps
            atomic(self.work/f'{run}.trained.json',{'run':run,'checkpoint':str(ck),'dataset':str(dataset),
                   'frames':info['total_frames'],'episodes':info['total_episodes'],'steps':steps,
                   'effective_passes':steps*16/info['total_frames'],'batch_size':16})
        finally:
            self.release(gpu)
        return run,ck

    def collect(self, round_id, checkpoint):
        out=ROOT/'datasets'/f'{self.tag}_collection_r{round_id}'
        done=self.work/f'collection_r{round_id}.json'
        if done.exists():
            assert read(done)['motion_ids']==self.ids
            return out
        out.mkdir(exist_ok=True)
        count=getattr(self, 'collection_workers', 2 if self.budget==10 else 4)
        gpus=self.slots(count,17000)
        tasks=[]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                for k,gpu in enumerate(gpus):
                    ids=self.ids[k::count]
                    idsfile=self.work/f'collection_r{round_id}_ids{k}.txt'
                    idsfile.write_text(' '.join(map(str,ids))+'\n')
                    label=f'collect_r{round_id}_shard{k}'
                    args=[sys.executable,self.sim/'collect_sonic_adapter.py','--headless',
                          '--task','Isaac-Motion-Tracking-Pick-Cam-HOI-v0','--waist-dof','29',
                          '--sonic-pt',self.pt,'--encoder-mode','g1','--residual-transform','additive_free',
                          '--residual-scale','0.1','--checkpoint-path',self.expert,'--ref-motions-path',REFS,
                          '--rollout-length',str(self.rollout_steps),'--num-samples','100000','--seed',str(6+round_id),
                          '--output-directory',out,'--dagger-vla-checkpoint',checkpoint,
                          '--dagger-demo-root',self.work/'demo_root','--dagger-motion-ids',idsfile,
                          '--dagger-chunk','8','--dagger-rescue','--dagger-rescue-box','file:'+str(self.work/'box.json'),
                          '--dagger-rescue-k-frac','1.0','--dagger-rescue-blend','0',
                          '--dagger-expert-base','reference','--dagger-rollouts','1']
                    tasks.append(pool.submit(self.execute,args,label,gpu,self.training_cwd))
                for future in tasks:future.result()
            result=verify_collection_logs([(self.work/f'collect_r{round_id}_shard{k}.log').read_text()
                                           for k in range(count)],self.ids)
            atomic(done,dict(result,checkpoint=str(checkpoint),output=str(out)))
        finally:
            for gpu in gpus:self.release(gpu)
        return out

    def dataset(self, round_id, raw, prior):
        import h5py
        import numpy as np
        sys.path.insert(0,str(self.sim))
        from vla_sonic.grasp_success import score
        box=read(self.work/'box.json');kept=[]
        for path in sorted(raw.rglob('*.hdf5')):
            assert not path.name.startswith('_passA')
            with h5py.File(path,'r',locking=False) as f:
                g=f['data/demo_0']
                ok=score(g['obs/object_pos'][()],g['obs/object_quat'][()],g['obs/robot0_root_pos_w'][()],
                         g['obs/robot0_root_quat_w'][()],g['teleop/right_wrist'][()],
                         box_lo=np.array(box['lo']),box_hi=np.array(box['hi']))['success']
            if ok:kept.append(path)
        assert len(kept)<=self.budget
        self.sources.extend(kept)
        dst=ROOT/'datasets'/f'{self.tag}_r{round_id}';data=dst/'lerobot/ds'
        if (dst/'built.json').exists():
            assert read(data/'meta/info.json')['total_episodes']==len(self.sources)
            return data
        inputs=dst/'hdf5';inputs.mkdir(parents=True,exist_ok=True)
        for i,path in enumerate(self.sources):
            link=inputs/f'{i:05d}__{path.name}'
            if not link.exists():link.symlink_to(path)
            assert link.resolve()==path.resolve()
        args=[sys.executable,HERE/'convert_incremental.py','--hdf5-root',inputs,
              '--prefix-hdf5-root',prior.parent.parent/'hdf5','--prefix-dataset',prior,
              '--out',data,'--task-prompt','pick up the mustard bottle','--fps','50','--shards','4']
        self.execute(args,f'convert_r{round_id}')
        assert read(data/'meta/info.json')['total_episodes']==len(self.sources)
        atomic(dst/'built.json',{'episodes':len(self.sources),'new_passes':len(kept),'new_attempts':self.budget})
        return data

    def evaluate(self, run, checkpoint):
        done=self.work/f'{run}.eval.json'
        if done.exists():return
        out=ROOT/'eval_videos'/run;out.mkdir(exist_ok=True)
        if self.external('evaluate',run,{'checkpoint':str(checkpoint),'output':str(out),
                         'pool':EVAL_REFS,'episodes':100,'box':read(self.work/'box.json')}):
            report=read(out/'results.json')
            assert report['pool']==EVAL_REFS and report['episodes']==100
            assert sorted(r['motion_id'] for r in report['per_episode'])==list(range(100))
            atomic(done,report)
            return
        gpus=self.slots(4,17000)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                tasks=[]
                for k,gpu in enumerate(gpus):
                    ids=self.work/f'eval_ids{k}.txt';ids.write_text(' '.join(map(str,range(k,100,4)))+'\n')
                    args=['env',f'HS_EVAL_NO_EE_TERM={int(self.eval_no_ee_term)}',sys.executable,self.sim/'eval_vla_sonic.py',
                          '--headless','--sonic-pt',self.pt,'--ref-motions-path',EVAL_REFS,
                          '--vla-checkpoint',checkpoint,'--waist-dof','29','--motions-from',ids,
                          '--num-episodes','25','--seed',str(100+k),'--traj-dump',out/f'b{k}',
                          '--max-steps-per-episode',str(self.rollout_steps)]
                    tasks.append(pool.submit(self.execute,args,f'eval_{run}_shard{k}',gpu,self.training_cwd))
                for f in tasks:f.result()
            self.execute([sys.executable,self.score_script,out,self.work/'box.json'],f'score_{run}')
            report=read(out/'results.json');assert report['episodes']==100
            atomic(done,report)
        finally:
            for gpu in gpus:self.release(gpu)

    def run(self):
        lock=(self.work/'controller.lock').open('w')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.write(str(os.getpid()));lock.flush()
        data=self.prepare()
        evaluations=[]
        # Evaluation errors propagate at the end; collection/training can proceed independently.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            for round_id in range(4):
                if round_id:
                    raw=self.collect(round_id,checkpoint)
                    data=self.dataset(round_id,raw,data)
                    # Collection/conversion can overlap the preceding evaluation,
                    # but its verified score must exist before the next fine-tune.
                    evaluations[-1].result()
                run,checkpoint=self.finetune(data,round_id)
                evaluations.append(pool.submit(self.evaluate,run,checkpoint))
            for f in evaluations:f.result()
        atomic(self.work/'done.json',{'time':time.time(),'rounds':3,'evals':4,'budget':self.budget})
        self.log('COMPLETE: base + 3 DAgger fine-tunes, all four held-out evaluations verified')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--budget',type=int,choices=[10,50],required=True)
    a=p.parse_args();chain=Chain(a.budget)
    try:chain.run()
    except BaseException as e:
        atomic(chain.work/'abort.json',{'time':time.time(),'error':repr(e),'pid':os.getpid()})
        raise


if __name__=='__main__':main()
