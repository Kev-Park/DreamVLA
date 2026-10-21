"""Original-only 200-reference base plus three fresh 5.7-pass DAgger rounds.

Uses the reviewed pick-budget pipeline. At most two devices are acquired on
demand, through the shared reservation registry and full-account capacity check.
An incomplete original-reference base cannot enter training.
"""
import argparse
import copy
import glob
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import threading
import time

from pick_budget_chain import Chain, ROOT, HERE, atomic, read
from pick_original_bluesclues_queue import capacity, reserve, release as release_reservation


def original_sources(request, collection):
    refs = request['references']
    if len(refs) != 200 or len({r['source_motion_id'] for r in refs}) != 200:
        raise ValueError('Exactly 200 distinct original sources are required')
    if any(r['reference_file'] != f"pick_{r['source_motion_id']}.pkl" for r in refs):
        raise ValueError('Augmented reference in original-only study')
    passing = [r for r in collection['recordings'] if r['passing']]
    names = [r['reference_file'] for r in passing]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate passing recordings need reconciliation')
    added = {r['reference_file']: r['path'] for r in passing}
    expected_new = {r['reference_file'] for r in refs if not r['existing_original_demo']}
    if not set(added) <= expected_new:
        raise ValueError('Unexpected reference in missing-original collection')
    old, new = [], []
    missing = []
    for row in refs:
        if row['existing_original_demo']:
            old.append(row['existing_original_demo'])
        elif row['reference_file'] in added:
            new.append(added[row['reference_file']])
        else:
            missing.append(row['reference_file'])
    if missing:
        raise ValueError('Original-source coverage is incomplete: '+', '.join(missing))
    if len(old) != 173 or len(new) != 27:
        raise ValueError('Unexpected base provenance; review before training')
    return old + new


def completed_retry_request_matches(previous, expected):
    """Replay a completed sweep across an orchestration-only controller update.

    Preserve its original as-run request. Every reference, seed, expert, scoring
    source and environment pin must still match; only this controller may differ.
    This exception never permits launching an incomplete old request.
    """
    if previous == expected:
        return True
    old, new = copy.deepcopy(previous), copy.deepcopy(expected)
    if not isinstance(old.get('runtime_hashes'), dict) or not isinstance(new.get('runtime_hashes'), dict):
        return False
    pins = [r['runtime_hashes'].pop('cluster/pick_original_chain.py', None) for r in [old, new]]
    return all(isinstance(p, str) and len(p) == 64 and set(p) <= set('0123456789abcdef') for p in pins) and old == new


class OriginalChain(Chain):
    def __init__(self, request):
        self.request = request
        assert request['launch_authorized'] is True
        assert request['run'] in ['pick200_original_dagger57_v1','pick200_original_spares57_v2']
        if request['run']=='pick200_original_spares57_v2':
            assert request.get('spare_replacements_authorized') is True
            assert request['training_reference_pool']=='Holosoma_Pick_29_original200_spares_v2'
        self.reference_pool=request.get('training_reference_pool','Holosoma_Pick_29_latband60max')
        self.budget=200
        self.tag=request['run']
        self.work=ROOT/'dline'/self.tag
        self.sim=HERE.parent/'Training/scripts/reinforcement_learning/rsl_rl'
        self.pt=ROOT/'sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000'
        self.expert=ROOT/'residuals/af60v8c_model_8998.pt'
        self.local_eval_lock=threading.Lock()
        self.condition=threading.Condition()
        self.leases={}
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
        self.env={**os.environ,**request['environment'],'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                  'OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false',
                  'PYTHONPATH':str(self.sim)}

    def slots(self,count,mb):
        with self.condition:
            while True:
                candidates=[]
                for gpu,leases in self.leases.items():
                    candidates.extend([gpu]*max(0,(45000-sum(leases))//mb))
                required=1 if count==4 else count
                # Collection may use fewer workers, while preserving one attempt
                # for every selected reference. Eval keeps its two fixed workers.
                if len(candidates)>=required and (count!=4 or len(candidates)>=count or len(self.leases)>=2):
                    chosen=candidates[:count]
                    for gpu in chosen:self.leases[gpu].append(mb)
                    return chosen
                if len(self.leases)<2:
                    gpu=reserve(capacity())
                    if gpu is not None:
                        self.leases[gpu]=[]
                        continue
                if len(candidates)>=required:
                    chosen=candidates[:count]
                    for gpu in chosen:self.leases[gpu].append(mb)
                    return chosen
                self.log(f'WAIT capacity count={count} memory={mb}MiB')
                self.condition.wait(timeout=30)

    def release(self,gpu):
        with self.condition:
            self.leases[gpu].pop()
            if not self.leases[gpu]:
                release_reservation(gpu)
                del self.leases[gpu]
            self.condition.notify_all()

    def verify_eval_pool(self):
        files=[Path(p) for p in glob.glob(str(ROOT/'ref_motions/Holosoma_Pick_29_latband60max_eval/*.pkl'))]
        expected=self.request['eval_reference_manifest']['files']
        assert len(files)==len(expected)==100
        assert [p.name for p in files]==[r['name'] for r in expected], 'Held-out ordering changed'
        for path,row in zip(files,expected):
            assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']

    def evaluate(self,run,checkpoint):
        self.verify_eval_pool()
        return super().evaluate(run,checkpoint)

    def finetune(self,dataset,round_id):
        if not (self.work/f'{self.tag}_r{round_id}.trained.json').exists():
            free=shutil.disk_usage(ROOT).free
            if free<50*1024**3:
                raise RuntimeError(f'Insufficient checkpoint headroom: {free} bytes free; archive/review before continuing')
        return super().finetune(dataset,round_id)

    def collect(self,round_id,checkpoint):
        if not (self.work/f'collection_r{round_id}.json').exists():
            mean_base_bytes=sum(p.stat().st_size for p in self.sources[:200])/200
            required=int(200*mean_base_bytes)+20*1024**3
            free=shutil.disk_usage(ROOT).free
            if free<required:
                raise RuntimeError(f'Insufficient raw-collection headroom: {free} bytes free; need {required}; archive/review first')
        return super().collect(round_id,checkpoint)

    def complete_original_coverage(self):
        """Keep the first passing demo per missing source; retry only failures."""
        final=self.work/'original_coverage.result.json'
        if final.exists():
            result=read(final)
            original_sources(self.request,result)
            return
        first=read(self.work/'collection.result.json')
        selected={r['reference_file'] for r in self.request['references'] if not r['existing_original_demo']}
        accepted={r['reference_file']:dict(r,collection_seed=self.request['seed'])
                  for r in first['recordings'] if r['passing']}
        receipts=[str(self.work/'collection.result.json')]
        stagnant=0
        attempt=0
        while selected-set(accepted):
            assert self.request.get('retry_failed_originals_authorized') is True, 'Missing originals require retry approval'
            attempt+=1
            seed=self.request['seed']+attempt
            missing=selected-set(accepted)
            atomic(self.work/'chain.status.json',dict(time=time.time(),pid=os.getpid(),phase='retrying_missing_originals',
                   missing=sorted(missing),seed=seed,passing_originals=173+len(accepted)))
            retry=copy.deepcopy(self.request)
            retry.update(run=f'{self.tag}_retry{attempt:02d}',seed=seed,expected_attempts=len(missing))
            for row in retry['references']:
                row['requires_original_demo_collection']=row['reference_file'] in missing
            directory=self.work/'original_retries'/f'{attempt:02d}'
            directory.mkdir(parents=True,exist_ok=True)
            request=directory/'collection.request.json'
            receipt=directory/'collection.result.json'
            if request.exists():
                previous=read(request)
                assert previous==retry or (receipt.exists() and completed_retry_request_matches(previous,retry)), 'Retry request changed; reconcile preserved evidence'
                if previous!=retry:
                    self.log(f'REPLAY completed seed {seed}; preserving its as-run orchestration pin')
            else:
                atomic(request,retry)
            if not receipt.exists():
                # Repeated failures may still save large raw recordings. Preserve
                # checkpoint headroom before another authorized sweep begins.
                required=50*1024**3+len(missing)*max(1024**3,max((r['bytes'] for r in first['recordings']),default=0))
                free=shutil.disk_usage(ROOT).free
                if free<required:
                    raise RuntimeError(f'Insufficient retry-collection headroom: {free} bytes free; need {required}; archive/review first')
                gpu=self.slots(1,17000)[0]
                try:
                    self.execute([sys.executable,'-u',HERE/'pick_original_base_collect.py',request],
                                 f'collect_missing_originals_retry{attempt:02d}',gpu,ROOT/'DreamVLA/Training')
                finally:
                    self.release(gpu)
            result=read(receipt)
            assert result['state']=='attempts_completed', 'Incomplete retry receipt'
            new=[dict(r,collection_seed=seed) for r in result['recordings'] if r['passing']]
            assert len({r['reference_file'] for r in new})==len(new)
            assert {r['reference_file'] for r in new}<=missing
            accepted.update({r['reference_file']:r for r in new})
            receipts.append(str(receipt))
            stagnant=0 if new else stagnant+1
            self.log(f'ORIGINAL COVERAGE {173+len(accepted)}/200 after seed {seed}; added {len(new)}')
            if stagnant and stagnant%3==0:
                atomic(self.work/'original_retry_plateau.json',dict(time=time.time(),seed=seed,
                       consecutive_zero_gain_sweeps=stagnant,missing=sorted(selected-set(accepted)),
                       action='Continue authorized retries with unchanged expert and success criteria'))
                self.log(f'ORIGINAL RETRY PLATEAU {stagnant} zero-gain sweeps; continuing authorized seed retries')
        result=dict(time=time.time(),state='complete_original_coverage',recordings=list(accepted.values()),
                    passing=len(accepted),missing_originals=[],retry_receipts=receipts,
                    selection='First passing demonstration per original; fixed expert and acceptance criteria',
                    retry_failed_originals_authorized=self.request.get('retry_failed_originals_authorized',False))
        original_sources(self.request,result)
        atomic(final,result)

    def prepare(self):
        for relative,expected in self.request['runtime_hashes'].items():
            prefix='Training/source/isaaclab_tasks/isaaclab_tasks/'
            if relative.startswith(prefix):
                spec=importlib.util.find_spec('isaaclab_tasks')
                assert spec is not None and spec.origin
                source=Path(spec.origin).parent/relative[len(prefix):]
            else:
                source=HERE.parent/relative
            assert hashlib.sha256(source.read_bytes()).hexdigest()==expected,(relative,str(source))
        collection=read(self.work/'original_coverage.result.json')
        sources=original_sources(self.request,collection)
        self.sources=[Path(p).expanduser().resolve(strict=True) for p in sources]
        assert len(set(self.sources))==200
        assert hashlib.sha256(self.expert.read_bytes()).hexdigest()==self.request['expert_sha256']
        commit=subprocess.check_output(['git','-C',str(ROOT/'Isaac-GR00T'),'rev-parse','HEAD'],text=True).strip()
        assert commit=='29c95b9cda2249b5285c84460c789d4fe3841d97'
        assert subprocess.run(['git','-C',str(ROOT/'Isaac-GR00T'),'diff','--quiet'],check=False).returncode==0
        self.verify_eval_pool()
        pool=ROOT/'ref_motions'/self.reference_pool
        ordered=[Path(p) for p in glob.glob(str(pool/'*.pkl'))]
        assert [p.name for p in ordered]==self.request['training_reference_order'], 'Training reference ordering changed'
        by_name={p.name:i for i,p in enumerate(ordered)}
        self.ids=[]
        for row in self.request['references']:
            p=pool/row['reference_file']
            assert hashlib.sha256(p.read_bytes()).hexdigest()==row['reference_sha256']
            self.ids.append(by_name[p.name])
        self.ids.sort()
        prefix=ROOT/'datasets/pick200_original_base_prefix'
        prefix_info=read(prefix/'lerobot/ds/meta/info.json')
        assert (prefix_info['total_episodes'],prefix_info['total_frames'])==(173,86327)
        output=ROOT/'datasets'/f'{self.tag}_r0'
        inputs=output/'hdf5';inputs.mkdir(parents=True,exist_ok=True)
        for i,source in enumerate(self.sources):
            p=inputs/f'{i:05d}__{source.name}'
            if not p.exists():p.symlink_to(source)
            assert p.resolve(strict=True)==source
        base=output/'lerobot/ds'
        if not (output/'built.json').exists():
            self.execute([sys.executable,HERE/'convert_incremental.py','--hdf5-root',inputs,
                          '--prefix-hdf5-root',prefix/'hdf5','--prefix-dataset',prefix/'lerobot/ds',
                          '--out',base,'--task-prompt','pick up the mustard bottle','--fps','50','--shards','4'],
                         'convert_base_original200')
            info=read(base/'meta/info.json');assert info['total_episodes']==200 and not info.get('discarded_episode_indices')
            atomic(output/'built.json',dict(episodes=200,frames=info['total_frames'],original_reference_count=200))
        assert read(base/'meta/info.json')['total_episodes']==200
        demo=self.work/'demo_root';demo.mkdir(exist_ok=True)
        for p in sorted(inputs.glob('*.hdf5')):
            dest=demo/p.name
            if not dest.exists():dest.symlink_to(p.resolve(strict=True))
            assert dest.resolve(strict=True)==p.resolve(strict=True)
        (self.work/'box.json').write_text(json.dumps(self.request['box'])+'\n')
        (self.work/'motion_ids.txt').write_text(' '.join(map(str,self.ids))+'\n')
        atomic(self.work/'eval_schedule.json',{f'{self.tag}_r{i}':{'workers':2} for i in range(4)})
        atomic(self.work/'recipe.json',dict(budget=200,original_only=True,base_episodes=200,
               rounds=3,student_attempts_per_round=200,batch_size=16,passes=5.7,
               steps_rule='ceil(cumulative_actual_frames*5.7/16)',initialization='Independent pretrained GR00T-N1.7-3B each stage',
               expert=str(self.expert),eval_references=100,eval_seeds=[100,101,102,103],
               source_episodes=list(map(str,self.sources)),motion_ids=self.ids))
        return base


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('request',type=Path);a=p.parse_args()
    chain=OriginalChain(read(a.request))
    try:
        while not (chain.work/'collection.result.json').exists():
            state=read(chain.work/'blue_queue.json')
            if state['phase'] in ['failed','finished']:
                raise RuntimeError('Missing-original collection ended without a valid result receipt')
            atomic(chain.work/'chain.status.json',dict(time=time.time(),pid=os.getpid(),phase='waiting_for_original_collection'))
            time.sleep(30)
        chain.complete_original_coverage()
        atomic(chain.work/'chain.status.json',dict(time=time.time(),pid=os.getpid(),phase='running'))
        chain.run()
        atomic(chain.work/'chain.status.json',dict(time=time.time(),pid=os.getpid(),phase='done'))
    except BaseException as error:
        atomic(chain.work/'abort.json',dict(time=time.time(),error=repr(error),pid=os.getpid()))
        atomic(chain.work/'chain.status.json',dict(time=time.time(),pid=os.getpid(),phase='needs_attention',error=repr(error)))
        raise


if __name__=='__main__':main()
