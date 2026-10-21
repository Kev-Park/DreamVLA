"""Approved 193-original prefix plus seven passing, unused hf60v3 originals.

Screen only the remaining deficit in a seeded, recorded candidate order. Keep
every attempt; train only after 200 unique original sources are verified.
"""
import argparse
import copy
import fcntl
import glob
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from pick_budget_chain import ROOT, HERE, atomic, read
from pick_original_chain import OriginalChain, original_sources


def candidate_batch(candidates, accepted, cursor, needed):
    assert 0 < needed <= len(candidates) - len(accepted)
    chosen = []
    while len(chosen) < needed:
        row = candidates[cursor % len(candidates)]
        cursor += 1
        if row['reference_file'] not in accepted:
            chosen.append(row)
    assert len({r['reference_file'] for r in chosen}) == needed
    return chosen, cursor


def assemble_references(retained, replacements, missing):
    assert len(retained) == 193 and len(replacements) == len(missing) == 7
    rows = retained + replacements
    assert len({r['source_motion_id'] for r in rows}) == 200
    assert not {r['reference_file'] for r in rows} & set(missing)
    assert all(r['reference_file'] == f"pick_{r['source_motion_id']}.pkl" for r in rows)
    return rows


def link_reference(destination, source):
    if not destination.exists():
        destination.symlink_to(source)
    assert destination.is_symlink() and destination.resolve(strict=True) == source


class SpareChain(OriginalChain):
    def verify_reused(self):
        import h5py
        import numpy as np
        sys.path.insert(0, str(self.sim))
        scorer = self.sim/'vla_sonic/grasp_success.py'
        assert hashlib.sha256(scorer.read_bytes()).hexdigest() == self.request['runtime_hashes']['Training/scripts/reinforcement_learning/rsl_rl/vla_sonic/grasp_success.py']
        from vla_sonic.grasp_success import score
        request = self.request
        assert len(request['references']) == 193
        old = [r for r in request['references'] if r['existing_original_demo']]
        added = request['reused_original_recordings']
        assert len(old) == 173 and len(added) == 20
        assert {r['reference_file'] for r in added} == {r['reference_file'] for r in request['references'] if not r['existing_original_demo']}
        source_paths = [(r['reference_file'], r['existing_original_demo']) for r in old]
        source_paths += [(r['reference_file'], r['path']) for r in added]
        checked = []
        for name, filename in source_paths:
            path = Path(filename).expanduser().resolve(strict=True)
            assert path.is_relative_to((ROOT/'datasets').resolve())
            with h5py.File(path, 'r', locking=False) as handle:
                data = handle['data/demo_0']
                result = score(data['obs/object_pos'][()], data['obs/object_quat'][()],
                               data['obs/robot0_root_pos_w'][()], data['obs/robot0_root_quat_w'][()],
                               data['teleop/right_wrist'][()], box_lo=np.array(request['box']['lo']),
                               box_hi=np.array(request['box']['hi']))
                assert result['success'], ('Reused demo no longer passes', name)
                frames = len(data['obs/object_pos'])
            checked.append(dict(reference_file=name,path=str(path),bytes=path.stat().st_size,frames=frames,passing=True))
        atomic(self.work/'reused193.verified.json',dict(time=time.time(),recordings=checked,passing=193))

    def complete_spares(self, request_path):
        final = self.work/'original_coverage.result.json'
        if final.exists():
            completed_request = read(self.work/'chain.request.final.json')
            assert completed_request['run'] == self.tag
            assert completed_request['runtime_hashes'] == self.request['runtime_hashes']
            original_sources(completed_request, read(final))
            self.request = completed_request
            atomic(request_path,self.request)
            return
        self.verify_reused()
        self.verify_eval_pool()
        initial = copy.deepcopy(self.request)
        assert initial['spare_replacements_authorized'] is True
        candidates = initial['spare_candidates']
        assert len(candidates) == len({r['source_motion_id'] for r in candidates}) == 19
        forbidden = {r['reference_file'] for r in initial['references']}
        forbidden |= {r['name'] for r in initial['eval_reference_manifest']['files']}
        assert not forbidden & {r['reference_file'] for r in candidates}
        accepted = {}
        receipts = []
        cursor = 0
        batch_index = 0
        while len(accepted) < 7:
            batch, cursor = candidate_batch(candidates, accepted, cursor, 7-len(accepted))
            directory = self.work/'spare_batches'/f'{batch_index:03d}'
            directory.mkdir(parents=True,exist_ok=True)
            spec = copy.deepcopy(initial)
            spec.update(purpose='approved_spare_screening',run=f'{self.tag}_spare{batch_index:03d}',
                        references=batch,expected_attempts=len(batch),seed=initial['seed']+batch_index,
                        reference_pool='Holosoma_Pick_29_hf60v3',eligible_spare_references=candidates)
            path = directory/'collection.request.json'
            result_path = directory/'collection.result.json'
            if path.exists():
                assert read(path) == spec, 'Preserved spare batch request changed'
            else:
                atomic(path,spec)
            atomic(self.work/'chain.status.json',dict(time=time.time(),pid=os.getpid(),phase='screening_spare_originals',
                   seed=spec['seed'],passing_originals=193+len(accepted),batch=batch_index,
                   selected_references=[r['reference_file'] for r in batch]))
            if not result_path.exists():
                assert shutil.disk_usage(ROOT).free >= (50+len(batch))*1024**3, 'Insufficient spare recording headroom'
                gpu = self.slots(1,17000)[0]
                try:
                    self.execute([sys.executable,'-u',HERE/'pick_original_base_collect.py',path],
                                 f'collect_spare_batch{batch_index:03d}',gpu,ROOT/'DreamVLA/Training')
                finally:
                    self.release(gpu)
            result = read(result_path)
            assert result['state'] == 'attempts_completed'
            assert len(result['attempted_motion_ids']) == len(batch)
            allowed = {r['reference_file'] for r in batch}
            for row in result['recordings']:
                assert row['reference_file'] in allowed
                if row['passing']:
                    assert row['reference_file'] not in accepted
                    accepted[row['reference_file']] = dict(row,collection_seed=spec['seed'],screening_receipt=str(result_path))
            receipts.append(str(result_path))
            atomic(self.work/'spare_selection.partial.json',dict(time=time.time(),accepted=list(accepted.values()),
                   coverage=193+len(accepted),receipts=receipts,cursor=cursor))
            batch_index += 1
        chosen = [r for r in candidates if r['reference_file'] in accepted]
        selected = assemble_references(initial['references'],chosen,initial['replaced_reference_names'])
        pool = ROOT/'ref_motions'/self.reference_pool
        pool.mkdir(exist_ok=True)
        for row in selected:
            source = Path(row['reference_source_path']).expanduser().resolve(strict=True)
            assert hashlib.sha256(source.read_bytes()).hexdigest() == row['reference_sha256']
            destination = pool/row['reference_file']
            link_reference(destination,source)
        ordered = [Path(p) for p in glob.glob(str(pool/'*.pkl'))]
        assert len(ordered) == 200 and {p.name for p in ordered} == {r['reference_file'] for r in selected}
        self.request = dict(initial,references=selected,training_reference_order=[p.name for p in ordered])
        atomic(self.work/'chain.request.approved193.json',initial)
        atomic(self.work/'chain.request.final.json',self.request)
        coverage = dict(time=time.time(),state='complete_original_coverage',
                        recordings=initial['reused_original_recordings']+list(accepted.values()),passing=27,
                        original_coverage=200,missing_originals=[],spare_receipts=receipts,
                        selection='193 verified retained originals + first seven passing spare originals in recorded candidate order')
        original_sources(self.request,coverage)
        atomic(self.work/'collection.result.json',coverage)
        atomic(final,coverage)
        atomic(request_path,self.request)
        atomic(self.work/'spare_selection.final.json',dict(time=time.time(),replaced=initial['replaced_reference_names'],
               selected=[r['reference_file'] for r in chosen],retained_originals=193,final_originals=200,
               heldout_manifest_unchanged=True,receipts=receipts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request',type=Path)
    args = parser.parse_args()
    chain = SpareChain(read(args.request))
    with (chain.work/'spare_driver.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            chain.complete_spares(args.request)
            atomic(chain.work/'chain.status.json',dict(time=time.time(),pid=os.getpid(),phase='running',passing_originals=200))
            chain.run()
            atomic(chain.work/'chain.status.json',dict(time=time.time(),pid=os.getpid(),phase='done',passing_originals=200))
        except BaseException as error:
            atomic(chain.work/'abort.json',dict(time=time.time(),error=repr(error),pid=os.getpid()))
            atomic(chain.work/'chain.status.json',dict(time=time.time(),pid=os.getpid(),phase='needs_attention',error=repr(error)))
            raise


if __name__ == '__main__':
    main()
