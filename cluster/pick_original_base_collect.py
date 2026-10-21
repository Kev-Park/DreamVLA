"""Collect an explicitly approved set of missing original-reference expert demos.

One attempt per reference. This worker records incomplete coverage rather than
retrying failures, substituting augmentations, or changing acceptance criteria.
"""
import argparse
import fcntl
import glob
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def reference_request(request):
    refs = request['references']
    assert len(refs) == len({r['source_motion_id'] for r in refs})
    assert all(r['reference_file'] == f"pick_{r['source_motion_id']}.pkl" for r in refs)
    if request.get('purpose') == 'approved_spare_screening':
        assert request.get('spare_replacements_authorized') is True
        assert 1 <= len(refs) <= 19
        allowed = {r['reference_file']: r['reference_sha256'] for r in request['eligible_spare_references']}
        assert len(allowed) == 19
        assert all(r['requires_original_demo_collection'] and allowed.get(r['reference_file']) == r['reference_sha256'] for r in refs)
        assert request['reference_pool'] == 'Holosoma_Pick_29_hf60v3'
        return refs, request['reference_pool']
    assert len(refs) == 200
    return refs, 'Holosoma_Pick_29_latband60max'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    assert request['launch_authorized'] is True
    root = Path.home() / 'kevin'
    work = args.request.parent
    with (work / 'collection.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (work / 'collection.started.json').exists(), 'Existing attempt needs reconciliation'
        refs, pool_name = reference_request(request)
        pool = root / 'ref_motions' / pool_name
        ordered = [Path(p) for p in glob.glob(str(pool / '*.pkl'))]
        names = {p.name: (i, p) for i, p in enumerate(ordered)}
        for row in refs:
            assert digest(names[row['reference_file']][1]) == row['reference_sha256']
        selected = [r for r in refs if r['requires_original_demo_collection']]
        assert len(selected) == request['expected_attempts']
        ids = [names[r['reference_file']][0] for r in selected]
        assert len(ids) == len(set(ids))
        repo = Path(__file__).resolve().parents[1]
        collector = repo / 'Training/scripts/reinforcement_learning/rsl_rl/collect_sonic_adapter.py'
        # The pinned sparse checkout supplies code; the production Training
        # directory supplies the same relative USD assets as the 10/50 runs.
        asset_working_directory = root / 'DreamVLA/Training'
        robot_asset = asset_working_directory / 'HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof_with_hands_min_collisions_flat_white.usd'
        assert robot_asset.is_file(), str(robot_asset)
        for relative, expected in request['runtime_hashes'].items():
            task_prefix = 'Training/source/isaaclab_tasks/isaaclab_tasks/'
            if relative.startswith(task_prefix):
                spec = importlib.util.find_spec('isaaclab_tasks')
                assert spec is not None and spec.origin, 'Isaac task package is unavailable'
                source = Path(spec.origin).parent / relative[len(task_prefix):]
            else:
                source = repo / relative
            assert digest(source) == expected, (relative, str(source))
        expert = root / 'residuals/af60v8c_model_8998.pt'
        assert digest(expert) == request['expert_sha256']
        output = root / 'datasets' / request['run'] / 'missing_original_attempt0'
        assert not output.exists(), 'Never overwrite an earlier collection'
        idsfile = work / 'missing_motion_ids.txt'
        idsfile.write_text(' '.join(map(str, ids)) + '\n')
        save(work / 'runtime_reference_order.json', dict(files=[p.name for p in ordered], selected_ids=ids))
        env = dict(os.environ, **request['environment'])
        assert env['HS_OBS_NOISE'] == '0'
        command = [sys.executable, str(collector), '--headless', '--task',
                   'Isaac-Motion-Tracking-Pick-Cam-HOI-v0', '--waist-dof', '29',
                   '--sonic-pt', str(root / 'sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000'),
                   '--encoder-mode', 'g1', '--residual-transform', 'additive_free',
                   '--residual-scale', '0.1', '--checkpoint-path', str(expert),
                   '--ref-motions-path', pool_name,
                   '--motions-from', str(idsfile), '--rollout-length', '500',
                   '--num-samples', str(len(ids)), '--seed', str(request['seed']),
                   '--output-directory', str(output)]
        save(work / 'collection.started.json', dict(time=time.time(), pid=os.getpid(), command=command,
             expected_attempts=len(ids), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
             working_directory=str(asset_working_directory), robot_asset=str(robot_asset),
             robot_asset_sha256=digest(robot_asset)))
        with (work / 'collection.log').open('w') as log:
            process = subprocess.Popen(command, cwd=asset_working_directory, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            save(work / 'collection.worker.json', dict(pid=process.pid, started=time.time()))
            code = process.wait()
        (work / 'collection.exit_code.txt').write_text(str(code) + '\n')
        if code:
            raise RuntimeError(f'Collector exited {code}; inspect collection.log, no automatic retry')
        text = (work / 'collection.log').read_text()
        attempted = list(map(int, re.findall(r'\[INFO\] motion (\d+)/\d+', text)))
        assert attempted == ids, f'Attempt coverage mismatch: {attempted} != {ids}'
        assert re.findall(r'Collection finished\..*errored=(\d+)', text) == ['0']
        import h5py
        import numpy as np
        sys.path.insert(0, str(collector.parent))
        from vla_sonic.grasp_success import score
        rows = []
        paths = list(output.rglob('*.hdf5'))
        seen = set()
        for path in paths:
            mid = int(re.search(r'__motion_(\d+)\.hdf5$', path.name)[1])
            assert mid in ids and mid not in seen
            seen.add(mid)
            with h5py.File(path, 'r', locking=False) as f:
                g = f['data/demo_0']
                result = score(g['obs/object_pos'][()], g['obs/object_quat'][()],
                               g['obs/robot0_root_pos_w'][()], g['obs/robot0_root_quat_w'][()],
                               g['teleop/right_wrist'][()], box_lo=np.array(request['box']['lo']),
                               box_hi=np.array(request['box']['hi']))
            rows.append(dict(motion_id=mid, reference_file=ordered[mid].name,
                             path=str(path), bytes=path.stat().st_size,
                             sha256=digest(path), passing=bool(result['success'])))
        passing = [r for r in rows if r['passing']]
        save(work / 'collection.result.json', dict(time=time.time(), state='attempts_completed',
             attempted_motion_ids=attempted, recordings=rows, passing=len(passing),
             missing_originals=[r['reference_file'] for r in selected
                                if r['reference_file'] not in {p['reference_file'] for p in passing}],
             automatic_retries=False))
        print(json.dumps(dict(attempts=len(ids), passing=len(passing), missing=len(ids)-len(passing))), flush=True)


if __name__ == '__main__':
    main()
