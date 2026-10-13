"""Recompute held-out pick metrics on Biped without storing trajectory copies.

Only completed evaluations enter the uncompressed stream. Every trajectory is
checksummed, and both per-episode and aggregate metrics must match its report.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

TAG = 'dagger57_20261013'


def send():
    root = Path.home() / 'kevin'
    manifest = {'runs': {}, 'files': {}}
    for budget in (10, 50):
        work = root / f'dline/pick{budget}_{TAG}'
        for round_id in range(4):
            run = f'pick{budget}_{TAG}_r{round_id}'
            done = work / (run + '.eval.json')
            if not done.exists():
                continue
            report = json.loads(done.read_text())
            assert report['episodes'] == 100
            manifest['runs'][run] = {'report': report,
                                     'box': json.loads((work / 'box.json').read_text())}
            paths = sorted((root / 'eval_videos' / run).glob('b*_traj.npz'))
            assert len(paths) == 100
            for path in paths:
                content = path.read_bytes()
                manifest['files'][f'{run}/{path.name}'] = {
                    'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    assert manifest['runs']
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
        data = json.dumps(manifest).encode()
        member = tarfile.TarInfo('manifest.json'); member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
        for name, expected in manifest['files'].items():
            content = (root / 'eval_videos' / name).read_bytes()
            assert len(content) == expected['bytes']
            assert hashlib.sha256(content).hexdigest() == expected['sha256']
            member = tarfile.TarInfo(name); member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def audit_stream(stream):
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                          'Training/scripts/reinforcement_learning/rsl_rl'))
    from vla_sonic.grasp_success import score
    seen = set(); rows = {}; names = None
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        members = iter(archive)
        first = next(members)
        assert first.name == 'manifest.json' and first.isfile()
        manifest = json.load(archive.extractfile(first))
        for member in members:
            assert member.isfile() and member.name in manifest['files']
            assert member.name not in seen
            seen.add(member.name)
            expected = manifest['files'][member.name]
            assert member.size == expected['bytes']
            content = archive.extractfile(member).read()
            assert hashlib.sha256(content).hexdigest() == expected['sha256']
            run = member.name.split('/')[0]
            report = manifest['runs'][run]['report']
            expected_row = next(x for x in report['per_episode']
                                if member.name.endswith(f"_m{x['motion_id']}_traj.npz"))
            box = manifest['runs'][run]['box']
            with np.load(io.BytesIO(content), allow_pickle=False) as z:
                current_names = z['body_names'].tolist()
                if names is None:
                    names = current_names
                assert names == current_names
                body, ref = z['body_pos'], z['ref_body_pos']
                root, ref_root = z['root_pos'], z['ref_root_pos']
                assert body.shape == ref.shape and body.ndim == 3 and body.shape[-1] == 3
                assert root.shape == ref_root.shape == (len(body), 3) and len(body) > 0
                assert len(names) == body.shape[1]
                # Independent calculation, in float64, of the specified metric.
                delta = body.astype('float64') - ref.astype('float64')
                aligned = delta - (root.astype('float64') - ref_root.astype('float64'))[:, None]
                world = float(np.sqrt(np.sum(delta * delta, axis=-1)).mean() * 100)
                relative = float(np.sqrt(np.sum(aligned * aligned, axis=-1)).mean() * 100)
                assert np.isfinite(world) and np.isfinite(relative)
                success = bool(score(z['obj_pos'], z['obj_quat'], root, z['root_quat'], z['wrist'],
                                     box_lo=np.array(box['lo']), box_hi=np.array(box['hi']))['success'])
            assert abs(world - expected_row['world_mpjpe_cm']) < 1e-5
            assert abs(relative - expected_row['root_aligned_mpjpe_cm']) < 1e-5
            assert success == expected_row['success']
            assert len(body) == expected_row['frames'] and body.shape[1] == expected_row['bodies']
            rows.setdefault(run, []).append({'motion_id': expected_row['motion_id'],
                                            'success': success, 'world': world, 'root': relative})
    assert seen == set(manifest['files'])
    results = []
    for run, data in manifest['runs'].items():
        episodes = rows[run]; report = data['report']
        assert sorted(x['motion_id'] for x in episodes) == list(range(100))
        world = sum(x['world'] for x in episodes) / 100
        relative = sum(x['root'] for x in episodes) / 100
        successes = sum(x['success'] for x in episodes)
        assert successes == report['successes']
        assert abs(world - report['world_mpjpe_cm']) < 1e-5
        assert abs(relative - report['root_aligned_mpjpe_cm']) < 1e-5
        results.append({'run': run, 'episodes': 100, 'successes': successes,
                        'world_mpjpe_cm': world, 'root_aligned_mpjpe_cm': relative})
    return {'time': time.time(), 'host': __import__('socket').gethostname(),
            'verified_trajectories': len(seen), 'body_names': names, 'runs': results,
            'trajectory_copies_stored': 0, 'metric_tolerance_cm': 1e-5}


def bridge():
    from stream_directory import ssh
    flags = {'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0)}
    sender = subprocess.Popen(ssh('bluesclues', 'python3 ~/kevin/wt/dagger-watch-sparse/cluster/'
                                  'audit_pick_budget_stream.py send'),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, **flags)
    receiver = subprocess.Popen(ssh('biped', '~/miniconda3/envs/dreamcontrol_51/bin/python '
                                    '~/kevin/DreamVLA/cluster/audit_pick_budget_stream.py receive'),
                                stdin=sender.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **flags)
    sender.stdout.close()
    output, errors = receiver.communicate(timeout=1800)
    source_errors = sender.stderr.read(); code = sender.wait(timeout=30)
    assert receiver.returncode == code == 0, (source_errors, errors)
    result = json.loads(output)
    path = Path(__file__).resolve().parents[2] / 'out/pick_budget_biped_trajectory_audit.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['send', 'receive', 'bridge'])
    mode = parser.parse_args().mode
    if mode == 'send':
        send()
    elif mode == 'receive':
        result = audit_stream(sys.stdin.buffer)
        path = Path.home() / 'kevin/eval_videos/pick_budget_trajectory_audit.json'
        path.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))
    else:
        bridge()


if __name__ == '__main__':
    main()
