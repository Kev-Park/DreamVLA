"""Export raw numeric columns, then audit existing filter chronology off-host.

The export performs IO and integrity checks only: no images, policy queries,
success scoring, source changes, or compression. Analysis runs separately.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time


TAG = 'walk200_v3_late1s_20261024'


def owned(relative):
    root = (Path.home() / 'kevin').resolve()
    rel = PurePosixPath(relative)
    assert not rel.is_absolute() and '..' not in rel.parts
    path = root.joinpath(*rel.parts)
    assert path.resolve().is_relative_to(root)
    return path


def stamp(path):
    s = path.stat()
    return {k: getattr(s, k) for k in
            ['st_size', 'st_ino', 'st_dev', 'st_mtime_ns']}


def export(plan_relative, bundle_relative):
    import h5py
    import numpy as np

    plan_path = owned(plan_relative)
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    records = plan['episodes']
    attempted = plan['attempted_motion_ids']
    skipped = [r['motion_id'] for r in plan['unrecorded_attempts']]
    observed = [r['motion_id'] for r in records]
    assert sorted(attempted) == list(range(200))
    assert len(observed) == len(set(observed))
    assert not set(observed).intersection(skipped)
    assert sorted(observed + skipped) == attempted
    bundle = owned(bundle_relative)
    assert not bundle.exists(), 'Inspect an existing export before restarting'
    bundle.mkdir(parents=True)
    output = bundle / 'object_traces.f32'
    digest = hashlib.sha256()
    rows = []
    offset = 0
    with output.open('xb') as stream:
        for record in records:
            relative = record['source_relative']
            assert relative.startswith(f'dline/{TAG}/raw/')
            source = owned(relative)
            assert not source.is_symlink()
            before = stamp(source)
            assert before['st_size'] == record['bytes']
            with h5py.File(source, 'r', locking=False) as f:
                raw_metadata = f.attrs['metadata_json']
                metadata = json.loads(raw_metadata)
                assert metadata['motion_id'] == record['motion_id']
                assert metadata['rescue'] == record['rescue']
                obs = f['data/demo_0/obs']
                position = obs['object_pos'][()]
                orientation = obs['object_quat'][()]
                n = record['frames']
                assert position.shape == (n, 3) and orientation.shape == (n, 4)
                assert position.dtype == np.float32 and orientation.dtype == np.float32
                values = np.concatenate([position, orientation], axis=1)
                payload = values.astype('<f4', copy=False).tobytes(order='C')
                assert len(payload) == n * 7 * 4
                stream.write(payload)
                digest.update(payload)
            assert stamp(source) == before, 'Source changed during numeric export'
            rows.append({'record': record, 'offset_bytes': offset,
                         'trace_bytes': len(payload), 'source_stat': before,
                         'metadata_sha256': hashlib.sha256(raw_metadata.encode()).hexdigest(),
                         'recorded_phys_held': metadata.get('phys_held')})
            offset += len(payload)
    assert output.stat().st_size == offset
    manifest = {'format': 'uncompressed_f32_object_trace_v1', 'at': time.time(),
                'dtype': '<f4', 'columns': ['object_x', 'object_y', 'object_z',
                                          'quat_0', 'quat_1', 'quat_2', 'quat_3'],
                'bytes': offset, 'sha256': digest.hexdigest(), 'rows': rows,
                'source_plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
                'attempted_motion_ids': attempted,
                'unrecorded_attempts': plan['unrecorded_attempts'],
                'compression': False, 'images_decoded': False,
                'policy_queries_executed': False, 'analysis_executed': False,
                'cpu_affinity': sorted(os.sched_getaffinity(0))}
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'complete': True, 'episodes': len(rows), 'bytes': offset,
                      'sha256': digest.hexdigest()}))


def summarize(records):
    rescues = [r for r in records if r['pass'] == 'B']
    rejected = [r for r in rescues if not r['retained']]
    return {'recorded_episodes': len(records), 'saved_rescues': len(rescues),
            'retained_rescues': sum(r['retained'] for r in rescues),
            'rejected_rescues': len(rejected),
            'rejected_with_any_irreversible_rule_failure': sum(
                r['first_irreversible_rule_failure_frame'] is not None for r in rejected),
            'rejected_irreversible_strictly_before_takeover': sum(
                r['irreversible_failure_strictly_before_takeover'] for r in rejected),
            'rejected_irreversible_at_takeover': sum(
                r['irreversible_failure_at_takeover_frame'] for r in rejected),
            'rejected_irreversible_after_takeover': sum(
                r['irreversible_failure_after_takeover_frame'] for r in rejected),
            'before_takeover_rule_failure_and_recorded_phys_held': sum(
                r['irreversible_failure_strictly_before_takeover'] and
                r['recorded_phys_held'] is True for r in rejected)}


def analyze(bundle_relative):
    import numpy as np

    base = Path.home() / 'kevin'
    native = base / 'wt/locomotion/DreamVLA/Training/scripts/reinforcement_learning/rsl_rl'
    sys.path.insert(0, str(native))
    from vla_sonic.grasp_success import (
        object_tilt_deg, TABLE_TOP_Z, FALLEN_FRAC, HORIZ_TOL, HORIZ_DEG)

    bundle = owned(bundle_relative)
    manifest_path = bundle / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['format'] == 'uncompressed_f32_object_trace_v1'
    payload = (bundle / 'object_traces.f32').read_bytes()
    assert len(payload) == manifest['bytes']
    assert hashlib.sha256(payload).hexdigest() == manifest['sha256']
    records = []
    offset = 0
    for row in manifest['rows']:
        episode = row['record']
        assert row['offset_bytes'] == offset
        n = episode['frames']
        values = np.frombuffer(payload, dtype='<f4', count=n * 7, offset=offset).reshape(n, 7)
        offset += row['trace_bytes']
        bad_height = values[:, 2] < FALLEN_FRAC * TABLE_TOP_Z
        bad_horizontal = np.abs(object_tilt_deg(values[:, 3:]) - HORIZ_DEG) <= HORIZ_TOL

        def first(values):
            indices = np.flatnonzero(values)
            return int(indices[0]) if len(indices) else None

        h, o = first(bad_height), first(bad_horizontal)
        failures = [t for t in (h, o) if t is not None]
        first_bad = min(failures) if failures else None
        rescue = episode['rescue']
        k = rescue['takeover_at'] if rescue['pass'] == 'B' else None
        if episode['four_rule_pass']:
            assert first_bad is None
        records.append({'motion_id': episode['motion_id'], 'pass': rescue['pass'],
                        'retained': episode['four_rule_pass'], 'frames': n,
                        'takeover_frame': k, 'first_height_rule_failure_frame': h,
                        'first_horizontal_rule_failure_frame': o,
                        'first_irreversible_rule_failure_frame': first_bad,
                        'irreversible_failure_strictly_before_takeover':
                            k is not None and first_bad is not None and first_bad < k,
                        'irreversible_failure_at_takeover_frame':
                            k is not None and first_bad == k,
                        'irreversible_failure_after_takeover_frame':
                            k is not None and first_bad is not None and first_bad > k,
                        'recorded_phys_held': row['recorded_phys_held'],
                        'expert_controlled_frames': n - episode['student_frames']})
    assert len(records) == len(manifest['rows']) and offset == len(payload)
    skipped = manifest['unrecorded_attempts']
    assert sorted([r['motion_id'] for r in records] +
                  [r['motion_id'] for r in skipped]) == list(range(200))
    report = {'at': time.time(), 'provisional': False, 'attempted_references': 200,
              'summary': summarize(records), 'per_episode': records,
              'unrecorded_attempts': skipped,
              'first50_summary': summarize([r for r in records if r['motion_id'] < 50]),
              'filter_thresholds': {'minimum_object_height': FALLEN_FRAC * TABLE_TOP_Z,
                                    'horizontal_degrees': HORIZ_DEG,
                                    'horizontal_tolerance_degrees': HORIZ_TOL},
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'native_filter_source_sha256': hashlib.sha256(
                  (native / 'vla_sonic/grasp_success.py').read_bytes()).hexdigest(),
              'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'GPU_used': False, 'data_changed': False,
              'limits': ['Numerical chronology of the existing unchanged episode-wide filter.',
                         'A rule violation before takeover precludes passing this episode filter; '
                         'it does not establish that an expert could have repaired the state.',
                         'Recorded phys_held is an existing separate episode-level proxy; '
                         'it does not establish successful post-takeover recovery.',
                         'Attempt141 terminated before a reference grasp was observed and '
                         'has no saved rescue recording; its event chronology is unavailable.',
                         'No held-out performance result or timing-only causal conclusion.']}
    target = bundle / 'filter_chronology_full200.json'
    assert not target.exists(), 'Inspect an existing analysis before restarting'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'summary': report['summary'], 'first50_summary': report['first50_summary']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['export', 'analyze'])
    parser.add_argument('--bundle-relative', required=True)
    parser.add_argument('--plan-relative')
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if args.mode == 'export':
        assert args.plan_relative
        export(args.plan_relative, args.bundle_relative)
    else:
        analyze(args.bundle_relative)


if __name__ == '__main__':
    main()
