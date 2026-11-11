"""Export the fixed, previously selected replay points without image arrays.

This is transfer preparation: one CPU, checksums, and unmodified numeric fields.
No policy inference, simulation, image decoding, or reference reconstruction.
"""
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    import numpy as np
    base = Path.home() / 'kevin'
    out = base / 'eval_videos/walk_teacher_reconstruction_validation_20261027'
    out.mkdir(exist_ok=True)
    destination = out / 'captured_numeric_inputs.json'
    assert not destination.exists(), 'Inspect the existing export before another launch'
    workflow = out / 'numeric_export.workflow.json'
    sources = [('current_r2', 'walk_gap_eval_input_capture20261022', 78),
               ('historical_r2', 'walk_gap_hist_eval_input_capture20261022', 76)]
    rows, provenance = [], []
    for label, name, expected in sources:
        source = base / 'eval_videos' / name
        audit_path, plan_path = source / 'capture_independent_audit.json', source / 'replay_sample_plan.json'
        audit, plan = json.loads(audit_path.read_text()), json.loads(plan_path.read_text())
        assert audit['all_queries_complete'] and audit['all_executed_hand_commands_match']
        points = plan['points']; assert len(points) == expected
        inventory = {r['path']: r for r in audit['inventory']}
        provenance.append({'capture_model': label, 'capture_root': str(source.relative_to(base)),
            'audit_sha256': hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(), 'points': expected})
        for point in points:
            file = source / point['path']; digest = hashlib.sha256(file.read_bytes()).hexdigest()
            assert digest == inventory[point['path']]['sha256'], 'Captured input changed: ' + str(file)
            with np.load(file, allow_pickle=False) as z:
                md = json.loads(str(z['metadata_json']))
                task = z['task_observation/policy'].copy()
                assert task.shape == (1, 951) and task.dtype == np.float32 and np.isfinite(task).all()
                assert md['decoder_contract'] == 'previous_full_fsq'
                scene = {key.split('/', 1)[1]: z[key].tolist() for key in z.files if key.startswith('scene/')}
                state = {key.removeprefix('observation/state/'): z[key].tolist()
                         for key in z.files if key.startswith('observation/state/')}
                count = point['executed_steps']; assert 0 < count <= 40
                rows.append({'capture_model': label, 'sample_plan': point, 'capture_sha256': digest,
                    'metadata': {k: md[k] for k in ('episode', 'motion_id', 'step', 'motion_time', 'step_dt',
                        'joint_names', 'eval_source_sha256', 'decoder_source_sha256', 'decoder_contract')},
                    'task_policy': task.tolist(), 'task_policy_dtype': str(task.dtype),
                    'task_policy_sha256': hashlib.sha256(task.tobytes()).hexdigest(),
                    'scene': scene, 'vla_observation_state': state,
                    'student_first_motion_token': z['prediction/motion_token'][0, 0].tolist(),
                    'student_executed_hand_predictions': z['prediction/right_hand_joints'][0, :count].tolist()})
            if len(rows) % 8 == 0:
                workflow.write_text(json.dumps({'pid': os.getpid(), 'status': 'exporting', 'rows': len(rows), 'heartbeat': time.time()}))
                print('exported', len(rows), flush=True)
    assert len(rows) == 154
    value = {'at': time.time(), 'format': 'unchanged_numeric_vla_replay_inputs_v1', 'rows': rows,
        'provenance': provenance, 'GPU_used': False, 'image_arrays_decoded': False,
        'policy_queries_executed': False, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'limits': ['Fixed selected diagnostic points from10replicas per source policy; not an unbiased success estimate.',
                   'Source RGB bytes participate in integrity hashes, but image arrays are not decoded or exported.',
                   'Reference identity and timing must still be validated before expert inference.']}
    temporary = destination.with_suffix('.json.tmp'); assert not temporary.exists()
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(destination)
    workflow.write_text(json.dumps({'pid': os.getpid(), 'status': 'complete', 'rows': len(rows),
        'heartbeat': time.time(), 'bytes': destination.stat().st_size,
        'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}))
    print(json.dumps({'rows': len(rows), 'bytes': destination.stat().st_size}), flush=True)


if __name__ == '__main__':
    main()
