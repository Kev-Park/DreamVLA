"""Audit saved aligned stitches numerically; never authorize or launch training.

Hand errors use the reanchored static path INCLUDING the existing calibrated shift.
Review flags are geometric diagnostics, not policy-success assessments.
"""
import json
import os
import time

import numpy as np
import reconstruct_own_motion_stitches as build


def audit(t, chain, limits):
    st, sh = build.stitcher, build.shared
    for path, digest in [(t['output'], t['sha256']), (t['walk'], t['walk_sha256']),
                         (t['static'], t['static_sha256'])]:
        assert sh.sha(path) == digest, path
    original = st.load(t['walk'], chain)
    w = build.shifted_walk(original, t['object_shift_m'], t['root_alignment_shift_m'])
    s, o = st.load(t['static'], chain), st.load(t['output'], chain)
    fw, on = t['diagnostics']['footfall'], t['diagnostics']['onset']
    start = fw + st.TRANSITION
    gs, go = s['g'], o['g']
    assert len(o['J']) == start + len(s['J']) - on
    assert go == start + gs - on
    assert all(np.isfinite(v).all() for v in o.values())
    assert np.allclose(o['J'][:fw, :12], original['J'][:fw, :12], atol=1e-6)
    assert np.allclose(o['P'][:fw], w['P'][:fw], atol=1e-6)
    assert np.allclose(o['Q'][:fw], original['Q'][:fw], atol=1e-6)
    assert np.allclose(o['P'][fw:], w['P'][fw], atol=1e-6)
    assert np.allclose(o['J'][fw:, :12], w['J'][fw, :12], atol=1e-6)
    assert np.allclose(np.linalg.norm(o['Q'], axis=1), 1, atol=1e-5)
    assert np.allclose(np.linalg.norm(o['O'][:, 3:], axis=1), 1, atol=1e-5)

    yaw = st.yaw_of(w['Q'][fw])
    angle = yaw - st.yaw_of(s['Q'][gs])
    c, z = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -z, 0], [z, c, 0], [0, 0, 1]])
    target = (s['hand'][on:] - s['O'][gs, :3]) @ rotation.T + w['O'][w['g'], :3]
    shift = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]) @ np.asarray(sh.SHIFT)
    ramp = np.clip((np.arange(on, len(s['J'])) - on) / max(gs-on, 1), 0, 1)
    target += (ramp*ramp*(3-2*ramp))[:, None] * shift
    error = o['hand'][start:] - target
    norms = np.linalg.norm(error, axis=1)
    grab = gs - on
    last_second = error[max(0, grab-19):grab+1]
    target_R = rotation @ s['handR'][on:]
    relative_R = o['handR'][start:] @ np.swapaxes(target_R, 1, 2)
    angles = np.degrees(np.arccos(np.clip((np.trace(relative_R, axis1=1, axis2=2)-1)/2, -1, 1)))
    expected_object = (s['O'][on:, :3] - s['O'][gs, :3]) @ rotation.T + w['O'][w['g'], :3]
    assert np.allclose(o['O'][start:, :3], expected_object, atol=1e-6)
    violation = np.maximum(np.maximum(limits[0] - o['J'], o['J'] - limits[1]), 0)
    m = {'id': t['id'], 'split': t['split'], 'name': t['name'],
         'root_translation_cm': float(100*np.linalg.norm(t['root_alignment_shift_m'])),
         'frames': len(o['J']), 'grab_frame': go,
         'grab_hand_error_cm': float(100*norms[grab]),
         'grab_hand_max_axis_error_cm': float(100*np.abs(error[grab]).max()),
         'approach_hand_max_error_cm': float(100*norms[:grab+1].max()),
         'postgrab_hand_max_error_cm': float(100*norms[grab:].max()),
         'last_second_hand_rms_cm': float(100*np.sqrt((last_second**2).sum(1).mean())),
         'grab_hand_orientation_error_deg': float(angles[grab]),
         'joint_limit_violation_rad': float(violation.max()),
         'max_joint_step_rad': float(np.abs(np.diff(o['J'], axis=0)).max()),
         'stitch_root_step_cm': float(100*np.linalg.norm(o['P'][fw]-o['P'][fw-1])) if fw else 0,
         'stitch_hand_step_cm': float(100*np.linalg.norm(o['hand'][fw]-o['hand'][fw-1])) if fw else 0,
         'object_transition_jump_cm': float(100*np.linalg.norm(o['O'][start, :3]-o['O'][start-1, :3]))}
    m['review_flags'] = [label for label, fail in [
        ('grab_hand_axis_gt_1cm', m['grab_hand_max_axis_error_cm'] > 1),
        ('last_second_hand_rms_gt_1p5cm', m['last_second_hand_rms_cm'] > 1.5),
        ('joint_limit_violation', m['joint_limit_violation_rad'] > 1e-5)] if fail]
    return m


def main():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:26])
    chain, limits = build.shared.chain()
    tasks = json.loads((build.REPORT/'generated.json').read_text())
    assert len(tasks) == 500
    assert {s: sum(t['split'] == s for t in tasks) for s in build.COUNTS} == build.COUNTS
    train = [t for t in tasks if t['split'] == 'train']
    test = [t for t in tasks if t['split'] == 'eval']
    assert not ({t['id'] for t in train} & {t['id'] for t in test})
    assert not ({t['raw_sha256'] for t in train} & {t['raw_sha256'] for t in test})
    parents = {t['id']: t for t in train}
    for t in tasks:
        if t['split'] == 'aug':
            assert t['root_alignment_shift_m'] == parents[t['id']]['root_alignment_shift_m']
            assert t['walk_sha256'] == parents[t['id']]['walk_sha256']
    results = []
    for t in tasks:
        results.append(audit(t, chain, limits))
        if len(results) % 25 == 0:
            print(json.dumps({'audited': len(results), 'total': 500}), flush=True)
    keys = [k for k in results[0] if k.endswith(('_cm', '_deg', '_rad'))]
    summary = {'created': time.time(), 'phase': 'numerical_audit_complete_pending_replay',
               'count': 500, 'counts': build.COUNTS, 'source_split_disjoint': True,
               'own_walk_preservation_checks_passed': True,
               'training_authorized_to_start': False,
               'review_flag_count': sum(bool(r['review_flags']) for r in results),
               'metric_quantiles_min_median_p90_max': {k: np.quantile([r[k] for r in results], [0,.5,.9,1]).tolist() for k in keys},
               'rows': results}
    build.shared.dump(build.REPORT/'numerical_validation.json', summary)
    build.status(**{k:v for k,v in summary.items() if k != 'rows'})
    print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        build.status(phase='failed', operation='numerical_validation', error=repr(exc))
        raise
