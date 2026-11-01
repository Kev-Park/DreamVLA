"""Expose saved R2/R3 per-rule metrics and logged terminations for user review."""
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time


def run():
    os.sched_setaffinity(0, {44})
    import numpy as np
    root = Path.home()/'kevin'
    runtime = root/'wt/pick200-original-spares'
    sys.path.insert(0, str(runtime/'Training/scripts/reinforcement_learning/rsl_rl'))
    from vla_sonic.grasp_success import score
    from vla_sonic import grasp_success
    tag = 'pick200_original_spares57_v2'
    work = root/'dline'/tag
    request = json.loads((work/'chain.request.final.json').read_text())
    names = [Path(p).name for p in glob.glob(str(root/'ref_motions/Holosoma_Pick_29_latband60max_eval/*.pkl'))]
    assert names == [r['name'] for r in request['eval_reference_manifest']['files']]
    all_rows = {}
    for round_id in [2, 3]:
        run_name = f'{tag}_r{round_id}'
        directory = root/'eval_videos'/run_name
        report = json.loads((directory/'results.json').read_text())
        by_id = {r['motion_id']: r for r in report['per_episode']}
        terminations = {}
        for log in work.glob(f'eval_{run_name}_shard*.log'):
            ids = {}
            for line in log.read_text(errors='replace').splitlines():
                match = re.search(r'\[episode (\d+)\] motion_id=(\d+)', line)
                if match:
                    ids[int(match[1])] = int(match[2])
                ended = re.search(r'\[episode (\d+)\] ended at step (\d+) \(([^)]+)\)', line)
                if ended:
                    mid = ids[int(ended[1])]
                    assert mid not in terminations
                    terminations[mid] = dict(steps=int(ended[2]), reason=ended[3])
        assert sorted(terminations) == list(range(100))
        rows = {}
        for path in directory.glob('b*_traj.npz'):
            mid = int(re.search(r'_m(\d+)_traj.npz$', path.name)[1])
            with np.load(path, allow_pickle=False) as z:
                scored = score(z['obj_pos'], z['obj_quat'], z['root_pos'],
                    z['root_quat'], z['wrist'], box_lo=np.array(request['box']['lo']),
                    box_hi=np.array(request['box']['hi']))
                assert bool(scored['success']) == by_id[mid]['success']
                frames = len(z['obj_pos'])
            rows[mid] = dict(by_id[mid], termination=terminations[mid],
                trajectory=path.name, trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                frames=frames, rules={key:bool(scored[key]) for key in
                    ['fallen_height', 'fallen_horizontal', 'in_box_at_end', 'slipping']},
                slip_spread_mm=1000*scored['slip_spread'],
                min_object_height_m=scored['min_z'], max_object_tilt_deg=scored['max_tilt'],
                in_box_frames=scored['in_box_frames'], last_in_box_frame=scored['last_in_box'])
        assert sorted(rows) == list(range(100))
        all_rows[round_id] = rows
    pairs = []
    for mid, name in enumerate(names):
        a, b = all_rows[2][mid], all_rows[3][mid]
        transition = ('both_success' if a['success'] else 'gained') if b['success'] else (
            'lost' if a['success'] else 'both_failure')
        pairs.append(dict(motion_index=mid, reference_file=name,
                          transition=transition, dagger2=a, dagger3=b))
    groups = {name:[r['motion_index'] for r in pairs if r['transition']==name]
              for name in ['gained', 'lost', 'both_success', 'both_failure']}
    def rule_counts(which, subset):
        counts = {name:0 for name in ['fallen_height', 'fallen_horizontal', 'outside_box_at_end', 'slipping']}
        for row in subset:
            rules = row[which]['rules']
            for key in ['fallen_height','fallen_horizontal','slipping']:
                counts[key] += rules[key]
            counts['outside_box_at_end'] += not rules['in_box_at_end']
        return counts
    summaries = {group:{which:rule_counts(which,[r for r in pairs if r['transition']==group])
                         for which in ['dagger2','dagger3']} for group in groups}
    result = dict(time=time.time(), runs=[f'{tag}_r2',f'{tag}_r3'],
        pool='Holosoma_Pick_29_latband60max_eval', episodes=100,
        transition_counts={name:len(ids) for name,ids in groups.items()},
        transitions=groups, per_rule_counts=summaries, per_reference=pairs,
        thresholds=dict(slip_max_mm=1000*grasp_success.SLIP_MAX,
                        slip_window_frames=grasp_success.SLIP_WIN,
                        box=request['box'], box_margin_m=grasp_success.BOX_MARGIN),
        interpretation='Metrics and logged termination reasons only. Rules may overlap. No visual behavior judgment; user review required.')
    output = root/'eval_videos/pick200_r23_case_comparison_20261101'
    output.mkdir(exist_ok=True)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(output/'results.json'),
                         transition_counts=result['transition_counts'],
                         per_rule_counts=summaries)))


if __name__ == '__main__':
    run()
