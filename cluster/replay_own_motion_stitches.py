"""Run headless kinematic replay checks for the aligned own-motion dataset.

No policy training, retries, deletion, compression, or success classification.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import watch_t20max_v2 as recipe

ROOT, REPO = recipe.ROOT, recipe.REPO
OUT = ROOT/'eval_videos/StitchOwn_v3'


def write(name, value):
    recipe.write(OUT/name, value)


def check_dump(path, count, steps):
    sys.path.insert(0, str(REPO/'TrajGen/holosoma_adapters'))
    import build_stitchmax_v2 as shared
    import torch
    chain, _ = shared.chain()
    with np.load(path, allow_pickle=False) as d:
        assert len(d['step']) == count*steps
        assert len(np.unique(d['mid'])) == count
        for key in ['hand_pos', 'hand_quat', 'root_pos', 'joint_pos', 'ref_joint_pos', 'ref_obj_pose']:
            assert np.isfinite(d[key]).all(), key
        joint_error = float(np.max(np.abs(d['joint_pos']-d['ref_joint_pos'])))
        root_error = float(np.max(np.linalg.norm(d['root_pos']-d['ref_root_pos'], axis=1)))
        assert joint_error < 1e-4 and root_error < 1e-4, (joint_error, root_error)
        # Isaac merges the fixed rubber-hand link; its parent wrist remains addressable.
        hand_link = 'right_wrist_yaw_link'
        hand_index = list(d['names']).index(hand_link)
        hand_error = 0.0
        joints, roots, quats, hands = d['joint_pos'], d['root_pos'], d['root_quat'], d['hand_pos'][:, hand_index]
        for lo in range(0, len(joints), 512):
            hi = lo+512
            fk = chain.forward_kinematics(torch.tensor(joints[lo:hi], dtype=torch.float32))
            local = fk[hand_link].get_matrix()[:, :3, 3].numpy()
            expected = np.einsum('nij,nj->ni', shared.stitcher.quat_to_R(quats[lo:hi]), local)+roots[lo:hi]
            hand_error = max(hand_error, float(np.linalg.norm(expected-hands[lo:hi], axis=1).max()))
        assert hand_error < .001, f'Isaac vs URDF hand FK discrepancy: {hand_error} m'
        return {'frames': len(d['step']), 'motions': count,
                'max_joint_replay_error_rad': joint_error, 'max_root_replay_error_m': root_error,
                'max_isaac_vs_urdf_hand_error_m': hand_error,
                'bytes': path.stat().st_size}


def main(mode):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:26])
    tasks = json.loads((OUT/'generated.json').read_text())
    if mode == 'smoke':
        tasks = [tasks[0]]
        ngpu = 1
    else:
        validation = json.loads((OUT/'numerical_validation.json').read_text())
        assert validation['count'] == len(tasks) == 500
        ngpu = 8
    run = OUT/('replay_'+mode)
    if run.exists():
        run = OUT/('replay_'+mode+'_'+str(time.time_ns()))
    run.mkdir(exist_ok=False)
    jobs = []
    state = {'phase': 'launching', 'mode': mode, 'started': time.time(),
             'training_authorized_to_start': False, 'jobs': []}
    try:
        for gpu in range(ngpu):
            assert recipe.gpu_free(gpu), f'GPU {gpu} occupied; no launch/retry'
            rows = sorted(tasks, key=lambda t: (t['split'],t['name']))[gpu::ngpu]
            pool = run/f'pool{gpu}'
            pool.mkdir()
            for t in rows:
                source = Path(t['output'])
                (pool/(t['split']+'_'+t['name'])).symlink_to(os.path.relpath(source, pool))
            recipe.write(run/f'manifest{gpu}.json', rows)
            steps = int(np.ceil(max(t['frames'] for t in rows)*.05/.02))+5
            dump = run/f'track{gpu}.npz'
            cmd = recipe.command('play_motion_montage.py', str(pool)) + [
                '--reference-playback','--no-video','--no-stability','--no-ref-overlay',
                '--num-motions',str(len(rows)), '--segment-steps',str(steps),
                '--dump-track',str(dump), '--name',str(run/f'unused{gpu}.mp4'),
                '--render-width','320','--render-height','240','--device',f'cuda:{gpu}']
            # Keep CUDA and Vulkan on the same physical index (Vulkan ignores CUDA masks).
            env = recipe.environment(gpu)
            env.pop('CUDA_VISIBLE_DEVICES', None)
            env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', JAX_PLATFORMS='cpu',
                       OMNI_KIT_ACCEPT_EULA='YES')
            log = run/f'replay{gpu}.log'
            with log.open('w') as f:
                proc = subprocess.Popen(cmd, cwd=REPO/'Training', env=env, stdout=f, stderr=subprocess.STDOUT)
            row = {'gpu':gpu,'pid':proc.pid,'log':str(log),'dump':str(dump),
                   'count':len(rows),'steps_per_motion':steps,'command':cmd,'status':'running'}
            jobs.append((proc,row))
            state['jobs'].append(row)
            write('replay_'+mode+'_status.json',state)
        state['phase'] = 'replaying'
        while True:
            active = False
            for proc,row in jobs:
                if row['status'] != 'running':
                    continue
                rc = proc.poll()
                if rc is None:
                    active = True
                    row['last_log'] = recipe.tail(Path(row['log']),2000)
                elif rc:
                    row.update(status='failed',returncode=rc,last_log=recipe.tail(Path(row['log']),4000))
                else:
                    try:
                        row.update(status='complete',validation=check_dump(Path(row['dump']),row['count'],row['steps_per_motion']))
                    except Exception as exc:
                        row.update(status='failed',error=repr(exc))
            state['updated'] = time.time()
            if not active:
                state['phase'] = 'failed' if any(r['status']=='failed' for _,r in jobs) else 'complete'
            write('replay_'+mode+'_status.json',state)
            if not active:
                break
            time.sleep(15)
        assert state['phase'] == 'complete', 'Replay failures: inspect status and logs'
    except BaseException as exc:
        state.update(phase='failed',error=repr(exc),updated=time.time())
        write('replay_'+mode+'_status.json',state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['smoke','full'])
    main(parser.parse_args().mode)
