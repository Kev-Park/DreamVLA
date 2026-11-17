"""Render the evaluated experts on a common reference selected by recorded held metrics."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback


def main():
    base = Path.home()/'kevin'
    root = Path(__file__).resolve().parents[2]
    output = base/'eval_videos/isaac62_expert_parity_20261116_v2'
    evaluation = json.loads((output/'receipt.json').read_text())
    assert evaluation['completed'], 'Complete the production comparisons first'
    videos = output/'videos'
    videos.mkdir(exist_ok=False)
    os.sched_setaffinity(0, {0, 1, 2, 3})
    slots = [sys.executable, str(root/'cluster/gpu_slots.py')]
    owner = str(os.getpid())
    receipt = dict(started=time.time(), pid=os.getpid(), completed=False, runs=[])
    reserved = False

    def save():
        (videos/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')

    try:
        result = subprocess.run([*slots, 'reserve', '--owner', owner, '--count', '1',
                                 '--mb', '9500', '--gpu', '0'], check=True,
                                capture_output=True, text=True, timeout=20)
        assert result.stdout.strip() == '0'
        reserved = True
        for task in ['static', 'walk']:
            runs = [r for r in evaluation['runs'] if r['name'].startswith(task+'_')]
            rankings = []
            for r in runs:
                text = (output/r['name']/'eval.log').read_text()
                rows = re.findall(r'^\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(pick_\d+\.pkl)\s*$', text, re.M)
                rankings.append({x[6]: dict(mid=int(x[0]), episodes=int(x[1]), held=float(x[3])) for x in rows})
            common = set(rankings[0]) & set(rankings[1])
            assert common, 'No paired evaluation reference names'
            filename = max(sorted(common), key=lambda n: (min(d[n]['held'] for d in rankings), sum(d[n]['episodes'] for d in rankings)))
            for run, rank in zip(runs, rankings):
                name = run['name']
                command = run['command']
                env = {k:v for k,v in os.environ.items() if not k.startswith('HS_')}
                env.update(CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_PLATFORMS='cpu',
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           PXR_WORK_THREAD_LIMIT='2', PYTHONUNBUFFERED='1', HYBRID_RARM='0',
                           PYTHONPATH=run['pythonpath'])
                env.update({k:v for k,v in run['hs_settings'].items() if not k.startswith('HS_EVAL_')})
                # Current presentation plumbing, with the evaluated SDK and task package.
                cmd = [command[0], str(root/'cluster/isaac62/run_expert_entrypoint.py'),
                       '--entrypoint', str(root/'Training/scripts/reinforcement_learning/rsl_rl/play_sonic_adapter.py'),
                       '--receipt', str(videos/(name+'_entrypoint.json')), '--', '--headless',
                       '--device', 'cuda:0', '--num_envs', '1', '--seed', '0',
                       '--task', 'Isaac-Motion-Tracking-Pick-HOI-v0', '--waist-dof', '29',
                       '--keep-terms', '--no-overlay', '--force-motion-id', str(rank[filename]['mid']),
                       '--video_length', '1000' if task == 'walk' else '650',
                       '--name', str(videos/(name+'_m'+filename[5:-4]+'.mp4')),
                       '--traj-dump', str(videos/(name+'.npz'))]
                for flag in ['--sonic-pt', '--residual-transform', '--residual-scale', '--ref-motions-path', '--path']:
                    cmd += [flag, command[command.index(flag)+1]]
                row = dict(name=name, started=time.time(), command=cmd, reference=filename,
                           selection='Highest minimum recorded held rate across both SDKs; no visual judgment',
                           per_reference_eval=rank[filename])
                receipt['runs'].append(row)
                save()
                with (videos/(name+'.log')).open('w') as stream:
                    result = subprocess.run(cmd, cwd=base/'DreamVLA/Training', env=env,
                                            stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
                row.update(finished=time.time(), returncode=result.returncode,
                           entrypoint=json.loads((videos/(name+'_entrypoint.json')).read_text()))
                save()
                assert result.returncode == 0 and row['entrypoint']['completed'], name
        receipt['completed'] = True
    except BaseException:
        receipt['error'] = traceback.format_exc()
        raise
    finally:
        if reserved:
            subprocess.run([*slots, 'release', '--owner', owner, '--gpu', '0'], check=True, timeout=20)
        receipt.update(finished=time.time(), reservation_released=reserved)
        save()


if __name__ == '__main__':
    main()
