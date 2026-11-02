"""Run the approved walking calibration fine-tune on an operator-reviewed GPU.

The command matches walk_gr00t_finetune.sbatch, including rolling recovery
checkpoints. This controller never retries training or launches evaluation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from gr00t_budget_chain import atomic, read
from walk_checkpoint_integrity import validate_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    base = Path.home() / 'kevin'
    tag = 'walk200_v3_histwindow65_20261029'
    work = base / 'dline' / tag
    run = tag + '_r3'
    dataset = base / 'datasets' / run / 'lerobot/ds'
    ready = read(work / 'dataset.ready.json')
    info = read(dataset / 'meta/info.json')
    assert info['total_episodes'] == ready['episodes'] == 530
    assert info['total_frames'] == ready['frames'] == 330664
    steps = (info['total_frames'] * 57 + 159) // 160
    assert steps == ready['steps'] == 117800
    assert not (base / 'checkpoints' / run).exists(), 'Reconcile an existing run'
    state_path = work / 'finetune_blue.workflow.json'
    assert not state_path.exists(), 'Never duplicate or silently restart training'
    log_path = work / 'finetune_blue.log'
    assert not log_path.exists(), 'Preserve existing training evidence'
    here = Path(__file__).resolve().parent
    slots = [str(base / 'wt/locomotion/.venv/bin/python'), str(here / 'gpu_slots.py')]
    reservation = subprocess.check_output(
        slots + ['reserve', '--owner', str(os.getpid()), '--count', '1',
                 '--mb', '40000', '--gpu', str(args.gpu)], text=True)
    assert list(map(int, reservation.split())) == [args.gpu]
    command = [str(base / 'Isaac-GR00T/.venv/bin/python'),
               'gr00t/experiment/launch_finetune.py',
               '--base-model-path', 'nvidia/GR00T-N1.7-3B',
               '--dataset-path', str(dataset), '--embodiment-tag', 'unitree_g1_sonic',
               '--num-gpus', '1', '--global-batch-size', '16',
               '--gradient-accumulation-steps', '1', '--learning-rate', '0.0001',
               '--dataloader-num-workers', '4', '--output-dir', str(base / 'checkpoints'),
               '--experiment-name', run, '--max-steps', str(steps),
               '--save-steps', '5000', '--save-total-limit', '1']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               XLA_PYTHON_CLIENT_PREALLOCATE='false', WANDB_MODE='offline',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    state = {'status': 'starting', 'pid': os.getpid(), 'tag': tag,
             'training_host': 'bluesclues', 'gpu': args.gpu,
             'cpu_mask': sorted(os.sched_getaffinity(0)), 'command': command,
             'steps': steps, 'dataset': str(dataset), 'started': time.time()}
    atomic(state_path, state)
    child = None
    try:
        with log_path.open('xb') as output:
            child = subprocess.Popen(command, cwd=base / 'Isaac-GR00T', env=env,
                                     stdout=output, stderr=subprocess.STDOUT)
            state.update(status='running', worker_pid=child.pid)
            atomic(state_path, state)
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    state.update(heartbeat=time.time(), log_bytes=log_path.stat().st_size,
                                 log_mtime=log_path.stat().st_mtime)
                    atomic(state_path, state)
        assert code == 0, ('Trainer failed; inspect preserved log', code)
        relative = f'checkpoints/{run}/checkpoint-{steps}'
        integrity = validate_checkpoint(base / relative, steps)
        result = {'completed': time.time(), 'steps': steps,
                  'checkpoint_relative': relative, 'training_host': 'bluesclues',
                  'evaluation_host': 'bluesclues', 'checkpoint_integrity': integrity}
        atomic(work / 'finetune.ready.json', result)
        state.update(status='complete', completed=time.time(), result=result)
        atomic(state_path, state)
    except BaseException as exc:
        state.update(status='stopped_for_review', error=repr(exc), heartbeat=time.time())
        atomic(state_path, state)
        raise
    finally:
        if child is None or child.poll() is not None:
            subprocess.run(slots + ['release', '--owner', str(os.getpid()),
                                   '--gpu', str(args.gpu)], check=True)


if __name__ == '__main__':
    main()
