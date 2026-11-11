"""Short real SONIC-adapter optimizer/save/resume gate on a reviewed Linux GPU.

This invokes the actual trainer twice. The tiny rollout/optimizer budgets are
smoke-test settings, not a scientific training run or performance comparison.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    if os.name != 'posix' or sys.version_info[:2] != (3, 12):
        raise SystemExit('Run on the reviewed Linux host in the isolated Python 3.12 environment')
    if importlib.metadata.version('isaacsim') != '6.2.0.0':
        raise SystemExit('This gate requires Isaac Sim 6.2')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--task', default='Isaac-Motion-Tracking-Pick-BinaryFingers-v0')
    parser.add_argument('--ref-motions-path', type=Path, required=True)
    parser.add_argument('--sonic-pt', type=Path, required=True)
    parser.add_argument('--asset-cwd', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('GPU must be a nonnegative physical-device index')
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ('ref_motions_path', 'sonic_pt', 'asset_cwd'):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=True))
    root = Path(__file__).resolve().parents[2]
    trainer = root/'Training/scripts/reinforcement_learning/rsl_rl/train_sonic_adapter.py'
    slots = [sys.executable, str(root/'cluster/gpu_slots.py')]
    owner = str(os.getpid())
    receipt = dict(started=time.time(), completed=False, gpu=args.gpu,
                   task=args.task, scope='Two smoke updates, save, then one resumed update',
                   scientific_training_run=False, phases=[])
    reserved = False
    try:
        result = subprocess.run([*slots, 'reserve', '--owner', owner, '--count', '1',
                                 '--mb', '16000', '--gpu', str(args.gpu)],
                                capture_output=True, text=True, check=True, timeout=20)
        if result.stdout.strip() != str(args.gpu):
            raise RuntimeError('Reviewed GPU reservation unavailable; trainer not launched')
        reserved = True
        import torch

        experiment = str(output/'smoke')
        run_root = Path(experiment+'_sonic_adapter')
        common = [sys.executable, str(trainer), '--headless', '--device', f'cuda:{args.gpu}',
                  '--num_envs', '4', '--seed', '17', '--logger', 'tensorboard',
                  '--task', args.task, '--sonic-pt', str(args.sonic_pt),
                  '--ref-motions-path', str(args.ref_motions_path),
                  '--waist-dof', '29', '--skip-start-frames', '0']
        overrides = [f'agent.device=cuda:{args.gpu}', f'agent.experiment_name={experiment}',
                     'agent.num_steps_per_env=16', 'agent.algorithm.num_mini_batches=1',
                     'agent.algorithm.num_learning_epochs=1', 'agent.save_interval=1']

        def execute(phase, iterations, extra):
            log = output/(phase+'.log')
            command = [*common, '--max_iterations', str(iterations), '--run_name', phase,
                       *extra, *overrides, f'hydra.run.dir={output/phase}']
            with log.open('w') as stream:
                result = subprocess.run(command, cwd=args.asset_cwd, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=1200)
            receipt['phases'].append(dict(phase=phase, command=command,
                                          returncode=result.returncode, log=str(log)))
            if result.returncode:
                raise RuntimeError(f'{phase} failed; inspect {log}')
            candidates = list(run_root.glob('*_'+phase+'/model_*.pt'))
            if not candidates:
                raise RuntimeError('Trainer produced no checkpoint: '+phase)
            return max(candidates, key=lambda p: int(p.stem.split('_')[-1]))

        first = execute('initial', 2, [])
        before = torch.load(first, map_location='cpu', weights_only=False)
        if int(before['iter']) != 1:
            raise RuntimeError('Two fresh updates did not produce zero-based iteration1')
        second = execute('resumed', 3, ['--resume', '--load_run', first.parent.name,
                                      '--checkpoint', first.name])
        after = torch.load(second, map_location='cpu', weights_only=False)
        if int(after['iter']) != 2:
            raise RuntimeError('Resume did not reach exactly three total updates')
        if 'RESUME: 2 completed updates, target 3' not in (output/'resumed.log').read_text():
            raise RuntimeError('Actual trainer did not report restoring two completed updates')

        def optimizer_steps(checkpoint):
            state = checkpoint['optimizer_state_dict']['state']
            if not state:
                raise RuntimeError('Empty optimizer state')
            return sorted(float(item['step']) for item in state.values())

        before_steps, after_steps = optimizer_steps(before), optimizer_steps(after)
        if len(before_steps) != len(after_steps) or not all(
                b > a for a, b in zip(before_steps, after_steps)):
            raise RuntimeError('Optimizer steps did not continue from restored state')
        changed = []
        for name, value in after['model_state_dict'].items():
            if not torch.isfinite(value).all():
                raise RuntimeError('Nonfinite resumed model state: '+name)
            if not torch.equal(value, before['model_state_dict'][name]):
                changed.append(name)
        if not changed:
            raise RuntimeError('Resumed update changed no model parameters')
        receipt.update(completed=True, initial_checkpoint=str(first),
                       resumed_checkpoint=str(second), initial_iteration=int(before['iter']),
                       resumed_iteration=int(after['iter']), optimizer_steps_before=before_steps,
                       optimizer_steps_after=after_steps, changed_model_tensors=changed,
                       finished=time.time())
    except BaseException:
        receipt.update(completed=False, error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        try:
            if reserved:
                subprocess.run([*slots, 'release', '--owner', owner, '--gpu', str(args.gpu)],
                               check=True, timeout=20)
                receipt['reservation_released'] = True
        finally:
            (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
