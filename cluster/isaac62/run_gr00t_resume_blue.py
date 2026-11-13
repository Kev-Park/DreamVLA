"""Run the bounded GR00T save/resume fixture on one manually reviewed GPU.

Run only after full-account CPU/GPU and pending-allocation review. This launcher
reserves its own memory and never throttles or changes another account job.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cuda-home', type=Path)
    args = parser.parse_args()
    assert os.name == 'posix' and args.gpu >= 0
    base = Path.home()/'kevin'
    output = args.output.expanduser().resolve()
    assert output.is_relative_to(base.resolve())
    output.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {0, 3})
    root = Path(__file__).resolve().parents[2]
    python = base/'Isaac-GR00T/.venv/bin/python'
    assert python.is_file()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', WANDB_MODE='offline',
               HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0', PYTHONUNBUFFERED='1')
    if args.cuda_home:
        toolkit = args.cuda_home.expanduser().resolve(strict=True)
        assert (toolkit/'bin/nvcc').is_file()
        env['CUDA_HOME'] = str(toolkit)
    slots = [str(python), str(root/'cluster/gpu_slots.py')]
    owner = str(os.getpid())
    receipt = dict(started=time.time(), completed=False, pid=os.getpid(),
                   physical_gpu=args.gpu, affinity=sorted(os.sched_getaffinity(0)),
                   dataset=str(args.dataset), checkpoint=str(args.checkpoint),
                   phases=[], reservation_released=False,
                   scope='Two full GR00T Adam updates with save and fresh-process resume; no research run')
    reserved = False
    try:
        result = subprocess.run([*slots, 'reserve', '--owner', owner, '--count', '1',
                                 '--mb', '45000', '--gpu', str(args.gpu)],
                                env=env, capture_output=True, text=True, check=True, timeout=20)
        assert result.stdout.strip() == str(args.gpu), result.stdout
        reserved = True
        for phase in ('initial', 'resume'):
            command = [str(python), str(root/'cluster/isaac62/probe_gr00t_train_resume.py'),
                       '--checkpoint', str(args.checkpoint.expanduser().resolve(strict=True)),
                       '--dataset', str(args.dataset.expanduser().resolve(strict=True)),
                       '--output', str(output/'run'), '--phase', phase]
            with (output/(phase+'.log')).open('w') as stream:
                # Explicit execution bound for this two-update validation fixture.
                result = subprocess.run(command, cwd=base/'Isaac-GR00T', env=env,
                                        stdout=stream, stderr=subprocess.STDOUT, timeout=1200)
            row = dict(phase=phase, returncode=result.returncode)
            receipt['phases'].append(row)
            (output/'progress.json').write_text(json.dumps(receipt, indent=2)+'\n')
            assert result.returncode == 0, phase+' failed; inspect phase log'
            report = json.loads((output/'run'/(phase+'.json')).read_text())
            assert report['completed'], phase
            row.update(global_step=report['global_step'], optimizer_step=report['optimizer_step'])
        receipt['completed'] = True
    except BaseException:
        receipt['error'] = traceback.format_exc()
        raise
    finally:
        try:
            if reserved:
                subprocess.run([*slots, 'release', '--owner', owner, '--gpu', str(args.gpu)],
                               env=env, check=True, timeout=20)
                receipt['reservation_released'] = True
        finally:
            receipt['finished'] = time.time()
            (output/'terminal.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
