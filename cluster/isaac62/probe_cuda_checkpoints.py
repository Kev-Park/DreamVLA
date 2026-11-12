"""Run frozen checkpoint CUDA gates sequentially within one reviewed allocation.

No Isaac/Vulkan imports, training, checkpoint mutation or task-success claims.
The caller reviews aggregate account resources before launch and after startup.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--physical-gpu', type=int, required=True)
    args = parser.parse_args()
    if os.name != 'posix' or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        parser.error('Run on the reviewed Linux host with one GPU explicitly visible')
    cases = json.loads(args.manifest.read_text())['cases']
    labels = [case['label'] for case in cases]
    if not cases or len(set(labels)) != len(labels) or any(
            not re.fullmatch(r'[A-Za-z0-9_-]+', label) for label in labels):
        parser.error('Manifest needs distinct safe labels and at least one case')
    if any(case['kind'] not in ('residual', 'sonic') for case in cases):
        parser.error('Unsupported checkpoint kind')
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    slots = [sys.executable, str(here.parent/'gpu_slots.py')]
    owner = str(os.getpid())
    receipt = dict(started=time.time(), completed=False, reservation_owner=os.getpid(),
                   physical_gpu=args.physical_gpu, cuda_visibility=os.environ['CUDA_VISIBLE_DEVICES'],
                   scope='Frozen model execution on fixed synthetic inputs only', cases=[])
    reserved = False
    try:
        r = subprocess.run([*slots, 'reserve', '--owner', owner, '--count', '1',
                            '--mb', '8000', '--gpu', str(args.physical_gpu)],
                           capture_output=True, text=True, check=True, timeout=20)
        if r.stdout.strip() != str(args.physical_gpu):
            raise RuntimeError('Reviewed GPU reservation unavailable')
        reserved = True
        for case in cases:
            kind, label = case['kind'], case['label']
            checkpoint = Path(case['checkpoint']).expanduser().resolve(strict=True)
            script = here/('probe_residual_checkpoint.py' if kind == 'residual'
                            else 'probe_sonic_checkpoint.py')
            command = [sys.executable, str(script), '--checkpoint', str(checkpoint),
                       '--device', 'cuda:0', '--output', str(output/label)]
            command += (['--activation', case['activation']] if kind == 'residual'
                        else ['--encoder', case['encoder']])
            with (output/(label+'.log')).open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            receipt['cases'].append(dict(label=label, kind=kind, command=command,
                                          returncode=result.returncode))
            if result.returncode:
                raise RuntimeError('Checkpoint probe failed; inspect '+label+'.log')
        receipt.update(completed=True, finished=time.time())
    except BaseException:
        receipt.update(completed=False, error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        try:
            if reserved:
                subprocess.run([*slots, 'release', '--owner', owner, '--gpu', str(args.physical_gpu)],
                               check=True, timeout=20)
                receipt['reservation_released'] = True
        except BaseException:
            receipt.update(completed=False, cleanup_error=traceback.format_exc())
            raise
        finally:
            (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
