"""Write a separate Isaac 6.2 USD copy of cooked legacy volume deformables.

Run in the approved Isaac environment on a cluster, never on the local desktop.
Surface/cloth, attachments, animated solver arrays and lossy material mappings
are rejected. The original USD and referenced layers are never edited.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    if args.destination.exists() or args.destination.is_symlink():
        parser.error('Destination already exists; choose a new file')
    if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        parser.error('Run on the reviewed cluster GPU with CUDA_VISIBLE_DEVICES=0')
    slots = [sys.executable, str(Path(__file__).resolve().parents[1] / 'gpu_slots.py')]
    reservation = subprocess.run([*slots, 'reserve', '--owner', str(os.getpid()),
                                  '--count', '1', '--mb', '8000', '--gpu', '0'],
                                 check=True, capture_output=True, text=True, timeout=20)
    if reservation.stdout.strip() != '0':
        raise RuntimeError('Requested GPU reservation unavailable')
    app = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, active_gpu=0, physics_gpu=0,
                                 multi_gpu=False, fast_shutdown=True))
        from legacy_volume_usd import convert_copy
        receipt = convert_copy(args.source, args.destination)
        print(json.dumps(receipt, indent=2), flush=True)
    finally:
        try:
            if app is not None:
                app.close()
        finally:
            subprocess.run([*slots, 'release', '--owner', str(os.getpid()), '--gpu', '0'],
                           check=True, timeout=20)


if __name__ == '__main__':
    main()
