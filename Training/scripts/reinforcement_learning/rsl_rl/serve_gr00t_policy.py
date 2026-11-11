"""Serve an unchanged GR00T checkpoint from its existing model environment.

Run on the same cluster host as Isaac 6.2, after CPU/GPU capacity review. This
does not start a simulator or submit a job. The caller owns service shutdown.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--ready-file', type=Path, required=True)
    parser.add_argument('--embodiment-tag', default='unitree_g1_sonic')
    args = parser.parse_args()
    ready_file = args.ready_file.expanduser().resolve()
    if ready_file.exists():
        parser.error('Refusing to overwrite an existing service descriptor')
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    import torch
    if torch.device(args.device).type == 'cuda':
        torch.cuda.set_device(args.device)
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from vla_sonic.gr00t_service import PolicyService

    policy = Gr00tPolicy(embodiment_tag=args.embodiment_tag, model_path=str(checkpoint), device=args.device)
    service = PolicyService(policy, checkpoint, args.device, args.embodiment_tag)

    def ready(descriptor):
        ready_file.parent.mkdir(parents=True, exist_ok=True)
        # The token stays in an owner-only descriptor, never in a shell command.
        fd = os.open(ready_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(descriptor, stream, indent=2)
        print(f'GR00T inference ready; descriptor: {ready_file}', flush=True)

    service.serve(ready)


if __name__ == '__main__':
    main()
