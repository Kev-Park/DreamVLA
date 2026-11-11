"""Serve an unchanged GR00T checkpoint from its existing model environment.

Run on the same cluster host as Isaac 6.2, after CPU/GPU capacity review. This
does not start a simulator or submit a job. The caller owns service shutdown.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--ready-file', type=Path, required=True)
    parser.add_argument('--embodiment-tag', default='unitree_g1_sonic')
    args = parser.parse_args()
    ready_file = args.ready_file.expanduser().resolve()
    if ready_file.exists() or ready_file.is_symlink():
        parser.error('Refusing to overwrite an existing service descriptor')
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    import torch
    if torch.device(args.device).type == 'cuda':
        torch.cuda.set_device(args.device)
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from vla_sonic.gr00t_service import PolicyService, publish_descriptor

    policy = Gr00tPolicy(embodiment_tag=args.embodiment_tag, model_path=str(checkpoint), device=args.device)
    service = PolicyService(policy, checkpoint, args.device, args.embodiment_tag)

    def ready(descriptor):
        publish_descriptor(ready_file, descriptor)
        print(f'GR00T inference ready; descriptor: {ready_file}', flush=True)

    service.serve(ready)


if __name__ == '__main__':
    main()
