"""Native SONIC encoder/decoder compatibility gate on saved weights.

Uses the project's existing loader without changing token conventions. Fixed
synthetic inputs test model execution, not control behavior or reference tracking.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--encoder', choices=['g1', 'teleop'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    args = parser.parse_args()
    if args.device == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Launch with CUDA_VISIBLE_DEVICES empty for this CPU-only gate')
    if not args.checkpoint.is_file():
        parser.error('Select the exact checkpoint file, not a directory')
    if args.output.exists():
        parser.error('Output exists; inspect the prior receipt before proceeding')

    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count() != 1:
            parser.error('CUDA gate requires exactly one explicitly visible reviewed GPU')
        torch.cuda.set_device(device)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'Training/scripts/reinforcement_learning/rsl_rl'))
    from vla_sonic.sonic_pt import ENCODER_IN_DIMS, load_sonic_pt

    checkpoint = args.checkpoint.resolve(strict=True)
    digest = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    encoder, decoder = load_sonic_pt(str(checkpoint), str(device), encoder=args.encoder)
    assert encoder[-1].out_features == 64 and decoder[-1].out_features == 29
    assert not encoder.training and not decoder.training
    assert all(not parameter.requires_grad for module in (encoder, decoder)
               for parameter in module.parameters())

    def inputs(width):
        index = np.arange(4*width, dtype=np.float64).reshape(4, width)
        return torch.from_numpy((np.sin(index*.013)*.5).astype(np.float32)).to(device)

    encoder_input, decoder_input = inputs(ENCODER_IN_DIMS[args.encoder]), inputs(994)
    with torch.inference_mode():
        tokens, actions = encoder(encoder_input), decoder(decoder_input)
    if not torch.isfinite(tokens).all() or not torch.isfinite(actions).all():
        raise ValueError('Native SONIC produced nonfinite outputs')
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output/'predictions.npz', encoder_input=encoder_input.cpu().numpy(),
             decoder_input=decoder_input.cpu().numpy(), tokens=tokens.cpu().numpy(), actions=actions.cpu().numpy())
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest.hexdigest(),
                  python=sys.version, encoder=args.encoder,
                  packages={name: importlib.metadata.version(name) for name in ['torch', 'numpy']},
                  encoder_output=list(tokens.shape), decoder_output=list(actions.shape),
                  frozen=True, gpu_used=device.type == 'cuda', device=str(device), task_evaluated=False,
                  cuda_runtime=torch.version.cuda,
                  input_scope='Four fixed synthetic inputs per module; no FSQ/control/task claim')
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
