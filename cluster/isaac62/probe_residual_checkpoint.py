"""Trained residual checkpoint gate; not a task-performance evaluation.

Run the same checkpoint under both runtimes, then compare the saved arrays.
Activation is explicit because a state_dict does not encode activation functions.
CUDA requires a separately reviewed, single visible GPU allocation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--activation', choices=['elu', 'relu', 'selu', 'tanh'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    args = parser.parse_args()
    if args.device == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Launch with CUDA_VISIBLE_DEVICES empty for this CPU-only gate')
    if args.output.exists():
        parser.error('Output already exists; inspect the existing receipt first')

    import numpy as np
    import torch
    from tensordict import TensorDict

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count() != 1:
            parser.error('CUDA gate requires exactly one explicitly visible reviewed GPU')
        torch.cuda.set_device(device)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'Training/scripts/reinforcement_learning/rsl_rl'))
    from vla_sonic.adapter_actor_critic import AdapterActorCritic

    checkpoint = args.checkpoint.resolve(strict=True)
    digest = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = saved['model_state_dict']

    def layers(prefix):
        selected = [(int(match.group(1)), value) for key, value in state.items()
                    if (match := re.fullmatch(prefix+r'\.(\d+)\.weight', key))]
        selected.sort(key=lambda row: row[0])
        if not selected or any(value.ndim != 2 for _, value in selected):
            raise ValueError('Checkpoint is not a supported stock MLP: '+prefix)
        return [value for _, value in selected]

    actor, critic = layers('actor'), layers('critic')
    # Formula-defined inputs avoid differences in random-number generators.
    def inputs(width):
        index = np.arange(8*width, dtype=np.float64).reshape(8, width)
        return torch.from_numpy((np.sin(index*.013)*.5).astype(np.float32)).to(device)

    observations = TensorDict({'policy': inputs(actor[0].shape[1]),
                               'critic': inputs(critic[0].shape[1])}, batch_size=[8])
    model = AdapterActorCritic(
        observations, {'policy': ['policy'], 'critic': ['critic']}, actor[-1].shape[0],
        actor_hidden_dims=[value.shape[0] for value in actor[:-1]],
        critic_hidden_dims=[value.shape[0] for value in critic[:-1]],
        activation=args.activation,
        actor_obs_normalization=any(key.startswith('actor_obs_normalizer.') for key in state),
        critic_obs_normalization=any(key.startswith('critic_obs_normalizer.') for key in state),
        noise_std_type='log' if 'log_std' in state else 'scalar',
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    assert all(parameter.device == device for parameter in model.parameters())
    with torch.inference_mode():
        actions = model.act_inference(observations)
        values = model.evaluate(observations)
    for tensor in (actions, values):
        assert tensor.device == device
        if not torch.isfinite(tensor).all():
            raise ValueError('Nonfinite checkpoint predictions')

    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output/'predictions.npz', policy=observations['policy'].cpu().numpy(),
             critic=observations['critic'].cpu().numpy(), actions=actions.cpu().numpy(), values=values.cpu().numpy())
    report = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest.hexdigest(),
                  iteration=saved.get('iter'), activation=args.activation,
                  python=sys.version, packages={name: importlib.metadata.version(name)
                                               for name in ['torch', 'numpy', 'rsl-rl-lib', 'tensordict']},
                  strict_load=True, gpu_used=device.type == 'cuda', device=str(device), task_evaluated=False,
                  cuda_runtime=torch.version.cuda,
                  input_scope='Eight fixed synthetic observations; no claim about task behavior',
                  actor_input_width=actor[0].shape[1], actions=list(actions.shape),
                  critic_input_width=critic[0].shape[1], values=list(values.shape))
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
