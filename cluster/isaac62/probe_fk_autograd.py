"""CPU FK derivative contract for 27/29-DOF historical reference optimizers.

Exercises all links and selected links with tensor/dictionary joint inputs.
This is component coverage, not execution of interactive legacy generators.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--urdf-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import numpy as np
    import torch
    import pytorch_kinematics as pk
    torch.set_num_threads(1)
    sys.path.insert(0, str(args.source_root / 'TrajGen/holosoma_adapters'))
    from kinematics_autograd import autograd_forward_kinematics
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(started=time.time(), completed=False, scope=__doc__, cases=[],
                  packages={n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'pytorch-kinematics')})
    arrays = {}
    for dof in (27, 29):
        chain = pk.build_chain_from_urdf((args.urdf_dir / f'g1_{dof}dof.urdf').read_bytes()).to(dtype=torch.float32)
        names = chain.get_joint_parameter_names()
        assert len(names) == dof
        forward = autograd_forward_kinematics(chain)
        for batch in (1, 7):
            for layout in ('tensor', 'dict'):
                for selected in (False, True):
                    label = f'dof{dof}_batch{batch}_{layout}_' + ('selected' if selected else 'all')
                    q = ((torch.arange(batch * dof).reshape(batch, dof) % 17 - 8) / 16.).requires_grad_(True)
                    inputs = {name: q[:, i] for i, name in enumerate(names)} if layout == 'dict' else q
                    indices = chain.get_frame_indices('torso_link', 'right_rubber_hand', 'left_ankle_roll_link') if selected else None
                    fk = forward(inputs, frame_indices=indices)
                    order = list(fk)
                    matrices = torch.stack([value.get_matrix() for value in fk.values()])
                    weights = ((torch.arange(matrices.numel()).reshape(matrices.shape) % 7 + 1) / 8.)
                    cost = (matrices * weights).sum() + (matrices.square()).sum() / 8.
                    cost.backward()
                    assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.norm() > 0
                    # Full chain name order and explicit selection order are contracts.
                    if selected:
                        assert order == ['torso_link', 'right_rubber_hand', 'left_ankle_roll_link']
                    for name, value in [('input', q), ('matrices', matrices), ('cost', cost), ('gradient', q.grad)]:
                        arrays[label + '_' + name] = value.detach().numpy().copy()
                    report['cases'].append(dict(label=label, links=order, gradient_norm=float(q.grad.norm())))
    np.savez(args.output / 'fk.npz', **arrays)
    report.update(completed=True, arrays=len(arrays), finished=time.time())
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
