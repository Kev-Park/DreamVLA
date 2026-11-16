"""Bounded CPU checks of the original SMPL converter and HumanML data loader.

Run in the existing OmniControl environment. This does not use the contributed
XYZ-only adapter, sample diffusion, train, render, install, or edit input assets.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2]/'TrajGen'
    assets = args.assets.resolve(strict=True)
    os.chdir(str(assets))
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    inputs = [assets/p for p in ('body_models/smpl/SMPL_NEUTRAL.pkl',
        'body_models/smpl/J_regressor_extra.npy', 'body_models/smpl/kintree_table.pkl',
        'dataset/humanml_opt.txt', 'dataset/HumanML3D/test.txt')]
    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    before = hashes()
    report = dict(completed=False, scope=__doc__, torch=torch.__version__, numpy=np.__version__,
                  device='cpu', input_sha256=before, smpl={}, loaders={})
    arrays = {}
    try:
        try:
            from model.rotation2xyz import Rotation2xyz
            from utils import rotation_conversions as rotations
            converter = Rotation2xyz(device=torch.device('cpu'))
            vector = torch.zeros(1, 4, 24, 3)
            vector[0, 1, 18, 0] = .15
            vector[0, 3, 20, 1] = -.12
            matrices = rotations.axis_angle_to_matrix(vector)
            representations = dict(rotvec=vector, rotmat=matrices.flatten(-2),
                rotquat=rotations.matrix_to_quaternion(matrices),
                rot6d=rotations.matrix_to_rotation_6d(matrices))
            mask = torch.tensor([[True, True, False, True]])
            for kind, value in representations.items():
                x = value.permute(0, 2, 3, 1).contiguous()
                with torch.no_grad():
                    y = converter(x, mask=mask, pose_rep=kind, translation=False,
                                  glob=True, jointstype='smpl', vertstrans=False)
                assert y.shape == (1, 24, 3, 4) and torch.isfinite(y).all()
                assert (y[..., 2] == 0).all()
                arrays[kind] = y.numpy()
            errors = {kind: float(np.max(np.abs(value-arrays['rotvec']))) for kind, value in arrays.items()}
            x = representations['rotvec'].permute(0, 2, 3, 1).contiguous()
            with torch.no_grad():
                for joints in ('a2m', 'a2mpl', 'vibe', 'vertices'):
                    y = converter(x, mask=mask, pose_rep='rotvec', translation=False,
                                  glob=True, jointstype=joints, vertstrans=False)
                    assert y.shape[0] == 1 and y.shape[-2:] == (3, 4) and torch.isfinite(y).all()
                    arrays[joints] = y.numpy()
                trans = torch.zeros(1, 1, 3, 4)
                trans[0, 0, 0] = torch.tensor([0., .1, 0., .3])
                y = converter(torch.cat((x, trans), dim=1), mask=mask, pose_rep='rotvec',
                              translation=True, glob=True, jointstype='smpl', vertstrans=True)
                translation_error = float((y-torch.from_numpy(arrays['rotvec'])-trans).abs().max())
            report['smpl'] = dict(completed=True, representation_errors=errors,
                output_shapes={k: list(v.shape) for k, v in arrays.items()},
                translation_error=translation_error, preservation_pass=max(errors.values()) <= 1e-5 and translation_error <= 1e-6)
        except Exception:
            report['smpl'] = dict(completed=False, error=traceback.format_exc())
        for mode in ('train', 'eval', 'gt', 'text_only'):
            iterator = None
            try:
                from data_loaders.get_data import get_dataset_loader
                loader = get_dataset_loader(name='humanml', batch_size=1, num_frames=196,
                    split='test', hml_mode=mode, control_joint=0, density=100)
                iterator = iter(loader)
                batch = next(iterator)
                if mode == 'gt':
                    tensors = [x for x in batch if torch.is_tensor(x)]
                    assert tensors and all(torch.isfinite(x).all() for x in tensors)
                    result = dict(tensor_shapes=[list(x.shape) for x in tensors])
                else:
                    motion, cond = batch
                    assert torch.isfinite(motion).all() and len(cond['y']['text']) == 1
                    result = dict(motion_shape=list(motion.shape),
                        lengths=cond['y']['lengths'].tolist(), conditioning_keys=sorted(cond['y']))
                report['loaders'][mode] = dict(completed=True, dataset_length=len(loader.dataset),
                    original_num_workers=loader.num_workers, **result)
            except Exception:
                report['loaders'][mode] = dict(completed=False, error=traceback.format_exc())
            finally:
                if iterator is not None:
                    iterator._shutdown_workers()
        report['input_files_unchanged'] = before == hashes()
        report['completed'] = (report['smpl'].get('preservation_pass', False)
            and all(v.get('completed') for v in report['loaders'].values()) and report['input_files_unchanged'])
        np.savez(str(output/'outputs.npz'), **arrays)
    finally:
        (output/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    if not report['completed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
