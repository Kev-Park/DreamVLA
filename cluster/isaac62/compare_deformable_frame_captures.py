"""Compare uniform affine diagnostic captures against the original native SDK.

Cooked topology differs between SDKs, so this compares element-mean tensors,
not unrelated individual tetrahedra. The capture independently asserts exact
kinematic target application. This is not a solver-trajectory parity test.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def rotation_matrices(quaternions):
    x, y, z, w = np.moveaxis(quaternions.reshape(-1, 4).astype(np.float64), -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], axis=-1).reshape(-1, 3, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old, new = np.load(args.legacy), np.load(args.target)
    rows = {}
    for kind in ('simulation', 'collision'):
        for mode in ('', 'stretch_', 'shear_', 'rigid_'):
            tag = 'nomat_default_'+kind+'_'+mode
            row = {}
            for diagnostic in ('deformation_gradients', 'stresses'):
                left = old[tag+diagnostic].astype(np.float64).reshape(-1, 3, 3).mean(0)
                right = new[tag+diagnostic].astype(np.float64).reshape(-1, 3, 3).mean(0)
                row[diagnostic+'_mean_max_error'] = float(np.abs(left-right).max())
            left = old[tag+'rotations'].reshape(-1, 4)
            right = new[tag+'rotations'].reshape(-1, 4)
            row['original_quaternion_norm_range'] = [float(np.linalg.norm(left, axis=-1).min()),
                                                     float(np.linalg.norm(left, axis=-1).max())]
            row['target_quaternion_norm_range'] = [float(np.linalg.norm(right, axis=-1).min()),
                                                   float(np.linalg.norm(right, axis=-1).max())]
            if kind == 'simulation':
                row['rotation_matrix_mean_max_error'] = float(np.abs(
                    rotation_matrices(left).mean(0)-rotation_matrices(right).mean(0)).max())
            rows[kind+'_'+(mode or 'initial')] = row
    # Retain existing gradient and stiffness-normalized stress bounds.
    tolerances = dict(gradient=1e-5, rotation=1e-5, stress_pa=50e6*2e-4)
    passed = all(row['deformation_gradients_mean_max_error'] <= tolerances['gradient']
                 and row['stresses_mean_max_error'] <= tolerances['stress_pa']
                 and row.get('rotation_matrix_mean_max_error', 0) <= tolerances['rotation']
                 for row in rows.values())
    result = dict(passed=passed, scope=__doc__, legacy=str(args.legacy), target=str(args.target),
                  tolerances=tolerances, comparisons=rows,
                  collision_quaternion_note='Original native quaternions are zero; no quaternion parity claim')
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    assert passed, 'Native transformed-body diagnostic comparison failed'


if __name__ == '__main__':
    main()
