"""Compare actual human-motion export in the existing legacy/target profiles."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--baseline-python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('Run CPU-only on the reviewed cluster')
    source = args.results.expanduser().resolve(strict=True)
    baseline = args.baseline_python.expanduser().absolute()
    assert baseline.is_file()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    exporter = Path(__file__).resolve().parents[2] / 'TrajGen/holosoma_adapters/export_to_holosoma.py'
    receipt = dict(started=time.time(), completed=False, input_sha256=digest(source),
                   input=str(source), exporter_sha256=digest(exporter), motion_id=0,
                   scope='Actual Adapter A export; no upstream sampling or holosoma optimization',
                   phases=[], atol=1e-6, rtol=0.)
    try:
        for label, python in [('legacy', baseline), ('target', Path(sys.executable).absolute())]:
            destination = output / label
            command = [str(python), str(exporter), '0', str(source), str(destination)]
            with (output / (label + '.log')).open('w') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                        check=False, timeout=90)
            receipt['phases'].append(dict(label=label, command=command, returncode=result.returncode))
            if result.returncode:
                raise RuntimeError('Exporter failed: ' + label)
        import numpy as np
        with np.load(output/'legacy/pick_0.npz', allow_pickle=False) as legacy, \
                np.load(output/'target/pick_0.npz', allow_pickle=False) as target:
            assert set(legacy.files) == set(target.files) == {'global_joint_positions', 'height', 'object_poses'}
            errors = {}
            for key in legacy.files:
                left, right = legacy[key], target[key]
                assert left.shape == right.shape and left.dtype == right.dtype, key
                assert np.isfinite(left).all() and np.isfinite(right).all(), key
                errors[key] = float(np.max(np.abs(left-right)))
                np.testing.assert_allclose(right, left, atol=1e-6, rtol=0)
            points, poses = target['global_joint_positions'], target['object_poses']
            assert points.ndim == 3 and points.shape[1:] == (22, 3)
            assert poses.shape == (len(points), 7) and len(points) > 1
            assert float(target['height']) > 0
            np.testing.assert_array_equal(poses[:, :4], np.tile([1., 0., 0., 0.], (len(points), 1)))
            receipt.update(frames=len(points), joints=22, max_abs_errors=errors)
        receipt['input_unchanged'] = digest(source) == receipt['input_sha256']
        assert receipt['input_unchanged']
        receipt['completed'] = True
    except BaseException as error:
        receipt['error'] = repr(error)
        raise
    finally:
        receipt['finished'] = time.time()
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
