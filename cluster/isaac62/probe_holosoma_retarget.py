"""One original headless holosoma retarget in its approved existing environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retarget-root', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--retarget-python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.name == 'posix' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    source = args.input.expanduser().resolve(strict=True)
    root = args.retarget_root.expanduser().resolve(strict=True)
    python = args.retarget_python.expanduser().absolute()
    assert python.is_file()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output/'inputs'; inputs.mkdir()
    (inputs/'pick_0.npz').symlink_to(source)
    entrypoint = root/'examples/robot_retarget.py'
    command = [str(python), str(entrypoint), '--task-type', 'object_interaction',
               '--robot', 'g1', '--data-format', 'smplx', '--task-name', 'pick_0',
               '--data-path', str(inputs), '--save-dir', str(output/'raw'),
               '--task-config.object-name', 'mustard']
    receipt = dict(started=time.time(), completed=False, command=command,
                   input_sha256=sha(source), entrypoint_sha256=sha(entrypoint),
                   scope='Full original one-clip default SQP/foot-sticking retarget; no policy evaluation')
    try:
        env = os.environ.copy()
        env.pop('DISPLAY', None)
        env['PYTHONPATH'] = str(root.parent)
        with (output/'retarget.log').open('w') as stream:
            result = subprocess.run(command, cwd=root, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=400)
        receipt['returncode'] = result.returncode
        if result.returncode:
            raise RuntimeError('Original retargeter failed; inspect retarget.log')
        import numpy as np
        with np.load(source, allow_pickle=False) as original, \
                np.load(output/'raw/pick_0_original.npz', allow_pickle=False) as generated:
            qpos = generated['qpos']
            assert len(qpos) == len(original['global_joint_positions'])
            assert qpos.ndim == 2 and qpos.shape[1] == 43 and np.isfinite(qpos).all()
            np.testing.assert_allclose(np.linalg.norm(qpos[:, 3:7], axis=-1), 1, atol=1e-5, rtol=0)
            receipt.update(frames=len(qpos), qpos_columns=qpos.shape[1],
                           output_sha256=sha(output/'raw/pick_0_original.npz'))
        assert sha(source) == receipt['input_sha256']
        receipt.update(completed=True, input_unchanged=True)
    except BaseException as error:
        receipt['error'] = repr(error)
        raise
    finally:
        receipt['finished'] = time.time()
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
