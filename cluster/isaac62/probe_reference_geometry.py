"""CPU-only saved-reference/FK/stitch geometry gate; never rewrites references."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import xml.etree.ElementTree as ET


def xml_fk(urdf, joint_names, joints, link):
    """Independent URDF origin/axis traversal, separate from pytorch_kinematics."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    root = ET.parse(urdf).getroot()
    incoming = {j.find('child').get('link'): j for j in root.findall('joint')}
    path = []
    while link in incoming:
        j = incoming[link]
        path.append(j)
        link = j.find('parent').get('link')
    transform = np.broadcast_to(np.eye(4), (len(joints), 4, 4)).copy()
    for j in reversed(path):
        origin = j.find('origin')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        fixed = np.eye(4)
        fixed[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
        fixed[:3, 3] = xyz
        transform = transform @ fixed
        if j.get('type') == 'fixed':
            continue
        assert j.get('type') in ('revolute', 'continuous'), j.attrib
        axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
        motion = np.broadcast_to(np.eye(4), transform.shape).copy()
        motion[:, :3, :3] = Rotation.from_rotvec(joints[:, joint_names.index(j.get('name')), None] * axis).as_matrix()
        transform = transform @ motion
    return transform


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urdf', type=Path, required=True)
    p.add_argument('--reference', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only probe requires hidden CUDA devices'
    import numpy as np
    import torch
    import jax
    import jaxlie
    import pytorch_kinematics as pk
    torch.set_num_threads(1)
    assert jax.default_backend() == 'cpu'
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'TrajGen/holosoma_adapters'))
    import stitch_static_approach as st
    import reconstruct_own_motion_stitches as build
    chain = pk.build_chain_from_urdf(args.urdf.read_bytes()).to(dtype=torch.float32)
    names = chain.get_joint_parameter_names()
    report = dict(started=time.time(), completed=False, python=sys.version, torch=torch.__version__,
                  numpy=np.__version__, jax=jax.__version__, device='cpu', references=[])
    arrays = {}
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        for i, path in enumerate(args.reference):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            motion = st.load(str(path), chain)
            assert motion['J'].shape[1] == len(names) == 29
            assert 0 <= motion['g'] < len(motion['J'])
            assert all(np.isfinite(v).all() for v in motion.values())
            np.testing.assert_allclose(np.linalg.norm(motion['Q'], axis=1), 1, atol=1e-5)
            selected = np.unique(np.linspace(0, len(motion['J'])-1, 9, dtype=int))
            joints = motion['J'][selected]
            fk = chain.forward_kinematics(torch.tensor(joints, dtype=torch.float32))
            max_error = 0.
            for link in (st.HAND, 'left_ankle_roll_link', 'right_ankle_roll_link', 'torso_link'):
                actual = fk[link].get_matrix().numpy()
                expected = xml_fk(args.urdf, names, joints, link)
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)
                max_error = max(max_error, float(np.max(np.abs(actual-expected))))
            # Exercise own-motion alignment without solving IK or modifying a saved clip.
            shift = build.alignment_shift(motion, motion)
            np.testing.assert_allclose(shift, np.zeros(3), atol=1e-12)
            moved = build.shifted_walk(motion, [.03, -.02, .01], [.1, -.05, 0])
            np.testing.assert_allclose(moved['P'] - motion['P'], np.broadcast_to([.1, -.05, 0], motion['P'].shape), atol=1e-12)
            assert st.final_footfall(moved) == st.final_footfall(motion)
            # Existing saved JAXLie types must survive decode/re-encode on Python 3.12.
            original = pickle.loads(path.read_bytes())
            restored = pickle.loads(pickle.dumps(original))
            np.testing.assert_array_equal(original['global_pose'].translation(), restored['global_pose'].translation())
            for pose in (original['global_pose'], restored['global_pose']):
                q = pose.rotation().wxyz
                q = q() if callable(q) else q
                np.testing.assert_array_equal(q, motion['Q'])
            for key in ('J', 'P', 'Q', 'O', 'hand', 'handR', 'la', 'ra'):
                arrays[f'{i}_{key}'] = motion[key]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            report['references'].append(dict(path=str(path), sha256=digest, frames=len(motion['J']),
                                             grab_frame=motion['g'], final_footfall=st.final_footfall(motion),
                                             independent_fk_max_abs_error=max_error))
        report['completed'] = True
        np.savez(args.output/'geometry.npz', **arrays)
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        report['finished'] = time.time()
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
