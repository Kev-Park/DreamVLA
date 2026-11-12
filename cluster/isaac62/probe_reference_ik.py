"""Bounded CPU exercise of the real stitch IK optimizer on a saved reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import numpy as np
    import torch
    import jax
    import pytorch_kinematics as pk
    assert jax.default_backend() == 'cpu'
    torch.set_num_threads(1)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'TrajGen/holosoma_adapters'))
    import stitch_static_approach as st
    chain = pk.build_chain_from_urdf(args.urdf.read_bytes()).to(dtype=torch.float32)
    limits = [np.asarray(x) for x in chain.get_joint_limits()]
    digest = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    motion = st.load(str(args.reference), chain)
    frames = np.arange(max(0, motion['g']-1), min(len(motion['J']), motion['g']+2))
    assert len(frames) == 3
    joints = motion['J'][frames]
    indices = list(range(12, 15))+list(range(22, 29))
    q0 = np.clip(joints[:, indices]+.03, limits[0][indices], limits[1][indices])
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(started=time.time(), completed=False, reference=str(args.reference), sha256=digest,
                  frames=frames.tolist(), iterations=st.IK_ITERS, torch=torch.__version__, numpy=np.__version__)
    try:
        root_R = st.quat_to_R(motion['Q'][frames])
        q, position_error, angle_error, torso_error = st.solve_upper(
            chain, limits, joints, q0, motion['P'][frames], root_R,
            motion['hand'][frames], motion['handR'][frames], motion['torsoR'][frames])
        solved = joints.copy();solved[:, indices] = q
        assert np.isfinite(q).all()
        assert np.all(q >= limits[0][indices]-1e-6) and np.all(q <= limits[1][indices]+1e-6)
        # Recompute after the final optimizer update: solver's returned loss is pre-update.
        fk = chain.forward_kinematics(torch.tensor(solved, dtype=torch.float32))
        matrix = fk[st.HAND].get_matrix().double().numpy()
        actual = np.einsum('nij,nj->ni', root_R, matrix[:, :3, 3])+motion['P'][frames]
        residual = np.linalg.norm(actual-motion['hand'][frames], axis=1)
        assert np.max(residual) < .002, residual
        rotation = root_R @ matrix[:, :3, :3]
        difference = rotation @ np.swapaxes(motion['handR'][frames], 1, 2)
        degrees = np.degrees(np.arccos(np.clip((np.trace(difference, axis1=1, axis2=2)-1)/2, -1, 1)))
        assert np.max(degrees) < 1, degrees
        assert hashlib.sha256(args.reference.read_bytes()).hexdigest() == digest
        np.savez(args.output/'ik.npz', frames=frames, input_joints=joints, initial_upper=q0,
                 solved_joints=solved, target_hand=motion['hand'][frames], actual_hand=actual,
                 final_position_error=residual, final_angle_error_deg=degrees,
                 solver_preupdate_position_error=position_error, solver_preupdate_angle_error=angle_error,
                 solver_preupdate_torso_error=torso_error)
        report.update(completed=True, final_position_error_m=residual.tolist(), final_angle_error_deg=degrees.tolist())
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        report['finished'] = time.time()
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
