"""CPU-only paired fixture for actual Adapter-B refinement and export paths.

Simple refinement uses its full 1,500-step budget. AL uses two outer rounds of
25 Adam updates as a bounded migration test, not a converged reference solve.
The optional augmentation pass exercises source-path and orientation anchors.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import runpy
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('simple', 'al', 'al_aug'), required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import numpy as np
    import torch
    import jax
    assert jax.default_backend() == 'cpu'
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(started=time.time(), completed=False, mode=args.mode,
                  source_root=str(args.source_root), raw_sha256=hashlib.sha256(args.raw.read_bytes()).hexdigest(),
                  packages={name: importlib.metadata.version(name) for name in
                            ('torch', 'numpy', 'pytorch-kinematics', 'jax', 'jaxlie')})
    previous_trace = sys.gettrace()
    assert previous_trace is None
    traces = {}
    stage = 'base'
    adapters = args.source_root / 'TrajGen/holosoma_adapters'
    sys.path.insert(0, str(adapters))
    # These are test-fixture settings only; no production defaults are altered.
    for key in list(os.environ):
        if key.startswith('HS_'):
            del os.environ[key]
    os.environ.update(HS_PKL_DOF='29', HS_STAND_LOWER='0', HS_NO_LEADIN='1',
                      HS_REFINE_ARM='1', HS_REFINE_MODE='simple' if args.mode == 'simple' else 'al',
                      HS_RETABLE_Z='0.911', HS_FREEZE_FOR='10', HS_STAND_MIN_STANDOFF='0.35',
                      HS_STAND_LAT_BAND='-0.28,-0.15', HS_REFINE_AFTER_LEADIN='1',
                      HS_REFINE_VERBOSE='1')
    report['fixture_recipe'] = {k: v for k, v in os.environ.items() if k.startswith('HS_')}
    try:
        import refine_al_29 as al
        assert al.DEVICE.type == 'cpu'
        al.AL_OUTER_ITERS, al.AL_INNER_ITERS = 2, 25
        report['bounded_al_budget'] = {'outer': 2, 'inner': 25}
        lines = {}
        for filename, function in ((adapters / 'holosoma_to_pkl.py', 'refine_right_arm_table'),
                                   (adapters / 'refine_al_29.py', 'refine_arm')):
            # Observe source lines; never replace the production function body.
            text = filename.read_text().splitlines()
            start = next(i for i, line in enumerate(text) if line.startswith('def ' + function + '('))
            before = next(i + 1 for i in range(start, len(text)) if 'cost.backward()' in text[i])
            lines[(str(filename), function)] = (before, before + 1)

        def array(value):
            return value.detach().cpu().numpy().copy()

        def local_trace(frame, event, arg):
            before, after = lines[(frame.f_code.co_filename, frame.f_code.co_name)]
            if event == 'line' and frame.f_lineno in (before, after):
                values = frame.f_locals
                simple = frame.f_code.co_name == 'refine_right_arm_table'
                step = values['_step'] if simple else values['_']
                outer = 0 if simple else values['outer_iter']
                selected = {0, 1, 9, 99, 1499} if simple else {0, 1, 24}
                if step in selected:
                    prefix = f'{stage}_{outer:02d}_{step:04d}_'
                    parameter = values['arm'] if simple else values['joint_angles']
                    if frame.f_lineno == before:
                        traces[prefix + 'parameters'] = array(parameter)
                        traces[prefix + 'cost'] = array(values['cost'])
                    else:
                        traces[prefix + 'gradient'] = array(parameter.grad)
            return local_trace

        def trace(frame, event, arg):
            return local_trace if (frame.f_code.co_filename, frame.f_code.co_name) in lines else None

        sys.argv = [str(adapters / 'holosoma_to_pkl.py'), str(args.raw), str(args.output / 'reference')]
        sys.settrace(trace)
        try:
            runpy.run_path(sys.argv[0], run_name='__main__')
        finally:
            sys.settrace(previous_trace)
        with (args.output / 'reference.pkl').open('rb') as stream:
            motion = pickle.load(stream)
        values = {key: np.asarray(motion[key]) for key in ('joints', 'global_position', 'object_poses', 'grab_pos')}
        values['global_pose'] = np.asarray(motion['global_pose'].as_matrix())
        grab = int(motion['grab_idx'])
        assert all(np.isfinite(v).all() for v in values.values())
        assert values['joints'].shape[1] == 29 and 0 < grab < len(values['joints'])
        if args.mode == 'al_aug':
            stage = 'augmentation'
            joints = values['joints'].copy()
            root_p = values['global_position'].copy()
            rotation = motion['global_pose'].rotation()
            root_q = np.asarray(rotation.wxyz() if callable(rotation.wxyz) else rotation.wxyz)
            shift = np.array([.015, -.01, 0.], dtype=np.float32)
            objects = values['object_poses'][:, :3] + shift
            # Exercise the existing rebased source-path/orientation branches.
            al.AUG_REBASE = True
            sys.settrace(trace)
            try:
                values['augmented_joints'] = al.refine_arm(
                    joints, root_p, root_q, objects[grab], grab, fps=20.,
                    obj_traj=objects, src_joints=joints, palm_shift=shift)
            finally:
                sys.settrace(previous_trace)
            values['augmented_palm_path'] = al.aug_palm_path.detach().numpy()
            values['augmented_hand_rotation'] = al.aug_hand_R_src.detach().numpy()
            report['augmentation_fixture'] = {'shift': shift.tolist(), 'AUG_REBASE': True}
        assert traces and all(np.isfinite(v).all() for v in traces.values())
        np.savez(args.output / 'reference.npz', **values)
        np.savez(args.output / 'trace.npz', **traces)
        report.update(completed=True, frames=len(values['joints']), grab_idx=grab,
                      trace_arrays=len(traces), output_arrays=list(values))
        assert hashlib.sha256(args.raw.read_bytes()).hexdigest() == report['raw_sha256']
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        sys.settrace(previous_trace)
        report['finished'] = time.time()
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
