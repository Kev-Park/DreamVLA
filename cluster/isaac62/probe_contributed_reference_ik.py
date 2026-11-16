"""Run one real contributed reference-generator point with its full IK budget.

Uses original trusted retargeted inputs, the real plan/CLI/export path, and
observation-only traces. It does not run a grid or judge generated behavior.
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--kind', choices=('stand', 'squat', 'button'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import numpy as np
    import torch
    import jax
    assert jax.default_backend() == 'cpu'
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['DREAMVLA_ROOT'] = str(args.source_root)
    os.environ['DREAMVLA_BACKEND_ROOT'] = str(args.source_root)
    sys.path.insert(0, str(args.source_root / 'contrib/drawer_button'))
    sources = args.source_root / 'contrib/drawer_button/dreamvla_drawer_button'
    bases = {'stand': args.base_root / 'Open_Drawer_sim/stand_still.pkl',
             'squat': args.base_root / 'Open_Drawer_sim/squat.pkl',
             'button': args.base_root / 'Button_Press_sim1/0.pkl'}
    report = dict(started=time.time(), completed=False, kind=args.kind, source_root=str(args.source_root),
                  input_sha256={k: digest(p) for k, p in bases.items()}, full_budget=True,
                  packages={n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'pytorch-kinematics', 'jax', 'jaxlie')})
    previous_trace = sys.gettrace()
    assert previous_trace is None
    traces = {}
    updates_seen = 0
    try:
        generated = args.output / 'generated'
        plan = args.output / 'plan.json'
        command = ['dreamvla_drawer_button', 'prepare-references', '--task',
                   'standing_button' if args.kind == 'button' else 'drawer',
                   '--count', '1' if args.kind == 'button' else '2', '--seed', '17',
                   '--output', str(generated), '--plan', str(plan)]
        if args.kind == 'button':
            command += ['--base-button', str(bases['button'])]
        else:
            command += ['--base-stand', str(bases['stand']), '--base-squat', str(bases['squat'])]
        report['planning_command'] = command.copy()
        sys.argv = command
        runpy.run_module('dreamvla_drawer_button', run_name='__main__')
        filename = sources / ('generate_button.py' if args.kind == 'button' else 'generate_drawer_' + args.kind + '.py')
        text = filename.read_text().splitlines()
        before = next(i + 1 for i, line in enumerate(text) if 'cost.backward()' in line)
        after = before + 1

        def local_trace(frame, event, arg):
            nonlocal updates_seen
            if event == 'line' and frame.f_lineno in (before, after):
                values = frame.f_locals
                step = values['j' if args.kind == 'button' else 'i']
                if frame.f_lineno == after:
                    updates_seen += 1
                if step in {0, 1, 24, 99, 499, 1999, 9999}:
                    prefix = f'{step:05d}_'
                    keys = ('cost', 'joint_angles') if frame.f_lineno == before else ()
                    for key in keys:
                        traces[prefix + key] = values[key].detach().cpu().numpy().copy()
                    if frame.f_lineno == after:
                        traces[prefix + 'gradient'] = values['joint_angles'].grad.detach().cpu().numpy().copy()
            return local_trace

        def trace(frame, event, arg):
            return local_trace if frame.f_code.co_filename == str(filename) and frame.f_code.co_name == '<module>' else None

        command = ['dreamvla_drawer_button', 'refs-button' if args.kind == 'button' else 'refs-drawer-' + args.kind,
                   '--plan' if args.kind == 'button' else '--point-plan', str(plan)]
        if args.kind != 'button':
            command += ['--max-points', '1']
        report['generation_command'] = command.copy()
        sys.argv = command
        sys.settrace(trace)
        try:
            runpy.run_module('dreamvla_drawer_button', run_name='__main__')
        finally:
            sys.settrace(previous_trace)
        outputs = list(generated.rglob('*.pkl'))
        assert len(outputs) == 1, outputs
        with outputs[0].open('rb') as stream:
            motion = pickle.load(stream)
        rotation = motion['global_pose'].rotation()
        arrays = {k: np.asarray(motion[k]) for k in ('joints', 'global_position')}
        arrays['global_pose'] = np.asarray(motion['global_pose'].as_matrix())
        arrays['root_quaternion'] = np.asarray(rotation.wxyz() if callable(rotation.wxyz) else rotation.wxyz)
        for name in ('open_drawer_pos', 'hit_pos_real', '_ik_optimized_joints'):
            if name in motion:
                arrays[name] = np.asarray(motion[name])
        assert all(np.isfinite(v).all() for v in arrays.values())
        assert len(arrays['joints']) >= 200 and arrays['joints'].shape[1] == 27
        assert traces and updates_seen > 0 and all(np.isfinite(v).all() for v in traces.values())
        np.savez(args.output / 'reference.npz', **arrays)
        np.savez(args.output / 'trace.npz', **traces)
        report.update(completed=True, frames=len(arrays['joints']), grab_idx=int(motion['grab_idx']),
                      backward_steps=updates_seen, trace_arrays=len(traces), output=str(outputs[0]),
                      provenance=motion.get('_phase1_expansion', motion.get('_table4')),
                      viewer_modules_loaded={n: n in sys.modules for n in ('viser', 'yourdfpy', 'pyroki')})
        assert report['input_sha256'] == {k: digest(p) for k, p in bases.items()}
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
