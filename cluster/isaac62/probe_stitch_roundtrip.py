"""Re-stitch a saved walking prefix with its static reference; never overwrite inputs.

This tests the complete stitch/export/load path. It does not regenerate the raw
walking retarget or replace simulator replay and visual review.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import pickle
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('urdf', 'walking', 'static', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--trace-ik', action='store_true', help='Observe original IK inputs/updates without modifying the solver')
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
    import build_stitchmax_v2 as shared
    import reconstruct_own_motion_stitches as build
    chain = pk.build_chain_from_urdf(args.urdf.read_bytes()).to(dtype=torch.float32)
    limits = [np.asarray(x, dtype=np.float64) for x in chain.get_joint_limits()]
    hashes = {name:hashlib.sha256(getattr(args,name).read_bytes()).hexdigest() for name in ('walking','static')}
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(started=time.time(), completed=False, torch=torch.__version__, numpy=np.__version__,
                  source_hashes=hashes, walking=str(args.walking), static=str(args.static),
                  walking_source_kind='prefix retained from an existing stitched reference')
    report['packages'] = {name: importlib.metadata.version(name)
                          for name in ('torch', 'numpy', 'pytorch-kinematics', 'jax', 'jaxlie')}
    traces = {}
    previous_trace = sys.gettrace()
    if args.trace_ik:
        assert previous_trace is None, 'Do not replace an active debugger/profiler'
        lines, first = inspect.getsourcelines(st.solve_upper)
        before = first + next(i for i, line in enumerate(lines) if 'opt.zero_grad();' in line)
        after = before + 1
        selected = {0, 1, 2, 9, 99, 599, st.IK_ITERS-1}

        def array(value):
            return value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.asarray(value).copy()

        def local_trace(frame, event, arg):
            if event == 'line' and frame.f_lineno in (before, after):
                values = frame.f_locals
                step = values['_']
                if step in selected:
                    prefix = f'{step:04d}_' + ('before_' if frame.f_lineno == before else 'after_')
                    keys = ('q', 'M', 'p', 'R', 'RT', 'e_p', 'e_r', 'e_t', 'smooth', 'loss') if frame.f_lineno == before else ('q',)
                    for key in keys:
                        traces[prefix+key] = array(values[key])
                    if frame.f_lineno == after:
                        traces[prefix+'gradient'] = array(values['q'].grad)
                        for key, value in values['opt'].state[values['q']].items():
                            traces[prefix+'adam_'+key] = array(value)
                    if step == 0 and frame.f_lineno == before:
                        for key in ('Jfix', 'q0', 'root_P', 'root_R', 'tgt_p', 'tgt_R', 'torso_R'):
                            traces['input_'+key] = array(values[key])
            return local_trace

        def trace(frame, event, arg):
            return local_trace if frame.f_code is st.solve_upper.__code__ else None
    try:
        w, s = st.load(str(args.walking), chain), st.load(str(args.static), chain)
        shift = build.alignment_shift(w, s)
        w = build.shifted_walk(w, [0, 0, 0], shift)
        st.HAND_SHIFT = np.asarray(shared.SHIFT)
        if args.trace_ik:
            sys.settrace(trace)
        try:
            result, info = st.stitch(w, s, chain, limits)
        finally:
            if args.trace_ik:
                sys.settrace(previous_trace)
        assert result is not None, info
        path = args.output/'reference.pkl'
        with path.open('wb') as f:
            pickle.dump(result, f)
        actual = st.load(str(path), chain)
        fw, on = info['footfall'], info['onset']
        assert len(actual['J']) == fw+st.TRANSITION+len(s['J'])-on
        assert actual['g'] == fw+st.TRANSITION+s['g']-on
        assert all(np.isfinite(v).all() for v in actual.values())
        np.testing.assert_allclose(actual['J'][:fw,:12], w['J'][:fw,:12], atol=1e-6)
        np.testing.assert_allclose(actual['J'][fw:,:12], np.broadcast_to(w['J'][fw,:12],actual['J'][fw:,:12].shape), atol=1e-6)
        expected_root = np.concatenate([w['P'][:fw], np.repeat(w['P'][fw:fw+1],len(actual['J'])-fw,axis=0)])
        np.testing.assert_allclose(actual['P'],expected_root,rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(actual['Q'],axis=1),1,atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(actual['O'][:,3:],axis=1),1,atol=1e-5)
        for name,digest in hashes.items():
            assert hashlib.sha256(getattr(args,name).read_bytes()).hexdigest()==digest
        np.savez(args.output/'stitch.npz', **{k:v for k,v in actual.items() if k!='g'})
        report.update(completed=True, alignment_shift_m=shift.tolist(), hand_shift_m=shared.SHIFT,
                      frames=len(actual['J']), grab_frame=actual['g'], diagnostics=info,
                      output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), iterations=st.IK_ITERS)
    except BaseException as exc:
        report['error']=repr(exc)
        raise
    finally:
        if args.trace_ik:
            np.savez(args.output/'ik_trace.npz', **traces)
            report['trace_arrays'] = len(traces)
        report['finished']=time.time()
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
