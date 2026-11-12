"""Re-stitch a saved walking prefix with its static reference; never overwrite inputs.

This tests the complete stitch/export/load path. It does not regenerate the raw
walking retarget or replace simulator replay and visual review.
"""
import argparse
import hashlib
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
    try:
        w, s = st.load(str(args.walking), chain), st.load(str(args.static), chain)
        shift = build.alignment_shift(w, s)
        w = build.shifted_walk(w, [0, 0, 0], shift)
        st.HAND_SHIFT = np.asarray(shared.SHIFT)
        result, info = st.stitch(w, s, chain, limits)
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
        report['finished']=time.time()
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
