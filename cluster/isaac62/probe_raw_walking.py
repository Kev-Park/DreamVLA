"""Convert one original walking qpos clip using the accepted Adapter-B recipe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

INSPECT = r'''
import json,pickle,sys,numpy as np,importlib.metadata
from pathlib import Path
p=Path(sys.argv[1])
with p.open('rb') as stream:m=pickle.load(stream)
values={name:np.asarray(m[name]) for name in ('joints','global_position','object_poses','grab_pos')}
values['global_pose']=np.asarray(m['global_pose'].as_matrix())
for name,a in values.items():
 assert np.isfinite(a).all(),name
assert values['joints'].shape[1]==29
assert 0<=int(m['grab_idx'])<len(values['joints'])
assert len(values['object_poses'])==len(values['joints'])
np.savez(p.with_suffix('.npz'),**values)
p.with_suffix('.json').write_text(json.dumps({'frames':len(values['joints']),'grab_idx':int(m['grab_idx']),
 'grab_pos_is_object':bool(m['grab_pos_is_object']),'versions':{k:importlib.metadata.version(k) for k in ('numpy','torch','jax','jaxlie','pytorch-kinematics')},
 'root_xy_net_m':float(np.linalg.norm(values['global_position'][-1,:2]-values['global_position'][0,:2])),
 'shapes':{k:list(v.shape) for k,v in values.items()}},indent=2))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime-python',type=Path)
    args=parser.parse_args()
    assert os.name=='posix' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    fixture=args.fixture.expanduser().resolve(strict=True)
    output=args.output.expanduser().resolve();output.mkdir(exist_ok=False,parents=True)
    receipt=dict(started=time.time(),completed=False,gpu_used=False,scope='One raw walking clip, accepted lower-body conversion; no arm refinement or stitch')
    try:
        source=Path(__file__).resolve().parents[2]/'TrajGen/holosoma_adapters/holosoma_to_pkl.py'
        runtime=args.runtime_python.expanduser().absolute() if args.runtime_python else Path(sys.executable)
        assert runtime.is_file() and source.is_file()
        copy=json.loads((fixture/'copy_receipt.json').read_text())
        raw=fixture/'pick_0.npz'
        assert hashlib.sha256(raw.read_bytes()).hexdigest()==copy['raw_sha256']==copy['manifest_row']['sha256']
        env={k:v for k,v in os.environ.items() if not k.startswith('HS_')}
        env.update(HS_PKL_DOF='29',HS_STAND_LOWER='0',HS_NO_LEADIN='1',HS_REFINE_ARM='0',
                   HS_RETABLE_Z='0.911',HS_FREEZE_FOR='10',HS_STAND_MIN_STANDOFF='0.35',
                   HS_STAND_LAT_BAND='-0.28,-0.15',HS_REFINE_AFTER_LEADIN='1',
                   CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        command=[str(runtime),str(source),str(raw),str(output/'reference')]
        receipt.update(command=command,raw_sha256=copy['raw_sha256'],recipe={k:v for k,v in env.items() if k.startswith('HS_')})
        with (output/'conversion.log').open('w') as stream:
            run=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        receipt['conversion_returncode']=run.returncode
        assert run.returncode==0,'Raw conversion failed; inspect conversion.log'
        with (output/'inspection.log').open('w') as stream:
            run=subprocess.run([str(runtime),'-c',INSPECT,str(output/'reference.pkl')],env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=60)
        assert run.returncode==0,'Converted-reference inspection failed'
        receipt.update(completed=True,converted=json.loads((output/'reference.json').read_text()))
    except BaseException:
        receipt['error']=traceback.format_exc()
        raise
    finally:
        receipt['finished']=time.time()
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
