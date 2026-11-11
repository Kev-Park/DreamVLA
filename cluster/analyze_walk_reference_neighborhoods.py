"""Read-only numerical walking-prefix neighborhoods; no donor identity claims."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import numpy as np


def nearest(query,candidates):
    distances=np.sqrt(np.mean((candidates-query[None])**2,axis=(1,2)))
    assert np.isfinite(distances).all()
    index=int(np.argmin(distances))
    return index,float(distances[index])


def analyze(root):
    counts={'StitchMAX200':157,'StitchMAX200v3':200,'StitchMAX100v3_eval':100}
    pools={}
    for name,count in counts.items():
        rows=[]
        for path in sorted((root/'ref_motions'/name).glob('*.pkl')):
            raw=path.read_bytes();value=pickle.loads(raw)
            joints=np.asarray(value['joints'],dtype=np.float64)
            assert joints.ndim==2 and joints.shape[0]>=50 and joints.shape[1]==29
            assert np.isfinite(joints[:50]).all()
            rows.append({'name':path.name,'sha256':hashlib.sha256(raw).hexdigest(),'legs':joints[:50,:12]})
        assert len(rows)==count,(name,len(rows));pools[name]=rows
    rows=[]
    for query in pools['StitchMAX100v3_eval']:
        row={'name':query['name'],'sha256':query['sha256'],'nearest':{}}
        for name in ('StitchMAX200','StitchMAX200v3'):
            candidates=pools[name];per_window={}
            for frames in (10,20,40):
                features=np.stack([r['legs'][:frames] for r in candidates])
                index,distance=nearest(query['legs'][:frames],features)
                per_window[str(frames)]={'name':candidates[index]['name'],'joint_rms_rad':distance}
            shifted=[]
            for offset in range(11):
                features=np.stack([r['legs'][offset:offset+40] for r in candidates])
                index,distance=nearest(query['legs'][:40],features)
                shifted.append({'name':candidates[index]['name'],'candidate_offset_frames':offset,'joint_rms_rad':distance})
            per_window['40_candidate_offset_0_to_10']=min(shifted,key=lambda r:r['joint_rms_rad'])
            row['nearest'][name]=per_window
        rows.append(row)
    return {'at':time.time(),'host':os.uname().nodename,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'method':'Unweighted RMS joint-angle difference over first12leg DOFs;10/20/40stored-frame prefixes;additional40-frame comparison allows candidate offset0..10.',
        'pools':{n:{'references':len(v),'files':[{'name':r['name'],'sha256':r['sha256']} for r in v]} for n,v in pools.items()},
        'rows':rows,
        'limits':['Reference pool candidates include files whose expert episode may not have been retained for GR00T training.',
            'Distances describe numerical prefix similarity; they neither certify donor reuse nor estimate model-state training support.',
            'First12joint indices are the unchanged left/right leg order;arms,global yaw,root translation and object geometry are excluded.',
            'No temporal resampling;candidate offsets are stored reference frames,not simulator steps.',
            'Any outcome association is descriptive and does not isolate a causal performance mechanism.'],
        'jobs_changed':False,'datasets_changed':False}


if __name__=='__main__':
    root=Path.home()/'kevin';value=analyze(root)
    output=root/'eval_videos/walk_reference_prefix_neighborhoods_20261027.json'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'output':str(output),'references':len(value['rows']),'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
