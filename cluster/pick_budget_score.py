"""Require exactly one evaluation of every held-out reference; score success and MPJPE."""
import json
from pathlib import Path
import sys
import re
import numpy as np
from mpjpe_report import measure

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Training/scripts/reinforcement_learning/rsl_rl'))
from vla_sonic.grasp_success import score

out,boxfile=map(Path,sys.argv[1:]);box=json.loads(boxfile.read_text())
paths=sorted(out.glob('b*_traj.npz'))
ids=[int(re.search(r'_m(\d+)_traj.npz$',p.name)[1]) for p in paths]
assert sorted(ids)==list(range(100)),f'Missing or duplicated held-out references: {ids}'
rows=[]
for p,mid in zip(paths,ids):
    with np.load(p,allow_pickle=False) as z:
        s=score(z['obj_pos'],z['obj_quat'],z['root_pos'],z['root_quat'],z['wrist'],
                box_lo=np.array(box['lo']),box_hi=np.array(box['hi']))
        metrics=measure(z['body_pos'],z['ref_body_pos'],z['root_pos'],z['ref_root_pos'])
        rows.append(dict(motion_id=mid,success=bool(s['success']),**metrics))
report={'pool':'Holosoma_Pick_29_latband60max_eval','episodes':100,
        'successes':sum(r['success'] for r in rows),'criterion':'calibrated four-rule',
        'world_mpjpe_cm':float(np.mean([r['world_mpjpe_cm'] for r in rows])),
        'root_aligned_mpjpe_cm':float(np.mean([r['root_aligned_mpjpe_cm'] for r in rows])),
        'aggregation':'episode mean; root translation only','per_episode':rows}
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='per_episode'}))
