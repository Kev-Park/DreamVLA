"""Two workers on one full A100; preserve four held-out shards and seeds."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

request=Path(sys.argv[1]);spec=json.loads(request.read_text());work=request.parent
root=Path.home()/'kevin';runtime=root/'DreamVLA';run=spec['run']
checkpoint=root/'checkpoints'/run/'checkpoint-158000'
output=root/'eval_videos'/run
assert spec['eval_pool']=='Holosoma_Pick_29_latband60max_eval'
for relative,digest in spec['runtime_hashes'].items():
    assert hashlib.sha256((runtime/relative).read_bytes()).hexdigest()==digest,relative
assert json.loads((checkpoint/'trainer_state.json').read_text())['global_step']==158000
assert len(list((root/'ref_motions'/spec['eval_pool']).glob('pick_*.pkl')))==100
assert not output.exists();output.mkdir(parents=True)
env=dict(os.environ,XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',
         MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
         PYTHONPATH=str(runtime/'Training/scripts/reinforcement_learning/rsl_rl'))
for key in list(env):
    if key.startswith('HS_'):del env[key]
env.update(HS_REWORK='1',HS_REWORK_OBJ_PC='1',HS_REWORK_OBJ_GATE='1',
           HS_REWORK_CONTACT_LEAD='2',HS_EVAL_NO_EE_TERM='1')

def worker(index):
    for shard in range(index,4,2):
        ids=work/f'eval_ids{shard}.txt'
        ids.write_text(' '.join(map(str,range(shard,100,4)))+'\n')
        args=[sys.executable,runtime/'Training/scripts/reinforcement_learning/rsl_rl/eval_vla_sonic.py',
              '--headless','--sonic-pt',root/'sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000',
              '--ref-motions-path',spec['eval_pool'],'--vla-checkpoint',checkpoint,'--waist-dof','29',
              '--motions-from',ids,'--num-episodes','25','--seed',str(100+shard),
              '--traj-dump',output/f'b{shard}']
        with (work/f'eval_shard{shard}.log').open('x') as log:
            p=subprocess.Popen(list(map(str,args)),env=env,cwd=runtime/'Training',stdout=log,stderr=subprocess.STDOUT)
            (work/f'eval_shard{shard}.process.json').write_text(json.dumps(dict(pid=p.pid,started=time.time(),command=list(map(str,args)))))
            code=p.wait()
        (work/f'eval_shard{shard}.exit.json').write_text(json.dumps(dict(code=code,time=time.time())))
        assert code==0,f'Shard {shard} exited {code}'

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures=[executor.submit(worker,index) for index in range(2)]
    for future in futures:future.result()
subprocess.run([sys.executable,str(runtime/'cluster/pick_budget_score.py'),str(output),str(work/'box.json')],env=env,check=True)
report=json.loads((output/'results.json').read_text())
assert report['episodes']==100 and sorted(row['motion_id'] for row in report['per_episode'])==list(range(100))
(work/'done.json').write_text(json.dumps(dict(time=time.time(),results=report),indent=2)+'\n')
