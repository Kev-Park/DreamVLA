"""Run unchanged Pick50 holdouts on Biped or a Slurm-assigned Adroit GPU.

Checkpoint and trajectory transfers are uncompressed and checksummed. Workers
use the same four reference subsets and seeds as Bluesclues. Occupied GPUs are
never used on Biped; Adroit workers stay on their one assigned full GPU.
"""
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from types import SimpleNamespace

from bridge_pick_budget_biped import remote
from stream_directory import transfer

RUN = 'pick50_dagger57_20261013_r2'
FINAL_RUN = 'pick50_dagger57_20261013_r3'
POOL = 'Holosoma_Pick_29_latband60max_eval'
OUT = Path(__file__).resolve().parents[2] / 'out'
STATE = OUT / 'pick_budget_biped_eval_bridge.json'


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)); temp.replace(path)


def validate_report(report):
    assert report['pool'] == POOL and report['episodes'] == 100
    assert sorted(x['motion_id'] for x in report['per_episode']) == list(range(100))
    assert sum(x['success'] for x in report['per_episode']) == report['successes']
    for key in ['world_mpjpe_cm', 'root_aligned_mpjpe_cm']:
        assert math.isfinite(report[key]) and report[key] >= 0
        assert all(math.isfinite(x[key]) and x[key] >= 0 for x in report['per_episode'])
        assert abs(sum(x[key] for x in report['per_episode']) / 100 - report[key]) < 1e-8


def slurm_worker_gpus(environment):
    """Share the one assigned full GPU; never discover other users' devices."""
    visible = environment.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not environment.get('SLURM_JOB_ID') or not visible or ',' in visible:
        raise ValueError('Adroit evaluation requires exactly one Slurm-assigned GPU')
    if environment.get('SLURM_GPUS_ON_NODE', '1') != '1':
        raise ValueError('Expected a one-GPU Slurm allocation')
    return [visible, visible]


def run_worker(request_file):
    root = Path.home() / 'kevin'; work = request_file.parent
    request = json.loads(request_file.read_text())
    assert request['host'] in ['adroit', 'biped']
    adroit = request['host'] == 'adroit'
    assert request['run'] == (FINAL_RUN if adroit else RUN)
    assert request['pool'] == POOL and request['episodes'] == 100
    relative = Path(request.get('runtime_relative', 'DreamVLA'))
    assert not relative.is_absolute() and '..' not in relative.parts
    runtime = root / relative
    assert runtime.resolve().is_relative_to(root.resolve())
    if adroit:
        assert request['runtime_files'], 'Pin the evaluation runtime before submission'
        for name, expected in request['runtime_files'].items():
            path = runtime / name
            assert path.resolve().is_relative_to(runtime.resolve())
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    checkpoint = root / request['checkpoint_relative']
    output = root / request['output_relative']; output.mkdir(parents=True, exist_ok=True)
    assert not list(output.iterdir()), 'Evaluation output already exists; diagnose before resuming'
    assert len(list((root / 'ref_motions' / POOL).glob('pick_*.pkl'))) == 100
    box = work / 'box.json'; box.write_text(json.dumps(request['box']))
    state_file = work / 'state.json'
    gpus = slurm_worker_gpus(os.environ) if adroit else []
    while not gpus:
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                       '--format=csv,noheader,nounits'], text=True)
        gpus = [int(a) for a, b, c in (row.split(',') for row in raw.splitlines())
                if int(b) < 2000 and int(c) <= 5]
        if gpus:
            break
        write(state_file, {'phase': 'waiting_for_idle_gpu', 'time': time.time(), 'gpu_status': raw})
        time.sleep(30)
    write(state_file, {'phase': 'running', 'time': time.time(), 'gpus': gpus})
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false', HS_REWORK='1', HS_REWORK_OBJ_PC='1',
               HS_REWORK_OBJ_GATE='1', HS_REWORK_CONTACT_LEAD='2', HS_EVAL_NO_EE_TERM='1',
               PYTHONPATH=str(runtime / 'Training/scripts/reinforcement_learning/rsl_rl'))
    # Host login/tmux settings must not introduce extra experiment switches.
    for key in list(env):
        if key.startswith('HS_'):
            del env[key]
    env.update(request['eval_environment'])
    assert env['HS_EVAL_NO_EE_TERM'] == '1'
    script = runtime / 'Training/scripts/reinforcement_learning/rsl_rl/eval_vla_sonic.py'
    sonic = root / 'sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000'

    def gpu_worker(index, gpu):
        for k in range(index, 4, len(gpus)):
            ids = work / f'eval_ids{k}.txt'; ids.write_text(' '.join(map(str, range(k, 100, 4))) + '\n')
            logfile = work / f'shard{k}.log'; assert not logfile.exists()
            args = [sys.executable, str(script), '--headless', '--sonic-pt', str(sonic),
                    '--ref-motions-path', POOL, '--vla-checkpoint', str(checkpoint), '--waist-dof', '29',
                    '--motions-from', str(ids), '--num-episodes', '25', '--seed', str(100 + k),
                    '--traj-dump', str(output / f'b{k}')]
            with logfile.open('w') as log:
                p = subprocess.Popen(args, cwd=runtime / 'Training',
                                     env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=log,
                                     stderr=subprocess.STDOUT)
                write(work / f'shard{k}.process.json', {'pid': p.pid, 'gpu': gpu,
                                                       'command': args, 'started': time.time()})
                code = p.wait()
            write(work / f'shard{k}.exit.json', {'code': code, 'time': time.time()})
            assert code == 0, f'Shard {k} exited {code}; see {logfile}'
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        tasks = [executor.submit(gpu_worker, i, gpu) for i, gpu in enumerate(gpus)]
        for task in tasks:
            task.result()
    subprocess.run([sys.executable, str(Path(__file__).with_name('pick_budget_score.py')),
                    str(output), str(box)], env=env, check=True)
    report = json.loads((output / 'results.json').read_text()); validate_report(report)
    write(state_file, {'phase': 'complete', 'time': time.time(), 'episodes': 100,
                       'successes': report['successes']})


def launch(request, blue_root):
    import shlex
    assert request['host'] == 'biped' and request['phase'] == 'evaluate'
    assert request['run'] == RUN and request['pool'] == POOL and request['episodes'] == 100
    ck = str(PurePosixPath(request['checkpoint']).relative_to(blue_root))
    assert ck == f'checkpoints/{RUN}/checkpoint-24888'
    output = str(PurePosixPath(request['output']).relative_to(blue_root))
    assert output == 'eval_videos/' + RUN
    transfer(SimpleNamespace(source_host='bluesclues', destination_host='biped', source=ck,
                             destination=ck, inference=True, receipt=OUT / (RUN + '_eval_weights_transfer.json')))
    payload = dict(request, checkpoint_relative=ck, output_relative=output)
    return remote('biped', r'''
from pathlib import Path
import json,subprocess,shlex,time
r=Path.home()/'kevin';p=PAYLOAD;w=r/'dline'/('pick50_dagger57_20261013_r2_biped_eval');w.mkdir(exist_ok=True)
assert not(w/'request.json').exists() and not(w/'exit_code.txt').exists()
idx=json.loads((r/p['checkpoint_relative']/'model.safetensors.index.json').read_text())
assert all((r/p['checkpoint_relative']/n).stat().st_size>8 for n in set(idx['weight_map'].values()))
(w/'request.json').write_text(json.dumps(p,indent=2))
session='eval'
if subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode:
 subprocess.run(['tmux','new-session','-d','-s',session,'export XLA_PYTHON_CLIENT_PREALLOCATE=false; exec bash'],check=True)
body='source '+shlex.quote(str(Path.home()/'miniconda3/etc/profile.d/conda.sh'))+'; conda activate dreamcontrol_51; export XLA_PYTHON_CLIENT_PREALLOCATE=false; echo $$ > '+shlex.quote(str(w/'worker.pid'))+'; '+shlex.join([str(Path.home()/'miniconda3/envs/dreamcontrol_51/bin/python'),'-u',str(r/'DreamVLA/cluster/pick_budget_biped_eval.py'),'run',str(w/'request.json')])+' > '+shlex.quote(str(w/'supervisor.log'))+' 2>&1; rc=$?; echo "$rc" > '+shlex.quote(str(w/'exit_code.txt'))
subprocess.run(['tmux','new-window','-d','-t',session,'-n','pick50_r2_holdout',body],check=True)
print(json.dumps(dict(p,work=str(w),started=time.time())))
'''.replace('PAYLOAD', repr(payload)))


def status(job):
    return remote('biped', r'''
from pathlib import Path
import json,time
w=Path(WORK);pid=int((w/'worker.pid').read_text());proc=Path('/proc')/str(pid)/'cmdline'
out={'exit_code':int((w/'exit_code.txt').read_text()) if (w/'exit_code.txt').exists() else None,
     'pid':pid,'args':proc.read_bytes().replace(b'\0',b' ').decode() if proc.exists() else '',
     'state':json.loads((w/'state.json').read_text()) if (w/'state.json').exists() else None,'workers':[],
     'failed_shards':[f.name for f in w.glob('shard*.exit.json') if json.loads(f.read_text())['code']!=0]}
for f in w.glob('shard*.process.json'):
 d=json.loads(f.read_text());p=Path('/proc')/str(d['pid'])/'cmdline';log=w/f.name.replace('.process.json','.log')
 if p.exists() and all(str(a) in p.read_bytes().replace(b'\0',b' ').decode() for a in d['command'][1:]):
  out['workers'].append({'pid':d['pid'],'gpu':d['gpu'],'log_age':time.time()-log.stat().st_mtime})
print(json.dumps(out))
'''.replace('WORK', repr(job['work'])))


def deliver(job):
    output = job['output_relative']; staging = output + '_biped_verified'
    transfer(SimpleNamespace(source_host='biped', destination_host='bluesclues', source=output,
                             destination=staging, inference=False,
                             receipt=OUT / (RUN + '_eval_results_transfer.json')))
    return remote('bluesclues', r'''
from pathlib import Path
import json,time
r=Path.home()/'kevin';p=PAYLOAD;src=r/(p['output_relative']+'_biped_verified');dst=r/p['output_relative']
d=json.loads((src/'results.json').read_text());assert d['episodes']==100 and d['pool']==p['pool']
assert sorted(x['motion_id'] for x in d['per_episode'])==list(range(100))
assert len(list(src.glob('b*_traj.npz')))==100 and not list(dst.iterdir())
src.rename(dst)
f=r/'dline/pick50_dagger57_20261013/external'/('evaluate_'+p['run']+'.reply.json')
assert not f.exists();reply={'state':'completed','host':'biped','run':p['run'],'time':time.time()}
tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(reply));tmp.replace(f);print(json.dumps(reply))
'''.replace('PAYLOAD', repr(job)))


def serve():
    OUT.mkdir(exist_ok=True)
    lock = (OUT / 'pick_budget_biped_eval_bridge.lock').open('a+b')
    import msvcrt
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    state = json.loads(STATE.read_text()) if STATE.exists() else {'phase': 'waiting_request'}
    assert state['phase'] in ['waiting_request', 'evaluating', 'complete'], 'Prior bridge state needs diagnosis'
    try:
        while state['phase'] != 'complete':
            state.update(time=time.time(), bridge_pid=os.getpid()); write(STATE, state)
            if state['phase'] == 'waiting_request':
                data = remote('bluesclues', r'''
from pathlib import Path
import json
r=Path.home()/'kevin';w=r/'dline/pick50_dagger57_20261013';p=w/'external/evaluate_pick50_dagger57_20261013_r2.request.json'
d=json.loads(p.read_text()) if p.exists() else None
if d:d['eval_environment']=json.loads((w/'eval_environment.json').read_text())
print(json.dumps({'root':str(r),'request':d}))
''')
                if data['request']:
                    state['phase'] = 'staging_checkpoint'; write(STATE, state)
                    state['job'] = launch(data['request'], data['root']); state['phase'] = 'evaluating'
            elif state['phase'] == 'evaluating':
                result = status(state['job']); state['last_status'] = result
                assert result['exit_code'] is not None or state['job']['work'] in result['args'], 'Supervisor disappeared without exit record'
                assert not result['failed_shards'], result
                assert all(w['log_age'] < 1800 for w in result['workers']), 'Evaluation log stalled 30 minutes'
                if result['exit_code'] is not None:
                    assert result['exit_code'] == 0, result
                    state['phase'] = 'returning_results'; write(STATE, state)
                    state['reply'] = deliver(state['job']); state['phase'] = 'complete'
            state.update(time=time.time(), bridge_pid=os.getpid()); write(STATE, state)
            print(json.dumps({'phase': state['phase'], 'time': state['time']}), flush=True)
            if state['phase'] != 'complete':
                time.sleep(30)
    except BaseException as error:
        state.update(phase='error', error=repr(error), time=time.time()); write(STATE, state)
        raise


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run_worker(Path(sys.argv[2]))
    else:
        assert sys.argv[1] == 'serve'
        serve()
