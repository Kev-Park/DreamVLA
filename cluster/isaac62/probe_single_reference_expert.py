"""Recheck one recorded expert render using the unchanged production evaluator."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    base = Path.home()/'kevin'
    root = Path(__file__).resolve().parents[2]
    previous = base/'eval_videos/isaac62_expert_parity_20261116_v2'
    batch = json.loads((previous/'receipt.json').read_text())
    reference = next(r for r in json.loads((previous/'videos/receipt.json').read_text())['runs'] if r['name']=='walk_target')
    run = next(r for r in batch['runs'] if r['name']=='walk_target')
    output = previous/'single_reference_eval_seed0'
    output.mkdir(exist_ok=False)
    os.sched_setaffinity(0, {0,1,2,3})
    owner = str(os.getpid())
    slots = [sys.executable, str(root/'cluster/gpu_slots.py')]
    receipt = dict(started=time.time(), pid=os.getpid(), completed=False,
                   reference=reference['reference'], reference_index=reference['per_reference_eval']['mid'],
                   scope='One production-evaluator episode matching the rendered reference and seed')
    reserved = False
    try:
        result = subprocess.run([*slots,'reserve','--owner',owner,'--count','1','--mb','4000','--gpu','0'],
                                capture_output=True,text=True,timeout=20,check=True)
        assert result.stdout.strip()=='0'
        reserved = True
        command = list(run['command'])
        for flag,value in [('--num_envs','1'),('--num_episodes','1'),('--max_steps','2000'),
                           ('--receipt',str(output/'entrypoint.json'))]:
            command[command.index(flag)+1] = value
        command += ['--motion-id',str(receipt['reference_index'])]
        env = {k:v for k,v in os.environ.items() if not k.startswith('HS_')}
        env.update(run['hs_settings'])
        env.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_PLATFORMS='cpu',
                   OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                   PXR_WORK_THREAD_LIMIT='2',PYTHONUNBUFFERED='1',HYBRID_RARM='0',
                   PYTHONPATH=run['pythonpath'],HS_EVAL_DUMP_EPS=str(output/'episodes.npz'))
        receipt['command'] = command
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2))
        with (output/'eval.log').open('w') as stream:
            result = subprocess.run(command,cwd=base/'DreamVLA/Training',env=env,
                                    stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
        entry = json.loads((output/'entrypoint.json').read_text())
        receipt.update(returncode=result.returncode,entrypoint=entry,
                       completed=result.returncode==0 and entry['completed'] and (output/'episodes.npz').exists(),
                       metric_lines=[line for line in (output/'eval.log').read_text().splitlines()
                                     if '[PHYS] held (lift>=5cm' in line or line.startswith('[MPJPE] ')])
    except BaseException:
        receipt['error'] = traceback.format_exc()
        raise
    finally:
        if reserved:
            subprocess.run([*slots,'release','--owner',owner,'--gpu','0'],check=True,timeout=20)
        receipt.update(finished=time.time(),reservation_released=reserved)
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
