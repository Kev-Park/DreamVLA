"""Collect one disjoint 50-attempt group on one full Slurm A40.

Four groups cover the original 200 references exactly once. Each allocation
runs two unchanged native collectors; all full recordings stay on Adroit.
Launch only after operator review of the A40 exception, camera startup, and
the verified input manifest. This entry point never submits Slurm jobs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid as uuidlib


TAG = 'walk200_v3_dagger57_20261013'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', required=True, type=int, choices=range(4))
    args = parser.parse_args()
    assert os.name == 'posix' and os.environ.get('SLURM_JOB_ID')
    visible = os.environ['CUDA_VISIBLE_DEVICES']
    assert visible and ',' not in visible and not visible.startswith('MIG-')
    b = Path.home() / 'kevin'
    work = b / 'dline' / TAG
    ready = json.loads((work / 'adroit_round3_inputs.ready.json').read_text())
    assert ready['round'] == 3 and ready['student_run'] == TAG + '_r2'
    assert ready['reference_pool'] == 'StitchMAX200v3'
    assert ready['student_attempts'] == 200
    assert ready['teacher_sha256'] == '4912a72813cbaa70cf04b678c5e312fde35c88c88838d2c4d1e5e1ac6813c57b'
    root = b.resolve(strict=True)
    verified = {}
    for row in ready['input_files']:
        relative = Path(row['relative'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = (b / relative).resolve(strict=True)
        assert path.is_relative_to(root)
        assert path not in verified, 'Duplicate canonical input'
        assert path.stat().st_size == row['bytes'] and digest(path) == row['sha256'], str(path)
        verified[path] = row['sha256']
    checkpoint = (b / ready['student_checkpoint_relative']).resolve(strict=True)
    assert checkpoint.is_relative_to(root) and checkpoint.parent.name == TAG + '_r2'
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert state['global_step'] == ready['student_steps']
    assert ready['student_steps'] == math.ceil(ready['student_frames'] * 57 / 160)
    assert checkpoint.name == 'checkpoint-' + str(ready['student_steps'])
    assert digest(work / 'box.json') == 'fdd462b05e1fd3671778d5e10f0a9378bdb2bde453434e979b632fcf4ba25a5b'
    assert len(list((b / 'ref_motions/StitchMAX200v3').glob('*.pkl'))) == 200
    assert len(list((b / 'ref_motions/StitchMAX100v3_eval').glob('*.pkl'))) == 100
    demos = sorted((work / 'demo_root').glob('*.hdf5'))
    assert len(demos) == 8 and all(p.is_file() for p in demos)
    teacher = (b / 'eval_videos/T20_max_v3_env2048/early_stop/best_evaluated.pt').resolve(strict=True)
    required = [teacher, work / 'box.json', *demos]
    for pool in ('StitchMAX200v3', 'StitchMAX100v3_eval'):
        required.extend((b / 'ref_motions' / pool).glob('*.pkl'))
    excluded = {'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin'}
    required.extend(p for p in checkpoint.rglob('*') if p.is_file() and p.name not in excluded)
    assert all(p.resolve(strict=True) in verified for p in required), 'Input inventory does not cover every runtime input'
    assert verified[teacher] == ready['teacher_sha256']

    # Preserve Slurm's physical-device/UUID selection. Replacing it with "0"
    # could select a GPU outside this allocation on a multi-GPU node.
    # CUDA indices can be remapped by Slurm's device cgroup. Resolve the UUID
    # through CUDA before asking NVML/nvidia-smi about physical memory.
    import torch
    assert torch.cuda.device_count() == 1
    # PyTorch 2.7's _CUuuid.__str__ returns bare 8-4-4-4-12 text, while
    # nvidia-smi expects its GPU- prefix (torch/csrc/cuda/Module.cpp, v2.7.0).
    cuda_uuid = 'GPU-' + str(uuidlib.UUID(str(torch.cuda.get_device_properties(0).uuid)))
    gpu = subprocess.check_output([
        'nvidia-smi', '-i', cuda_uuid,
        '--query-gpu=name,uuid,memory.free,memory.total',
        '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    assert len(gpu) == 1
    name, uuid, free, total = [part.strip() for part in gpu[0].split(',')]
    assert uuid.casefold() == cuda_uuid.casefold()
    assert 'A40' in name and int(total) > 44000 and int(free) > 36000, gpu
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gr00t_budget_chain import atomic, checkpoint_ok
    from walk_gr00t_chain import WalkChain
    assert checkpoint_ok(checkpoint)

    class Allocation(WalkChain):
        def slots(self, count, mb):
            assert count == 2 and mb == 17000
            return [visible, visible]

        def release(self, gpu):
            # Slurm owns the allocation; no gpu_slots lease was acquired.
            assert gpu == visible

    chain = Allocation(200)
    chain.work = work / 'adroit_round3_groups' / str(args.group)
    chain.work.mkdir(parents=True, exist_ok=True)
    assert not (chain.work / 'group.launch.json').exists(), 'Reconcile prior launch; never repeat attempts automatically'
    for name in ('box.json', 'demo_root'):
        target = chain.work / name
        assert not target.exists() and not target.is_symlink()
        target.symlink_to(work / name)
    chain.ids = list(range(args.group * 50, (args.group + 1) * 50))
    chain.collection_workers = 2
    chain.sim_task_threads = max(1, len(os.sched_getaffinity(0)) // 2)
    chain.env.update(CUDA_VISIBLE_DEVICES=visible,
                     PXR_WORK_THREAD_LIMIT=str(chain.sim_task_threads))
    atomic(chain.work / 'group.launch.json', {
        'started': time.time(), 'host': os.uname().nodename,
        'pid': os.getpid(), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'group': args.group, 'motion_ids': chain.ids, 'checkpoint': str(checkpoint),
        'cuda_visible_devices': visible, 'gpu_uuid': uuid, 'gpu_name': name,
        'collection_workers': 2, 'seed': 9, 'rollout_cap': 1200,
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'source_manifest_sha256': digest(work / 'adroit_round3_inputs.ready.json'),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                        cwd=Path(__file__).resolve().parent, text=True).strip(),
        'retain_full_recordings': True})
    try:
        raw = chain.collect(3, checkpoint)
        result = json.loads((chain.work / 'collection_r3.json').read_text())
        assert result['motion_ids'] == chain.ids and result['student_attempts'] == 50
        assert result['errors'] == 0
        atomic(chain.work / 'group.done.json', {'group': args.group,
               'slurm_job_id': os.environ['SLURM_JOB_ID'], 'raw': str(raw), **result})
    except BaseException as exc:
        atomic(chain.work / 'group.abort.json', {'error': repr(exc),
               'slurm_job_id': os.environ['SLURM_JOB_ID'], 'pid': os.getpid()})
        raise


if __name__ == '__main__':
    main()
