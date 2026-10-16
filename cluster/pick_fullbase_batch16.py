"""One reviewed GPU: historical Pick DAgger-3 batch-16 ablation, then evaluation.

Reuses the original LeRobot export and initializes from pretrained GR00T.
Failures are recorded and never retried. Resource placement is supplied by a
manually reviewed request; this script does not enforce account-wide policies.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = Path.home() / 'kevin'
POOL = 'Holosoma_Pick_29_latband60max_eval'
RESERVATION_MB = 44000


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class Run:
    def __init__(self, request):
        self.request = read(request)
        self.work = Path(request).resolve().parent
        self.name = self.request['run']
        assert re.fullmatch(r'af60v8f57_dagger3_bs16_\d{8}', self.name)
        assert self.work == ROOT / 'dline' / self.name
        self.gpu = self.request['gpu']
        # Keep the exact source data while isolating the verified metadata cache.
        self.dataset = self.work / 'dataset'
        self.checkpoint = ROOT / 'checkpoints' / self.name / 'checkpoint-158000'
        self.output = ROOT / 'eval_videos' / self.name
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpu),
                        XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='2',
                        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='2',
                        TOKENIZERS_PARALLELISM='false',
                        PYTHONPATH=str(HERE.parent / 'Training/scripts/reinforcement_learning/rsl_rl'))
        for key in list(self.env):
            if key.startswith('HS_'):
                del self.env[key]
        self.env.update(HS_REWORK='1', HS_REWORK_OBJ_PC='1', HS_REWORK_OBJ_GATE='1',
                        HS_REWORK_CONTACT_LEAD='2', HS_EVAL_NO_EE_TERM='1')
        self.state = dict(run=self.name, controller_pid=os.getpid(), gpu=self.gpu,
                          dataset=str(self.dataset), checkpoint=str(self.checkpoint),
                          output=str(self.output), phase='validating')

    def status(self, **values):
        self.state.update(time=time.time(), **values)
        write(self.work / 'state.json', self.state)

    def reserve(self):
        # Coordinate only this explicitly reviewed placement with other launchers.
        owner = os.getpid()
        start = Path(f'/proc/{owner}/stat').read_text().rsplit(')', 1)[1].split()[19]
        placement = dict(owner=owner, owner_start=start, count=1, mb=RESERVATION_MB,
                         gpus=[self.gpu], reason='Manually reviewed historical Pick batch-16 ablation')
        with (ROOT / 'dline/gpu_slots.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = ROOT / 'dline/gpu_reservation_placements.json'
            rows = read(path) if path.exists() else []
            assert not any(row['owner'] == owner for row in rows)
            write(path, rows + [placement])
        selected = subprocess.check_output([sys.executable, str(HERE / 'gpu_slots.py'),
            'reserve', '--owner', str(owner), '--count', '1', '--mb', str(RESERVATION_MB)], text=True)
        assert selected.split() == [str(self.gpu)], 'Reviewed GPU no longer has sufficient free memory; no retry'
        self.status(reserved=True)

    def release(self):
        subprocess.run([sys.executable, str(HERE / 'gpu_slots.py'), 'release',
                        '--owner', str(os.getpid()), '--gpu', str(self.gpu)], check=True)
        with (ROOT / 'dline/gpu_slots.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = ROOT / 'dline/gpu_reservation_placements.json'
            if path.exists():
                write(path, [row for row in read(path) if row['owner'] != os.getpid()])

    def execute(self, label, command, cwd):
        logfile = self.work / f'{label}.log'
        assert not logfile.exists(), 'Existing stage requires diagnosis; refusing to overwrite'
        with logfile.open('w') as log:
            process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=self.env,
                                       stdout=log, stderr=subprocess.STDOUT)
            record = dict(label=label, pid=process.pid, command=list(map(str, command)),
                          started=time.time(), gpu=self.gpu, cpu_mask=sorted(os.sched_getaffinity(process.pid)))
            write(self.work / f'{label}.process.json', record)
            self.status(phase=label, worker_pid=process.pid, log=str(logfile))
            while process.poll() is None:
                self.status(log_bytes=logfile.stat().st_size, log_mtime=logfile.stat().st_mtime)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            write(self.work / f'{label}.exit.json', dict(code=process.returncode, time=time.time()))
            if process.returncode:
                raise RuntimeError(f'{label} exited {process.returncode}; see {logfile}')

    def validate(self):
        assert self.request['batch_size'] == 16 and self.request['steps'] == 158000
        os.sched_setaffinity(0, self.request['cpu_mask'])
        assert sorted(os.sched_getaffinity(0)) == sorted(self.request['cpu_mask'])
        info = read(self.dataset / 'meta/info.json')
        assert (info['total_episodes'], info['total_frames']) == (892, 443502)
        source = ROOT / 'datasets/af60v8f_ds_dagger3/lerobot/ds'
        assert all((self.dataset / key).resolve() == (source / key).resolve()
                   for key in ['data', 'videos'])
        assert subprocess.check_output(['git', '-C', str(ROOT / 'Isaac-GR00T'),
            'rev-parse', 'HEAD'], text=True).strip() == self.request['gr00t_commit']
        for relative, digest in self.request['dataset_meta_hashes'].items():
            assert hashlib.sha256((self.dataset / relative).read_bytes()).hexdigest() == digest, relative
        for relative, digest in self.request['runtime_hashes'].items():
            assert hashlib.sha256((HERE.parent / relative).read_bytes()).hexdigest() == digest, relative
        assert len(list((ROOT / 'ref_motions' / POOL).glob('pick_*.pkl'))) == 100
        assert not self.checkpoint.parent.exists() and not self.output.exists()
        original_box = ROOT / 'dline/af60v8f57/box.json'
        (self.work / 'box.json').write_bytes(original_box.read_bytes())
        write(self.work / 'recipe.json', dict(self.request,
              dataset=str(self.dataset), episodes=892, frames=443502,
              effective_raw_frame_passes=158000 * 16 / 443502,
              initialization='nvidia/GR00T-N1.7-3B independently from pretrained',
              learning_rate=0.0001, optimizer='adamw_torch', precision='bf16',
              warmup_ratio=0.05, scheduler='cosine', weight_decay=1e-5, seed=42,
              collection_reused=True, evaluation_pool=POOL, evaluation_episodes=100))

    def run(self):
        with (self.work / 'controller.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
            try:
                self.validate()
                self.reserve()
                groot = ROOT / 'Isaac-GR00T'
                self.execute('train', [groot / '.venv/bin/python',
                    'gr00t/experiment/launch_finetune.py', '--base-model-path', 'nvidia/GR00T-N1.7-3B',
                    '--dataset-path', self.dataset, '--embodiment-tag', 'unitree_g1_sonic',
                    '--num-gpus', '1', '--global-batch-size', '16', '--gradient-accumulation-steps', '1',
                    '--learning-rate', '0.0001', '--weight-decay', '0.00001', '--warmup-ratio', '0.05',
                    '--dataloader-num-workers', '8', '--output-dir', ROOT / 'checkpoints',
                    '--experiment-name', self.name, '--max-steps', '158000',
                    '--save-steps', '20000', '--save-total-limit', '1'], groot)
                assert read(self.checkpoint / 'trainer_state.json')['global_step'] == 158000
                index = read(self.checkpoint / 'model.safetensors.index.json')['weight_map']
                assert all((self.checkpoint / name).stat().st_size > 8 for name in set(index.values()))
                assert (self.checkpoint / 'optimizer.pt').stat().st_size > 0
                write(self.work / 'trained.json', dict(checkpoint=str(self.checkpoint), steps=158000,
                                                       batch_size=16, time=time.time()))
                self.release()
                # Agent review is required before a new Isaac allocation, including
                # its graphics context. Do not hold an idle GPU while awaiting it.
                self.status(phase='await_eval_review', worker_pid=None, reserved=False)
                while not (self.work / 'eval_release.json').exists():
                    time.sleep(10)
                release = read(self.work / 'eval_release.json')
                assert release['run'] == self.name and release['gpu'] == self.gpu
                assert release['reviewed_at'] >= read(self.work / 'trained.json')['time']
                self.reserve()
                self.output.mkdir(parents=True)
                sim = HERE.parent / 'Training/scripts/reinforcement_learning/rsl_rl'
                sonic = ROOT / 'sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000'
                for shard in range(4):
                    ids = self.work / f'eval_ids{shard}.txt'
                    ids.write_text(' '.join(map(str, range(shard, 100, 4))) + '\n')
                    self.execute(f'eval_shard{shard}', [sys.executable, sim / 'eval_vla_sonic.py',
                        '--headless', '--sonic-pt', sonic, '--ref-motions-path', POOL,
                        '--vla-checkpoint', self.checkpoint, '--waist-dof', '29', '--motions-from', ids,
                        '--num-episodes', '25', '--seed', str(100 + shard),
                        '--traj-dump', self.output / f'b{shard}'], HERE.parent / 'Training')
                self.execute('score', [sys.executable, HERE / 'pick_budget_score.py',
                                      self.output, self.work / 'box.json'], HERE.parent)
                report = read(self.output / 'results.json')
                assert report['episodes'] == 100 and report['pool'] == POOL
                assert sorted(row['motion_id'] for row in report['per_episode']) == list(range(100))
                write(self.work / 'done.json', dict(time=time.time(), results=report))
                self.status(phase='complete', worker_pid=None)
            except BaseException as error:
                self.status(phase='failed', error=repr(error))
                write(self.work / 'abort.json', self.state)
                raise
            finally:
                self.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    Run(parser.parse_args().request).run()
