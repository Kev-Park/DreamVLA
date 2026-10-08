"""Bounded real-data batch throughput test; never saves production weights.

A separate watchdog resumes the identified production process on every exit,
including a hung benchmark. Production optimizer state stays resident in memory.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--batch', type=int, choices=[32, 64, 128], required=True)
p.add_argument('--output', required=True)
p.add_argument('--production-pid', type=int, required=True)
p.add_argument('--warmup', type=int, default=30)
p.add_argument('--measure', type=int, default=100)
p.add_argument('--pause-limit', type=int, default=180)
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=False)
proc = Path(f'/proc/{a.production_pid}')
assert proc.stat().st_uid == os.getuid()
assert b'launch_finetune.py' in (proc/'cmdline').read_bytes()
assert b'af60v8f57_dagger2_run01' in (proc/'cmdline').read_bytes()
birth = (proc/'stat').read_text().split()[21]
paused = False
guardian = None

import torch
from transformers import TrainerCallback
from transformers.trainer_utils import SaveStrategy
from gr00t.configs.base_config import Config
import gr00t.experiment.experiment as experiment


def resume():
    global paused
    if paused and proc.exists() and (proc/'stat').read_text().split()[21] == birth:
        os.kill(a.production_pid, signal.SIGCONT)
        paused = False
        print('Production resumed with optimizer state intact', flush=True)


class Timing(TrainerCallback):
    def on_train_begin(self, args, state, control, **kw):
        global paused, guardian
        # This independent process kills only this benchmark on timeout, then
        # resumes the original PID if its creation identity still matches.
        guard_code = '''import os,sys,time,signal,pathlib
time.sleep(float(sys.argv[1]))
try: os.kill(int(sys.argv[2]),signal.SIGKILL)
except ProcessLookupError: pass
p=pathlib.Path('/proc/'+sys.argv[3]+'/stat')
if p.exists() and p.read_text().split()[21]==sys.argv[4]:
    os.kill(int(sys.argv[3]),signal.SIGCONT)
'''
        guardian = subprocess.Popen([sys.executable, '-c', guard_code,
            str(a.pause_limit), str(os.getpid()), str(a.production_pid), birth],
            start_new_session=True)
        os.kill(a.production_pid, signal.SIGSTOP)
        paused = True
        time.sleep(1)  # Let already submitted production GPU work drain.
        self.times = []
        self.started = None
        self.previous = None
        self.losses = []
        torch.cuda.reset_peak_memory_stats()
        print('Production paused; independent resume watchdog armed', flush=True)

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        now = time.monotonic()
        if state.global_step == a.warmup:
            self.started = self.previous = now
        elif state.global_step > a.warmup:
            self.times.append(now-self.previous)
            self.previous = now

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and 'loss' in logs:
            self.losses.append(logs['loss'])

    def on_train_end(self, args, state, control, **kw):
        seconds = self.previous-self.started
        result = dict(batch=a.batch, measured_steps=len(self.times),
            warmup_steps=a.warmup, seconds=seconds,
            steps_per_second=len(self.times)/seconds,
            frames_per_second=len(self.times)*a.batch/seconds,
            median_step_seconds=statistics.median(self.times),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu=torch.cuda.get_device_name(), losses=self.losses,
            production_memory_retained=True, precision='bf16', status='completed')
        (out/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        resume()


class BenchTrainer(experiment.Gr00tTrainer):
    def __init__(self, *args, **kwargs):
        kwargs['args'].save_strategy = SaveStrategy.NO
        super().__init__(*args, **kwargs)
        self.add_callback(Timing())

    def save_model(self, *args, **kwargs):
        pass


cfg = Config().load(Path(a.config))
assert cfg.training.optim == 'adamw_torch'
assert cfg.training.bf16
assert cfg.training.gradient_accumulation_steps == 1
cfg.training.global_batch_size = a.batch
cfg.training.batch_size = None
cfg.training.num_gpus = 1
cfg.training.use_ddp = False
cfg.training.max_steps = a.warmup+a.measure
cfg.training.save_steps = cfg.training.max_steps+1
cfg.training.output_dir = str(out/'training')
cfg.training.experiment_name = 'batch_benchmark'
cfg.training.use_wandb = False
cfg.training.enable_profiling = False
experiment.Gr00tTrainer = BenchTrainer
try:
    experiment.run(cfg)
except Exception as error:
    (out/'result.json').write_text(json.dumps(dict(batch=a.batch,
        status='failed', error=str(error), production_memory_retained=True), indent=2))
    raise
finally:
    resume()
    if guardian is not None:
        guardian.terminate()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
