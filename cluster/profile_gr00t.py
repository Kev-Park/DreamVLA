"""Isolated short throughput benchmark using a production run's saved config.

Never saves model weights or modifies production checkpoints. DDP/ZeRO and data
worker count are the only tuning variables; batch, optimizer and model stay fixed.
"""
import argparse
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--mode', choices=['ddp', 'zero2'], required=True)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--steps', type=int, default=80)
parser.add_argument('--output', required=True)
parser.add_argument('--cpu-grid', action='store_true')
parser.add_argument('--compile-action', action='store_true')
parser.add_argument('--cache-geometry', action='store_true')
args = parser.parse_args()

import torch
from transformers import TrainerCallback
from transformers.trainer_utils import SaveStrategy
from gr00t.configs.base_config import Config
import gr00t.experiment.experiment as experiment

rank = int(os.environ.get('RANK', '0'))
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)


def trace_ready(prof):
    prof.export_chrome_trace(str(output/f'trace_rank{rank}.json'))
    averages = prof.key_averages()
    (output/f'profile_rank{rank}.txt').write_text(
        averages.table(sort_by='self_cpu_time_total', row_limit=40)+'\n'+
        averages.table(sort_by='self_cuda_time_total', row_limit=40))


class Timing(TrainerCallback):
    def __init__(self):
        self.times = []
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(skip_first=20, wait=1, warmup=1, active=4, repeat=1),
            on_trace_ready=trace_ready)

    def on_train_begin(self, trainer_args, state, control, **kwargs):
        self.started = time.monotonic()
        self.prof.start()

    def on_step_begin(self, trainer_args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.step_started = time.monotonic()

    def on_step_end(self, trainer_args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.times.append(time.monotonic()-self.step_started)
        self.prof.step()
        if state.global_step == 30:
            self.steady_started = time.monotonic()

    def on_train_end(self, trainer_args, state, control, **kwargs):
        self.prof.stop()
        result = dict(mode=args.mode, workers=args.workers, rank=rank,
            cpu_grid=args.cpu_grid,
            compile_action=args.compile_action,
            cache_geometry=args.cache_geometry,
            steps=state.global_step, steady_steps_per_second=(state.global_step-30)/(time.monotonic()-self.steady_started),
            steady_compute_step_seconds=sum(self.times[30:])/len(self.times[30:]),
            max_memory_bytes=torch.cuda.max_memory_allocated())
        (output/f'result_rank{rank}.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


class BenchmarkTrainer(experiment.Gr00tTrainer):
    def __init__(self, *trainer_args, **kwargs):
        kwargs['args'].save_strategy = SaveStrategy.NO
        super().__init__(*trainer_args, **kwargs)
        if args.cpu_grid:
            visual = self.model.backbone.model.visual
            # These helpers only use grid sizes as Python/CPU shape metadata.
            # Leave the original GPU grid intact for flash-attention cu_seqlens.
            verified = set()
            for name in ('rot_pos_emb', 'fast_pos_embed_interpolate'):
                original = getattr(visual, name)
                cache = {}
                def cpu_shape(grid, original=original, name=name, cache=cache):
                    cpu_grid = grid.cpu()
                    key = (cpu_grid.numpy().tobytes(), str(grid.device), torch.is_autocast_enabled('cuda'))
                    if args.cache_geometry and key in cache:
                        return cache[key]
                    actual = original(cpu_grid)
                    if name not in verified:
                        with torch.no_grad():
                            expected = original(grid)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        verified.add(name)
                        print(f'{name}: outputs verified bitwise equal on {grid.device}', flush=True)
                    if args.cache_geometry:
                        assert not any(p.requires_grad for p in visual.parameters())
                        cache[key] = actual.detach()
                    return actual
                setattr(visual, name, cpu_shape)
        if args.cache_geometry:
            vlm = self.model.backbone.model
            position_cache = {}
            def position_inputs(module, positional, inputs):
                ids = inputs['input_ids'].cpu()
                mask = inputs['attention_mask'].cpu()
                grid = inputs['image_grid_thw'].cpu()
                key = (tuple(ids.shape), ids.numpy().tobytes(), mask.numpy().tobytes(), grid.numpy().tobytes())
                if key not in position_cache:
                    position_ids, deltas = module.model.get_rope_index(
                        input_ids=ids, attention_mask=mask, image_grid_thw=grid)
                    expected, expected_deltas = module.model.get_rope_index(
                        input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                        image_grid_thw=inputs['image_grid_thw'])
                    position_ids = position_ids.to(inputs['input_ids'].device)
                    deltas = deltas.to(inputs['input_ids'].device)
                    torch.testing.assert_close(position_ids, expected, rtol=0, atol=0)
                    torch.testing.assert_close(deltas, expected_deltas, rtol=0, atol=0)
                    if len(position_cache) >= 8:
                        position_cache.clear()
                    position_cache[key] = position_ids, deltas
                    print('Cached position IDs verified bitwise equal', flush=True)
                inputs['position_ids'], module.model.rope_deltas = position_cache[key]
                return positional, inputs
            vlm.register_forward_pre_hook(position_inputs, with_kwargs=True)
        if args.compile_action:
            # Preserve parameter names and optimizer; compile the forward callable.
            # Keep ATen random operations so RNG algorithms stay unchanged.
            torch._inductor.config.fallback_random = True
            self.model.action_head.forward = torch.compile(
                self.model.action_head.forward, mode='reduce-overhead')
        self.add_callback(Timing())

    def save_model(self, *args, **kwargs):
        pass


config = Config().load(Path(args.config))
assert config.training.global_batch_size == 16
assert config.training.optim == 'adamw_torch'
config.training.use_ddp = args.mode == 'ddp'
config.training.dataloader_num_workers = args.workers
config.training.max_steps = args.steps
config.training.save_steps = args.steps+1
config.training.output_dir = str(output)
config.training.experiment_name = 'benchmark'
config.training.use_wandb = False
config.training.enable_profiling = False
experiment.Gr00tTrainer = BenchmarkTrainer
try:
    experiment.run(config)
finally:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
