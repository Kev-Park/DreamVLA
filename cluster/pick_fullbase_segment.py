"""Checkpointed slices of one 158000-update Pick run; never reset the LR horizon."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

TOTAL_STEPS = 158000
SEGMENT_STEPS = 20000


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def checkpoint_step(checkpoint):
    """Reject partial/weights-only checkpoints before allowing continuation."""
    checkpoint = Path(checkpoint)
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    step = int(state['global_step'])
    assert checkpoint.name == f'checkpoint-{step}'
    assert state['max_steps'] == TOTAL_STEPS, 'LR horizon must remain 158000'
    assert 0 < step <= TOTAL_STEPS
    required = ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin',
                'config.json', 'processor_config.json']
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
    required += sorted(set(index['weight_map'].values()))
    assert index['weight_map']
    for name in required:
        assert (checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0, name
    return step


def latest_checkpoint(output):
    candidates = sorted(Path(output).glob('checkpoint-*'),
                        key=lambda p: int(p.name.split('-')[-1]))
    if not candidates:
        return None, 0
    latest = candidates[-1]
    return latest, checkpoint_step(latest)


def segment_callback(base_class, start, stop, deadline, clock=time.monotonic):
    class Boundary(base_class):
        reason = None

        def on_train_begin(self, args, state, control, **kwargs):
            assert args.max_steps == state.max_steps == TOTAL_STEPS
            assert state.global_step == start, (state.global_step, start)
            assert not args.save_only_model
            assert args.per_device_train_batch_size == 16
            assert args.gradient_accumulation_steps == 1

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= stop or clock() >= deadline:
                self.reason = 'target' if state.global_step >= stop else 'allocation_budget'
                control.should_save = True
                control.should_training_stop = True
            return control

    return Boundary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('run')
    parser.add_argument('segment', type=int)
    args = parser.parse_args()
    root = Path.home() / 'kevin'
    work = root / 'dline' / args.run
    repo = root / 'Isaac-GR00T'
    output = root / 'checkpoints' / args.run
    spec = json.loads((work / 'request.json').read_text())
    assert spec['run'] == args.run
    info = json.loads((args.dataset / 'meta/info.json').read_text())
    assert (info['total_episodes'], info['total_frames']) == (892, 443502)
    for name, digest in spec['dataset_meta_hashes'].items():
        assert hashlib.sha256((args.dataset / name).read_bytes()).hexdigest() == digest, name
    import subprocess
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == spec['gr00t_commit']
    receipt = work / f'segment-{args.segment:03d}.json'
    assert not receipt.exists(), 'A successful segment must never be rerun blindly'
    checkpoint, start = latest_checkpoint(output)
    assert start < TOTAL_STEPS
    if args.segment == 1:
        assert checkpoint is None, 'First segment must start from pretrained'
    else:
        previous = json.loads((work / f'segment-{args.segment - 1:03d}.json').read_text())
        assert checkpoint is not None and start == previous['end_step']
    stop = min(start + SEGMENT_STEPS, TOTAL_STEPS)
    # Allow 40 minutes for saving and shutdown; include model/data startup in the budget.
    end = int(os.environ['SLURM_JOB_END_TIME'])
    seconds = end - time.time() - 2400
    assert seconds > 0
    deadline = time.monotonic() + seconds
    sys.path.insert(0, str(repo))
    from transformers import TrainerCallback
    from gr00t.experiment.trainer import Gr00tTrainer
    boundary = segment_callback(TrainerCallback, start, stop, deadline)
    original_train = Gr00tTrainer.train

    def train_slice(trainer, *positional, **kwargs):
        trainer.add_callback(boundary)
        result = original_train(trainer, *positional, **kwargs)
        assert start < trainer.state.global_step <= stop
        return result

    Gr00tTrainer.train = train_slice
    write_json(work / f'segment-{args.segment:03d}.start.json', dict(
        time=time.time(), job_id=os.environ['SLURM_JOB_ID'], start_step=start,
        target_step=stop, checkpoint=str(checkpoint) if checkpoint else None,
        total_schedule_steps=TOTAL_STEPS, allocation_end=end,
        resume_data_note='Official GR00T resume reseeds the iterable data stream; not bitwise equivalent to uninterrupted training.'))
    sys.argv = [str(repo / 'gr00t/experiment/launch_finetune.py'),
                '--base-model-path', 'nvidia/GR00T-N1.7-3B', '--dataset-path', str(args.dataset),
                '--embodiment-tag', 'unitree_g1_sonic', '--num-gpus', '1',
                '--global-batch-size', '16', '--gradient-accumulation-steps', '1',
                '--learning-rate', '0.0001', '--weight-decay', '0.00001',
                '--warmup-ratio', '0.05', '--dataloader-num-workers', '8',
                '--output-dir', str(root / 'checkpoints'), '--experiment-name', args.run,
                '--max-steps', str(TOTAL_STEPS), '--save-steps', '20000', '--save-total-limit', '1']
    runpy.run_path(sys.argv[0], run_name='__main__')
    checkpoint, step = latest_checkpoint(output)
    assert start < step <= stop and boundary.reason is not None
    result = dict(time=time.time(), job_id=os.environ['SLURM_JOB_ID'], start_step=start,
                  end_step=step, target_step=stop, stop_reason=boundary.reason,
                  checkpoint=str(checkpoint), total_schedule_steps=TOTAL_STEPS, batch_size=16)
    write_json(receipt, result)
    if step == TOTAL_STEPS:
        write_json(work / 'trained.json', dict(time=time.time(), checkpoint=str(checkpoint),
                                              steps=step, batch_size=16, segmented=True))
        (work / 'train.exit_code.txt').write_text('0\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
