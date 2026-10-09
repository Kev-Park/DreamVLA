"""Explicitly approved recovery of weights-only GR00T checkpoints.

Preserves global step and the original LR schedule, but cannot recover AdamW
moments or RNG state. Writes that limitation into the run and saves full states
on subsequent checkpoints. Run inside the existing GR00T environment/repo.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import warnings


def restore_schedule(scheduler, step):
    # GR00T uses a LambdaLR cosine schedule. Fail closed for other schedulers.
    if not hasattr(scheduler, "lr_lambdas"):
        raise TypeError("Recovery requires the original LambdaLR scheduler")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler.step(step)
    return scheduler.get_last_lr()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining.pop(0)
    checkpoint = args.checkpoint.resolve(strict=True)
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    step = state["global_step"]
    assert step > 0
    assert not (checkpoint / "optimizer.pt").exists(), "Use normal resume for full checkpoints"
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    assert all((checkpoint / f).stat().st_size > 0 for f in set(index["weight_map"].values()))

    from transformers import TrainerCallback
    from gr00t.experiment.trainer import Gr00tTrainer

    class RecoverSchedule(TrainerCallback):
        def on_train_begin(self, args, state, control, lr_scheduler=None, **kwargs):
            assert state.global_step == step, (state.global_step, step)
            assert state.max_steps > step
            rates = restore_schedule(lr_scheduler, step)
            report = dict(checkpoint=str(checkpoint), resumed_global_step=step,
                target_steps=state.max_steps, learning_rates=rates,
                optimizer_state="reset: unavailable in weights-only checkpoint",
                rng_state="reset: unavailable in weights-only checkpoint",
                lr_schedule="restored to original global step",
                subsequent_checkpoints="full optimizer/scheduler/RNG states")
            (Path(args.output_dir) / "recovery.json").write_text(json.dumps(report, indent=2))
            print("RECOVERY " + json.dumps(report), flush=True)

    original = Gr00tTrainer.train

    def train(self, resume_from_checkpoint=None, **kwargs):
        self.args.save_only_model = False
        self.add_callback(RecoverSchedule())
        return original(self, resume_from_checkpoint=str(checkpoint), **kwargs)

    Gr00tTrainer.train = train
    launch = Path.cwd() / "gr00t/experiment/launch_finetune.py"
    sys.argv = [str(launch), *remaining]
    runpy.run_path(str(launch), run_name="__main__")


if __name__ == "__main__":
    main()
