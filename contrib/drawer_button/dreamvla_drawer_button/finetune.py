"""Explicit native NVIDIA GR00T fine-tuning configuration for SONIC actions.

This launcher uses the upstream online trainer, not historical cached-feature
head experiments. Frame-based budgets are nominal sample exposure, not a promise
that a sharded distributed sampler visits every frame exactly equally.
"""

import argparse
import json
import math
import os
from pathlib import Path

from .io import read
from decimal import Decimal, ROUND_CEILING


def training_budget(dataset, epochs=5.7, batch_size=16):
    info = read(Path(dataset) / "meta/info.json")
    if (
        epochs <= 0
        or not math.isfinite(epochs)
        or batch_size < 1
        or info["total_frames"] < 1
    ):
        raise ValueError("Invalid frame-based training budget")
    return int(
        (
            Decimal(info["total_frames"]) * Decimal(str(epochs)) / Decimal(batch_size)
        ).to_integral_value(rounding=ROUND_CEILING)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=5.7)
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Effective optimizer batch, across all ranks",
    )
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--scheduler", choices=("constant", "cosine"), default="constant")
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if (
        a.learning_rate <= 0
        or not math.isfinite(a.learning_rate)
        or not 0 <= a.warmup_ratio < 1
        or a.weight_decay < 0
    ):
        p.error("Invalid optimizer configuration")
    steps = training_budget(a.dataset, a.epochs, a.batch_size)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if a.batch_size % world:
        p.error("Effective batch size must be divisible by WORLD_SIZE")
    print(
        json.dumps(
            dict(
                max_steps=steps,
                raw_frame_epochs=a.epochs,
                effective_batch=a.batch_size,
                learning_rate=a.learning_rate,
                scheduler=a.scheduler,
                vlm_frozen=True,
            ),
            indent=2,
        )
    )
    if a.dry_run:
        return
    if not a.base_model.is_dir():
        raise FileNotFoundError(a.base_model)
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:
        raise FileExistsError("Choose a new output or explicitly resume this run")
    from .paths import add_external_paths

    add_external_paths()
    from gr00t.configs.base_config import get_default_config
    from gr00t.experiment.experiment import run

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    dict(
                        dataset_paths=[str(a.dataset.resolve())],
                        mix_ratio=1.0,
                        embodiment_tag="unitree_g1_sonic",
                    )
                ],
            }
        }
    )
    config.load_config_path = None
    modality = config.data.modality_configs["unitree_g1_sonic"]
    if modality["action"].delta_indices != list(range(40)) or modality[
        "action"
    ].modality_keys != ["motion_token", "left_hand_joints", "right_hand_joints"]:
        raise ValueError(
            "Upstream SONIC modality is incompatible; do not silently change action semantics"
        )
    config.data.seed = a.seed
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.tune_vlln = True
    config.model.action_horizon = 40
    config.model.state_dropout_prob = 0.2
    config.model.use_percentiles = True
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = False
    t = config.training
    t.start_from_checkpoint = str(a.base_model.resolve())
    t.output_dir = str(a.output.resolve())
    t.optim = "adamw_torch"
    t.global_batch_size = world
    t.gradient_accumulation_steps = a.batch_size // world
    t.num_gpus = world
    t.dataloader_num_workers = 2
    t.learning_rate = a.learning_rate
    t.lr_scheduler_type = a.scheduler
    t.warmup_ratio = a.warmup_ratio
    t.weight_decay = a.weight_decay
    t.max_steps = steps
    t.use_wandb = False
    t.save_steps = min(1000, steps)
    t.save_total_limit = 2
    t.save_only_model = False
    t.resume_from_checkpoint = a.resume
    run(config)


if __name__ == "__main__":
    main()
