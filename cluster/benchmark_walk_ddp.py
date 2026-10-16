"""Short two-GPU throughput check; its weights are not study results.

Launch with torchrun after an operator reviews account CPU/GPU occupancy.
Use a saved fine-tune configuration to preserve model/data/optimizer settings.
The shortened schedule measures throughput only and does not replace any
5.7-pass experimental fine-tune.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert int(os.environ["WORLD_SIZE"]) == 2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"

    from gr00t.configs.base_config import get_default_config
    from gr00t.experiment.experiment import run

    config = get_default_config().load(args.config)
    assert config.training.global_batch_size == 16
    assert config.training.gradient_accumulation_steps == 1
    assert config.training.optim == "adamw_torch"
    assert config.training.learning_rate == 1e-4
    assert config.training.start_from_checkpoint == "nvidia/GR00T-N1.7-3B"
    assert len(config.data.datasets) == 1
    config.load_config_path = None
    config.training.use_ddp = True
    config.training.num_gpus = 2
    config.training.dataloader_num_workers = 2  # Four workers across two ranks.
    config.training.max_steps = 80
    config.training.save_steps = 80
    config.training.save_total_limit = 1
    config.training.save_only_model = True
    config.training.use_wandb = False
    config.training.output_dir = str(args.output)
    config.training.experiment_name = None
    # The launcher checks that the output is new. A sibling rank may already
    # have created its directory, but neither rank may resume old weights.
    assert not list(args.output.glob("checkpoint-*")), "Never resume a diagnostic run"
    assert not (args.output / "model.safetensors.index.json").exists()
    if int(os.environ["RANK"]) == 0:
        print(json.dumps({"purpose": "throughput only; not a study fine-tune",
                          "global_batch": 16, "per_rank_batch": 8,
                          "optimizer": "adamw_torch", "learning_rate": 1e-4,
                          "steps": 80, "full_config": str(args.config)}), flush=True)
    run(config)


if __name__ == "__main__":
    main()
