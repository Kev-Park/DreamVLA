"""One real contributed native GR00T update and model-only export on a reviewed GPU.

Use the existing converted synthetic fixture. Smoke overrides are recorded; this
is not a scientific training run. Full Adam save/resume is validated separately.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.name == 'posix' and os.environ.get('CUDA_VISIBLE_DEVICES', '').isdecimal()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(started=time.time(), completed=False, synthetic_dataset=True,
                  scientific_training_run=False, scope=__doc__)
    try:
        import torch
        from transformers import TrainingArguments
        from gr00t.experiment import experiment
        from gr00t.experiment.trainer import Gr00tTrainer
        from dreamvla_drawer_button.finetune import main as finetune
        torch.set_num_threads(1)
        assert shutil.disk_usage(args.output).free > 15*1024**3
        captured = []
        student = args.output/'student'
        with patch.object(experiment, 'run', captured.append), patch.object(sys, 'argv', [
            'finetune', '--dataset', str(args.dataset), '--base-model', str(args.base_model),
            '--output', str(student), '--batch-size', '1']):
            finetune()
        assert len(captured) == 1
        config = captured[0]
        assert not config.model.use_relative_action and config.model.action_horizon == 40
        assert not config.model.tune_visual and not config.model.tune_llm
        config.data.num_shards_per_epoch = 1
        config.training.max_steps = 1
        config.training.logging_steps = 1
        config.training.dataloader_num_workers = 0
        config.training.transformers_local_files_only = True
        config.training.save_only_model = True
        config.training.save_total_limit = 1
        report['overrides'] = dict(effective_batch=1, max_steps=1, workers=0,
                                   num_shards_per_epoch=1, checkpoint='final model only',
                                   periodic_checkpointing=False)

        def samples(model):
            selected = [(name, p) for name, p in model.named_parameters()
                        if p.requires_grad and name.endswith('weight')][:8]
            assert selected
            return {name: p.detach().flatten()[:64].float().cpu().clone() for name, p in selected}

        original_train = Gr00tTrainer.train

        def measured_train(trainer, *positional, **keywords):
            before = samples(trainer.model)
            report['trainable_parameters'] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            result = original_train(trainer, *positional, **keywords)
            after = samples(trainer.model)
            change = max(float((after[key]-before[key]).abs().max()) for key in before)
            assert change > 0 and all(torch.isfinite(value).all() for value in after.values())
            assert trainer.state.global_step == 1 and math.isfinite(result.training_loss)
            steps = [float(state['step']) for state in trainer.optimizer.state.values() if 'step' in state]
            assert steps and min(steps) == max(steps) == 1
            trainer.save_state()
            report.update(global_step=1, optimizer_step=1, loss=result.training_loss,
                          max_sampled_parameter_change=change)
            return result

        # The upstream run saves again at the output root after training. For
        # this bounded test save that final model once, avoiding a second full
        # model/Adam snapshot. No production launcher defaults are changed.
        def smoke_training_arguments(**keywords):
            keywords['save_strategy'] = 'no'
            return TrainingArguments(**keywords)

        with patch.object(experiment, 'TrainingArguments', smoke_training_arguments), \
             patch.object(Gr00tTrainer, 'train', measured_train):
            experiment.run(config)
        weights = list(student.glob('*.safetensors'))
        assert weights and (student/'config.json').is_file()
        assert (student/'processor').is_dir() and (student/'experiment_cfg/config.yaml').is_file()
        assert not list(student.glob('checkpoint-*')) and not (student/'optimizer.pt').exists()
        assert json.loads((student/'trainer_state.json').read_text())['global_step'] == 1
        report.update(completed=True, model_path=str(student),
                      weight_bytes=sum(path.stat().st_size for path in weights),
                      weight_files=[path.name for path in weights],
                      limitation='Model-only export; no optimizer-resume or policy-success claim')
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['finished'] = time.time()
        if 'torch' in locals() and torch.cuda.is_initialized():
            report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        (args.output/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
