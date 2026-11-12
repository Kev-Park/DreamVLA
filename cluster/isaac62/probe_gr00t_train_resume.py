"""Two bounded real GR00T updates with a full Trainer checkpoint and fresh-process resume.

Run in the existing GR00T environment, not the simulator environment. Batch one,
no warmup and a two-update schedule are validation settings, not a study recipe.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=('initial', 'resume'), required=True)
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0', '1'), 'Expose one reviewed physical GPU'
    import torch
    from transformers import TrainerCallback, TrainingArguments, set_seed
    from gr00t.configs.base_config import get_default_config
    from gr00t.experiment.trainer import Gr00tTrainer
    from gr00t.experiment.utils import CheckpointFormatCallback
    from gr00t.model import MODEL_REGISTRY
    torch.set_num_threads(1)
    set_seed(42)
    args.output.mkdir(parents=True, exist_ok=True)
    resume = args.phase == 'resume'
    checkpoint = args.output/'checkpoint-1'
    assert checkpoint.is_dir() if resume else not checkpoint.exists()
    report = dict(started=time.time(), phase=args.phase, completed=False, torch=torch.__version__,
                  source_checkpoint=str(args.checkpoint), dataset=str(args.dataset), batch_size=1,
                  maximum_steps=2, warmup_ratio=0, device='cuda:0')
    config = get_default_config().load_dict({'data': {'download_cache': False, 'datasets': [
        {'dataset_paths': [str(args.dataset)], 'mix_ratio': 1., 'embodiment_tag': 'unitree_g1_sonic'}]}})
    config.training.start_from_checkpoint = str(args.checkpoint)
    config.training.transformers_local_files_only = True
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.load_bf16 = False
    config.model.backbone_trainable_params_fp32 = True
    config.model.model_name = 'nvidia/Cosmos-Reason2-2B'
    config.model.use_relative_action = True
    config.model.reproject_vision = False
    # Keep the production shard size/split rate: shards smaller than an episode
    # subsequence can create empty greedy-assignment shards in the real loader.
    config.data.num_shards_per_epoch = 1
    cfg = args.output/'experiment_cfg';cfg.mkdir(exist_ok=True)
    config.save(cfg/'config.yaml')
    pipeline = MODEL_REGISTRY.get(type(config.model))(config, cfg)

    def samples(model):
        return {name: param.detach().flatten()[:128].float().cpu().tolist()
                for name, param in model.named_parameters() if param.requires_grad}

    def optimizer_steps(optimizer):
        values = [int(s['step'].item()) if torch.is_tensor(s['step']) else int(s['step'])
                  for s in optimizer.state.values() if 'step' in s]
        assert values and min(values) == max(values), (min(values, default=-1), max(values, default=-1))
        return values[0]

    try:
        pipeline.setup()
        model = pipeline.return_model()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
        # One full model + Adam moments and a small safety margin, never two snapshots.
        required = model_bytes+8*trainable+2*1024**3
        assert resume or shutil.disk_usage(args.output).free > required, (required, shutil.disk_usage(args.output).free)
        report.update(trainable_parameters=trainable, model_bytes=model_bytes,
                      planned_checkpoint_bytes_upper_bound=required)
        processor = pipeline.return_processor()
        processor_dir = args.output/'processor';processor.save_pretrained(processor_dir)
        train_dataset, eval_dataset = pipeline.return_dataset()
        initial = samples(model)

        class Gate(TrainerCallback):
            def on_train_begin(self, train_args, state, control, model=None, optimizer=None, **kwargs):
                report['initial_global_step'] = state.global_step
                if resume:
                    assert state.global_step == 1
                    report['loaded_optimizer_step'] = optimizer_steps(optimizer)
                    assert report['loaded_optimizer_step'] == 1
                    expected = json.loads((args.output/'initial.json').read_text())['parameter_samples']
                    assert samples(model) == expected, 'Resumed weights differ from saved step-one weights'
                else:
                    assert state.global_step == 0

            def on_step_end(self, train_args, state, control, **kwargs):
                if not resume and state.global_step == 1:
                    control.should_training_stop = True
                return control

        training_args = TrainingArguments(output_dir=str(args.output), max_steps=2,
            per_device_train_batch_size=1, learning_rate=1e-4, warmup_ratio=0,
            lr_scheduler_type='cosine', optim='adamw_torch', weight_decay=1e-5,
            bf16=True, tf32=True, dataloader_num_workers=0, logging_steps=1,
            save_steps=1, save_total_limit=1, save_only_model=False,
            save_strategy='no' if resume else 'steps', report_to='none',
            remove_unused_columns=False, ignore_data_skip=True, seed=42)
        trainer = Gr00tTrainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=eval_dataset, data_collator=pipeline.return_collator(),
            multiprocessing_context=config.data.multiprocessing_context, callbacks=[Gate()])
        trainer.add_callback(CheckpointFormatCallback(run_name='isaac62_validation',
            exp_cfg_dir=cfg, processor_dir=processor_dir))
        result = trainer.train(resume_from_checkpoint=str(checkpoint) if resume else False)
        expected_step = 2 if resume else 1
        assert trainer.state.global_step == expected_step
        assert math.isfinite(result.training_loss), result
        final = samples(model)
        reference = json.loads((args.output/'initial.json').read_text())['parameter_samples'] if resume else initial
        differences = {k:max(abs(a-b) for a,b in zip(final[k],reference[k])) for k in final}
        assert max(differences.values()) > 0, 'No sampled trainable parameter changed'
        assert all(math.isfinite(x) for values in final.values() for x in values)
        step = optimizer_steps(trainer.optimizer)
        assert step == expected_step, step
        if not resume:
            assert all((checkpoint/n).is_file() for n in ('optimizer.pt','scheduler.pt','trainer_state.json'))
            assert json.loads((checkpoint/'trainer_state.json').read_text())['global_step'] == 1
        report.update(completed=True, global_step=trainer.state.global_step, optimizer_step=step,
                      loss=result.training_loss, parameter_samples=final,
                      max_sampled_parameter_change=max(differences.values()),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        report['error'] = repr(exc)
        raise
    finally:
        report['finished'] = time.time()
        (args.output/(args.phase+'.json')).write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k != 'parameter_samples'}), flush=True)


if __name__ == '__main__':
    main()
