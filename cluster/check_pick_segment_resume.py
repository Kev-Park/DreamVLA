"""Small CPU-only serialization check using Adroit's installed GR00T Trainer.

Run on a Slurm CPU allocation, never on the local editing machine. This checks
model/optimizer/scheduler/RNG continuation, not real-data stream equivalence.
"""
import json
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn
from torch.utils.data import IterableDataset
from transformers import TrainingArguments, TrainerCallback
from gr00t.experiment.trainer import Gr00tTrainer
import pick_fullbase_segment as segment


class RepeatedData(IterableDataset):
    seed = 42

    def reset_seed(self, seed):
        self.seed = seed

    def __iter__(self):
        for _ in range(512):
            yield {'features': torch.ones(3)}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, features):
        return {'loss': (self.linear(self.dropout(features)) - 0.3).square().mean()}


def run(output, start, stop):
    torch.manual_seed(7)
    model = TinyModel()
    args = TrainingArguments(output_dir=str(output), max_steps=6,
        per_device_train_batch_size=16, gradient_accumulation_steps=1,
        learning_rate=1e-4, weight_decay=1e-5, warmup_ratio=0.05,
        lr_scheduler_type='cosine', optim='adamw_torch', seed=42,
        dataloader_num_workers=0, save_steps=20000, save_total_limit=1,
        report_to='none', logging_steps=20000, use_cpu=True,
        save_only_model=False, remove_unused_columns=False)
    trainer = Gr00tTrainer(model=model, args=args, train_dataset=RepeatedData())
    trainer.add_callback(segment.segment_callback(TrainerCallback, start, stop, float('inf')))
    trainer.train(resume_from_checkpoint=True)
    assert trainer.state.global_step == stop
    ck = output / f'checkpoint-{stop}'
    for name in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'training_args.bin']:
        assert (ck / name).stat().st_size > 0, name
    assert json.loads((ck / 'trainer_state.json').read_text())['max_steps'] == 6
    return trainer


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), (a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b, (a, b)


def main():
    torch.set_num_threads(1)
    segment.TOTAL_STEPS = 6
    with tempfile.TemporaryDirectory(prefix='pick-resume-check-', dir=Path.home() / 'kevin/dline') as temp:
        root = Path(temp)
        continuous = run(root / 'continuous', 0, 6)
        run(root / 'segmented', 0, 2)
        run(root / 'segmented', 2, 4)
        resumed = run(root / 'segmented', 4, 6)
        equal(continuous.model.state_dict(), resumed.model.state_dict())
        equal(continuous.optimizer.state_dict(), resumed.optimizer.state_dict())
        equal(continuous.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
        rng_a = torch.load(root / 'continuous/checkpoint-6/rng_state.pth', weights_only=False)
        rng_b = torch.load(root / 'segmented/checkpoint-6/rng_state.pth', weights_only=False)
        equal(rng_a['cpu'], rng_b['cpu'])
        result = dict(passed=True, total_updates=6, segment_boundaries=[2, 4, 6],
                      model_exact=True, optimizer_exact=True, scheduler_exact=True, torch_rng_exact=True,
                      cuda_used=torch.cuda.is_initialized(),
                      limitation='Repeated toy inputs; real GR00T iterable data reseeds on resume.')
        assert not result['cuda_used']
        Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
