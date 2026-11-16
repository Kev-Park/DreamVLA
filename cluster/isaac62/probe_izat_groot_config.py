"""Check the real contributed launcher against the installed native GR00T config.

The execution boundary is intercepted: no model is loaded and no training runs.
This uses stdlib mocking so the production GR00T environment needs no pytest.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    args.output.mkdir(parents=True, exist_ok=False)
    from dreamvla_drawer_button.io import write_json
    from dreamvla_drawer_button.finetune import main as finetune
    dataset = args.output/'data'
    write_json(dataset/'meta/info.json', {'total_frames': 499*200})
    base = args.output/'base'; base.mkdir()
    captured = []
    with patch.dict(sys.modules, {'gr00t.experiment.experiment': SimpleNamespace(run=captured.append)}), \
         patch.dict(os.environ, WORLD_SIZE='1'), \
         patch.object(sys, 'argv', ['finetune', '--dataset', str(dataset),
                                   '--base-model', str(base), '--output', str(args.output/'student')]):
        finetune()
    assert len(captured) == 1
    cfg = captured[0]
    assert cfg.training.max_steps == 35554
    assert cfg.training.global_batch_size*cfg.training.gradient_accumulation_steps == 16
    assert cfg.training.lr_scheduler_type == 'constant'
    assert cfg.training.learning_rate == 1e-4
    assert not cfg.model.tune_visual and not cfg.model.tune_llm
    assert cfg.model.action_horizon == 40 and not cfg.model.use_relative_action
    cfg.save(args.output/'config.yaml')
    report = dict(completed=True, finished=time.time(), max_steps=cfg.training.max_steps,
                  effective_batch=16, action_horizon=40, use_relative_action=False,
                  scope='Actual native configuration assembly; no model loading, optimizer or serving')
    (args.output/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
