"""Opt-in native config assembly test; does not load weights or train a model."""

import os
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.skipif(
    os.getenv("DREAMVLA_TEST_GROOT") != "1",
    reason="Requires the separate GR00T environment",
)
def test_native_finetune_configuration(tmp_path, monkeypatch):
    from dreamvla_drawer_button.io import write_json
    from dreamvla_drawer_button.paths import add_external_paths

    add_external_paths()
    from dreamvla_drawer_button.finetune import main

    captured = []
    monkeypatch.setitem(
        sys.modules, "gr00t.experiment.experiment", SimpleNamespace(run=captured.append)
    )
    dataset = tmp_path / "data"
    write_json(dataset / "meta/info.json", {"total_frames": 499 * 200})
    base = tmp_path / "base"
    base.mkdir()
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finetune",
            "--dataset",
            str(dataset),
            "--base-model",
            str(base),
            "--output",
            str(tmp_path / "student"),
        ],
    )
    main()
    config = captured[0]
    assert config.training.max_steps == 35554
    assert (
        config.training.global_batch_size * config.training.gradient_accumulation_steps
        == 16
    )
    assert config.training.lr_scheduler_type == "constant"
    assert config.training.learning_rate == 1e-4
    assert config.model.tune_visual is False and config.model.tune_llm is False
    assert (
        config.model.action_horizon == 40 and config.model.use_relative_action is False
    )
