import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from dreamvla_drawer_button.isaac.sonic_teacher_repair_contract import (
    RepairSpec,
    compose,
)
from dreamvla_drawer_button.actions import expert_token
from dreamvla_drawer_button.model_head import ROW_KEYS, restore_state
from dreamvla_drawer_button.checkpoint import capture_rng, restore_rng, atomic_save


def test_torch_numpy_controller_parity():
    rng = np.random.default_rng(19)
    ref = rng.normal(size=(4, 64)).astype(np.float32)
    residual = rng.normal(size=(4, 64)).astype(np.float32)
    got = compose(
        torch.from_numpy(ref), torch.from_numpy(residual), RepairSpec(scale=0.5)
    )
    np.testing.assert_array_equal(got.numpy(), expert_token(ref, residual))


def test_restore_only_sonic_row_and_shared_parameters():
    target = {k: torch.zeros(16, 2, 3) for k in ROW_KEYS}
    source = {k: torch.ones(1, 2, 3) for k in ROW_KEYS}
    target["shared"] = torch.zeros(3)
    source["shared"] = torch.ones(3)
    restore_state(target, source)
    assert len(ROW_KEYS) == 14
    for k in ROW_KEYS:
        assert target[k][11].eq(1).all()
        assert target[k][:11].eq(0).all() and target[k][12:].eq(0).all()
    assert target["shared"].eq(1).all()


def test_rng_resume_repeats_next_draw():
    rng = np.random.default_rng(91)
    state = capture_rng(rng)
    expected = (random.random(), np.random.rand(), rng.random(), torch.rand(3))
    restore_rng(state, rng)
    actual = (random.random(), np.random.rand(), rng.random(), torch.rand(3))
    assert expected[:3] == actual[:3]
    assert torch.equal(expected[3], actual[3])


def test_atomic_torch_checkpoint(tmp_path):
    path = tmp_path / "weights.pt"
    atomic_save(dict(x=torch.ones(3)), path)
    atomic_save(dict(x=torch.zeros(3)), path, rotate=True)
    assert torch.load(path, weights_only=True)["x"].eq(0).all()
    assert (
        torch.load(tmp_path / "resume_previous.pt", weights_only=True)["x"].eq(1).all()
    )
