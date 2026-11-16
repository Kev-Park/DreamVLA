from types import SimpleNamespace

import numpy as np
import pytest

from dreamvla_drawer_button.actions import (
    fsq,
    expert_token,
    previous_action_proxy,
    nearest_binary,
)
from dreamvla_drawer_button.contracts import (
    accepted,
    scene,
    validate_scene_contract,
    STATE_SLICES,
)
from dreamvla_drawer_button.exposure import budget, next_batch
from dreamvla_drawer_button.isaac.evaluation_observations import (
    disable_observation_noise,
    validate_noise_free_resume,
)


def test_shape_contract():
    assert sum(hi - lo for lo, hi in STATE_SLICES.values()) == 43
    assert sorted(i for lo, hi in STATE_SLICES.values() for i in range(lo, hi)) == list(
        range(43)
    )


@pytest.mark.parametrize("task", ["drawer", "standing_button"])
def test_scene(task):
    cfg = scene(task)
    validate_scene_contract(cfg)
    cfg["debug_markers"] = True
    with pytest.raises(ValueError):
        validate_scene_contract(cfg)


@pytest.mark.parametrize(
    "value", [-10.0, -1.0, -0.53125, 0.0, 0.03125, 15 / 16, 1.0, 10.0]
)
def test_fsq(value):
    token = fsq(np.full(64, value))
    assert np.array_equal(fsq(token), token)
    assert -1 <= token.min() <= token.max() <= 15 / 16
    assert np.array_equal(token * 16, np.round(token * 16))


@pytest.mark.parametrize(
    "value", [np.zeros(63), np.array(0.0), np.full(64, np.nan), np.full(64, np.inf)]
)
def test_bad_tokens(value):
    with pytest.raises(ValueError):
        fsq(value)


def test_residual_semantics_and_unrepresentable_history():
    ref = np.linspace(-1, 15 / 16, 64, dtype=np.float32)
    residual = np.ones(64, np.float32)
    assert np.array_equal(expert_token(ref, residual), fsq(ref * 1.5))
    assert np.array_equal(expert_token(ref, np.zeros(64)), fsq(ref))
    token = np.zeros(64, np.float32)
    token[0] = 0.5
    proxy, bad = previous_action_proxy(np.zeros(64), token)
    assert proxy[0] == 0 and bad.sum() == 1
    # A zero reference cannot generate this nonzero student token at any scale.
    assert not np.array_equal(expert_token(np.zeros(64), proxy), token)


def test_hands():
    assert nearest_binary(np.zeros(7), np.zeros(7), np.ones(7)) == 1
    assert nearest_binary(np.ones(7), np.zeros(7), np.ones(7)) == -1
    with pytest.raises(ValueError):
        nearest_binary(np.zeros(6), np.zeros(7), np.ones(7))


@pytest.mark.parametrize(
    "touch,early,xy,lift,ok",
    [
        (True, False, 0.05, 0.05, True),
        (False, False, 0, 0, False),
        (True, True, 0, 0, False),
        (True, False, 0.051, 0, False),
        (True, False, 0, 0.051, False),
    ],
)
def test_button_success(touch, early, xy, lift, ok):
    assert accepted(touch, early, xy, lift) is ok


@pytest.mark.parametrize("episodes", [1, 10, 50, 200, 1000])
def test_frame_budget(episodes):
    b = budget([499] * episodes)
    assert b["frames"] == episodes * 499
    assert b["windows"] == episodes * 460
    assert b["samples"] == (episodes * 499 * 57 + 9) // 10
    assert b["updates"] == (b["samples"] + 15) // 16
    assert (b["updates"] - 1) * 16 + b["final_batch_size"] == b["samples"]


@pytest.mark.parametrize(
    "lengths,epochs,batch",
    [
        ([], 5.7, 16),
        ([39], 5.7, 16),
        ([40], 0, 16),
        ([40], float("nan"), 16),
        ([40], 5.7, 0),
    ],
)
def test_bad_budgets(lengths, epochs, batch):
    with pytest.raises(ValueError):
        budget(lengths, epochs, batch)


def test_exact_sampler_crosses_boundaries_without_dropping_samples():
    order = list(range(7))
    rng = np.random.default_rng(12)
    samples = []
    for seen in range(0, 23, 5):
        samples += next_batch(order, rng, seen, 23, 5)
    assert len(samples) == 23
    for start in (0, 7, 14):
        assert sorted(samples[start : start + 7]) == list(range(7))


def test_eval_noise_is_process_local_and_does_not_disable_physics():
    policy = SimpleNamespace(enable_corruption=True)
    cfg = SimpleNamespace(
        observations=SimpleNamespace(policy=policy), physics_randomization=True
    )
    assert disable_observation_noise(cfg) == ["policy"]
    assert policy.enable_corruption is False and cfg.physics_randomization is True
    validate_noise_free_resume({"observation_noise_enabled": False})
    with pytest.raises(RuntimeError):
        validate_noise_free_resume({})
