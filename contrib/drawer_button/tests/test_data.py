import numpy as np
import pytest

from dreamvla_drawer_button.contracts import (
    scene,
    validate_episode,
    validate_recording_resume,
)
from dreamvla_drawer_button.io import read, write_json
from dreamvla_drawer_button.run_config import prepare, load_run
from dreamvla_drawer_button.convert import raw_episodes
from dreamvla_drawer_button.finetune import training_budget
from dreamvla_drawer_button.dagger import replay_plan
from dreamvla_drawer_button.reference_config import points
from dreamvla_drawer_button.provenance import model_identity


def episode(tmp_path, **overrides):
    folder = tmp_path / "episode_000000"
    folder.mkdir()
    meta = dict(
        accepted=True,
        evaluation_only=False,
        frames=50,
        fps=50,
        environment_contract=scene("drawer"),
    )
    meta.update(overrides)
    write_json(folder / "metadata.json", meta)
    arrays = {
        key: np.zeros((50, width), np.float32)
        for key, width in (
            ("observation.state", 43),
            ("observation.projected_gravity", 3),
            ("action.motion_token", 64),
            ("teleop.left_hand_joints", 7),
            ("teleop.right_hand_joints", 7),
        )
    }
    arrays["diagnostic.control_step"] = np.arange(50)[:, None]
    np.savez(folder / "data.npz", **arrays)
    return folder


def test_valid_raw_episode(tmp_path):
    validate_episode(episode(tmp_path), "drawer")
    assert len(raw_episodes(tmp_path)) == 1


@pytest.mark.parametrize(
    "override",
    [
        dict(accepted=False),
        dict(evaluation_only=True),
        dict(requires_supervision_mask=True),
        dict(fps=25),
        dict(frames=10),
        dict(environment_contract=scene("standing_button")),
    ],
)
def test_bad_raw_episode(tmp_path, override):
    with pytest.raises(ValueError):
        validate_episode(episode(tmp_path, **override), "drawer")


def test_no_episode_gaps(tmp_path):
    folder = episode(tmp_path)
    folder.rename(tmp_path / "episode_000001")
    with pytest.raises(ValueError):
        raw_episodes(tmp_path)


def test_manifests_are_immutable_and_disjoint(tmp_path):
    train, test = tmp_path / "train", tmp_path / "test"
    train.mkdir()
    test.mkdir()
    (train / "a.pkl").write_bytes(b"trusted training reference")
    (test / "b.pkl").write_bytes(b"distinct held-out reference")
    target = tmp_path / "config"
    prepare("drawer", train, test, tmp_path / "ppo.pt", target, episodes=100, seed=4)
    cfg, bank = load_run(target)
    assert (
        cfg["n"] == 1 and len(bank["repeatability"]) == len(bank["fresh_start"]) == 100
    )
    assert (
        len({x["seed"] for key in ("repeatability", "fresh_start") for x in bank[key]})
        == 200
    )
    (test / "b.pkl").write_bytes(b"changed")
    with pytest.raises(ValueError):
        load_run(target)
    with pytest.raises(ValueError):
        prepare("drawer", train, train, tmp_path / "ppo.pt", tmp_path / "bad")


def test_atomic_json_failure_preserves_old_file(tmp_path):
    path = tmp_path / "a.json"
    write_json(path, dict(ok=1), durable=True)
    with pytest.raises(ValueError):
        write_json(path, dict(bad=float("nan")), durable=True)
    assert read(path) == dict(ok=1)
    assert list(tmp_path.iterdir()) == [path]


def test_frame_budget_from_lerobot_metadata(tmp_path):
    write_json(tmp_path / "meta/info.json", {"total_frames": 499 * 200})
    assert training_budget(tmp_path) == 35554


def test_resume_contract():
    report = dict(
        mode="eval",
        target=2,
        checkpoint_sha256="a",
        evaluation_bank_sha256="b",
        attempts=1,
        outcomes=[dict(attempt=0)],
        status="running",
    )
    validate_recording_resume(report, "eval", 2, "a", "b")
    with pytest.raises(ValueError):
        validate_recording_resume(report, "eval", 2, "changed", "b")


def test_dagger_failed_only_bounded_attempts():
    rows = [
        dict(accepted=True, attempt=0, reference_name="ok", seed=3, steps=500),
        dict(accepted=False, attempt=1, reference_name="fail", seed=4, steps=200),
    ]
    plan = replay_plan(rows, 10, seed=2)
    assert len(plan) == 10 and all(
        x["reference_name"] == "fail" and 0 <= x["takeover_step"] < 200 for x in plan
    )
    assert replay_plan(rows[:1], 10) == []


def test_nested_reference_points():
    bounds = ((0.28, 0.30, 10), (-0.2, -0.1, 25), (0.7, 0.8, 25))
    assert points(bounds, 10, 7) == points(bounds, 200, 7)[:10]
    assert points(bounds, 10, 7) != points(bounds, 10, 8)


def test_model_identity_changes_with_weights(tmp_path):
    write_json(tmp_path / "config.json", dict(horizon=40))
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"fixture-a")
    first = model_identity(tmp_path)
    weights.write_bytes(b"fixture-b")
    assert first != model_identity(tmp_path)
