"""Deployable observation/action, scene, and raw-episode contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import read, digest

TASKS = ("drawer", "standing_button")
STATE_SLICES = dict(
    left_leg=(0, 6),
    right_leg=(6, 12),
    waist=(12, 15),
    left_arm=(15, 22),
    left_hand=(22, 29),
    right_arm=(29, 36),
    right_hand=(36, 43),
)
ACTION_DIMS = dict(motion_token=64, left_hand_joints=7, right_hand_joints=7)
HORIZON = 40
EXECUTION_STEPS = 20
CONTROL_HZ = 50


def prompt(task):
    if task not in TASKS:
        raise ValueError("Unknown task: " + str(task))
    return "open the drawer" if task == "drawer" else "press the button"


def scene(task):
    prompt(task)
    return dict(
        version="drawer-button-shared-scene-v1",
        task=task,
        cabinet="stock NVIDIA Sektion, production corrected placement"
        if task == "drawer"
        else None,
        button="DreamVLA original visual-only panel"
        if task == "standing_button"
        else None,
        camera=dict(
            prim="Robot/d435_link/Camera",
            resolution=[640, 480],
            pos=[0.05, 0, 0],
            rot=[0.5, 0.5, -0.5, -0.5],
            convention="opengl",
            focal_length=9.1,
            horizontal_aperture=20.955,
        ),
        antialiasing="TAA",
        debug_markers=False,
        neighboring_environments=False,
        physics_hz=200,
        sonic_hz=50,
        groot_hz=2.5,
        horizon=40,
        execute=20,
        residual="multiplicative_free",
        scale=0.5,
        fsq=True,
        evaluation_observation_noise=False,
    )


def validate_scene_contract(contract):
    if not isinstance(contract, dict) or contract != scene(contract.get("task")):
        raise ValueError("Training and evaluation scene contracts must match exactly.")


def accepted(touch, early, foot_xy, foot_lift):
    """Standing-button dataset criterion: native touch, full episode, stable feet."""
    return bool(touch and not early and foot_xy <= 0.05 and foot_lift <= 0.05)


def validate_recording_resume(report, mode, target, checkpoint_hash, bank_hash):
    if (
        report.get("mode") != mode
        or report.get("target") != target
        or report.get("checkpoint_sha256") != checkpoint_hash
        or report.get("evaluation_bank_sha256") != bank_hash
        or report.get("attempts") != len(report.get("outcomes", []))
    ):
        raise ValueError("Resume mode, target, teacher or evaluation bank changed.")
    rows = report["outcomes"]
    if [row.get("attempt") for row in rows] != list(range(len(rows))):
        raise ValueError("Outcome history must be a contiguous prefix.")
    if report.get("status") == "complete":
        count = report.get("saved") if mode == "collect" else len(rows)
        if count != target:
            raise ValueError("Completed recording has a different episode count.")


def validate_episode(
    ep, task, checkpoint=None, *, allow_supervision_mask=False, checkpoint_sha256=None
):
    ep = Path(ep)
    meta = read(ep / "metadata.json")
    if meta.get("requires_supervision_mask") and not allow_supervision_mask:
        raise ValueError("Masked intervention data needs a mask-aware trainer.")
    if meta.get("evaluation_only") or not meta.get("accepted"):
        raise ValueError(
            "Do not export evaluation or failed episodes as demonstrations."
        )
    if meta.get("fps") != CONTROL_HZ or meta.get("frames", 0) < HORIZON:
        raise ValueError("Invalid demonstration timing or duration.")
    if meta.get("environment_contract") != scene(task):
        raise ValueError("Scene contract mismatch.")
    expected = (
        checkpoint_sha256
        if checkpoint_sha256 is not None
        else digest(checkpoint)
        if checkpoint
        else None
    )
    if expected is not None and meta.get("checkpoint_sha256") != expected:
        raise ValueError("Teacher checkpoint mismatch.")
    count = meta["frames"]
    with np.load(ep / "data.npz", allow_pickle=False) as data:
        for key, width in (
            ("observation.state", 43),
            ("observation.projected_gravity", 3),
            ("action.motion_token", 64),
            ("teleop.left_hand_joints", 7),
            ("teleop.right_hand_joints", 7),
        ):
            if data[key].shape != (count, width) or not np.isfinite(data[key]).all():
                raise ValueError("Invalid demonstration array: " + key)
        from .actions import fsq

        if not np.array_equal(
            data["action.motion_token"], fsq(data["action.motion_token"])
        ):
            raise ValueError("Labels are not executed FSQ tokens.")
        if not np.array_equal(
            data["diagnostic.control_step"].ravel(), np.arange(count)
        ):
            raise ValueError("Missing, repeated or terminal-reset frames.")
    return meta
