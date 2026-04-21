"""Adapter: Isaac Lab env obs → Gr00tPolicy input dict.

At eval time, ``env.step(action)`` returns a flat policy observation tensor
(what RSL-RL needs), while the richer state the VLA wants lives on
``env.unwrapped.scene[...]``. This adapter reads the rich state directly
and assembles the nested ``{video, state, language}`` dict Gr00tPolicy
expects.

Shape contract (see Isaac-GR00T/getting_started/policy.md):
    video:    {"ego_view": uint8 (B, T, H=480, W=640, 3)}
    state:    {"<group>": float32 (B, T, D)} per joint group + wrist + base keys
    language: {"annotation.human.task_description": [[str]] (B,)}

For single-env inference, B=1 and T=1 across the board.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from gear_sonic.data.features_sonic_vla import (
    EGO_VIEW_HEIGHT,
    EGO_VIEW_WIDTH,
    _JOINT_GROUPS_FOR_STATE,
)
from gear_sonic.data.robot_model import RobotModel

from .frame_transforms import (
    pelvis_relative_pose,
    quat_wxyz_to_xyzw,
)


@dataclass
class ObsAdapterConfig:
    """Runtime knobs for the adapter."""

    language_instruction: str
    """Task prompt supplied every step (e.g. 'pick up the mustard bottle')."""

    robot_model: RobotModel
    """Same RobotModel instance used by the converter, so joint ordering + group
    slices match what the VLA was trained on."""

    camera_scene_key: str = "camera_robot"
    """Scene entity name for the ego camera; matches G1PickCamEnvCfg."""

    left_wrist_body_name: str = "left_wrist_yaw_link"
    right_wrist_body_name: str = "right_wrist_yaw_link"


class ObsToPolicyAdapter:
    """Stateful adapter — bind to an env once, then call every step.

    Init computes the joint-name permutation and caches body indices; the
    per-step call (``__call__``) only does cheap tensor slicing + one image
    resize.
    """

    def __init__(self, env, cfg: ObsAdapterConfig) -> None:
        self.env = env
        self.cfg = cfg

        robot = env.unwrapped.scene["robot"]

        # Name-based joint permutation (Isaac order → gear_sonic order).
        isaac_names = list(robot.data.joint_names)
        gs_names = list(cfg.robot_model.joint_names)
        name_to_idx = {n: i for i, n in enumerate(isaac_names)}
        self._joint_perm = np.array(
            [name_to_idx.get(n, -1) for n in gs_names], dtype=np.int64
        )
        missing = [n for n, idx in zip(gs_names, self._joint_perm) if idx < 0]
        if missing:
            import warnings
            warnings.warn(
                f"{len(missing)} gear_sonic joints have no Isaac counterpart and will "
                f"be zero-filled in observation.state: {missing}"
            )

        # Per-group slices in gear_sonic joint order.
        self._group_slices: dict[str, slice] = {}
        for group in _JOINT_GROUPS_FOR_STATE:
            indices = sorted(cfg.robot_model.get_joint_group_indices(group))
            self._group_slices[group] = slice(indices[0], indices[-1] + 1)

        # Body indices for wrist poses (resolved once).
        left_ids, _ = robot.find_bodies([cfg.left_wrist_body_name])
        right_ids, _ = robot.find_bodies([cfg.right_wrist_body_name])
        if len(left_ids) != 1 or len(right_ids) != 1:
            raise RuntimeError(
                f"Expected exactly one body match for "
                f"{cfg.left_wrist_body_name}/{cfg.right_wrist_body_name}"
            )
        self._left_wrist_idx = int(left_ids[0])
        self._right_wrist_idx = int(right_ids[0])

        # Camera is only present in *-Cam env variants; defer hard failure
        # to __call__ so env-only smoke tests (Stage 2b) still work.
        try:
            self._camera = env.unwrapped.scene[cfg.camera_scene_key]
        except KeyError:
            self._camera = None

    def __call__(self) -> dict[str, Any]:
        """Build one Policy API obs dict (B=1, T=1)."""
        robot = self.env.unwrapped.scene["robot"]

        # ---- State: joint-group readings (pelvis-relative joint positions) ----
        q_isaac = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
        q_gs = np.zeros(self._joint_perm.shape[0], dtype=np.float32)
        valid = self._joint_perm >= 0
        q_gs[valid] = q_isaac[self._joint_perm[valid]]

        state: dict[str, np.ndarray] = {}
        for group, sl in self._group_slices.items():
            state[group] = q_gs[sl][None, None, :]  # (1, 1, D)

        # ---- Base attitude ----
        root_pos_w = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
        root_quat_w = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)  # wxyz
        state["root_orientation"] = root_quat_w[None, None, :]

        # Projected gravity: rotate world-frame [0, 0, -1] into body frame.
        from scipy.spatial.transform import Rotation as R  # noqa: N813
        grav_body = (
            R.from_quat(quat_wxyz_to_xyzw(root_quat_w)).inv().apply(
                np.array([0.0, 0.0, -1.0], dtype=np.float32)
            )
        ).astype(np.float32)
        state["projected_gravity"] = grav_body[None, None, :]

        # ---- Wrist Cartesian pose, pelvis-relative (matches dataset convention) ----
        body_pos_w = robot.data.body_pos_w[0].detach().cpu().numpy().astype(np.float32)
        body_quat_w = robot.data.body_quat_w[0].detach().cpu().numpy().astype(np.float32)

        lw_pos, lw_quat = pelvis_relative_pose(
            body_pos_w[self._left_wrist_idx], body_quat_w[self._left_wrist_idx],
            root_pos_w, root_quat_w,
        )
        rw_pos, rw_quat = pelvis_relative_pose(
            body_pos_w[self._right_wrist_idx], body_quat_w[self._right_wrist_idx],
            root_pos_w, root_quat_w,
        )
        state["left_wrist_pos"]       = lw_pos.astype(np.float32)[None, None, :]
        state["left_wrist_abs_quat"]  = lw_quat.astype(np.float32)[None, None, :]
        state["right_wrist_pos"]      = rw_pos.astype(np.float32)[None, None, :]
        state["right_wrist_abs_quat"] = rw_quat.astype(np.float32)[None, None, :]

        # ---- Video ----
        video = {"ego_view": self._read_ego_view()}

        # ---- Language ----
        language = {
            "annotation.human.task_description": [[self.cfg.language_instruction]]
        }

        return {"video": video, "state": state, "language": language}

    def _read_ego_view(self) -> np.ndarray:
        """Read the ego camera, resize to (480, 640, 3), return uint8 (1, 1, H, W, 3)."""
        if self._camera is None:
            raise RuntimeError(
                f"Scene has no '{self.cfg.camera_scene_key}'. "
                "Use G1PickCamEnvCfg or equivalent --enable_cameras env variant."
            )
        available = list(self._camera.data.output.keys())
        if "rgb" not in available:
            raise RuntimeError(
                f"Camera '{self.cfg.camera_scene_key}' has no 'rgb' output yet "
                f"(available: {available}). Call env.step() at least once after reset."
            )
        rgb = self._camera.data.output["rgb"][0]  # (H_src, W_src, 3 or 4), GPU tensor
        rgb = rgb[..., :3]

        # If dtype is float, assume [0, 255]-scaled (matches collect_pick_cam.py).
        if rgb.dtype != torch.uint8:
            rgb = rgb.clamp(0.0, 255.0)

        # Resize on-GPU with bilinear interpolation.
        rgb_chw = rgb.permute(2, 0, 1).unsqueeze(0).float()
        rgb_chw = F.interpolate(
            rgb_chw,
            size=(EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        rgb_hwc = rgb_chw[0].permute(1, 2, 0)

        rgb_np = rgb_hwc.cpu().numpy()
        if rgb_np.dtype != np.uint8:
            rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)

        return rgb_np[None, None, ...]  # (1, 1, H, W, 3)
