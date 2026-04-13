"""Utilities for writing robot_camera trajectories to HDF5
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


def format_rollout_state(raw_state: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Placeholder for future rollout-state formatting.

    The caller is expected to provide downstream constraints before this
    hook is enabled.
    """

    raise NotImplementedError(
        "Rollout state formatting is intentionally left generic for now. "
        "Provide the downstream state layout before enabling this hook."
    )


def _to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    return value


def _write_value(group: h5py.Group, key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, dict):
        child_group = group.create_group(key)
        for child_key, child_value in value.items():
            _write_value(child_group, child_key, child_value)
        return

    array_value = _to_numpy(value)
    if isinstance(array_value, str):
        group.create_dataset(key, data=np.array(array_value, dtype=h5py.string_dtype("utf-8")))
        return

    if isinstance(array_value, bytes):
        group.create_dataset(key, data=np.array(array_value, dtype=h5py.string_dtype("utf-8")))
        return

    if isinstance(array_value, np.ndarray) and array_value.dtype.kind in {"U", "O"}:
        group.create_dataset(key, data=np.array(array_value, dtype=h5py.string_dtype("utf-8")))
        return

    group.create_dataset(key, data=array_value, compression="gzip")


@dataclass
class RolloutRecorder:
    """Write a single rollout to a standalone HDF5 file."""

    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_rollout(
        self,
        file_name: str,
        *,
        frames: np.ndarray,
        raw_state: dict[str, Any] | None,
        metadata: dict[str, Any],
        teleop: dict[str, Any] | None = None,
    ) -> Path:
        """Write rollout data to a new HDF5 file and return the file path."""

        if frames is None or frames.size == 0:
            raise ValueError("Images are mandatory: expected non-empty frame tensor for every rollout.")

        file_path = self.output_dir / file_name
        with h5py.File(file_path, "w") as handle:
            handle.attrs["metadata_json"] = json.dumps(_json_safe(metadata))
            handle.create_dataset("images", data=frames, compression="gzip", chunks=True)

            if raw_state is not None:
                state_group = handle.create_group("state")
                _write_value(state_group, "raw", raw_state)

            if teleop is not None:
                self._write_teleop_group(handle, teleop)

        return file_path

    def _write_teleop_group(self, handle: h5py.File, teleop: dict[str, Any]) -> None:
        g = handle.create_group("teleop")

        lw = np.asarray(teleop["left_wrist"], dtype=np.float64)
        rw = np.asarray(teleop["right_wrist"], dtype=np.float64)
        if lw.ndim != 3 or lw.shape[1:] != (4, 4):
            raise ValueError(f"teleop.left_wrist: bad shape {lw.shape}, expected (N, 4, 4)")
        if rw.shape != lw.shape:
            raise ValueError(f"teleop shape mismatch: left={lw.shape} right={rw.shape}")

        self._validate_se3_batch(lw, "left_wrist")
        self._validate_se3_batch(rw, "right_wrist")

        g.create_dataset("left_wrist", data=lw, compression="gzip")
        g.create_dataset("right_wrist", data=rw, compression="gzip")
        g.create_dataset(
            "timestamps",
            data=np.asarray(teleop["timestamps"], dtype=np.float64),
            compression="gzip",
        )

        cal = g.create_group("calibration")
        cal.create_dataset("left_wrist", data=lw[0])
        cal.create_dataset("right_wrist", data=rw[0])

        finger_joints = teleop.get("finger_joints")
        if finger_joints is not None:
            fj = g.create_group("finger_joints")
            fj.create_dataset(
                "left",
                data=np.asarray(finger_joints["left"], dtype=np.float64),
                compression="gzip",
            )
            fj.create_dataset(
                "right",
                data=np.asarray(finger_joints["right"], dtype=np.float64),
                compression="gzip",
            )
            str_dt = h5py.string_dtype("utf-8")
            fj.create_dataset(
                "left_finger_joint_names",
                data=np.array(list(finger_joints["left_names"]), dtype=str_dt),
            )
            fj.create_dataset(
                "right_finger_joint_names",
                data=np.array(list(finger_joints["right_names"]), dtype=str_dt),
            )

        g.attrs["schema_version"] = 1
        g.attrs["frame"] = "pelvis"
        g.attrs["rotation_layout"] = "R|t; 0 0 0 1"
        g.attrs["quaternion_convention"] = "not_stored"
        g.attrs["source_robot"] = teleop.get("source_robot", "unknown")
        g.attrs["left_body_name"] = teleop.get("left_body_name", "left_wrist_yaw_link")
        g.attrs["right_body_name"] = teleop.get("right_body_name", "right_wrist_yaw_link")
        g.attrs["step_dt"] = float(teleop.get("step_dt", 0.0))

    @staticmethod
    def _validate_se3_batch(T: np.ndarray, label: str) -> None:
        bottom = T[:, 3, :]
        if not np.allclose(bottom, np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
            raise ValueError(f"{label}: bottom row not [0,0,0,1]")
        R = T[:, :3, :3]
        gram = np.einsum("nij,nkj->nik", R, R)
        if not np.allclose(gram, np.eye(3)[None], atol=1e-5):
            raise ValueError(f"{label}: rotation not orthonormal")
        dets = np.linalg.det(R)
        if not np.allclose(dets, 1.0, atol=1e-5):
            bad = np.where(np.abs(dets - 1.0) > 1e-5)[0]
            raise ValueError(f"{label}: det != +1 at frames {bad[:5].tolist()}")