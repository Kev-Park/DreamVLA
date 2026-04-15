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
        env_args: dict[str, Any] | None = None,
    ) -> Path:
        """Write rollout data in robomimic-style HDF5 layout (Part A v2,
        ``schema_version=2``).

        Layout::

            /data
              @total      int
              @env_args   json string
              /demo_0
                /obs
                  ego_view_image                  (N, H, W, 3) u8
                  robot0_joint_pos                (N, J)       f64
                  robot0_joint_vel                (N, J)       f64
                  robot0_root_pos_w               (N, 3)       f64
                  robot0_root_quat_w              (N, 4)       f64  wxyz
                  robot0_left_finger_joint_pos    (N, K_L)     f64
                  robot0_right_finger_joint_pos   (N, K_R)     f64
                  object_pos                      (N, 3)       f64
                  object_quat                     (N, 4)       f64  wxyz
                actions                           (N, A)       f64
                /teleop                           (see _write_teleop_group)
            /  @metadata_json    json string
        """

        if frames is None or frames.size == 0:
            raise ValueError("Images are mandatory: expected non-empty frame tensor for every rollout.")

        file_path = self.output_dir / file_name
        with h5py.File(file_path, "w") as handle:
            handle.attrs["metadata_json"] = json.dumps(_json_safe(metadata))

            data_grp = handle.create_group("data")
            n_frames = int(frames.shape[0])
            data_grp.attrs["total"] = n_frames
            if env_args is not None:
                data_grp.attrs["env_args"] = json.dumps(_json_safe(env_args))

            demo_grp = data_grp.create_group("demo_0")
            obs_grp = demo_grp.create_group("obs")
            obs_grp.create_dataset(
                "ego_view_image", data=frames, compression="gzip", chunks=True
            )

            obs_rename = {
                "joint_pos": "robot0_joint_pos",
                "joint_vel": "robot0_joint_vel",
                "root_pos_w": "robot0_root_pos_w",
                "root_quat_w": "robot0_root_quat_w",
                "left_finger_joint_pos": "robot0_left_finger_joint_pos",
                "right_finger_joint_pos": "robot0_right_finger_joint_pos",
            }
            if raw_state is not None and isinstance(raw_state.get("robot"), dict):
                for src, dst in obs_rename.items():
                    if src in raw_state["robot"]:
                        obs_grp.create_dataset(
                            dst,
                            data=np.asarray(
                                _to_numpy(raw_state["robot"][src]), dtype=np.float64
                            ),
                            compression="gzip",
                        )

            if raw_state is not None and isinstance(raw_state.get("object"), dict):
                obj = raw_state["object"]
                if "root_pos_w" in obj:
                    obs_grp.create_dataset(
                        "object_pos",
                        data=np.asarray(_to_numpy(obj["root_pos_w"]), dtype=np.float64),
                        compression="gzip",
                    )
                if "root_quat_w" in obj:
                    obs_grp.create_dataset(
                        "object_quat",
                        data=np.asarray(_to_numpy(obj["root_quat_w"]), dtype=np.float64),
                        compression="gzip",
                    )

            if raw_state is not None and "action" in raw_state:
                demo_grp.create_dataset(
                    "actions",
                    data=np.asarray(_to_numpy(raw_state["action"]), dtype=np.float64),
                    compression="gzip",
                )

            if teleop is not None:
                self._write_teleop_group(demo_grp, teleop)

        return file_path

    def _write_teleop_group(self, parent: h5py.Group, teleop: dict[str, Any]) -> None:
        g = parent.create_group("teleop")

        lw = np.asarray(teleop["left_wrist"], dtype=np.float64)
        rw = np.asarray(teleop["right_wrist"], dtype=np.float64)
        if "torso_pose" not in teleop:
            raise ValueError(
                "teleop payload is missing 'torso_pose' — required by Part A v2."
            )
        tp = np.asarray(teleop["torso_pose"], dtype=np.float64)

        for arr, label in ((lw, "left_wrist"), (rw, "right_wrist"), (tp, "torso_pose")):
            if arr.ndim != 3 or arr.shape[1:] != (4, 4):
                raise ValueError(f"teleop.{label}: bad shape {arr.shape}, expected (N, 4, 4)")
            self._validate_se3_batch(arr, label)
        if rw.shape != lw.shape or tp.shape != lw.shape:
            raise ValueError(
                f"teleop frame count mismatch: left={lw.shape[0]} "
                f"right={rw.shape[0]} torso={tp.shape[0]}"
            )

        g.create_dataset("left_wrist", data=lw, compression="gzip")
        g.create_dataset("right_wrist", data=rw, compression="gzip")
        g.create_dataset("torso_pose", data=tp, compression="gzip")
        g.create_dataset(
            "timestamps",
            data=np.asarray(teleop["timestamps"], dtype=np.float64),
            compression="gzip",
        )

        cal = g.create_group("calibration")
        cal.create_dataset("left_wrist", data=lw[0])
        cal.create_dataset("right_wrist", data=rw[0])
        cal.create_dataset("torso_pose", data=tp[0])

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

        g.attrs["schema_version"] = 2
        g.attrs["frame"] = "pelvis"
        g.attrs["rotation_layout"] = "R|t; 0 0 0 1"
        g.attrs["quaternion_convention"] = "wxyz"
        g.attrs["source_robot"] = teleop.get("source_robot", "unknown")
        g.attrs["left_body_name"] = teleop.get("left_body_name", "left_wrist_yaw_link")
        g.attrs["right_body_name"] = teleop.get("right_body_name", "right_wrist_yaw_link")
        g.attrs["torso_body_name"] = teleop.get("torso_body_name", "torso_link")
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


def replay_hdf5(
    hdf5_path: Path,
    output_path: Path | None = None,
    fps: float | None = None,
) -> Path:
    """Write the ego_view_image frames stored in a rollout HDF5 to an mp4."""
    import cv2  # lazy: only required when replaying

    with h5py.File(hdf5_path, "r") as handle:
        frames = np.asarray(handle["data/demo_0/obs/ego_view_image"])
        if fps is None:
            teleop = handle.get("data/demo_0/teleop")
            if teleop is not None:
                step_dt = float(teleop.attrs.get("step_dt", 0.0) or 0.0)
                if step_dt > 0:
                    fps = 1.0 / step_dt

    if fps is None:
        fps = 50.0
    if output_path is None:
        output_path = hdf5_path.with_suffix(".mp4")

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"ego_view_image has unexpected shape {frames.shape}; expected (N, H, W, 3)."
        )
    if frames.dtype != np.uint8:
        raise ValueError(f"ego_view_image has dtype {frames.dtype}; expected uint8.")

    n_frames, height, width, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")
    try:
        for frame_rgb in frames:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    print(f"Wrote {n_frames} frames at {fps:.2f} fps -> {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Replay a rollout HDF5 by writing its ego_view_image frames to an mp4.",
    )
    parser.add_argument("hdf5_path", type=Path, help="Path to the rollout HDF5 file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path. Defaults to <hdf5_path>.mp4.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback fps. Defaults to 1/teleop.step_dt, else 50.",
    )
    args = parser.parse_args()

    replay_hdf5(args.hdf5_path, args.output, args.fps)