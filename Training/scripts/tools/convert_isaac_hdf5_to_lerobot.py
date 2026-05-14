"""Offline converter: producer Isaac Lab HDF5 (Part A v2) -> LeRobot v2.1 SONIC VLA dataset.

Reads robomimic-style HDF5 rollouts written by the TrajGen ``collect_pick_cam.py``
producer (``schema_version=2``, see Part A v2 of the cross-repo plan) and emits
a LeRobot v2.1 dataset whose feature schema matches
``gear_sonic.data.features_sonic_vla.get_features_sonic_vla`` exactly.

Run under ``.venv_data_collection`` (provisioned via
``bash install_scripts/install_data_collection.sh``).

Several feature slots that have no Isaac-Lab analog are zero-filled
(see ``ZERO_FILLED_FIELDS`` and the dataset's ``meta/info.json`` script_config).
A later revision may run ``model_encoder.onnx`` to populate
``action.motion_token`` from the recorded teleop fields.

Converter v2 (current): planner_{movement,speed,facing,height,mode} are
derived per-frame from ``root_pos_w`` / ``root_quat_w`` so the VLA learns
real locomotion commands. v1 hardcoded them to stationary-task constants.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_sonic_vla import (
    EGO_VIEW_HEIGHT,
    EGO_VIEW_WIDTH,
    get_features_sonic_vla,
    get_g1_robot_model,
    get_modality_config_sonic_vla,
)
from gear_sonic.utils.data_collection.transforms import (
    compute_projected_gravity,
    quat_to_rot6d,
)


SCHEMA_VERSION = 2
TELEOP_GROUP_PATH = "data/demo_0/teleop"
OBS_GROUP_PATH = "data/demo_0/obs"
DATA_GROUP_PATH = "data"
DEMO_GROUP_PATH = "data/demo_0"

# Locked-in zero-fill set (see plan Part C: "Zero-fill locked in").
ZERO_FILLED_FIELDS = [
    "action.motion_token",
    "teleop.smpl_joints",
    "teleop.smpl_pose",
    "teleop.body_quat_w",
    "teleop.target_body_orientation",
    "observation.cpp_rotation_offset",
    "teleop.smpl_frame_index",
]

# Local-frame offsets applied by gear_sonic for the VR 3-point representation.
# Mirrors gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer.G1_KEY_FRAME_OFFSETS.
LW_LOCAL_OFFSET = np.array([0.18, -0.025, 0.0], dtype=np.float64)
RW_LOCAL_OFFSET = np.array([0.18, 0.025, 0.0], dtype=np.float64)
TORSO_LOCAL_OFFSET = np.array([0.0, 0.0, 0.35], dtype=np.float64)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _mat_to_quat_wxyz(rot_mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> wxyz quaternion (scalar-first)."""
    quat_xyzw = R.from_matrix(rot_mat).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )


def _apply_local_offset(T: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Translate by an offset expressed in the local frame of T (4x4 SE3)."""
    return T[:3, 3] + T[:3, :3] @ offset


def _build_joint_permutation(
    isaac_joint_names: list[str],
    gear_sonic_joint_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Return ``perm`` such that ``q_gs = q_il[perm]`` for joints that exist in
    both name lists, plus the list of gear_sonic joint names that are NOT
    present in Isaac Lab (these will be zero-filled).

    ``perm[i]`` is the Isaac-Lab index for ``gear_sonic_joint_names[i]``, or -1
    when the name is missing on the Isaac side.
    """
    isaac_index = {name: i for i, name in enumerate(isaac_joint_names)}
    perm = np.full(len(gear_sonic_joint_names), -1, dtype=np.int64)
    missing: list[str] = []
    for i, name in enumerate(gear_sonic_joint_names):
        idx = isaac_index.get(name)
        if idx is None:
            missing.append(name)
        else:
            perm[i] = idx
    return perm, missing


def _permute_with_zero_fill(
    q_il_batch: np.ndarray, perm: np.ndarray
) -> np.ndarray:
    """Apply a name-based permutation; missing entries (perm == -1) become 0.0."""
    n_frames = q_il_batch.shape[0]
    n_out = perm.shape[0]
    out = np.zeros((n_frames, n_out), dtype=np.float64)
    valid = perm >= 0
    out[:, valid] = q_il_batch[:, perm[valid]]
    return out


def _resize_ego_view(frame: np.ndarray) -> np.ndarray:
    """Resize 720x1280 producer ego view -> 480x640 dataset feature."""
    if frame.shape[:2] == (EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH):
        return frame
    return cv2.resize(
        frame,
        (EGO_VIEW_WIDTH, EGO_VIEW_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def _read_v2_hdf5(path: Path) -> dict[str, Any]:
    """Slurp every field the converter needs out of one rollout HDF5."""
    with h5py.File(path, "r") as f:
        if DATA_GROUP_PATH not in f:
            raise RuntimeError(
                f"{path}: missing top-level 'data' group. Not a Part A v2 HDF5."
            )
        if TELEOP_GROUP_PATH not in f:
            raise RuntimeError(
                f"{path}: missing '{TELEOP_GROUP_PATH}' group."
            )

        teleop_grp = f[TELEOP_GROUP_PATH]
        schema = int(teleop_grp.attrs.get("schema_version", 0))
        if schema != SCHEMA_VERSION:
            raise RuntimeError(
                f"{path}: teleop schema_version={schema}, expected {SCHEMA_VERSION}. "
                "Regenerate this file with the Part A v2 producer."
            )

        env_args_attr = f[DATA_GROUP_PATH].attrs.get("env_args")
        if env_args_attr is None:
            raise RuntimeError(
                f"{path}: data/@env_args is missing — not a Part A v2 HDF5."
            )
        env_args = json.loads(_decode(env_args_attr))

        obs_grp = f[OBS_GROUP_PATH]
        joint_pos = obs_grp["robot0_joint_pos"][...].astype(np.float64)
        root_pos_w = obs_grp["robot0_root_pos_w"][...].astype(np.float64)
        root_quat_w = obs_grp["robot0_root_quat_w"][...].astype(np.float64)
        ego_view = obs_grp["ego_view_image"][...]
        object_pos_w = (
            obs_grp["object_pos"][...].astype(np.float64)
            if "object_pos" in obs_grp else None
        )
        object_quat_w = (
            obs_grp["object_quat"][...].astype(np.float64)
            if "object_quat" in obs_grp else None
        )

        left_wrist = teleop_grp["left_wrist"][...].astype(np.float64)
        right_wrist = teleop_grp["right_wrist"][...].astype(np.float64)
        torso_pose = teleop_grp["torso_pose"][...].astype(np.float64)
        timestamps = teleop_grp["timestamps"][...].astype(np.float64)

        if "finger_joints" not in teleop_grp:
            raise RuntimeError(
                f"{path}: missing teleop/finger_joints group — converter requires fingers."
            )
        fj_grp = teleop_grp["finger_joints"]
        finger_left = fj_grp["left"][...].astype(np.float64)
        finger_right = fj_grp["right"][...].astype(np.float64)
        finger_left_names = [_decode(n) for n in fj_grp["left_finger_joint_names"][...]]
        finger_right_names = [_decode(n) for n in fj_grp["right_finger_joint_names"][...]]

        step_dt = float(teleop_grp.attrs.get("step_dt", 0.0))

    return {
        "env_args": env_args,
        "joint_pos": joint_pos,
        "root_pos_w": root_pos_w,
        "root_quat_w": root_quat_w,
        "ego_view": ego_view,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "torso_pose": torso_pose,
        "timestamps": timestamps,
        "finger_left": finger_left,
        "finger_right": finger_right,
        "finger_left_names": finger_left_names,
        "finger_right_names": finger_right_names,
        "step_dt": step_dt,
        "object_pos_w": object_pos_w,
        "object_quat_w": object_quat_w,
    }


def _build_canonical_finger_perm(
    hdf5_names: list[str], canonical_names: list[str], side: str
) -> np.ndarray:
    missing = [n for n in canonical_names if n not in hdf5_names]
    if missing:
        raise RuntimeError(
            f"HDF5 {side} finger joint names are missing canonical entries: "
            f"{missing}\nHDF5 names: {hdf5_names}\nCanonical: {canonical_names}"
        )
    return np.array([hdf5_names.index(n) for n in canonical_names], dtype=np.int64)


def _wrist_joint_slot_indices(
    robot_model, side: str
) -> tuple[int, int, int]:
    """Return ``(roll_idx, pitch_idx, yaw_idx)`` in gear_sonic full-q space."""
    return (
        robot_model.dof_index(f"{side}_wrist_roll_joint"),
        robot_model.dof_index(f"{side}_wrist_pitch_joint"),
        robot_model.dof_index(f"{side}_wrist_yaw_joint"),
    )


def convert_one_rollout(
    hdf5_path: Path,
    exporter: Gr00tDataExporter,
    robot_model,
    body_perm: np.ndarray,
    body_missing: list[str],
    canonical_left_finger_names: list[str],
    canonical_right_finger_names: list[str],
    left_wrist_slots: tuple[int, int, int],
    right_wrist_slots: tuple[int, int, int],
    expected_left_body_name: str,
    expected_right_body_name: str,
    rollout_index: int,
    max_frames: int | None = None,
) -> int:
    print(f"[rollout {rollout_index}] loading {hdf5_path.name}")
    raw = _read_v2_hdf5(hdf5_path)

    n_frames = raw["joint_pos"].shape[0]
    if max_frames is not None and max_frames > 0:
        n_frames = min(n_frames, max_frames)

    # Sanity: HDF5 wrist body names must match what gear_sonic's pinocchio uses.
    with h5py.File(hdf5_path, "r") as _f:
        teleop_attrs = dict(_f[TELEOP_GROUP_PATH].attrs)
    actual_left = _decode(teleop_attrs.get("left_body_name", ""))
    actual_right = _decode(teleop_attrs.get("right_body_name", ""))
    if actual_left != expected_left_body_name:
        raise RuntimeError(
            f"{hdf5_path}: left_body_name attr {actual_left!r} != expected "
            f"{expected_left_body_name!r}"
        )
    if actual_right != expected_right_body_name:
        raise RuntimeError(
            f"{hdf5_path}: right_body_name attr {actual_right!r} != expected "
            f"{expected_right_body_name!r}"
        )

    # Pre-compute permutations & per-rollout fields.
    left_finger_perm = _build_canonical_finger_perm(
        raw["finger_left_names"], canonical_left_finger_names, "left"
    )
    right_finger_perm = _build_canonical_finger_perm(
        raw["finger_right_names"], canonical_right_finger_names, "right"
    )

    q_gs_all = _permute_with_zero_fill(raw["joint_pos"], body_perm)

    # Frame zero init base quat for the whole episode.
    init_base_quat = raw["root_quat_w"][0].astype(np.float64).copy()

    # Constant zero-fill arrays we can reuse per frame.
    zero_motion_token = np.zeros(64, dtype=np.float64)
    zero_smpl_joints = np.zeros(72, dtype=np.float32)
    zero_smpl_pose = np.zeros(63, dtype=np.float32)
    identity_body_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    identity_target_rot6d = quat_to_rot6d(identity_body_quat).astype(np.float32)
    identity_cpp_offset = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    zero_smpl_frame_idx = np.array([0], dtype=np.int64)
    zero_delta_heading = np.zeros(1, dtype=np.float64)

    # Per-frame planner commands derived from recorded root motion, matching
    # the kinematic planner ONNX input semantics (see gear_sonic_deploy's
    # localmotion_kplanner.hpp for the mode-speed range table).
    step_dt = raw["step_dt"] if raw["step_dt"] > 0.0 else 1.0 / 50.0
    root_pos = raw["root_pos_w"][:n_frames].astype(np.float64)
    root_quat_wxyz = raw["root_quat_w"][:n_frames].astype(np.float64)

    # 3D root velocity via central differences (world frame, MuJoCo Z-up).
    _vel_all = np.gradient(root_pos, step_dt, axis=0).astype(np.float32)

    # Horizontal speed magnitude (derived from velocity, before normalization).
    # Zeroed below the IDLE threshold (0.1 m/s) so target_vel=0 for IDLE frames —
    # passing a small positive target_vel with mode=IDLE is incoherent.
    _raw_speed = np.linalg.norm(_vel_all[:, :2], axis=1, keepdims=True).astype(np.float32)
    planner_speed_all = np.where(_raw_speed >= 0.1, _raw_speed, 0.0).astype(np.float32)

    # Forward unit vector: world-frame direction the torso faces, derived by
    # rotating [1, 0, 0] (body forward) by the root quaternion.
    root_quat_xyzw = np.stack(
        [
            root_quat_wxyz[:, 1],
            root_quat_wxyz[:, 2],
            root_quat_wxyz[:, 3],
            root_quat_wxyz[:, 0],
        ],
        axis=1,
    )
    planner_facing_all = (
        R.from_quat(root_quat_xyzw).apply(np.array([1.0, 0.0, 0.0])).astype(np.float32)
    )

    # Movement direction = facing direction (body forward, Z zeroed and renormalized).
    # The kinematic planner only generates forward locomotion — it cannot strafe.
    # Deriving movement from root velocity would capture physics drift and produce a
    # direction the planner can't act on.  Using facing ensures both inputs are
    # self-consistent and the planner walks in the direction the body was oriented.
    _moving_mask = _raw_speed.reshape(-1) >= 0.1
    _facing_xy = planner_facing_all[:, :2].copy()
    _facing_xy_norm = np.linalg.norm(_facing_xy, axis=1, keepdims=True)
    _facing_xy_unit = np.where(_facing_xy_norm > 1e-6, _facing_xy / _facing_xy_norm, np.array([[1.0, 0.0]]))
    planner_movement_all = np.zeros((n_frames, 3), dtype=np.float32)
    planner_movement_all[_moving_mask, :2] = _facing_xy_unit[_moving_mask]

    # Height sentinel: -1.0 means "use mode default" for all standard modes.
    # From localmotion_kplanner.hpp: height=-1 for IDLE/SLOW_WALK/WALK/RUN (all pick-task modes).
    planner_height_all = np.full((n_frames, 1), -1.0, dtype=np.float32)

    # Mode from horizontal speed: 0=IDLE, 1=SLOW_WALK, 2=WALK, 3=RUN.
    def _speed_to_mode(s: float) -> int:
        s = abs(float(s))
        if s < 0.1:
            return 0
        if s < 0.8:
            return 1
        if s < 2.5:
            return 2
        return 3

    planner_mode_all = np.array(
        [[_speed_to_mode(float(s))] for s in planner_speed_all.flatten()],
        dtype=np.int32,
    )

    # Object pose arrays — None when the HDF5 predates object state recording.
    _obj_pos_src = raw.get("object_pos_w")
    _obj_quat_src = raw.get("object_quat_w")
    _has_object = _obj_pos_src is not None and _obj_quat_src is not None
    if _has_object:
        object_pos_all  = _obj_pos_src[:n_frames].astype(np.float32)
        object_quat_all = _obj_quat_src[:n_frames].astype(np.float32)
    else:
        object_pos_all  = np.zeros((n_frames, 3), dtype=np.float32)
        object_quat_all = np.tile(np.array([1., 0., 0., 0.], dtype=np.float32), (n_frames, 1))

    stream_mode = np.array([5], dtype=np.int32)

    written = 0
    for i in range(n_frames):
        q_gs = q_gs_all[i]

        lw = raw["left_wrist"][i]
        rw = raw["right_wrist"][i]
        tp = raw["torso_pose"][i]

        lw_pos = lw[:3, 3]
        rw_pos = rw[:3, 3]
        lw_quat_wxyz = _mat_to_quat_wxyz(lw[:3, :3])
        rw_quat_wxyz = _mat_to_quat_wxyz(rw[:3, :3])
        observation_eef_state = np.concatenate(
            [lw_pos, lw_quat_wxyz, rw_pos, rw_quat_wxyz], dtype=np.float64
        )

        root_quat = raw["root_quat_w"][i].astype(np.float64)
        projected_gravity = compute_projected_gravity(root_quat).astype(np.float64)

        vr_3pt_position = np.concatenate(
            [
                _apply_local_offset(lw, LW_LOCAL_OFFSET),
                _apply_local_offset(rw, RW_LOCAL_OFFSET),
                _apply_local_offset(tp, TORSO_LOCAL_OFFSET),
            ],
            dtype=np.float32,
        )
        vr_3pt_orientation = np.concatenate(
            [
                quat_to_rot6d(_mat_to_quat_wxyz(lw[:3, :3]).astype(np.float32)),
                quat_to_rot6d(_mat_to_quat_wxyz(rw[:3, :3]).astype(np.float32)),
                quat_to_rot6d(_mat_to_quat_wxyz(tp[:3, :3]).astype(np.float32)),
            ],
            dtype=np.float32,
        )

        left_finger_canonical = raw["finger_left"][i, left_finger_perm].astype(np.float32)
        right_finger_canonical = raw["finger_right"][i, right_finger_perm].astype(np.float32)

        left_wrist_joints = np.array(
            [q_gs[idx] for idx in left_wrist_slots], dtype=np.float32
        )
        right_wrist_joints = np.array(
            [q_gs[idx] for idx in right_wrist_slots], dtype=np.float32
        )

        ego_image = _resize_ego_view(raw["ego_view"][i])

        frame_data = {
            "observation.images.ego_view": ego_image,
            "observation.state": q_gs,
            "observation.eef_state": observation_eef_state,
            "action.wbc": q_gs.copy(),
            "observation.root_orientation": root_quat,
            "observation.projected_gravity": projected_gravity,
            "observation.cpp_rotation_offset": identity_cpp_offset.copy(),
            "observation.init_base_quat": init_base_quat.copy(),
            "teleop.delta_heading": zero_delta_heading.copy(),
            "action.motion_token": zero_motion_token.copy(),
            "teleop.smpl_joints": zero_smpl_joints.copy(),
            "teleop.smpl_pose": zero_smpl_pose.copy(),
            "teleop.body_quat_w": identity_body_quat.copy(),
            "teleop.target_body_orientation": identity_target_rot6d.copy(),
            "teleop.left_hand_joints": left_finger_canonical,
            "teleop.right_hand_joints": right_finger_canonical,
            "teleop.smpl_frame_index": zero_smpl_frame_idx.copy(),
            "teleop.left_wrist_joints": left_wrist_joints,
            "teleop.right_wrist_joints": right_wrist_joints,
            "teleop.stream_mode": stream_mode.copy(),
            "teleop.planner_mode": planner_mode_all[i].copy(),
            "teleop.planner_movement": planner_movement_all[i].copy(),
            "teleop.planner_facing": planner_facing_all[i].copy(),
            "teleop.planner_speed": planner_speed_all[i].copy(),
            "teleop.planner_height": planner_height_all[i].copy(),
            "teleop.root_pos_w": root_pos[i].astype(np.float32),
            "teleop.object_pos_w": object_pos_all[i].copy(),
            "teleop.object_quat_w": object_quat_all[i].copy(),
            "teleop.vr_3pt_position": vr_3pt_position,
            "teleop.vr_3pt_orientation": vr_3pt_orientation,
        }

        exporter.add_frame(frame_data)
        written += 1

    exporter.save_episode()
    print(f"[rollout {rollout_index}] wrote {written} frames -> episode saved")
    if body_missing and rollout_index == 0:
        print(
            f"[info] {len(body_missing)} gear_sonic joints have no Isaac Lab "
            f"counterpart and are zero-filled in observation.state / action.wbc: "
            f"{body_missing}"
        )
    return written


def _list_rollouts(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.hdf5" if recursive else "*.hdf5"
    files = sorted(root.glob(pattern))
    if not files:
        raise RuntimeError(f"No .hdf5 files found under {root} (recursive={recursive})")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Part A v2 Isaac Lab rollout HDF5 files to a LeRobot v2.1 "
            "Sonic VLA dataset."
        )
    )
    parser.add_argument(
        "--hdf5-root",
        type=Path,
        required=True,
        help="Directory containing producer rollout .hdf5 files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output dataset root directory.",
    )
    parser.add_argument(
        "--task-prompt",
        type=str,
        required=True,
        help='Natural-language task description, e.g. "pick up the mustard bottle".',
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional dataset name; defaults to the output-path basename.",
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--waist-location",
        choices=["lower_body", "upper_body", "lower_and_upper_body"],
        default="lower_body",
    )
    parser.add_argument(
        "--high-elbow",
        action="store_true",
        help="Use the high elbow default joint pose for the gear_sonic robot model.",
    )
    parser.add_argument(
        "--rollout-limit",
        type=int,
        default=-1,
        help="Process at most this many rollout files. -1 = all.",
    )
    parser.add_argument(
        "--max-frames-per-rollout",
        type=int,
        default=-1,
        help="Process at most this many frames per rollout. -1 = all.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Glob .hdf5 files recursively under --hdf5-root.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Wipe --output-path before writing.",
    )
    args = parser.parse_args()

    rollouts = _list_rollouts(args.hdf5_root, args.recursive)
    if args.rollout_limit >= 0:
        rollouts = rollouts[: args.rollout_limit]
    print(f"[info] found {len(rollouts)} rollout file(s)")

    robot_model = get_g1_robot_model(
        waist_location=args.waist_location,
        high_elbow_pose=args.high_elbow,
    )
    expected_left_body = robot_model.supplemental_info.hand_frame_names["left"]
    expected_right_body = robot_model.supplemental_info.hand_frame_names["right"]
    canonical_left_finger = robot_model.supplemental_info.left_hand_actuated_joints
    canonical_right_finger = robot_model.supplemental_info.right_hand_actuated_joints
    left_wrist_slots = _wrist_joint_slot_indices(robot_model, "left")
    right_wrist_slots = _wrist_joint_slot_indices(robot_model, "right")

    # Use the first rollout's env_args to build the body-joint permutation.
    first_raw = _read_v2_hdf5(rollouts[0])
    isaac_joint_names = list(first_raw["env_args"]["joint_names"])
    body_perm, body_missing = _build_joint_permutation(
        isaac_joint_names, robot_model.joint_names
    )
    print(
        f"[info] gear_sonic joints: {len(robot_model.joint_names)}, "
        f"isaac joints: {len(isaac_joint_names)}, "
        f"missing-on-isaac (zero-filled): {len(body_missing)}"
    )

    features = get_features_sonic_vla(robot_model)
    modality_config = get_modality_config_sonic_vla(robot_model)

    script_config = {
        "source": "isaac_lab_hdf5_converted",
        "source_schema": "producer HDF5 v2 (Part A v2, robomimic-layout)",
        "converter": "convert_isaac_hdf5_to_lerobot.py",
        "converter_version": 2,
        "planner_fields_source": (
            "derived from root_pos_w / root_quat_w in the source HDF5: "
            "planner_movement = world-frame root velocity via np.gradient; "
            "planner_speed = horizontal speed; planner_facing = R(root_quat) @ [1,0,0]; "
            "planner_height = root_pos_w[:, 2]; "
            "planner_mode via speed_to_mode thresholds from "
            "gear_sonic_deploy/.../localmotion_kplanner.hpp"
        ),
        "producer": first_raw["env_args"].get("producer", "unknown"),
        "producer_version": first_raw["env_args"].get("producer_version", 0),
        "task_name": first_raw["env_args"].get("task_name"),
        "robot_name": first_raw["env_args"].get("robot_name"),
        "robot_usd": first_raw["env_args"].get("robot_usd"),
        "fps": args.fps,
        "waist_location": args.waist_location,
        "high_elbow_pose": bool(args.high_elbow),
        "isaac_joint_names": isaac_joint_names,
        "gear_sonic_joint_names": list(robot_model.joint_names),
        "joints_zero_filled_in_state": body_missing,
        "zero_filled_fields": ZERO_FILLED_FIELDS,
        "action_wbc_source": "robot0_joint_pos via name-based permutation",
        "notes": (
            "VLA fine-tuning primary targets per the SONIC paper are the "
            "teleop-format commands (vr_3pt_position, vr_3pt_orientation, "
            "planner_*, hand_joints). Zero-filled fields are auxiliary/token-"
            "space signals only. action.wbc is sourced from joint_pos because "
            "Isaac Lab collection rollouts do not store a separate WBC command."
        ),
    }

    save_root = args.output_path
    if args.dataset_name and not str(save_root).endswith(args.dataset_name):
        save_root = save_root / args.dataset_name
    save_root.parent.mkdir(parents=True, exist_ok=True)

    exporter = Gr00tDataExporter.create(
        save_root=str(save_root),
        fps=args.fps,
        features=features,
        modality_config=modality_config,
        task=args.task_prompt,
        script_config=script_config,
        robot_type="g1",
        overwrite_existing=args.overwrite_existing,
    )

    total_frames = 0
    max_frames = args.max_frames_per_rollout if args.max_frames_per_rollout > 0 else None
    for i, hdf5_path in enumerate(rollouts):
        try:
            written = convert_one_rollout(
                hdf5_path=hdf5_path,
                exporter=exporter,
                robot_model=robot_model,
                body_perm=body_perm,
                body_missing=body_missing,
                canonical_left_finger_names=canonical_left_finger,
                canonical_right_finger_names=canonical_right_finger,
                left_wrist_slots=left_wrist_slots,
                right_wrist_slots=right_wrist_slots,
                expected_left_body_name=expected_left_body,
                expected_right_body_name=expected_right_body,
                rollout_index=i,
                max_frames=max_frames,
            )
            total_frames += written
        except Exception as exc:  # noqa: BLE001
            print(f"[error] rollout {i} ({hdf5_path}): {exc}", file=sys.stderr)
            raise

    print(
        f"[done] wrote {total_frames} frames across {len(rollouts)} episode(s) -> {save_root}"
    )
    print(
        f"[done] zero-filled fields: {ZERO_FILLED_FIELDS}. "
        "Run `process_dataset.py --no-remove-stale-smpl` (or skip post-processing) "
        "to avoid dropping frames with zero SMPL."
    )


if __name__ == "__main__":
    main()
