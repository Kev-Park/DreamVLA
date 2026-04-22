"""Offline converter: refine_motions_al.py pickle files -> LeRobot v2.1 SONIC VLA dataset.

Reads *_n.pkl files produced by refine_motions_al.py and emits a LeRobot v2.1
dataset whose feature schema mirrors convert_isaac_hdf5_to_lerobot.py, using
the same VR 3-point representation strategy.

Wrist/torso SE3s are computed via pytorch_kinematics FK in the root (pelvis)
frame — identical to the pelvis-relative poses recorded in the HDF5 teleop group.

Fields with no pkl analog are zero-filled:
  - observation.images.ego_view -> black (zeros) image
  - teleop.left_hand_joints / right_hand_joints -> zeros
  - SMPL fields, motion tokens -> zeros (same as HDF5 converter)

Run under the same .venv_data_collection environment as convert_isaac_hdf5_to_lerobot.py.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import pytorch_kinematics as pk
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

URDF_PATH = "../../../Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf"

# G1 27-DOF joint names — order matches the 'joints' tensor column order in pkl files.
JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# FK link names for VR 3-point representation (root-relative SE3).
# Must match the body names used by collect_pick_cam.py and the consumer
# pinocchio model: left/right_wrist_yaw_link, torso_link.
RIGHT_WRIST_LINK = "right_wrist_yaw_link"
LEFT_WRIST_LINK = "left_wrist_yaw_link"
TORSO_LINK = "torso_link"

# Local-frame offsets matching convert_isaac_hdf5_to_lerobot.py / vr3pt_pose_visualizer.py.
LW_LOCAL_OFFSET = np.array([0.18, -0.025, 0.0], dtype=np.float64)
RW_LOCAL_OFFSET = np.array([0.18, 0.025, 0.0], dtype=np.float64)
TORSO_LOCAL_OFFSET = np.array([0.0, 0.0, 0.35], dtype=np.float64)

ZERO_FILLED_FIELDS = [
    "action.motion_token",
    "teleop.smpl_joints",
    "teleop.smpl_pose",
    "teleop.body_quat_w",
    "teleop.target_body_orientation",
    "observation.cpp_rotation_offset",
    "teleop.smpl_frame_index",
    "observation.images.ego_view",
    "teleop.left_hand_joints",
    "teleop.right_hand_joints",
]


def _mat_to_quat_wxyz(rot_mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> wxyz quaternion (scalar-first)."""
    quat_xyzw = R.from_matrix(rot_mat).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
    )


def _apply_local_offset(T: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Translate by an offset expressed in the local frame of T (4x4 SE3)."""
    return T[:3, 3] + T[:3, :3] @ offset


def _build_joint_permutation(
    pkl_joint_names: list[str], gear_sonic_joint_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Return perm such that q_gs = q_pkl[perm], with -1 for gear_sonic joints absent in pkl."""
    pkl_index = {name: i for i, name in enumerate(pkl_joint_names)}
    perm = np.full(len(gear_sonic_joint_names), -1, dtype=np.int64)
    missing: list[str] = []
    for i, name in enumerate(gear_sonic_joint_names):
        idx = pkl_index.get(name)
        if idx is None:
            missing.append(name)
        else:
            perm[i] = idx
    return perm, missing


def _permute_with_zero_fill(q_batch: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply name-based permutation; entries with perm == -1 become 0.0."""
    out = np.zeros((q_batch.shape[0], perm.shape[0]), dtype=np.float64)
    valid = perm >= 0
    out[:, valid] = q_batch[:, perm[valid]]
    return out


def _wrist_joint_slot_indices(robot_model, side: str) -> tuple[int, int, int]:
    """Return (roll_idx, pitch_idx, yaw_idx) in gear_sonic full-q space."""
    return (
        robot_model.dof_index(f"{side}_wrist_roll_joint"),
        robot_model.dof_index(f"{side}_wrist_pitch_joint"),
        robot_model.dof_index(f"{side}_wrist_yaw_joint"),
    )


def _speed_to_mode(s: float) -> int:
    s = abs(float(s))
    if s < 0.1:
        return 0
    if s < 0.8:
        return 1
    if s < 2.5:
        return 2
    return 3


def _run_fk_batch(chain, joints_np: np.ndarray) -> dict[str, np.ndarray]:
    """Run FK for all T frames; return root-relative SE3 (T, 4, 4) per link."""
    joints_t = torch.tensor(joints_np, dtype=torch.float32, device=DEVICE)
    q_dict = {name: joints_t[:, i] for i, name in enumerate(JOINT_NAMES)}
    fk = chain.forward_kinematics(q_dict)
    return {
        link: fk[link].get_matrix().cpu().numpy()
        for link in (LEFT_WRIST_LINK, RIGHT_WRIST_LINK, TORSO_LINK)
        if link in fk
    }


def convert_one_pkl(
    pkl_path: Path,
    chain,
    exporter: Gr00tDataExporter,
    body_perm: np.ndarray,
    body_missing: list[str],
    canonical_left_finger_names: list[str],
    canonical_right_finger_names: list[str],
    left_wrist_slots: tuple[int, int, int],
    right_wrist_slots: tuple[int, int, int],
    fps: float,
    rollout_index: int,
    max_frames: int | None = None,
) -> int:
    print(f"[rollout {rollout_index}] loading {pkl_path.name}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    joints_np = np.array(data["joints"])       # (T, 27)
    wxyz_xyz = np.array(data["global_pose"].wxyz_xyz)  # (T, 7): wxyz | xyz

    n_frames = joints_np.shape[0]
    if max_frames is not None and max_frames > 0:
        n_frames = min(n_frames, max_frames)
    joints_np = joints_np[:n_frames]
    wxyz_xyz = wxyz_xyz[:n_frames]

    root_quat_w = wxyz_xyz[:, :4].astype(np.float64)   # (T, 4) wxyz
    root_pos_w = wxyz_xyz[:, 4:].astype(np.float64)    # (T, 3)

    q_gs_all = _permute_with_zero_fill(joints_np.astype(np.float64), body_perm)

    # FK -> wrist/torso SE3 in root (pelvis) frame, same convention as HDF5 teleop group.
    link_se3 = _run_fk_batch(chain, joints_np)
    lw_se3 = link_se3.get(LEFT_WRIST_LINK)
    rw_se3 = link_se3.get(RIGHT_WRIST_LINK)
    torso_se3 = link_se3.get(TORSO_LINK)
    if lw_se3 is None or rw_se3 is None or torso_se3 is None:
        raise RuntimeError(
            f"FK missing one or more links. Got: {list(link_se3.keys())}. "
            f"Expected: {LEFT_WRIST_LINK}, {RIGHT_WRIST_LINK}, {TORSO_LINK}"
        )

    # Planner fields from root trajectory — same derivation as HDF5 converter.
    step_dt = 1.0 / fps
    planner_movement_all = np.gradient(root_pos_w, step_dt, axis=0).astype(np.float32)
    planner_speed_all = np.linalg.norm(
        planner_movement_all[:, :2], axis=1, keepdims=True
    ).astype(np.float32)
    root_quat_xyzw = root_quat_w[:, [1, 2, 3, 0]]
    planner_facing_all = (
        R.from_quat(root_quat_xyzw).apply(np.array([1.0, 0.0, 0.0])).astype(np.float32)
    )
    planner_height_all = root_pos_w[:, 2:3].astype(np.float32)
    planner_mode_all = np.array(
        [[_speed_to_mode(float(s))] for s in planner_speed_all.flatten()],
        dtype=np.int32,
    )

    # Constant zero-fill arrays (reused every frame).
    init_base_quat = root_quat_w[0].copy()
    zero_motion_token = np.zeros(64, dtype=np.float64)
    zero_smpl_joints = np.zeros(72, dtype=np.float32)
    zero_smpl_pose = np.zeros(63, dtype=np.float32)
    identity_body_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    identity_target_rot6d = quat_to_rot6d(identity_body_quat).astype(np.float32)
    identity_cpp_offset = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    zero_smpl_frame_idx = np.array([0], dtype=np.int64)
    zero_delta_heading = np.zeros(1, dtype=np.float64)
    zero_ego_view = np.zeros((EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3), dtype=np.uint8)
    zero_left_fingers = np.zeros(len(canonical_left_finger_names), dtype=np.float32)
    zero_right_fingers = np.zeros(len(canonical_right_finger_names), dtype=np.float32)
    stream_mode = np.array([5], dtype=np.int32)

    written = 0
    for i in range(n_frames):
        q_gs = q_gs_all[i]
        lw = lw_se3[i]      # (4, 4) root-relative SE3
        rw = rw_se3[i]
        tp = torso_se3[i]

        lw_pos = lw[:3, 3]
        rw_pos = rw[:3, 3]
        lw_quat_wxyz = _mat_to_quat_wxyz(lw[:3, :3])
        rw_quat_wxyz = _mat_to_quat_wxyz(rw[:3, :3])
        observation_eef_state = np.concatenate(
            [lw_pos, lw_quat_wxyz, rw_pos, rw_quat_wxyz], dtype=np.float64
        )

        root_quat = root_quat_w[i]
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

        left_wrist_joints = np.array([q_gs[idx] for idx in left_wrist_slots], dtype=np.float32)
        right_wrist_joints = np.array([q_gs[idx] for idx in right_wrist_slots], dtype=np.float32)

        frame_data = {
            "observation.images.ego_view": zero_ego_view,
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
            "teleop.left_hand_joints": zero_left_fingers.copy(),
            "teleop.right_hand_joints": zero_right_fingers.copy(),
            "teleop.smpl_frame_index": zero_smpl_frame_idx.copy(),
            "teleop.left_wrist_joints": left_wrist_joints,
            "teleop.right_wrist_joints": right_wrist_joints,
            "teleop.stream_mode": stream_mode.copy(),
            "teleop.planner_mode": planner_mode_all[i].copy(),
            "teleop.planner_movement": planner_movement_all[i].copy(),
            "teleop.planner_facing": planner_facing_all[i].copy(),
            "teleop.planner_speed": planner_speed_all[i].copy(),
            "teleop.planner_height": planner_height_all[i].copy(),
            "teleop.vr_3pt_position": vr_3pt_position,
            "teleop.vr_3pt_orientation": vr_3pt_orientation,
        }

        exporter.add_frame(frame_data)
        written += 1

    exporter.save_episode()
    print(f"[rollout {rollout_index}] wrote {written} frames -> episode saved")
    if body_missing and rollout_index == 0:
        print(
            f"[info] {len(body_missing)} gear_sonic joints absent in pkl "
            f"and zero-filled in observation.state / action.wbc: {body_missing}"
        )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert refine_motions_al.py *_n.pkl files to a LeRobot v2.1 "
            "Sonic VLA dataset."
        )
    )
    parser.add_argument(
        "--pkl-root", type=Path, required=True,
        help="Directory containing *_n.pkl files from refine_motions_al.py.",
    )
    parser.add_argument(
        "--output-path", type=Path, required=True,
        help="Output dataset root directory.",
    )
    parser.add_argument(
        "--task-prompt", type=str, required=True,
        help='Natural-language task description, e.g. "pick up the mustard bottle".',
    )
    parser.add_argument(
        "--urdf-path", type=str, default=URDF_PATH,
        help="Path to g1_27dof.urdf (default: relative path from this script's location).",
    )
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument(
        "--fps", type=float, default=20.0,
        help="Trajectory FPS matching TRAJ_FPS_DEFAULT in refine_motions_al.py (default: 20.0).",
    )
    parser.add_argument(
        "--waist-location",
        choices=["lower_body", "upper_body", "lower_and_upper_body"],
        default="lower_body",
    )
    parser.add_argument("--high-elbow", action="store_true")
    parser.add_argument("--rollout-limit", type=int, default=-1, help="-1 = all.")
    parser.add_argument("--max-frames-per-rollout", type=int, default=-1, help="-1 = all.")
    parser.add_argument("--recursive", action="store_true", help="Glob *_n.pkl recursively.")
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    pattern = "**/*_n.pkl" if args.recursive else "*_n.pkl"
    pkl_files = sorted(args.pkl_root.glob(pattern))
    if not pkl_files:
        raise RuntimeError(
            f"No *_n.pkl files found under {args.pkl_root} (recursive={args.recursive})"
        )
    if args.rollout_limit >= 0:
        pkl_files = pkl_files[: args.rollout_limit]
    print(f"[info] found {len(pkl_files)} pkl file(s)")

    chain = pk.build_chain_from_urdf(
        open(args.urdf_path).read()
    ).to(dtype=torch.float32, device=DEVICE)

    robot_model = get_g1_robot_model(
        waist_location=args.waist_location, high_elbow_pose=args.high_elbow
    )
    canonical_left_finger = robot_model.supplemental_info.left_hand_actuated_joints
    canonical_right_finger = robot_model.supplemental_info.right_hand_actuated_joints
    left_wrist_slots = _wrist_joint_slot_indices(robot_model, "left")
    right_wrist_slots = _wrist_joint_slot_indices(robot_model, "right")

    body_perm, body_missing = _build_joint_permutation(JOINT_NAMES, robot_model.joint_names)
    print(
        f"[info] gear_sonic joints: {len(robot_model.joint_names)}, "
        f"pkl joints: {len(JOINT_NAMES)}, "
        f"missing-in-pkl (zero-filled): {len(body_missing)}"
    )

    features = get_features_sonic_vla(robot_model)
    modality_config = get_modality_config_sonic_vla(robot_model)

    script_config = {
        "source": "refine_motions_al_pkl",
        "converter": "convert_to_vr.py",
        "fps": args.fps,
        "waist_location": args.waist_location,
        "high_elbow_pose": bool(args.high_elbow),
        "pkl_joint_names": JOINT_NAMES,
        "gear_sonic_joint_names": list(robot_model.joint_names),
        "joints_zero_filled_in_state": body_missing,
        "zero_filled_fields": ZERO_FILLED_FIELDS,
        "fk_links": {
            "left_wrist": LEFT_WRIST_LINK,
            "right_wrist": RIGHT_WRIST_LINK,
            "torso": TORSO_LINK,
        },
        "notes": (
            "Wrist/torso SE3 computed via pytorch_kinematics FK in root (pelvis) frame — "
            "equivalent to the pelvis-relative teleop poses in the HDF5 pipeline. "
            "ego_view is zero-filled (no camera in pkl). "
            "Finger joints are zero-filled (not recorded in pkl). "
            "Planner fields derived from root trajectory via np.gradient, same as HDF5 converter."
        ),
    }

    save_root = args.output_path
    if args.dataset_name and not str(save_root).endswith(args.dataset_name):
        save_root = save_root / args.dataset_name
    save_root.parent.mkdir(parents=True, exist_ok=True)

    exporter = Gr00tDataExporter.create(
        save_root=str(save_root),
        fps=int(args.fps),
        features=features,
        modality_config=modality_config,
        task=args.task_prompt,
        script_config=script_config,
        robot_type="g1",
        overwrite_existing=args.overwrite_existing,
    )

    total_frames = 0
    max_frames = args.max_frames_per_rollout if args.max_frames_per_rollout > 0 else None
    for i, pkl_path in enumerate(pkl_files):
        try:
            written = convert_one_pkl(
                pkl_path=pkl_path,
                chain=chain,
                exporter=exporter,
                body_perm=body_perm,
                body_missing=body_missing,
                canonical_left_finger_names=canonical_left_finger,
                canonical_right_finger_names=canonical_right_finger,
                left_wrist_slots=left_wrist_slots,
                right_wrist_slots=right_wrist_slots,
                fps=args.fps,
                rollout_index=i,
                max_frames=max_frames,
            )
            total_frames += written
        except Exception as exc:
            print(f"[error] rollout {i} ({pkl_path}): {exc}", file=sys.stderr)
            raise

    print(f"[done] wrote {total_frames} frames across {len(pkl_files)} episode(s) -> {save_root}")
    print(f"[done] zero-filled fields: {ZERO_FILLED_FIELDS}")


if __name__ == "__main__":
    main()
