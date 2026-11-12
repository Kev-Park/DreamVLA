"""Convert validated raw expert episodes to SONIC/LeRobot v2.1 datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .contracts import validate_episode
from .io import read
from .paths import add_external_paths


def optional_features(frame_index):
    """Exporter-required teleoperation fields, not VLA observations or targets."""
    from gear_sonic.utils.data_collection.transforms import quat_to_rot6d

    return {
        "teleop.smpl_joints": np.zeros(72, np.float32),
        "teleop.smpl_pose": np.zeros(63, np.float32),
        "teleop.body_quat_w": np.array([1, 0, 0, 0], np.float32),
        "teleop.target_body_orientation": quat_to_rot6d(
            np.array([1, 0, 0, 0], np.float32)
        ),
        "teleop.smpl_frame_index": np.array([frame_index], np.int64),
        "teleop.left_wrist_joints": np.zeros(3, np.float32),
        "teleop.right_wrist_joints": np.zeros(3, np.float32),
        "teleop.stream_mode": np.zeros(1, np.int32),
        "teleop.planner_mode": np.zeros(1, np.int32),
        "teleop.planner_movement": np.zeros(3, np.float32),
        "teleop.planner_facing": np.array([1, 0, 0], np.float32),
        "teleop.planner_speed": np.array([-1], np.float32),
        "teleop.planner_height": np.array([-1], np.float32),
        "teleop.vr_3pt_position": np.zeros(9, np.float32),
        "teleop.vr_3pt_orientation": np.zeros(18, np.float32),
    }


FIELDS = {
    "observation.state": 43,
    "observation.eef_state": 14,
    "action.wbc": 43,
    "observation.root_orientation": 4,
    "observation.projected_gravity": 3,
    "observation.cpp_rotation_offset": 4,
    "observation.init_base_quat": 4,
    "teleop.delta_heading": 1,
    "action.motion_token": 64,
    "teleop.left_hand_joints": 7,
    "teleop.right_hand_joints": 7,
}


def raw_episodes(root):
    folders = sorted(Path(root).glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if not folders or [p.name for p in folders] != [
        f"episode_{i:06d}" for i in range(len(folders))
    ]:
        raise ValueError("Expected contiguous episode_000000 directories")
    return folders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    episodes = raw_episodes(args.raw_root)
    first = read(episodes[0] / "metadata.json")
    task = first["environment_contract"]["task"]
    # Preflight every episode before creating or modifying an output dataset.
    hashes = {}
    from .io import digest

    for episode in episodes:
        meta = validate_episode(episode, task)
        with np.load(episode / "data.npz", allow_pickle=False) as data:
            for key, width in FIELDS.items():
                if (
                    data[key].shape != (meta["frames"], width)
                    or not np.isfinite(data[key]).all()
                ):
                    raise ValueError(f"Invalid {key}: {episode}")
        hashes[episode.name] = {
            name: digest(episode / name)
            for name in ("metadata.json", "data.npz", "ego_view.mp4")
        }
    source = dict(
        raw_root=str(args.raw_root.resolve()),
        source_hashes=hashes,
        raw_episode_count=len(episodes),
        environment_contract=first["environment_contract"],
        embodiment_tag="UNITREE_G1_SONIC",
        raw_format="dreamcontrol-sonic-raw-v1",
    )
    completed = 0
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(args.output)
        info = read(args.output / "meta/info.json")
        if info.get("script_config") != source:
            raise ValueError(
                "Raw source/provenance changed; refusing conversion resume"
            )
        completed = info["total_episodes"]
        if not 0 <= completed <= len(episodes):
            raise ValueError("Invalid completed episode count")
        if completed == len(episodes):
            print("Conversion already complete")
            return
    add_external_paths()
    import cv2
    import gear_sonic.data.exporter as exporter_module
    from .video import VideoWriter

    exporter_module.VideoWriter = VideoWriter
    Gr00tDataExporter = exporter_module.Gr00tDataExporter
    from gear_sonic.data.features_sonic_vla import (
        get_features_sonic_vla,
        get_modality_config_sonic_vla,
    )
    from .robot_model import get_project_g1_robot_model

    model = get_project_g1_robot_model()
    exporter = Gr00tDataExporter.create(
        save_root=args.output,
        fps=50,
        features=get_features_sonic_vla(model),
        modality_config=get_modality_config_sonic_vla(model),
        task=first["task"],
        script_config=source,
    )
    if exporter.meta.total_episodes != completed:
        raise ValueError("Exporter episode count changed")
    try:
        for index, episode in enumerate(episodes[completed:], start=completed):
            meta = read(episode / "metadata.json")
            capture = cv2.VideoCapture(str(episode / "ego_view.mp4"))
            if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - 50) > 0.01:
                raise ValueError(f"Missing or wrong-rate video: {episode}")
            try:
                with np.load(episode / "data.npz", allow_pickle=False) as arrays:
                    for frame_index in range(meta["frames"]):
                        ok, bgr = capture.read()
                        if not ok or bgr.shape != (480, 640, 3):
                            raise ValueError(
                                f"Missing/invalid RGB frame {frame_index}: {episode}"
                            )
                        frame = {
                            name: arrays[name][frame_index].astype(
                                exporter.features[name]["dtype"], copy=False
                            )
                            for name in FIELDS
                        }
                        frame["observation.images.ego_view"] = cv2.cvtColor(
                            bgr, cv2.COLOR_BGR2RGB
                        )
                        frame.update(optional_features(frame_index))
                        exporter.add_frame(frame)
                    if capture.read()[0]:
                        raise ValueError(
                            f"Video has more frames than labels: {episode}"
                        )
            finally:
                capture.release()
            exporter.save_episode()
            print(f"Converted {index + 1}/{len(episodes)}", flush=True)
    finally:
        exporter.stop_video_writers()


if __name__ == "__main__":
    main()
