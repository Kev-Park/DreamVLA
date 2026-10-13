"""Verify one complete walking episode survives the actual sharded converter.

Checks the fine-tuning action labels against the recorded executed SONIC tokens
and finger poses, with exact frame coverage. Runs remotely in the existing data
collection environment; keeps its small output and verification receipt.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    root = Path.home()/"kevin"
    out = root/"datasets/walk_v3_study/conversion_preflight"
    assert not out.exists(), "Inspect existing preflight before restarting"
    inputs = out/"hdf5"
    inputs.mkdir(parents=True)
    source = args.source.resolve(strict=True)
    assert source.is_relative_to(root.resolve()), str(source)
    (inputs/source.name).symlink_to(source)
    status = root/"eval_videos/walk_v3_study/conversion_preflight.json"
    report = {"state": "converting", "pid": os.getpid(), "started": time.time(), "source": str(source)}
    status.write_text(json.dumps(report, indent=2))
    try:
        with h5py.File(source, "r", locking=False) as f:
            group = f["data/demo_0"]
            tokens = group["obs/motion_token"][()].astype(np.float32)
            fingers = group["teleop/finger_joints/right"][()].astype(np.float32)
            finger_names = [n.decode() if isinstance(n, bytes) else str(n)
                            for n in group["teleop/finger_joints/right_finger_joint_names"][()]]
            frames = len(tokens)
            assert tokens.shape == (frames, 64) and fingers.shape == (frames, 7)
            assert group["obs/ego_view_image"].shape == (frames, 960, 1280, 3)
        subprocess.run(["bash", str(Path(__file__).with_name("convert_sharded.sh")),
            "--hdf5-root", str(inputs), "--recursive", "--output-path", str(out/"lerobot"),
            "--dataset-name", "ds", "--task-prompt", "pick up the mustard bottle",
            "--fps", "50", "--shards", "1"], check=True)
        dataset = out/"lerobot/ds"
        info = json.loads((dataset/"meta/info.json").read_text())
        assert info["total_episodes"] == 1 and info["total_frames"] == frames
        assert "recorded executed token" in info["script_config"]["motion_token_source"]
        parquet = sorted(dataset.glob("data/**/*.parquet"))
        assert len(parquet) == 1
        table = pq.read_table(parquet[0])
        actual_tokens = np.asarray(table["action.motion_token"].to_pylist(), dtype=np.float32)
        actual_fingers = np.asarray(table["teleop.right_hand_joints"].to_pylist(), dtype=np.float32)
        from gear_sonic.data.features_sonic_vla import get_g1_robot_model
        robot = get_g1_robot_model()
        canonical_names = list(robot.supplemental_info.right_hand_actuated_joints)
        assert set(canonical_names) == set(finger_names)
        fingers = fingers[:, [finger_names.index(n) for n in canonical_names]]
        assert np.array_equal(actual_tokens, tokens), "SONIC token labels changed during conversion"
        assert np.array_equal(actual_fingers, fingers), "Right-hand labels changed during conversion"
        assert len(list(dataset.glob("videos/**/*.mp4"))) == 1
        report.update(state="complete", completed=time.time(), frames=frames,
                      dataset=str(dataset), token_labels_exact=True, right_hand_labels_exact=True)
    except BaseException as exc:
        report.update(state="failed", error=repr(exc), completed=time.time())
        raise
    finally:
        status.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
