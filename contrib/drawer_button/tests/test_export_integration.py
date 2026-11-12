"""Opt-in CPU integration test; requires the separately installed exporter/assets."""

import os
import sys

import numpy as np
import pytest


@pytest.mark.skipif(
    os.getenv("DREAMVLA_TEST_EXPORT") != "1",
    reason="Opt in with the SONIC data environment and resolved robot URDF",
)
def test_real_exporter_on_synthetic_episode(tmp_path, monkeypatch):
    import cv2
    from dreamvla_drawer_button.contracts import scene
    from dreamvla_drawer_button.convert import FIELDS, main
    from dreamvla_drawer_button.io import read, write_json
    from dreamvla_drawer_button.video import VideoWriter

    raw = tmp_path / "raw"
    ep = raw / "episode_000000"
    ep.mkdir(parents=True)
    frames = 50
    arrays = {
        key: np.zeros((frames, width), np.float64) for key, width in FIELDS.items()
    }
    for key in (
        "observation.root_orientation",
        "observation.cpp_rotation_offset",
        "observation.init_base_quat",
    ):
        arrays[key][:, 0] = 1
    arrays["observation.projected_gravity"][:, 2] = -1
    arrays["diagnostic.control_step"] = np.arange(frames)[:, None]
    np.savez(ep / "data.npz", **arrays)
    write_json(
        ep / "metadata.json",
        dict(
            accepted=True,
            evaluation_only=False,
            frames=frames,
            fps=50,
            task="open the drawer",
            environment_contract=scene("drawer"),
            synthetic_fixture=True,
        ),
    )
    writer = VideoWriter(ep / "ego_view.mp4", 640, 480, 50)
    for i in range(frames):
        writer.add_frame(np.full((480, 640, 3), i * 4, np.uint8))
    writer.stop()
    output = tmp_path / "lerobot"
    monkeypatch.setattr(sys, "argv", ["convert", str(raw), str(output)])
    main()
    info = read(output / "meta/info.json")
    assert info["total_episodes"] == 1 and info["total_frames"] == frames
    assert info["script_config"]["raw_episode_count"] == 1
    videos = list((output / "videos").rglob("*.mp4"))
    assert len(videos) == 1
    cap = cv2.VideoCapture(str(videos[0]))
    try:
        decoded = []
        while True:
            ok, rgb = cap.read()
            if not ok:
                break
            decoded.append(rgb.mean())
        assert len(decoded) == frames and decoded[-1] > decoded[0] + 150
        assert abs(cap.get(cv2.CAP_PROP_FPS) - 50) < 0.01
    finally:
        cap.release()
    # Resume validates all source hashes and is a no-op once complete.
    monkeypatch.setattr(sys, "argv", ["convert", str(raw), str(output), "--resume"])
    main()
