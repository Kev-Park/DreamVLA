"""Array/video serialization shared by the native successful-episode recorder."""

from fractions import Fraction
import json
import subprocess

import cv2
import numpy as np

from .paths import ffmpeg_path, ffprobe_path


def write_episode_media(folder, data, frames):
    if len(frames) != 499:
        raise ValueError("Incomplete successful episode")
    np.savez_compressed(
        folder / "data.npz", **{k: np.asarray(v) for k, v in data.items()}
    )
    cv2.imwrite(
        str(folder / "first_frame.png"),
        cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR),
    )
    cmd = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "640x480",
        "-r",
        "50",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(folder / "ego_view.mp4"),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for rgb in frames:
            proc.stdin.write(rgb.tobytes())
    finally:
        proc.stdin.close()
    if proc.wait(timeout=60):
        raise RuntimeError("Video encoder failed")
    probe = subprocess.run(
        [ffprobe_path(), "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height",
         "-of", "json", str(folder / "ego_view.mp4")],
        check=True, capture_output=True, text=True, timeout=60,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    if (int(stream["nb_read_frames"]) != 499
            or Fraction(stream["avg_frame_rate"]) != 50
            or (int(stream["width"]), int(stream["height"])) != (640, 480)):
        raise ValueError("Encoded video mismatch")
