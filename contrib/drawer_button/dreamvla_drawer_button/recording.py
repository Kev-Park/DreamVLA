"""Array/video serialization shared by the native successful-episode recorder."""

import subprocess

import cv2
import numpy as np

from .paths import ffmpeg_path


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
    cap = cv2.VideoCapture(str(folder / "ego_view.mp4"))
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if count != 499 or fps != 50:
        raise ValueError("Encoded video mismatch")
