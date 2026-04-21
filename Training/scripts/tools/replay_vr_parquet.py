#!/usr/bin/env python3
"""Parquet VR data playback visualizer for SONIC VLA converted datasets.

Reads a LeRobot v2.1 parquet episode produced by convert_isaac_hdf5_to_lerobot.py
and replays all VR-derived fields live in a viser scene on port 8081.

Visualized
----------
- VR 3-point skeleton: left wrist / right wrist / neck — spheres + orientation frames
- Skeleton arm lines (neck → each wrist)
- Torso stub line (neck → estimated torso joint, reversing the TORSO_LOCAL_OFFSET)
- Raw EEF wrist positions (before local-frame offset) as smaller markers
- Robot root orientation frame positioned at (neck_xy, planner_height_z)
- Trajectory trails for all three VR points
- Planner HUD: mode / speed / height / facing arrow / projected gravity
- Wrist joint angles (roll/pitch/yaw) and finger joint angles per hand

Usage
-----
    python replay_vr_parquet.py path/to/episode_000000.parquet [--fps 50] [--trail 60] [--port 8081]
"""

from __future__ import annotations

import argparse
import time
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import viser


# ── Rotation helpers ──────────────────────────────────────────────────────────

def rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """First-two-columns 6D → 3×3 rotation matrix (Zhou et al. CVPR 2019).

    Storage order: [col0_x, col0_y, col0_z, col1_x, col1_y, col1_z].
    """
    col0 = r6d[:3].astype(np.float64)
    col1 = r6d[3:6].astype(np.float64)
    n0 = np.linalg.norm(col0)
    if n0 > 1e-8:
        col0 /= n0
    col1 -= np.dot(col1, col0) * col0
    n1 = np.linalg.norm(col1)
    if n1 > 1e-8:
        col1 /= n1
    return np.stack([col0, col1, np.cross(col0, col1)], axis=1)  # (3, 3)


def rot6d_to_wxyz(r6d: np.ndarray) -> np.ndarray:
    mat = rot6d_to_matrix(r6d)
    q = R.from_matrix(mat).as_quat()           # xyzw
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


# ── Parquet loading ───────────────────────────────────────────────────────────

def _cell_to_array(val, dtype) -> np.ndarray:
    if isinstance(val, np.ndarray):
        return val.astype(dtype)
    return np.array(val, dtype=dtype)


def load_episode(path: Path) -> dict:
    df = pd.read_parquet(path)
    n = len(df)

    def col(name: str, dtype=np.float32):
        if name not in df.columns:
            return None
        return np.stack([_cell_to_array(df[name].iloc[i], dtype) for i in range(n)])

    data: dict = {
        "n_frames":         n,
        "vr_3pt_pos":       col("teleop.vr_3pt_position"),                    # (N,9)
        "vr_3pt_ori":       col("teleop.vr_3pt_orientation"),                  # (N,18)
        "eef_state":        col("observation.eef_state",       np.float64),    # (N,14)
        "root_quat":        col("observation.root_orientation", np.float64),   # (N,4) wxyz
        "proj_gravity":     col("observation.projected_gravity", np.float64),  # (N,3)
        "planner_height":   col("teleop.planner_height"),                      # (N,1)
        "planner_mode":     col("teleop.planner_mode",         np.int32),      # (N,1)
        "planner_speed":    col("teleop.planner_speed"),                       # (N,1)
        "planner_movement": col("teleop.planner_movement"),                    # (N,3)
        "planner_facing":   col("teleop.planner_facing"),                      # (N,3)
        "lw_joints":        col("teleop.left_wrist_joints"),                   # (N,3)
        "rw_joints":        col("teleop.right_wrist_joints"),                  # (N,3)
        "lh_joints":        col("teleop.left_hand_joints"),                    # (N,7)
        "rh_joints":        col("teleop.right_hand_joints"),                   # (N,7)
    }

    for ts_key in ("timestamp", "timestamps"):
        if ts_key in df.columns:
            data["timestamps"] = col(ts_key, np.float64)
            break
    else:
        data["timestamps"] = None

    data["task"] = str(df["task"].iloc[0]) if "task" in df.columns else ""

    missing = [k for k, v in data.items() if v is None and k not in ("timestamps", "task")]
    if missing:
        print(f"[warn] columns not found in parquet (will be skipped): {missing}")

    return data


PLANNER_MODE_LABELS = {0: "IDLE", 1: "SLOW_WALK", 2: "WALK", 3: "RUN"}

# Local-frame offsets applied by convert_isaac_hdf5_to_lerobot.py.
# We reverse the neck offset to draw the torso stub.
TORSO_LOCAL_OFFSET = np.array([0.0, 0.0, 0.35], dtype=np.float64)


# ── Trail ring buffer ─────────────────────────────────────────────────────────

class TrailBuffer:
    def __init__(self, capacity: int):
        self.cap = capacity
        self._buf: list[np.ndarray] = []

    def push(self, pt: np.ndarray) -> np.ndarray | None:
        self._buf.append(pt.copy().astype(np.float32))
        if len(self._buf) > self.cap:
            self._buf.pop(0)
        if len(self._buf) < 2:
            return None
        pts = np.array(self._buf, dtype=np.float32)
        return np.stack([pts[:-1], pts[1:]], axis=1)   # (K-1, 2, 3)

    def reset(self):
        self._buf.clear()


# ── Colors ────────────────────────────────────────────────────────────────────

_C_LW    = (0x55, 0x88, 0xFF)  # blue  — left wrist VR point
_C_RW    = (0xFF, 0x55, 0x44)  # red   — right wrist VR point
_C_NK    = (0x44, 0xDD, 0x66)  # green — neck VR point
_C_EEF   = (0xCC, 0xCC, 0xFF)  # pale  — raw EEF wrists (no local offset)
_C_YEL   = (0xFF, 0xFF, 0x00)  # yellow — facing arrow
_C_GREY  = (0xAA, 0xAA, 0xAA)  # grey  — torso stub


def _seg_colors(c1, c2, n: int = 1) -> np.ndarray:
    """Build (n, 2, 3) uint8 color array for n line segments."""
    return np.tile(np.array([[c1, c2]], dtype=np.uint8), (n, 1, 1))


# ── Scene setup ───────────────────────────────────────────────────────────────

def setup_scene(server: viser.ViserServer) -> dict:
    sc = server.scene

    sc.add_grid("ground", width=8.0, height=8.0, cell_size=0.25, plane="xy")
    sc.add_frame("world/origin", axes_length=0.20, axes_radius=0.006,
                 wxyz=(1, 0, 0, 0), position=(0, 0, 0))

    _dummy_pts  = np.zeros((1, 2, 3), dtype=np.float32)
    _dummy_cols = np.zeros((1, 2, 3), dtype=np.uint8)

    # VR 3-point spheres
    lw_sph = sc.add_icosphere("vr/lwrist/sphere", radius=0.045, color=_C_LW)
    rw_sph = sc.add_icosphere("vr/rwrist/sphere", radius=0.045, color=_C_RW)
    nk_sph = sc.add_icosphere("vr/neck/sphere",   radius=0.045, color=_C_NK)

    # Raw EEF wrist positions (smaller, paler) — before local-frame offsets
    eef_lw = sc.add_icosphere("eef/lwrist", radius=0.025, color=_C_EEF)
    eef_rw = sc.add_icosphere("eef/rwrist", radius=0.025, color=_C_EEF)

    # VR orientation frames
    lw_frm = sc.add_frame("vr/lwrist/frame", axes_length=0.10, axes_radius=0.005,
                           wxyz=(1, 0, 0, 0), position=(0, 0, 0))
    rw_frm = sc.add_frame("vr/rwrist/frame", axes_length=0.10, axes_radius=0.005,
                           wxyz=(1, 0, 0, 0), position=(0, 0, 0))
    nk_frm = sc.add_frame("vr/neck/frame",   axes_length=0.10, axes_radius=0.005,
                           wxyz=(1, 0, 0, 0), position=(0, 0, 0))

    # Robot root orientation frame (approx: neck_xy + planner_height_z)
    root_frm = sc.add_frame("robot/root", axes_length=0.18, axes_radius=0.007,
                             wxyz=(1, 0, 0, 0), position=(0, 0, 0))

    # Skeleton lines
    line_l   = sc.add_line_segments("skeleton/left_arm",  points=_dummy_pts,
                                    colors=_dummy_cols, line_width=4.0)
    line_r   = sc.add_line_segments("skeleton/right_arm", points=_dummy_pts,
                                    colors=_dummy_cols, line_width=4.0)
    line_t   = sc.add_line_segments("skeleton/torso",     points=_dummy_pts,
                                    colors=_dummy_cols, line_width=3.0)

    # Facing arrow (yellow)
    facing_seg = sc.add_line_segments("planner/facing",   points=_dummy_pts,
                                      colors=_dummy_cols, line_width=5.0)

    # Trajectory trails
    trail_lw = sc.add_line_segments("trail/lwrist", points=_dummy_pts,
                                    colors=_dummy_cols, line_width=2.0)
    trail_rw = sc.add_line_segments("trail/rwrist", points=_dummy_pts,
                                    colors=_dummy_cols, line_width=2.0)
    trail_nk = sc.add_line_segments("trail/neck",   points=_dummy_pts,
                                    colors=_dummy_cols, line_width=2.0)

    return dict(
        lw_sph=lw_sph, rw_sph=rw_sph, nk_sph=nk_sph,
        eef_lw=eef_lw, eef_rw=eef_rw,
        lw_frm=lw_frm, rw_frm=rw_frm, nk_frm=nk_frm,
        root_frm=root_frm,
        line_l=line_l, line_r=line_r, line_t=line_t,
        facing_seg=facing_seg,
        trail_lw=trail_lw, trail_rw=trail_rw, trail_nk=trail_nk,
    )


# ── Per-frame update ──────────────────────────────────────────────────────────

def update_frame(
    handles: dict,
    data: dict,
    fi: int,
    trails: dict[str, TrailBuffer],
    gui_info,
    gui_wrist,
    gui_hand,
) -> None:
    vr_pos = data["vr_3pt_pos"][fi]   # (9,)
    vr_ori = data["vr_3pt_ori"][fi]   # (18,)

    lw_pos = vr_pos[0:3].astype(np.float64)
    rw_pos = vr_pos[3:6].astype(np.float64)
    nk_pos = vr_pos[6:9].astype(np.float64)

    lw_wxyz = rot6d_to_wxyz(vr_ori[0:6])
    rw_wxyz = rot6d_to_wxyz(vr_ori[6:12])
    nk_wxyz = rot6d_to_wxyz(vr_ori[12:18])

    # VR spheres
    handles["lw_sph"].position = tuple(lw_pos)
    handles["rw_sph"].position = tuple(rw_pos)
    handles["nk_sph"].position = tuple(nk_pos)

    # VR orientation frames
    for frm, pos, wxyz in [
        (handles["lw_frm"], lw_pos, lw_wxyz),
        (handles["rw_frm"], rw_pos, rw_wxyz),
        (handles["nk_frm"], nk_pos, nk_wxyz),
    ]:
        frm.position = tuple(pos)
        frm.wxyz     = tuple(wxyz)

    # Raw EEF wrist positions (before local-frame offsets)
    if data["eef_state"] is not None:
        eef = data["eef_state"][fi]
        handles["eef_lw"].position = tuple(eef[0:3])
        handles["eef_rw"].position = tuple(eef[7:10])

    # Root frame: neck_xy + planner_height_z, oriented by root_quat
    height = float(data["planner_height"][fi][0]) if data["planner_height"] is not None else 0.0
    root_pos = np.array([nk_pos[0], nk_pos[1], height])
    root_wxyz = (1.0, 0.0, 0.0, 0.0)
    if data["root_quat"] is not None:
        root_wxyz = tuple(data["root_quat"][fi].astype(float))
    handles["root_frm"].position = tuple(root_pos)
    handles["root_frm"].wxyz     = root_wxyz

    # Skeleton: neck→lwrist, neck→rwrist
    handles["line_l"].points = np.array([[lw_pos, nk_pos]], dtype=np.float32)
    handles["line_l"].colors = _seg_colors(_C_LW, _C_NK)
    handles["line_r"].points = np.array([[rw_pos, nk_pos]], dtype=np.float32)
    handles["line_r"].colors = _seg_colors(_C_RW, _C_NK)

    # Torso stub: neck → torso_link (reverse TORSO_LOCAL_OFFSET in neck local frame)
    nk_mat    = rot6d_to_matrix(vr_ori[12:18])
    torso_est = nk_pos - nk_mat @ TORSO_LOCAL_OFFSET
    handles["line_t"].points = np.array([[nk_pos, torso_est.astype(np.float32)]], dtype=np.float32)
    handles["line_t"].colors = _seg_colors(_C_NK, _C_GREY)

    # Facing arrow: from root_pos toward facing direction
    if data["planner_facing"] is not None:
        facing = data["planner_facing"][fi].astype(np.float64)
        arrow_end = root_pos + facing * 0.5
        handles["facing_seg"].points = np.array([[root_pos, arrow_end]], dtype=np.float32)
        handles["facing_seg"].colors = _seg_colors(_C_YEL, _C_YEL)

    # Trails
    for key, pt, handle, color in [
        ("lw", lw_pos, handles["trail_lw"], _C_LW),
        ("rw", rw_pos, handles["trail_rw"], _C_RW),
        ("nk", nk_pos, handles["trail_nk"], _C_NK),
    ]:
        segs = trails[key].push(pt.astype(np.float32))
        if segs is not None:
            k = len(segs)
            handle.points = segs
            handle.colors = _seg_colors(color, color, n=k)

    # ── HUD text ──────────────────────────────────────────────────────────────

    mode_int = int(data["planner_mode"][fi][0]) if data["planner_mode"] is not None else 0
    mode_str = PLANNER_MODE_LABELS.get(mode_int, f"MODE_{mode_int}")
    speed    = float(data["planner_speed"][fi][0]) if data["planner_speed"] is not None else 0.0
    mov      = data["planner_movement"][fi] if data["planner_movement"] is not None else np.zeros(3)
    facing_v = data["planner_facing"][fi]   if data["planner_facing"]   is not None else np.zeros(3)
    grav     = data["proj_gravity"][fi]     if data["proj_gravity"]     is not None else np.zeros(3)

    gui_info.content = (
        f"**Frame** {fi + 1} / {data['n_frames']}\n\n"
        f"**Planner:** `{mode_str}`  speed `{speed:.3f}` m/s  height `{height:.3f}` m\n\n"
        f"**Movement:**  `[{mov[0]:+.3f}, {mov[1]:+.3f}, {mov[2]:+.3f}]`\n\n"
        f"**Facing:**    `[{facing_v[0]:+.3f}, {facing_v[1]:+.3f}, {facing_v[2]:+.3f}]`\n\n"
        f"**ProjGrav:**  `[{grav[0]:+.3f}, {grav[1]:+.3f}, {grav[2]:+.3f}]`"
    )

    lw_j = data["lw_joints"][fi] if data["lw_joints"] is not None else np.zeros(3)
    rw_j = data["rw_joints"][fi] if data["rw_joints"] is not None else np.zeros(3)
    gui_wrist.content = (
        f"**Left wrist** &nbsp; roll `{np.degrees(lw_j[0]):+6.1f}°` "
        f"pitch `{np.degrees(lw_j[1]):+6.1f}°` yaw `{np.degrees(lw_j[2]):+6.1f}°`\n\n"
        f"**Right wrist** roll `{np.degrees(rw_j[0]):+6.1f}°` "
        f"pitch `{np.degrees(rw_j[1]):+6.1f}°` yaw `{np.degrees(rw_j[2]):+6.1f}°`"
    )

    lh = data["lh_joints"][fi] if data["lh_joints"] is not None else np.zeros(7)
    rh = data["rh_joints"][fi] if data["rh_joints"] is not None else np.zeros(7)
    lh_str = "  ".join(f"`{np.degrees(v):+5.1f}°`" for v in lh)
    rh_str = "  ".join(f"`{np.degrees(v):+5.1f}°`" for v in rh)
    gui_hand.content = f"**L hand:** {lh_str}\n\n**R hand:** {rh_str}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet", type=Path, help="Path to episode_*.parquet file")
    parser.add_argument("--fps",   type=float, default=50.0,
                        help="Nominal playback FPS (default: 50)")
    parser.add_argument("--trail", type=int,   default=60,
                        help="Trajectory trail length in frames (default: 60)")
    parser.add_argument("--port",  type=int,   default=8081,
                        help="Viser server port (default: 8081)")
    args = parser.parse_args()

    print(f"[load] {args.parquet}")
    data = load_episode(args.parquet)
    n    = data["n_frames"]
    task = data.get("task", "")
    print(f"[load] {n} frames  task='{task}'")

    server = viser.ViserServer(port=args.port)
    print(f"[viser] http://localhost:{args.port}")

    handles = setup_scene(server)
    trails: dict[str, TrailBuffer] = {
        "lw": TrailBuffer(args.trail),
        "rw": TrailBuffer(args.trail),
        "nk": TrailBuffer(args.trail),
    }

    # ── GUI panels ────────────────────────────────────────────────────────────
    with server.gui.add_folder("Playback"):
        gui_slider  = server.gui.add_slider("Frame", min=0, max=n - 1,
                                             step=1, initial_value=0)
        gui_playing = server.gui.add_checkbox("Playing", initial_value=True)
        gui_speed   = server.gui.add_slider("Speed ×", min=0.1, max=8.0,
                                             step=0.1, initial_value=1.0)
        gui_loop    = server.gui.add_checkbox("Loop", initial_value=True)
        gui_clr_tr  = server.gui.add_button("Clear Trails")

    with server.gui.add_folder("Info"):
        _task_md = server.gui.add_markdown(
            f"**Task:** {task}" if task else "*(no task string)*"
        )
        gui_info  = server.gui.add_markdown("—")
        gui_wrist = server.gui.add_markdown("—")
        gui_hand  = server.gui.add_markdown("—")

    with server.gui.add_folder("Visibility"):
        gui_eef    = server.gui.add_checkbox("Raw EEF wrists",      initial_value=True)
        gui_trails = server.gui.add_checkbox("Trails",              initial_value=True)
        gui_root   = server.gui.add_checkbox("Root frame",          initial_value=True)
        gui_facing = server.gui.add_checkbox("Facing arrow",        initial_value=True)
        gui_torso  = server.gui.add_checkbox("Torso stub line",     initial_value=True)

    @gui_clr_tr.on_click
    def _clear_trails(_):
        for t in trails.values():
            t.reset()

    # ── Playback state ────────────────────────────────────────────────────────
    _fi    = 0
    _dirty = threading.Event()   # set when slider moves while paused

    @gui_slider.on_update
    def _on_slider(ev):
        nonlocal _fi
        _fi = int(ev.target.value)
        _dirty.set()

    def _apply_visibility():
        for h in [handles["eef_lw"], handles["eef_rw"]]:
            h.visible = gui_eef.value
        for h in [handles["trail_lw"], handles["trail_rw"], handles["trail_nk"]]:
            h.visible = gui_trails.value
        handles["root_frm"].visible   = gui_root.value
        handles["facing_seg"].visible = gui_facing.value
        handles["line_t"].visible     = gui_torso.value

    def _render(fi: int):
        _apply_visibility()
        update_frame(handles, data, fi, trails, gui_info, gui_wrist, gui_hand)

    def _playback_loop():
        nonlocal _fi
        while True:
            if gui_playing.value:
                fi = _fi
                _render(fi)
                gui_slider.value = fi

                fi += 1
                if fi >= n:
                    if gui_loop.value:
                        fi = 0
                        for t in trails.values():
                            t.reset()
                    else:
                        fi = n - 1
                        gui_playing.value = False
                _fi = fi
            else:
                if _dirty.is_set():
                    _dirty.clear()
                    _render(_fi)
                    gui_slider.value = _fi

            dt = 1.0 / (args.fps * max(0.05, gui_speed.value))
            time.sleep(dt)

    threading.Thread(target=_playback_loop, daemon=True).start()

    print("[ready] open the URL above in your browser — Ctrl-C to exit")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
