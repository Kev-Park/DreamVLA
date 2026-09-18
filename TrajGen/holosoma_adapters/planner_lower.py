"""Re-synthesize the lower body with SONIC's own kinematic planner.

Motivation
----------
The retargeted lower body is dense per-joint targets taken from a human walk. latband60loco
showed the policy cannot track it: held-upright fell 67.9% -> 6.3%, 54% of episodes never
reached the object, and 99.9% of episodes ended in time_out rather than a fall -- the robot
does not fall over, it simply never gets where the reference goes.

This module drops those dense targets entirely and instead **reparameterises the reference as
commands the SONIC stack was trained to follow**: root waypoints (position + heading) and a
pelvis height command. `planner_sonic.onnx` -- the same kinematic planner the deploy path runs
-- turns those commands into a gait. The resulting legs and root are in-distribution for SONIC
by construction, because SONIC's own planner produced them.

What is kept from the retarget: the waist (3) and both arms (14), i.e. the manipulation, plus
the root path *as a target* (not as dense state). What is replaced: the root trajectory that is
actually written to the reference, and the 12 leg joints.

Conventions
-----------
planner_sonic.onnx works in the MuJoCo Z-up world frame. Its ``mujoco_qpos`` is
``[0:3] root pos | [3:7] root quat (w,x,y,z) | [7:36] 29 joints``, and those 29 joints are in
MUJOCO body-tree order, which is byte-identical to ``refine_al_29.JOINT_NAMES_29`` (left leg 6,
right leg 6, waist 3, left arm 7, right arm 7) -- so columns map 1:1 with no permutation.

The planner emits 30 Hz; one token is 4 frames. We replan at 10 Hz (every 3 emitted frames),
matching the deploy-side C++ cadence, and feed ``specific_target_positions`` /
``specific_target_headings`` (one token = the next 4 reference root poses) so the gait tracks
the reference path instead of free-running on a velocity command and drifting.
"""

from __future__ import annotations

import os

import numpy as np

PLANNER_HZ = 30.0
FRAMES_PER_TOKEN = 4          # one token's worth of waypoints
REPLAN_EVERY = 3              # emitted frames between replans (10 Hz)
_DEFAULT_ANGLES_29 = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     0.0,   0.0, 0.0,
     0.2,   0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
     0.2,  -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float32)


def _yaw_of(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_to_quat(yaw: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(yaw / 2.0), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2.0)], axis=1)


def _resample(a: np.ndarray, n_out: int) -> np.ndarray:
    """Linear resample along axis 0 to n_out samples (endpoints preserved)."""
    n_in = a.shape[0]
    if n_in == n_out:
        return a.copy()
    src = np.linspace(0.0, n_in - 1.0, n_out)
    lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, n_in - 1)
    t = (src - lo).reshape(-1, *([1] * (a.ndim - 1)))
    return a[lo] * (1.0 - t) + a[hi] * t


def _unwrap_resample_angle(ang: np.ndarray, n_out: int) -> np.ndarray:
    return _resample(np.unwrap(ang), n_out)


def plan_lower_body(base_pos: np.ndarray, base_quat: np.ndarray, fps: float = 20.0,
                    onnx_path: str | None = None, seed: int = 1234, verbose: bool = True):
    """Replace the root + legs with a planner-generated gait that tracks the reference path.

    Args:
        base_pos:  (F,3) reference root position (pre-grounding).
        base_quat: (F,4) reference root quaternion, wxyz.
        fps:       reference frame rate (holosoma refs are 20 Hz).

    Returns:
        (base_pos_new (F,3), base_quat_new (F,4 wxyz), legs (F,12)) at the reference rate/length.
    """
    from vla_sonic.planner_wrapper import PlannerWrapper
    from vla_sonic.frame_transforms import speed_to_mode
    from vla_sonic.repo_paths import gear_sonic_deploy

    if onnx_path is None:
        onnx_path = os.environ.get("HS_PLANNER_ONNX") or gear_sonic_deploy(
            "planner", "target_vel", "V2", "planner_sonic.onnx")

    F = base_pos.shape[0]
    n_plan = max(FRAMES_PER_TOKEN + 1, int(round(F * PLANNER_HZ / fps)))

    # Reference path as COMMANDS, at planner rate.
    ref_xy = _resample(base_pos[:, :2].astype(np.float64), n_plan)
    ref_z = _resample(base_pos[:, 2].astype(np.float64), n_plan)
    ref_yaw = _unwrap_resample_angle(_yaw_of(base_quat.astype(np.float64)), n_plan)
    ref_speed = np.concatenate([[0.0], np.linalg.norm(np.diff(ref_xy, axis=0), axis=1) * PLANNER_HZ])

    planner = PlannerWrapper(onnx_path)

    # Context bootstrap: 4 frames standing at the reference's START pose, so the planner is in
    # our world frame from frame 0 (the canned origin-standing context would make it walk in
    # from somewhere else and need a rigid fix-up afterwards).
    ctx = np.zeros((1, 4, 36), dtype=np.float32)
    ctx[0, :, 0:2] = ref_xy[0]
    ctx[0, :, 2] = ref_z[0]
    ctx[0, :, 3:7] = _yaw_to_quat(np.array([ref_yaw[0]]))[0]
    ctx[0, :, 7:36] = _DEFAULT_ANGLES_29

    out_qpos = np.zeros((n_plan, 36), dtype=np.float32)
    cached = None
    cache_i = 0
    n_replans = 0
    for k in range(n_plan):
        if k % REPLAN_EVERY == 0 or cached is None or cache_i >= cached.shape[0]:
            idx = np.clip(np.arange(k, k + FRAMES_PER_TOKEN), 0, n_plan - 1)
            wp = np.zeros((1, FRAMES_PER_TOKEN, 3), dtype=np.float32)
            wp[0, :, :2] = ref_xy[idx]
            wp[0, :, 2] = ref_z[idx]
            head = ref_yaw[idx].astype(np.float32)[None, :]
            here = ctx[0, -1, 0:2].astype(np.float64)
            to_goal = ref_xy[idx[-1]] - here
            nrm = float(np.linalg.norm(to_goal))
            move = np.array([[to_goal[0] / nrm, to_goal[1] / nrm, 0.0]], dtype=np.float32) if nrm > 1e-6 \
                else np.array([[np.cos(ref_yaw[k]), np.sin(ref_yaw[k]), 0.0]], dtype=np.float32)
            face = np.array([[np.cos(ref_yaw[k]), np.sin(ref_yaw[k]), 0.0]], dtype=np.float32)
            spd = float(ref_speed[k])
            res = planner.run(
                context_mujoco_qpos=ctx,
                target_vel=np.array([spd], dtype=np.float32),
                mode=np.array([speed_to_mode(spd)], dtype=np.int64),
                movement_direction=move,
                facing_direction=face,
                height=np.array([ref_z[k]], dtype=np.float32),
                random_seed=seed,
                has_specific_target=1,
                specific_target_positions=wp,
                specific_target_headings=head,
            )
            cached = res.mujoco_qpos[0][: res.num_pred_frames]
            cache_i = 0
            n_replans += 1
        out_qpos[k] = cached[cache_i]
        cache_i += 1
        ctx[0, :-1] = ctx[0, 1:]                 # rolling closed-loop context
        ctx[0, -1] = out_qpos[k]

    # Back to the reference rate/length.
    pos = _resample(out_qpos[:, 0:3].astype(np.float64), F)
    yaw = _unwrap_resample_angle(_yaw_of(out_qpos[:, 3:7].astype(np.float64)), F)
    legs = _resample(out_qpos[:, 7:19].astype(np.float64), F)      # cols 0..11 == both legs
    quat = _yaw_to_quat(yaw)                                       # upright root (roll/pitch dropped)
    if verbose:
        err = np.linalg.norm(pos[:, :2] - base_pos[:, :2], axis=1)
        print(f"[planner-lower] {n_plan} frames @ {PLANNER_HZ:.0f} Hz, {n_replans} replans "
              f"(1 per {REPLAN_EVERY}); path error vs reference: med {np.median(err):.3f} m "
              f"max {err.max():.3f} m, end {err[-1]:.3f} m")
    return pos, quat, legs
