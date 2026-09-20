"""Kinematic grasp-success criterion (replaces the [PHYS] held test).

WHY THE OLD TEST WAS REPLACED
  ``held = lifted >= 2 cm above rest AND within 0.15 m of the palm for >= 25 CONSECUTIVE steps``
  fails in both directions:
    * false positive  -- 0.15 m of the palm is proximity, not grasp: a bottle standing on the table
      beside the hand is inside that sphere. It never checks the object is still held at the END,
      so "lift then drop on the floor" scores as success (14/270 of the training set did).
    * false positive  -- ``obj_rest_z`` was a running MINIMUM over the first 50 steps, so an object
      knocked lower early (worst case: off the table) makes every later frame read as "lifted".
    * false negative  -- the run counter resets on ONE bad frame, so a real grasp hovering near the
      threshold never accumulates 25 consecutive steps; a grasp in the last <25 steps cannot register.

WHAT REPLACES IT
  A supported object is static in the WORLD frame; a held object is static in the PALM frame.
    rel(t)    = R_palm^T (p_obj - p_palm)           object expressed in the palm frame
    in_volume = rel(t) inside the palm-frame grasp box (from the demos' hold phase)
    stable    = || rel(t) - median(rel over W) || < eps
    comove    = || v_obj - v_palm || < tau, asserted ONLY while || v_palm || > v_min
    held(t)   = in_volume and stable and comove, LATCHED through near-static stretches
  Deliberately NOT used:
    * height above the table -- the arm can lift the object clear of the table and then carry it
      BELOW the table top (past the edge); a height test would reject that legitimate grasp. Height
      is used only for the unambiguous floor reject.
    * object tilt -- a bottle carried at 70 deg is still a successful pick. A bottle knocked over on
      the table is excluded by rel/comove instead, so no tilt threshold is needed.
    * contact forces -- the env's ContactSensor covers ``Robot/.*`` with no filter, so it cannot
      separate finger<->object from finger<->table, and the env's own comments call it unreliable
      ("robust to the dead force sensor").

Thresholds are CALIBRATION TARGETS, not settled values: tune them against human-labelled episodes
before this gates any dataset or eval.
"""

from __future__ import annotations

import numpy as np

DT = 0.02                                   # 50 Hz
# palm-frame grasp box, from the demos' hold-phase p05/p95 (see GraspGate), plus margin
BOX_LO = np.array([0.082, 0.062, -0.071]) - 0.045
BOX_HI = np.array([0.151, 0.099, -0.004]) + 0.045
EPS_STABLE = 0.030                          # m, palm-frame wander allowed
TAU_COMOVE = 0.150                          # m/s, |v_obj - v_palm| while carried
V_MIN = 0.050                               # m/s, palm speed above which comove is informative
WIN = 25                                    # frames (0.5 s) for the median/stability window
T_HELD = 25                                 # frames of held required somewhere
N_END = 25                                  # frames at the end that must be held
FLOOR_DROP = 0.20                           # m below rest => on the ground


def quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def palm_frame(root_pos: np.ndarray, root_quat: np.ndarray, wrist: np.ndarray):
    """World-frame palm position + rotation, from root pose composed with teleop/right_wrist (4x4)."""
    T = len(root_pos)
    p = np.zeros((T, 3)); R = np.zeros((T, 3, 3))
    for t in range(T):
        Rr = quat_to_R(root_quat[t])
        R[t] = Rr @ wrist[t][:3, :3]
        p[t] = root_pos[t] + Rr @ wrist[t][:3, 3]
    return p, R


def score(obj_pos, root_pos, root_quat, wrist, *, z_rest=None,
          box_lo=BOX_LO, box_hi=BOX_HI, eps=EPS_STABLE, tau=TAU_COMOVE,
          v_min=V_MIN, win=WIN, t_held=T_HELD, n_end=N_END):
    """Per-step held mask + episode verdict. All inputs are (T,·) arrays in world frame."""
    p_palm, R_palm = palm_frame(root_pos, root_quat, wrist)
    T = len(obj_pos)
    rel = np.zeros((T, 3))
    for t in range(T):
        rel[t] = R_palm[t].T @ (obj_pos[t] - p_palm[t])

    # z_rest from a settled early window (NOT a running min, which the old test used)
    if z_rest is None:
        z_rest = float(np.median(obj_pos[: min(20, T), 2]))

    v_obj = np.gradient(obj_pos, DT, axis=0)
    v_palm = np.gradient(p_palm, DT, axis=0)
    speed_palm = np.linalg.norm(v_palm, axis=1)
    comove_err = np.linalg.norm(v_obj - v_palm, axis=1)

    in_volume = np.all((rel >= box_lo) & (rel <= box_hi), axis=1)
    stable = np.zeros(T, bool)
    for t in range(T):
        a, b = max(0, t - win + 1), t + 1
        stable[t] = np.linalg.norm(rel[t] - np.median(rel[a:b], axis=0)) < eps

    held = np.zeros(T, bool)
    last = False
    for t in range(T):
        if speed_palm[t] > v_min:                      # palm moving -> comove is informative
            last = bool(in_volume[t] and stable[t] and comove_err[t] < tau)
        else:                                          # near-static -> latch previous verdict,
            last = bool(last and in_volume[t])         # but the object must still be in the hand
        held[t] = last

    on_floor = bool((obj_pos[:, 2] < z_rest - FLOOR_DROP).any())
    held_frames = int(held.sum())
    held_at_end = bool(held[-n_end:].all()) if T >= n_end else bool(held[-1])
    last_held = int(np.max(np.where(held)[0])) if held.any() else -1

    return dict(
        held=held, rel=rel, z_rest=z_rest,
        on_floor=on_floor, held_frames=held_frames, held_at_end=held_at_end,
        last_held=last_held,
        success=bool(not on_floor and held_frames >= t_held and held_at_end),
        # "lifted then lost it" -- kept separately so it is never silently counted as success
        dropped=bool(held_frames >= t_held and not held_at_end),
        truncate_at=last_held,                         # chop-before-the-drop point
    )
