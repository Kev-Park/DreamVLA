"""Grasp-success criterion for the pick task (user-specified, 2026-09-20).

Three rules, deliberately simple. Extra rules get added only when labelled motions reveal a hole.

  1. FALLEN (height)      reject if the object drops below half the table height at any point
  2. HELD  (palm box)     a rectangular box anchored at the right palm, extending along the palm
                          normal, just large enough to contain the mustard bottle while grasped.
                          The object must be inside it AT THE END of the trajectory, else it was
                          dropped.
  3. FALLEN (orientation) reject if the object is horizontal (90 +/- 10 deg from upright) at any
                          point -- it was knocked over.

  success = (not fallen_height) and (not fallen_horizontal) and in_box_at_end

The box is CALIBRATED, not guessed: p05/p95 of the object's palm-frame position over the demos'
hold phase (the longest contiguous run of frames the demo commanded the hand closed) -- 15175
frames over 54 demos. min/max and p01/p99 are unusable: they include frames where the hand still
commands closed after the bottle is already gone (z spread of 1.17 m).

NOTE the box tracks the object's ROOT (centre). Tipping the bottle swings its centre several cm,
so a heavily tilted grasp can fall outside a box this tight. Rule 3 rejects near-horizontal cases
outright, but tilts between roughly 30 and 80 degrees are neither rejected by rule 3 nor reliably
inside the box. Worth watching for while labelling.
"""

from __future__ import annotations

import numpy as np

# --- calibration constants ---------------------------------------------------------------------
BOX_LO = np.array([0.082, 0.062, -0.071])   # p05 of the demos' hold phase (palm frame, m)
BOX_HI = np.array([0.148, 0.099, -0.005])   # p95
BOX_MARGIN = 0.020                          # m, added to every face

OBJ_REST_Z = 0.946                          # bottle centre at rest on the table (measured)
BOTTLE_HALF_H = 0.095                       # => table top at ~0.851 m
TABLE_TOP_Z = OBJ_REST_Z - BOTTLE_HALF_H
FALLEN_FRAC = 0.5                           # "below half the table height"

HORIZ_DEG = 90.0                            # object axis perpendicular to world up
HORIZ_TOL = 10.0                            # +/- tolerance -> knocked over

N_END = 1                                   # frames at the end used for the in-box test


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def palm_frame(root_pos, root_quat, wrist):
    """World palm position + rotation, from the root pose composed with teleop/right_wrist (4x4)."""
    T = len(root_pos)
    p = np.zeros((T, 3)); R = np.zeros((T, 3, 3))
    for t in range(T):
        Rr = quat_to_R(root_quat[t])
        R[t] = Rr @ wrist[t][:3, :3]
        p[t] = root_pos[t] + Rr @ wrist[t][:3, 3]
    return p, R


def object_tilt_deg(obj_quat):
    """Angle between the object's body z-axis and world up, per frame."""
    q = np.asarray(obj_quat)
    zz = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    return np.degrees(np.arccos(np.clip(zz, -1.0, 1.0)))


def score(obj_pos, obj_quat, root_pos, root_quat, wrist, *,
          box_lo=BOX_LO, box_hi=BOX_HI, margin=BOX_MARGIN,
          table_top=TABLE_TOP_Z, fallen_frac=FALLEN_FRAC,
          horiz_tol=HORIZ_TOL, n_end=N_END):
    """Per-rule verdicts plus the overall success flag."""
    obj_pos = np.asarray(obj_pos, float)
    p_palm, R_palm = palm_frame(root_pos, root_quat, wrist)
    T = len(obj_pos)

    rel = np.empty((T, 3))
    for t in range(T):
        rel[t] = R_palm[t].T @ (obj_pos[t] - p_palm[t])

    lo, hi = box_lo - margin, box_hi + margin
    in_box = np.all((rel >= lo) & (rel <= hi), axis=1)

    fallen_height = bool((obj_pos[:, 2] < fallen_frac * table_top).any())            # rule 1
    tilt = object_tilt_deg(obj_quat)
    fallen_horizontal = bool((np.abs(tilt - HORIZ_DEG) <= horiz_tol).any())          # rule 3
    in_box_at_end = bool(in_box[-n_end:].all()) if T >= n_end else bool(in_box[-1])   # rule 2

    return dict(
        rel=rel, in_box=in_box, tilt=tilt,
        fallen_height=fallen_height,
        fallen_horizontal=fallen_horizontal,
        in_box_at_end=in_box_at_end,
        min_z=float(obj_pos[:, 2].min()),
        max_tilt=float(tilt.max()),
        in_box_frames=int(in_box.sum()),
        last_in_box=int(np.max(np.where(in_box)[0])) if in_box.any() else -1,
        success=bool(not fallen_height and not fallen_horizontal and in_box_at_end),
    )
