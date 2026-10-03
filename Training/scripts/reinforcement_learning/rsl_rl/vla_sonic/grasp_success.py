"""Grasp-success criterion for the pick task (user-specified, 2026-09-20).

Three rules, deliberately simple. Extra rules get added only when labelled motions reveal a hole.

  1. FALLEN (height)      reject if the object drops below half the table height at any point
  2. HELD  (palm box)     a rectangular box anchored at the right palm, extending along the palm
                          normal, just large enough to contain the mustard bottle while grasped.
                          The object must be inside it AT THE END of the trajectory, else it was
                          dropped.
  3. FALLEN (orientation) reject if the object is horizontal (90 +/- 10 deg from upright) at any
                          point -- it was knocked over.

  4. SLIPPING (rigidity) reject if the object moves vertically in the PALM frame over the last
                          second -- a solid grasp holds it rigid; "grasp + lean" is the object
                          sliding out of the hand down onto the table.

  success = (not fallen_height) and (not fallen_horizontal) and in_box_at_end and (not slipping)

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

# --- post-audit (af60v2) calibration ------------------------------------------------------------
# The box above was calibrated on the pre-audit demos. The af60v2 residual (additive_free, post-
# audit config 6996378) holds the bottle ~10 cm HIGHER in the palm: object z - palm z is -0.002
# against the old -0.045, so 97% of its frames sit above the old ceiling and the old box rejects
# 100% of the episodes. The palm itself is in the same place (palm pos in root frame differs by
# <5 cm, object absolute height by 8 mm), so this is a different GRASP, not a frame change.
#
# Same procedure as the original: p05/p95 of the object's palm-frame position over the hold phase
# (longest contiguous commanded-closed run, >=25 frames), pooled over episodes that pass rules
# 1/3/4. Source: 54 clean episodes, 16168 frames, 2026-09-29. Widths are essentially unchanged
# (x 0.066->0.063, z 0.066->0.058; y widens 0.037->0.057), i.e. an equally tight grasp, sited
# higher -- the shift is a translation, not a loosening.
#
# NOT yet validated against hand labels the way the original was (36/36, 0 FP / 0 FN). Spot-check
# before trusting it for anything comparative.
BOX_LO_AF60 = np.array([0.073, 0.041, 0.031])
BOX_HI_AF60 = np.array([0.136, 0.098, 0.089])

# --- af60v7 calibration (hands 15/3/3, 0.603 kg; 2026-10-03) --------------------------------------
# af60v7 seats the bottle lower in the palm than af60v2: its hold-phase z p05 is 0.006 against the
# AF60 floor of 0.031, so the AF60 box rejected 17 of 162 af60v7 demos on rule 2 ALONE (every other
# rule passed). Same procedure as above; source: 143 clean af60v7 episodes, 42456 frames. x is
# unchanged, y narrows (0.057 -> 0.031), z shifts down ~2.5 cm with a similar width (0.058 -> 0.072).
# Re-scoring the af60v7 GR00T evals with this box changes nothing (every failure also trips rule 1, 3
# or 4); it only affects which expert episodes enter training.
BOX_LO_AF60V7 = np.array([0.078, 0.054, 0.006])
BOX_HI_AF60V7 = np.array([0.137, 0.084, 0.078])

OBJ_REST_Z = 0.946                          # bottle centre at rest on the table (measured)
BOTTLE_HALF_H = 0.095                       # => table top at ~0.851 m
TABLE_TOP_Z = OBJ_REST_Z - BOTTLE_HALF_H
FALLEN_FRAC = 0.5                           # "below half the table height"

HORIZ_DEG = 90.0                            # object axis perpendicular to world up
HORIZ_TOL = 10.0                            # +/- tolerance -> knocked over

N_END = 1                                   # frames at the end used for the in-box test

# rule 4: a solid grasp holds the bottle RIGID in the palm frame. Over the last SLIP_WIN frames a
# good grasp moves <= 0.38 mm vertically in that frame; a "grasp + lean" (the object slipping out
# of the hand down onto the table) moves >= 4.13 mm. Calibrated on 24 human-labelled episodes
# (20 good / 4 lean); 1-3 mm all separate with zero errors. NOTE the RATE of slip does NOT work:
# two of the four leans slide 25 mm but slowly (0.002 m/s), inside the good range.
SLIP_WIN = 50                               # frames (1 s)
SLIP_MAX = 0.002                            # m of palm-frame vertical excursion allowed


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
          horiz_tol=HORIZ_TOL, n_end=N_END, slip_win=SLIP_WIN, slip_max=SLIP_MAX):
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
    rz = rel[-slip_win:, 2]                                                          # rule 4
    slip_spread = float(rz.max() - rz.min()) if len(rz) else 0.0
    slipping = bool(slip_spread > slip_max)

    return dict(
        rel=rel, in_box=in_box, tilt=tilt,
        fallen_height=fallen_height,
        fallen_horizontal=fallen_horizontal,
        in_box_at_end=in_box_at_end, slipping=slipping, slip_spread=slip_spread,
        min_z=float(obj_pos[:, 2].min()),
        max_tilt=float(tilt.max()),
        in_box_frames=int(in_box.sum()),
        last_in_box=int(np.max(np.where(in_box)[0])) if in_box.any() else -1,
        success=bool(not fallen_height and not fallen_horizontal and in_box_at_end
                     and not slipping),
    )
