"""Frame bounds for the original and fixed pre-grasp rescue schedules."""
import math


def takeover_bounds(grab_idx, *, k_frac=1.0, window_s=None, step_dt=0.02):
    """Return the inclusive lower and exclusive upper integer takeover bounds."""
    if grab_idx <= 0:
        raise ValueError("A reference grasp after frame zero is required")
    if window_s is None:
        return 0, max(1, int(round(float(k_frac) * grab_idx)))
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError("The pre-grasp window must be finite and positive")
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("The simulation step duration must be finite and positive")
    if k_frac != 1.0:
        raise ValueError("A fixed pre-grasp window cannot be combined with k_frac != 1")
    frames = max(1, int(round(window_s / step_dt)))
    return max(0, grab_idx - frames), grab_idx
