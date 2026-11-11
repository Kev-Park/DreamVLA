"""Reference-scheduled right-hand command for the frozen SONIC baseline."""
import torch


def reference_finger_command(env):
    # Sample before the action, using the same reference clock as the encoder.
    times = env.episode_length_buf * env.step_dt + env.start_motion_times.to(env.device)
    closed = env.motion_lib.get_motion_state(env.motion_ids, times)["is_closed"].bool()
    # Isaac BinaryJointPositionAction: negative = closed, nonnegative = open.
    return 1.0 - 2.0 * closed.to(torch.float32)
