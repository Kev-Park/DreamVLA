from isaaclab_tasks.utils.motion_lib.motion_lib_base import MotionLibBase
class MotionLibRobot(MotionLibBase):
    def __init__(self, num_envs, device, motion_file):
        super().__init__(motion_file = motion_file, num_envs = num_envs, device = device)
        return