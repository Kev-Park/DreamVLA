"""Frame-weighted body MPJPE over completed episodes, sampled before each action.

World error uses environment-local world coordinates and the paired reference.
Root-relative error removes each pose's pelvis translation, without rotating it.
Initial states are included; post-action terminal states are unavailable after
autoreset and excluded. Unfinished episodes are excluded from the final report.
"""
import torch


class EpisodeMPJPE:
    def __init__(self, env):
        from isaaclab_tasks.manager_based.motion_tracking.g1.motion_tracking_env import (
            HOI_BODY_NAMES, HOI_BODY_KEYPT_IDXS)
        self.env = env
        self.robot = env.scene["robot"]
        self.names = list(HOI_BODY_NAMES)
        self.keypoints = list(HOI_BODY_KEYPT_IDXS)
        assert self.names[0] == "pelvis"
        self.body_ids = [self.robot.find_bodies(name)[0][0] for name in self.names]
        self.sums = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float64)
        self.counts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.total = torch.zeros(2, device=env.device, dtype=torch.float64)
        self.frames = 0
        self.episodes = 0

    def sample(self):
        env = self.env
        times = env.episode_length_buf * env.step_dt + env.start_motion_times.to(env.device)
        ref = env.motion_lib.get_motion_state(env.motion_ids, times)["global_keypts"][:, self.keypoints]
        sim = self.robot.data.body_pos_w[:, self.body_ids] - env.scene.env_origins[:, None, :]
        world = torch.linalg.vector_norm(sim-ref, dim=-1).mean(dim=-1)
        relative = torch.linalg.vector_norm(
            (sim-sim[:, :1])-(ref-ref[:, :1]), dim=-1).mean(dim=-1)
        self.sums += torch.stack((world, relative), dim=-1)
        self.counts += 1

    def finish(self, ids):
        self.total += self.sums[ids].sum(dim=0)
        self.frames += int(self.counts[ids].sum().item())
        self.episodes += len(ids)
        self.sums[ids] = 0
        self.counts[ids] = 0

    def result(self):
        assert self.frames > 0 and bool(torch.isfinite(self.total).all())
        values = (100*self.total/self.frames).tolist()
        return {"world_cm": values[0], "root_relative_cm": values[1],
                "frames": self.frames, "completed_episodes": self.episodes,
                "body_count": len(self.names), "body_names": self.names,
                "aggregation": "frame_weighted_completed_episodes_pre_action",
                "root_alignment": "pelvis_translation_only"}
