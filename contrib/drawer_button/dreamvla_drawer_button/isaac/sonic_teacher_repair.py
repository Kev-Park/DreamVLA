"""Opt-in PPO diagnostics; production tasks and historical checkpoints untouched."""

import gymnasium as gym
import torch
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from .sonic_action import SonicLatentAction, SonicLatentActionCfg
from .sonic_teacher_repair_contract import RepairSpec, compose
from .motion_tracking_open_drawer_sonic_env import G1OpenDrawerSonicNvidiaEnvCfg
from .sonic_dreamvla_experiment import (
    G1DreamVlaStylePPORunnerCfg as G1DreamVlaStylePPORunnerCfg,
)

TASK = "DreamVLA-DrawerButton-Open-Drawer-SONIC-Teacher-Repair-v0"
EXPERIMENT = "g1_sonic_open_drawer_teacher_repair"


class TeacherRepairAction(SonicLatentAction):
    def process_actions(self, actions):
        self._raw_actions[:] = actions
        reference = self._encode_reference_motion()
        spec = RepairSpec(
            mode=self.cfg.residual_transform,
            scale=self.cfg.residual_scale,
            snap=self.cfg.token_postprocess == "fsq32_lattice_round",
        )
        self._full_latent[:] = compose(reference, actions, spec)
        self._append_history()
        self._assemble_decoder_input()
        if self._use_cuda_ort:
            self._ort_execution.run(self._decoder_session, self._io_binding)
        else:
            decoded = self._decoder_session.run(
                [self._decoder_output_name],
                {self._decoder_input_name: self._decoder_input.cpu().numpy()},
            )[0]
            self._decoded_actions[:] = torch.from_numpy(decoded).to(self.device)
        default_pos = self._asset.data.default_joint_pos[:, self._joint_ids]
        self._joint_targets[:] = (
            default_pos + self._decoded_actions * self._action_scale
        )


@configclass
class TeacherRepairActionCfg(SonicLatentActionCfg):
    class_type: type = TeacherRepairAction
    token_postprocess: str = "fsq32_lattice_round"


@configclass
class TeacherRepairEnvCfg(G1OpenDrawerSonicNvidiaEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        previous = self.actions.joint_pos
        self.actions.joint_pos = TeacherRepairActionCfg(
            asset_name=previous.asset_name,
            decoder_path=previous.decoder_path,
            encoder_path=previous.encoder_path,
            joint_names=list(previous.joint_names),
        )


def configure(cfg, spec):
    body = cfg.actions.joint_pos
    body.residual_transform = spec.mode
    body.residual_scale = spec.scale
    body.latent_clip = (-1.0, spec.upper)
    body.token_postprocess = "fsq32_lattice_round" if spec.snap else "none"
    cfg.rewards.action_rate_l2.weight = -0.005 * spec.action_rate_factor


class ReferenceTokenWrapper(RslRlVecEnvWrapper):
    """Append a freshly encoded CURRENT-state token, never previous-step cache."""

    def __init__(self, env, spec, **kwargs):
        self.repair_spec = spec
        super().__init__(env, **kwargs)
        if spec.base_observation:
            self.num_obs += 64

    def _append(self, obs, extras):
        if not self.repair_spec.base_observation:
            return obs, extras
        body = self.unwrapped.action_manager.get_term("joint_pos")
        reference = body._encode_reference_motion().clone()
        body._last_policy_reference = reference
        obs = torch.cat((obs, reference), dim=-1)
        extras = dict(extras)
        extras["observations"] = dict(extras["observations"], policy=obs)
        return obs, extras

    def get_observations(self):
        return self._append(*super().get_observations())

    def reset(self):
        return self._append(*super().reset())

    def step(self, actions):
        obs, reward, done, extras = super().step(actions)
        obs, extras = self._append(obs, extras)
        return obs, reward, done, extras


def register_task():
    if TASK not in gym.registry:
        gym.register(
            TASK,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"{__name__}:TeacherRepairEnvCfg"},
        )
