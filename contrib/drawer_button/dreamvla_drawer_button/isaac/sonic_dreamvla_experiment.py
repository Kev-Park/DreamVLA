"""Separate task for two DreamVLA implementation choices, not a full repo port.

Existing tasks, action terms and checkpoints remain unchanged. Use the dedicated
experiment entrypoint so training AND evaluation register the same actor/task.
"""

import gymnasium as gym
import torch

from isaaclab.utils import configclass
from .agents import G1SonicMultiplicativeFreePPORunnerCfg
from .motion_tracking_open_drawer_sonic_env import (
    G1OpenDrawerSonicNvidiaMultiplicativeFreeEnvCfg,
)
from .sonic_action import SonicLatentAction, SonicLatentActionCfg
from .sonic_dreamvla_contract import POSTPROCESS, compose_dreamvla_token

TASK = "DreamVLA-DrawerButton-Open-Drawer-SONIC-NVIDIA-Mul-Free-FSQ-v0"
EXPERIMENT = "g1_sonic_open_drawer_nvidia_mul_free_010_fsq_zero"


class DreamVlaSonicLatentAction(SonicLatentAction):
    def process_actions(self, actions: torch.Tensor) -> None:
        # Same execution sequence, ORT stream safety, proprioceptive history and
        # target scaling as SonicLatentAction. Only token composition differs.
        self._raw_actions[:] = actions
        reference = self._encode_reference_motion()
        self._full_latent[:] = compose_dreamvla_token(
            reference, actions, self.cfg.residual_scale
        )
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
class DreamVlaSonicLatentActionCfg(SonicLatentActionCfg):
    class_type: type = DreamVlaSonicLatentAction
    residual_transform: str = "multiplicative_free"
    residual_scale: float = 0.1
    latent_clip: tuple[float, float] = (-1.0, 15.0 / 16.0)
    token_postprocess: str = POSTPROCESS
    fsq_levels: int = 32


@configclass
class G1DreamVlaStyleEnvCfg(G1OpenDrawerSonicNvidiaMultiplicativeFreeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        previous = self.actions.joint_pos
        self.actions.joint_pos = DreamVlaSonicLatentActionCfg(
            asset_name=previous.asset_name,
            decoder_path=previous.decoder_path,
            encoder_path=previous.encoder_path,
            joint_names=list(previous.joint_names),
        )


@configclass
class G1DreamVlaStylePPORunnerCfg(G1SonicMultiplicativeFreePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = EXPERIMENT
        self.policy.class_name = "DreamVlaResidualActorCritic"
        self.resume = False


def register_task():
    if TASK not in gym.registry:
        gym.register(
            TASK,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}:G1DreamVlaStyleEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}:G1DreamVlaStylePPORunnerCfg",
            },
        )
