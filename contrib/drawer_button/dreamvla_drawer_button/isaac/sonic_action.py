"""SONIC latent action term for the DreamControl Open Drawer environment.

The PPO policy emits a 64-D residual around the encoded reference motion
(or the released safe-standing token when no reference encoder is used).
The released ONNX decoder converts the resulting full token and ten
frames of proprioceptive history into normalized targets for the 29 body
joints.  Dexterous hand primitives remain separate Isaac Lab action terms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from .sonic_ort import SonicCudaExecution
from .sonic_latent import action_contract, compose_latent


SONIC_BODY_JOINTS = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# This token is checkpoint-specific and comes from the public SONIC release.
SONIC_STANDING_TOKEN = [
    -0.0625,
    0.0000,
    -0.0625,
    -0.1250,
    -0.1875,
    -0.0625,
    0.1875,
    0.2500,
    0.1875,
    -0.1250,
    0.0625,
    -0.0625,
    -0.2500,
    -0.2500,
    -0.3125,
    -0.0625,
    0.0000,
    -0.0625,
    -0.1250,
    -0.1875,
    0.0000,
    -0.2500,
    0.0000,
    -0.2500,
    -0.0625,
    0.0625,
    0.1250,
    -0.1250,
    0.2500,
    0.1875,
    0.2500,
    -0.1250,
    0.1250,
    0.1875,
    -0.0625,
    0.0000,
    -0.1875,
    -0.1875,
    0.2500,
    0.0000,
    0.0000,
    -0.1250,
    0.0625,
    0.0000,
    -0.0625,
    -0.0625,
    0.1875,
    -0.0625,
    0.0000,
    0.0625,
    0.1250,
    0.0625,
    0.1250,
    0.0625,
    0.1250,
    0.0000,
    0.1250,
    0.1875,
    0.0000,
    0.0000,
    0.0625,
    0.0625,
    0.1875,
    0.0625,
]


def _sonic_action_scale(joint_name: str) -> float:
    """Return the public g1_model_12 normalized-action scale for one joint."""
    if "wrist_pitch" in joint_name or "wrist_yaw" in joint_name:
        return 0.07450087032524835
    if "hip_yaw" in joint_name or joint_name == "waist_yaw_joint":
        return 0.5475464651829304
    if any(key in joint_name for key in ("hip_pitch", "hip_roll", "knee")):
        return 0.35066146635877904
    return 0.43857731389829635


class SonicLatentAction(ActionTerm):
    """Decode a 64-D SONIC token residual into 29 body joint targets."""

    cfg: SonicLatentActionCfg
    _asset: Articulation

    def __init__(self, cfg: SonicLatentActionCfg, env) -> None:
        action_contract(cfg)  # Fail before allocating sessions for invalid settings.
        super().__init__(cfg, env)

        self._joint_ids, resolved_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=True
        )
        if list(resolved_names) != list(cfg.joint_names):
            raise RuntimeError(
                "SONIC joint contract mismatch. "
                f"Expected {cfg.joint_names}, resolved {resolved_names}."
            )

        self._raw_actions = torch.zeros(self.num_envs, 64, device=self.device)
        self._full_latent = torch.zeros_like(self._raw_actions)
        self._decoded_actions = torch.zeros(self.num_envs, 29, device=self.device)
        self._joint_targets = torch.zeros_like(self._decoded_actions)
        self._standing_token = torch.tensor(
            SONIC_STANDING_TOKEN, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        self._action_scale = torch.tensor(
            [_sonic_action_scale(name) for name in cfg.joint_names],
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        # Histories are flattened oldest-to-newest, matching the released C++
        # deploy runtime and the policy's 994-D decoder observation contract.
        self._base_ang_vel_history = torch.zeros(
            self.num_envs, 10, 3, device=self.device
        )
        self._joint_pos_history = torch.zeros(self.num_envs, 10, 29, device=self.device)
        self._joint_vel_history = torch.zeros(self.num_envs, 10, 29, device=self.device)
        self._last_action_history = torch.zeros(
            self.num_envs, 10, 29, device=self.device
        )
        self._gravity_history = torch.zeros(self.num_envs, 10, 3, device=self.device)
        self._decoder_input = torch.zeros(self.num_envs, 994, device=self.device)

        self._encoder_input = torch.zeros(self.num_envs, 1762, device=self.device)
        self._encoded_reference = torch.zeros(self.num_envs, 64, device=self.device)
        self._reference_offsets = (
            torch.arange(10, device=self.device, dtype=torch.float32)
            * 5.0
            * env.step_dt
        )
        legacy_joint_names = env.cfg.observations.policy.joint_pos.params[
            "asset_cfg"
        ].joint_names
        self._legacy_to_sonic = [
            (legacy_index, cfg.joint_names.index(name))
            for legacy_index, name in enumerate(legacy_joint_names)
        ]

        self._create_decoder_session(Path(cfg.decoder_path))
        if cfg.encoder_path:
            self._create_encoder_session(Path(cfg.encoder_path))
        else:
            self._encoder_session = None

    def _create_decoder_session(self, decoder_path: Path) -> None:
        if not decoder_path.is_file():
            raise FileNotFoundError(f"SONIC decoder not found: {decoder_path}")

        import onnxruntime as ort

        ort.preload_dlls()
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        torch_device = torch.device(self.device)
        self._use_cuda_ort = torch_device.type == "cuda"
        if self._use_cuda_ort:
            device_id = torch_device.index
            if device_id is None:
                device_id = torch.cuda.current_device()
            self._ort_execution = SonicCudaExecution(device_id)
            providers = self._ort_execution.providers()
        else:
            device_id = 0
            providers = ["CPUExecutionProvider"]

        self._decoder_session = ort.InferenceSession(
            str(decoder_path), sess_options=session_options, providers=providers
        )
        active_provider = self._decoder_session.get_providers()[0]
        expected_provider = (
            "CUDAExecutionProvider" if self._use_cuda_ort else "CPUExecutionProvider"
        )
        if active_provider != expected_provider:
            raise RuntimeError(
                f"SONIC decoder requested {expected_provider}, but ONNX Runtime selected {active_provider}."
            )
        self._decoder_input_name = self._decoder_session.get_inputs()[0].name
        self._decoder_output_name = self._decoder_session.get_outputs()[0].name
        self._ort_device_id = device_id

        if self._use_cuda_ort:
            self._ort_execution.validate(self._decoder_session)
            self._decoder_input.record_stream(self._ort_execution.stream)
            self._decoded_actions.record_stream(self._ort_execution.stream)
            self._io_binding = self._decoder_session.io_binding()
            self._io_binding.bind_input(
                self._decoder_input_name,
                "cuda",
                device_id,
                np.float32,
                tuple(self._decoder_input.shape),
                self._decoder_input.data_ptr(),
            )
            self._io_binding.bind_output(
                self._decoder_output_name,
                "cuda",
                device_id,
                np.float32,
                tuple(self._decoded_actions.shape),
                self._decoded_actions.data_ptr(),
            )

    def _create_encoder_session(self, encoder_path: Path) -> None:
        if not encoder_path.is_file():
            raise FileNotFoundError(f"SONIC encoder not found: {encoder_path}")

        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        if self._use_cuda_ort:
            providers = self._ort_execution.providers()
        else:
            providers = ["CPUExecutionProvider"]
        self._encoder_session = ort.InferenceSession(
            str(encoder_path), sess_options=session_options, providers=providers
        )
        self._encoder_input_name = self._encoder_session.get_inputs()[0].name
        self._encoder_output_name = self._encoder_session.get_outputs()[0].name
        if self._use_cuda_ort:
            self._ort_execution.validate(self._encoder_session)
            self._encoder_input.record_stream(self._ort_execution.stream)
            self._encoded_reference.record_stream(self._ort_execution.stream)
            self._encoder_io_binding = self._encoder_session.io_binding()
            self._encoder_io_binding.bind_input(
                self._encoder_input_name,
                "cuda",
                self._ort_device_id,
                np.float32,
                tuple(self._encoder_input.shape),
                self._encoder_input.data_ptr(),
            )
            self._encoder_io_binding.bind_output(
                self._encoder_output_name,
                "cuda",
                self._ort_device_id,
                np.float32,
                tuple(self._encoded_reference.shape),
                self._encoded_reference.data_ptr(),
            )

    @property
    def action_dim(self) -> int:
        return 64

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        # The full token, rather than the residual, is the supervision target
        # that should be saved in the eventual GR00T dataset.
        return self._full_latent

    @property
    def full_latent(self) -> torch.Tensor:
        return self._full_latent

    @property
    def decoded_actions(self) -> torch.Tensor:
        return self._decoded_actions

    @property
    def joint_targets(self) -> torch.Tensor:
        """29-D decoded position targets in :data:`SONIC_BODY_JOINTS` order."""
        return self._joint_targets

    def _append_history(self) -> None:
        histories_and_values = (
            (self._base_ang_vel_history, self._asset.data.root_ang_vel_b),
            (
                self._joint_pos_history,
                self._asset.data.joint_pos[:, self._joint_ids]
                - self._asset.data.default_joint_pos[:, self._joint_ids],
            ),
            (
                self._joint_vel_history,
                self._asset.data.joint_vel[:, self._joint_ids]
                - self._asset.data.default_joint_vel[:, self._joint_ids],
            ),
            (self._last_action_history, self._decoded_actions),
            (self._gravity_history, self._asset.data.projected_gravity_b),
        )
        for history, value in histories_and_values:
            history[:, :-1] = history[:, 1:].clone()
            history[:, -1] = value

    def _assemble_decoder_input(self) -> None:
        offset = 0
        values = (
            self._full_latent,
            self._base_ang_vel_history.flatten(1),
            self._joint_pos_history.flatten(1),
            self._joint_vel_history.flatten(1),
            self._last_action_history.flatten(1),
            self._gravity_history.flatten(1),
        )
        for value in values:
            width = value.shape[1]
            self._decoder_input[:, offset : offset + width] = value
            offset += width
        if offset != 994:
            raise RuntimeError(f"Invalid SONIC decoder input width: {offset}")

    def _encode_reference_motion(self) -> torch.Tensor:
        """Encode the same reference motion window used by DreamControl rewards."""
        if self._encoder_session is None or not hasattr(self._env, "motion_lib"):
            return self._standing_token

        current_time = self._env.episode_length_buf * self._env.step_dt
        current_time = current_time + self._env.start_motion_times
        future_times = current_time[:, None] + self._reference_offsets[None, :]
        future_ids = self._env.motion_ids[:, None].expand(-1, 10)
        motion = self._env.motion_lib.get_motion_state(
            future_ids.reshape(-1), future_times.reshape(-1)
        )
        next_motion = self._env.motion_lib.get_motion_state(
            future_ids.reshape(-1), (future_times + self._env.step_dt).reshape(-1)
        )

        legacy_pos = motion["dof_pos"].reshape(self.num_envs, 10, -1)
        legacy_next_pos = next_motion["dof_pos"].reshape(self.num_envs, 10, -1)
        legacy_vel = (legacy_next_pos - legacy_pos) / self._env.step_dt
        sonic_pos = torch.zeros(self.num_envs, 10, 29, device=self.device)
        sonic_vel = torch.zeros_like(sonic_pos)
        for legacy_index, sonic_index in self._legacy_to_sonic:
            sonic_pos[:, :, sonic_index] = legacy_pos[:, :, legacy_index]
            sonic_vel[:, :, sonic_index] = legacy_vel[:, :, legacy_index]

        reference_quat = motion["root_rot"].reshape(self.num_envs, 10, 4)
        robot_quat = self._asset.data.root_quat_w[:, None, :].expand(-1, 10, -1)
        relative_quat = math_utils.quat_mul(
            math_utils.quat_inv(robot_quat), reference_quat
        )
        orientation_6d = math_utils.matrix_from_quat(relative_quat)[..., :, :2].reshape(
            self.num_envs, 10, 6
        )

        # Encoder mode 0 (G1) is represented by four zeros. Only its required
        # observations are populated; the other supported modes remain zero.
        self._encoder_input.zero_()
        self._encoder_input[:, 4:294] = sonic_pos.flatten(1)
        self._encoder_input[:, 294:584] = sonic_vel.flatten(1)
        self._encoder_input[:, 601:661] = orientation_6d.flatten(1)

        if self._use_cuda_ort:
            self._ort_execution.run(self._encoder_session, self._encoder_io_binding)
        else:
            encoded = self._encoder_session.run(
                [self._encoder_output_name],
                {self._encoder_input_name: self._encoder_input.cpu().numpy()},
            )[0]
            self._encoded_reference[:] = torch.from_numpy(encoded).to(self.device)
        return self._encoded_reference

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        reference_latent = self._encode_reference_motion()
        self._full_latent[:] = compose_latent(reference_latent, actions, self.cfg)
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

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(
            self._joint_targets, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._full_latent[env_ids] = self._standing_token
        self._decoded_actions[env_ids] = 0.0
        self._joint_targets[env_ids] = 0.0
        self._base_ang_vel_history[env_ids] = 0.0
        self._joint_pos_history[env_ids] = 0.0
        self._joint_vel_history[env_ids] = 0.0
        self._last_action_history[env_ids] = 0.0
        self._gravity_history[env_ids] = 0.0


@configclass
class SonicLatentActionCfg(ActionTermCfg):
    """Configuration for :class:`SonicLatentAction`."""

    class_type: type[ActionTerm] = SonicLatentAction
    decoder_path: str = MISSING
    encoder_path: str | None = None
    joint_names: list[str] = MISSING
    # Keep existing checkpoints/tasks additive. New experiments opt in explicitly.
    residual_transform: str = "additive"
    residual_scale: float = 0.25
    latent_clip: tuple[float, float] = (-1.0, 1.0)
