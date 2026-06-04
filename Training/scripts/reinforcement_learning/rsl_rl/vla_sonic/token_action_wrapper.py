"""RL-train a custom encoder against the FROZEN SONIC decoder (ONNX-backed).

The SONIC release ships the decoder as ``model_decoder.onnx`` (the exact decoder
``eval_parquet_sonic.py`` already drives: a single ``obs_dict`` input of 994 dims →
``action`` of 29 dims). Here we freeze that decoder and let an RL (PPO) policy play the
role of the SONIC *encoder*: the rsl_rl actor maps the env observation to a 65-D latent —
64 dims for the SONIC body token, 1 dim for the right-hand open/close scalar — and this
wrapper feeds the body token (FSQ-quantized) and batched proprioception history to the
frozen decoder, then assembles the full 41-D env action.

Why ONNX (not the PyTorch ``last.pt``): instantiating the PyTorch ``UniversalTokenModule``
requires observation dims that only exist after building a live SONIC env (the heavy-dep
stack kept in a separate repo). The ONNX decoder carries identical frozen weights and runs
standalone. We convert it to a batched GPU torch module via ``onnx2torch`` (forward
inference only — PPO is model-free, so no gradient flows through the decoder).

Token quantization (FSQ): the decoder was trained against FSQ-quantized tokens. From
``gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml``:
    num_fsq_levels: 32, fsq_level_list: 32, max_num_tokens: 2 → levels = [32]*32
We reimplement FSQ exactly (≈ vector_quantize_pytorch.FSQ) so the policy's continuous
latent is mapped onto the same discrete grid the decoder was trained on. No
vector_quantize_pytorch dependency needed.

Fingers: SONIC has NO finger decoder in the release (only g1_kin + g1_dyn body decoders).
In deploy, fingers come from outside SONIC (VLA/teleop). For RL we mirror train.py's
BinaryJointPositionActionCfg: a single right-hand open/close scalar from the policy,
mapped via ``0.5*(1+tanh(scalar))`` to a blend factor that interpolates between fixed
canonical open and closed right-hand poses (values copied from the original binary cfg).
Left hand stays at its env-default pose (not used for the pick task). Existing rewards
(right_hand_state_target with interpolated target, object_above_the_ground) train it.

Decoder obs layout (994 = 64 + 930), matching ``vla_sonic.planner_to_utm.build_decoder_obs``:
    token_state(64),
    his_base_angular_velocity(30), his_body_joint_positions(290),
    his_body_joint_velocities(290), his_last_actions(290), his_gravity_dir(30).
The 29-D body command → env action transform mirrors
``vla_sonic.action_assembler.utm_body_29_to_env_27`` (batched here).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from .action_assembler import (
    G1_ACTION_SCALE_SONIC,
    G1_DEFAULT_ANGLES_SONIC,
    PERM_SONIC27_TO_ENV27,
    WAIST_PITCH_SONIC_IDX,
    WAIST_ROLL_SONIC_IDX,
)

# Left/right hand action-term joint order — must match ContinuousFingersActionsCfg.
_LEFT_FINGER_JOINTS = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
]
_RIGHT_FINGER_JOINTS = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]

# SONIC-IsaacLab 29-joint order (no fingers). Mirrors eval_parquet_sonic.UTM_29_JOINT_NAMES.
UTM_29_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
assert len(UTM_29_JOINT_NAMES) == 29

N_HISTORY_FRAMES = 10
N_BODY_JOINTS = 29
TOKEN_TOTAL_DIM = 64          # 2 tokens x 32 FSQ channels
PROPRIO_DIM = 930            # 10 frames x (3 + 29 + 29 + 29 + 3)
DECODER_OBS_DIM = TOKEN_TOTAL_DIM + PROPRIO_DIM  # 994
# FSQ config from gear_sonic/config/actor_critic/universal_token/all_mlp_v1.yaml.
FSQ_NUM_LEVELS = 32           # channels per token
FSQ_LEVEL_LIST = [32] * 32    # 32 quantization levels per channel
FSQ_MAX_NUM_TOKENS = 2        # 2 tokens × 32 channels = 64-D
# Finger control (NOT decoded by SONIC — direct policy passthrough).
# Mirrors train.py's BinaryJointPositionActionCfg: the policy emits a single right-hand
# open/close scalar; the wrapper synthesizes a 14-D env action by blending between fixed
# canonical poses (left hand stays at its default open pose; right hand interpolates
# between open and closed pose by the scalar).
N_FINGERS_PER_HAND = 7
ENV_FINGER_SLOTS = 2 * N_FINGERS_PER_HAND     # 14 env-action slots (7 left + 7 right)
TOTAL_FINGER_DIM = 1                          # policy-side: a single right-hand scalar
POLICY_ACTION_DIM = TOKEN_TOTAL_DIM + TOTAL_FINGER_DIM  # 65 (was 78)

# Right-hand canonical poses, in action-term order — values from
# motion_tracking_pick_env.py's BinaryJointPositionActionCfg.right_hand_action.
_RIGHT_FINGER_OPEN_POSE  = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_RIGHT_FINGER_CLOSED_POSE = (
    0.0,                # thumb_0
    -np.pi / 3.0,       # thumb_1
    -np.pi / 2.0,       # thumb_2
    np.pi / 2.0,        # index_0
    np.pi / 2.0,        # index_1
    np.pi / 2.0,        # middle_0
    np.pi / 2.0,        # middle_1
)


class FSQ(nn.Module):
    """Finite Scalar Quantization (Mentzer et al. 2023), matching vector_quantize_pytorch.FSQ.

    For each of ``len(levels)`` channels, bound the input via a tanh and round to one of
    ``levels[i]`` integer values, then normalize by ``levels[i]//2`` so the output lives
    on a discrete grid in roughly ``[-1, 1]``. The straight-through estimator is used so
    PPO's gradient flows through the bound only (the round is a constant in the backward
    pass). For SONIC: ``levels = [32]*32`` → 32 channels per token, each on a 32-point grid.

    Input shape: (..., len(levels)). Output shape: same. Returns ``(quantized, None)`` to
    mimic the vector_quantize_pytorch interface ``(out, indices)``.
    """

    def __init__(self, levels):
        super().__init__()
        lvl = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("levels", lvl)
        # Precompute constants — these never change after init.
        eps = 1e-3
        self.register_buffer("_half_l", (lvl - 1.0) * (1.0 - eps) / 2.0)
        is_even = (lvl.long() % 2 == 0).to(torch.float32)
        self.register_buffer("_offset", torch.where(is_even.bool(), 0.5 * torch.ones_like(lvl), torch.zeros_like(lvl)))
        self.register_buffer("_shift", (self._offset / self._half_l).atanh())
        self.register_buffer("_half_width", (lvl // 2).to(torch.float32))

    def forward(self, z: torch.Tensor):
        bounded = (z + self._shift).tanh() * self._half_l - self._offset
        # Straight-through round: forward is round(bounded); backward is identity on bounded.
        quantized = bounded + (bounded.round() - bounded).detach()
        return quantized / self._half_width, None


# =========================================================================
# Frozen decoder loader (ONNX → batched torch module)
# =========================================================================

def load_frozen_decoder(onnx_path: str, device):
    """Convert ``model_decoder.onnx`` to a frozen, batched GPU torch module.

    The ONNX has a single input ``obs_dict`` of shape (B, 994) = [token(64) | proprio(930)]
    and a single output ``action`` of shape (B, 29). ``onnx2torch`` produces an fx
    GraphModule whose forward takes one positional tensor and returns one tensor; the
    internal graph ops (last-dim slices, unsqueeze on a fixed axis, Linear/MLP, squeeze)
    are all batch-polymorphic, so it runs for any batch size despite the batch=1 export.
    """
    from onnx2torch import convert

    module = convert(onnx_path)
    module = module.to(device).eval()
    for p in module.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in module.parameters())
    n_buffers = sum(b.numel() for b in module.buffers())
    n_state = len(module.state_dict())
    # onnx2torch often attaches ONNX Initializer tensors as buffers, not nn.Parameter,
    # so `params=0` is normal — what matters is that buffers/state_dict are populated.
    print(f"[load_frozen_decoder] converted {onnx_path} → torch module "
          f"(params={n_params}, buffer values={n_buffers}, state_dict keys={n_state}), frozen+eval")
    if n_state == 0 and n_buffers == 0:
        raise RuntimeError("Converted decoder has NO weights (state_dict + buffers both empty) — conversion failed.")
    return module


# =========================================================================
# Batched token → action vec-env wrapper
# =========================================================================

class TokenActionDecoderVecEnvWrapper(RslRlVecEnvWrapper):
    """rsl_rl vec-env wrapper that makes the policy a SONIC *encoder*.

    The rsl_rl actor outputs a 64-D continuous latent per env. This wrapper squashes it to
    a token (tanh → [-1, 1]), runs the frozen ONNX decoder with batched proprioception
    history, and feeds the resulting body command to the env. ``num_actions`` is overridden
    to the token dim (64) so rsl_rl sizes the actor's output head correctly — the base
    wrapper would otherwise read the env's 41-D ``action_manager.total_action_dim``.
    """

    def __init__(self, env, decoder, device, *, clip_actions=None):
        # Base init: sets num_envs/device, num_actions(=env 41-D), modifies action space,
        # and calls env.reset() once. We override num_actions afterwards.
        super().__init__(env, clip_actions=clip_actions)

        self.decoder = decoder
        self._dev = self.device

        self.token_total_dim = TOKEN_TOTAL_DIM
        self.policy_action_dim = POLICY_ACTION_DIM   # 64 (body token) + 14 (fingers)
        self.num_actions = self.policy_action_dim    # OVERRIDE: actor emits 78-D latent

        # FSQ quantizer for the body-token portion of the latent (frozen — no params).
        self.fsq = FSQ(FSQ_LEVEL_LIST).to(self._dev)
        for p in self.fsq.parameters():
            p.requires_grad = False

        # rebuild the (cosmetic) action space to the policy latent dim
        import gymnasium as gym
        self.unwrapped.single_action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.policy_action_dim,)
        )
        self.unwrapped.action_space = gym.vector.utils.batch_space(
            self.unwrapped.single_action_space, self.num_envs
        )

        robot = self.unwrapped.scene["robot"]
        joint_names = list(robot.data.joint_names)
        name_to_idx = {n: i for i, n in enumerate(joint_names)}

        # ---- joint-order bridges (SONIC-IsaacLab <-> env articulation) ----
        perm = np.full(N_BODY_JOINTS, -1, dtype=np.int64)
        for i, nm in enumerate(UTM_29_JOINT_NAMES):
            perm[i] = name_to_idx.get(nm, -1)
        self._gather_idx = torch.as_tensor(np.clip(perm, 0, None), device=self._dev, dtype=torch.long)
        self._gather_valid = torch.as_tensor(perm >= 0, device=self._dev, dtype=torch.bool)
        missing = [nm for i, nm in enumerate(UTM_29_JOINT_NAMES) if perm[i] < 0]
        if missing:
            print(f"[token-wrapper] SONIC joints absent on env robot (zero-filled in history): {missing}")

        self._default_sonic = torch.as_tensor(G1_DEFAULT_ANGLES_SONIC, device=self._dev, dtype=torch.float32)
        self._scale_sonic = torch.as_tensor(G1_ACTION_SCALE_SONIC, device=self._dev, dtype=torch.float32)
        self._scale_inv = 1.0 / self._scale_sonic
        keep = [i for i in range(N_BODY_JOINTS) if i not in (WAIST_ROLL_SONIC_IDX, WAIST_PITCH_SONIC_IDX)]
        self._keep_idx = torch.as_tensor(keep, device=self._dev, dtype=torch.long)
        self._perm27 = torch.as_tensor(PERM_SONIC27_TO_ENV27, device=self._dev, dtype=torch.long)
        self._n_body_env = len(keep)  # 27

        # ---- finger action slots: train.py-style single open/close scalar ----
        # The policy emits ONE right-hand scalar; we blend between canonical open and
        # canonical closed pose by 0.5*(1+tanh(scalar)). Left hand is held at its default
        # open pose (it's not relevant for the pick task). This mirrors what
        # BinaryJointPositionActionCfg did in train.py, with a smooth blend instead of a
        # discrete step (the smoothness is needed for PPO's gradient through the policy).
        left_ids = [name_to_idx[n] for n in _LEFT_FINGER_JOINTS if n in name_to_idx]
        right_ids = [name_to_idx[n] for n in _RIGHT_FINGER_JOINTS if n in name_to_idx]
        if len(left_ids) != 7 or len(right_ids) != 7:
            print(f"[token-wrapper] WARNING finger joints found L={len(left_ids)} R={len(right_ids)} (expected 7/7)")
        # Right-hand canonical poses, as (7,) tensors on device.
        self._right_open_pose = torch.tensor(_RIGHT_FINGER_OPEN_POSE, device=self._dev, dtype=torch.float32)
        self._right_closed_pose = torch.tensor(_RIGHT_FINGER_CLOSED_POSE, device=self._dev, dtype=torch.float32)
        # Left hand held at robot's default open pose for its 7 finger joints.
        default_q = robot.data.default_joint_pos[0]
        self._left_default_pose = default_q[left_ids].clone().to(self._dev)  # (7,)
        self._env_action_dim = self.unwrapped.action_manager.total_action_dim
        expected = self._n_body_env + len(left_ids) + len(right_ids)
        assert self._env_action_dim == expected, (
            f"env action dim {self._env_action_dim} != body{self._n_body_env}+fingers — "
            "ContinuousFingersActionsCfg layout mismatch"
        )

        # ---- batched decoder history (oldest at index 0) ----
        N = self.num_envs
        self._h_jp = torch.zeros(N, N_HISTORY_FRAMES, N_BODY_JOINTS, device=self._dev)
        self._h_jv = torch.zeros_like(self._h_jp)
        self._h_la = torch.zeros_like(self._h_jp)
        self._h_av = torch.zeros(N, N_HISTORY_FRAMES, 3, device=self._dev)
        self._h_gv = torch.zeros_like(self._h_av)
        self._prev_body_29 = torch.zeros(N, N_BODY_JOINTS, device=self._dev)

        self._seed_history(torch.ones(N, dtype=torch.bool, device=self._dev))
        print(f"[token-wrapper] ready: num_actions={self.num_actions} (= {TOKEN_TOTAL_DIM} body token + "
              f"{TOTAL_FINGER_DIM} right-hand open/close scalar; FSQ levels=[{FSQ_LEVEL_LIST[0]}]*{len(FSQ_LEVEL_LIST)}), "
              f"env_action_dim={self._env_action_dim}, num_envs={N}")

    # ---------- helpers ----------

    def _gather_sonic(self, q_env: torch.Tensor) -> torch.Tensor:
        """(N, n_joints) env order → (N, 29) SONIC-IsaacLab order (zero-fill missing)."""
        out = q_env.index_select(1, self._gather_idx)
        out = out * self._gather_valid.to(out.dtype)
        return out

    def _read_state(self):
        robot = self.unwrapped.scene["robot"]
        q_sonic = self._gather_sonic(robot.data.joint_pos)
        qd_sonic = self._gather_sonic(robot.data.joint_vel)
        av = robot.data.root_ang_vel_b
        gv = robot.data.projected_gravity_b
        return q_sonic, qd_sonic, av, gv

    def _seed_history(self, mask: torch.Tensor):
        """Fill all 10 history frames for masked envs from current robot state."""
        if not bool(mask.any()):
            return
        q_sonic, qd_sonic, av, gv = self._read_state()
        jp = q_sonic - self._default_sonic
        la = jp * self._scale_inv
        m = mask
        self._h_jp[m] = jp[m].unsqueeze(1).expand(-1, N_HISTORY_FRAMES, -1)
        self._h_jv[m] = qd_sonic[m].unsqueeze(1).expand(-1, N_HISTORY_FRAMES, -1)
        self._h_la[m] = la[m].unsqueeze(1).expand(-1, N_HISTORY_FRAMES, -1)
        self._h_av[m] = av[m].unsqueeze(1).expand(-1, N_HISTORY_FRAMES, -1)
        self._h_gv[m] = gv[m].unsqueeze(1).expand(-1, N_HISTORY_FRAMES, -1)
        self._prev_body_29[m] = la[m]

    @staticmethod
    def _push(buf: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        """Roll a (N,T,D) buffer left (drop oldest) and write newest at index -1."""
        buf = torch.roll(buf, shifts=-1, dims=1)
        buf[:, -1] = frame
        return buf

    def _build_obs_dict(self, token: torch.Tensor) -> torch.Tensor:
        """(N, 994) decoder input = [token(64) | proprioception(930)] in build_decoder_obs order."""
        N = self.num_envs
        return torch.cat(
            [
                token,                       # 64
                self._h_av.reshape(N, -1),   # 30
                self._h_jp.reshape(N, -1),   # 290
                self._h_jv.reshape(N, -1),   # 290
                self._h_la.reshape(N, -1),   # 290
                self._h_gv.reshape(N, -1),   # 30
            ],
            dim=-1,
        )

    def _decode_body_29(self, body_latent: torch.Tensor) -> torch.Tensor:
        """(N, 64) body-token latent → (N, 29) SONIC-order body command via the frozen decoder.

        The latent is reshaped to (N, max_num_tokens, num_fsq_levels) = (N, 2, 32), passed
        through FSQ to land on the discrete grid the decoder was trained against, flattened
        back to (N, 64), and concatenated with proprioception history to form the (N, 994)
        decoder input. The decoder is run under no_grad — PPO is model-free.
        """
        N = self.num_envs
        # FSQ: (N, 64) → (N, 2, 32) → quantize → (N, 2, 32) → flatten → (N, 64)
        z = body_latent.reshape(N, FSQ_MAX_NUM_TOKENS, FSQ_NUM_LEVELS)
        token, _ = self.fsq(z)
        token = token.reshape(N, TOKEN_TOTAL_DIM)
        obs = self._build_obs_dict(token).to(torch.float32)
        out = self.decoder(obs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.reshape(self.num_envs, -1)[:, :N_BODY_JOINTS]

    def _fingers_from_latent(self, finger_latent: torch.Tensor) -> torch.Tensor:
        """(N, 1) right-hand open/close scalar → (N, 14) env-finger-action vector.

        ``blend = 0.5 * (1 + tanh(scalar - 1.5))`` maps the unbounded policy scalar to
        ``[0, 1]`` with a +1.5 BIAS so the **default at latent=0 is fully open**:
            blend = 0 → fully open canonical pose
            blend = 1 → fully closed canonical pose
            at scalar=0: tanh(-1.5) = -0.905 → blend ≈ 0.048 (effectively open)
            to close:    scalar ≈ +3.0 → tanh(+1.5) = +0.905 → blend ≈ 0.952 (closed)
        This breaks the "always closed" local optimum: from an open default, the policy
        has to *actively learn* to drive the scalar positive when the is_closed obs says
        the reference is grasping — exploration starts from "wrong only during the grasp
        phase" instead of "wrong only during the open phase" (which was getting reward
        ~0.6 from passing through the closed-phase frames undetected). Combined with the
        sharpened reward scale, the conditional strategy now has a much steeper gradient.
        Right hand interpolates linearly between open/closed. Left hand stays at its env-
        default pose (open by default; not used for the pick task). Output layout follows
        ContinuousFingersActionsCfg: ``[7 left | 7 right]`` after the 27 body slots.
        """
        N = finger_latent.shape[0]
        blend = 0.5 * (1.0 + torch.tanh(finger_latent[:, :1] - 1.5))         # (N, 1) in [0,1]
        right_fingers = (
            blend * self._right_closed_pose.unsqueeze(0)
            + (1.0 - blend) * self._right_open_pose.unsqueeze(0)
        )                                                                     # (N, 7)
        left_fingers = self._left_default_pose.unsqueeze(0).expand(N, -1)    # (N, 7) constant
        return torch.cat([left_fingers, right_fingers], dim=1)               # (N, 14)

    def _body29_to_env_action(self, body_29: torch.Tensor, finger_cmds: torch.Tensor) -> torch.Tensor:
        """(N,29) SONIC body + (N,14) finger cmds → (N, env_action_dim) full env action."""
        N = self.num_envs
        q_target = self._default_sonic + body_29 * self._scale_sonic
        sonic27 = q_target.index_select(1, self._keep_idx)
        env_body = sonic27.index_select(1, self._perm27)
        act = torch.empty(N, self._env_action_dim, device=self._dev, dtype=env_body.dtype)
        act[:, : self._n_body_env] = env_body
        act[:, self._n_body_env : self._n_body_env + ENV_FINGER_SLOTS] = finger_cmds
        return act

    # ---------- vec-env API ----------

    def reset(self):
        obs, extras = super().reset()
        self._seed_history(torch.ones(self.num_envs, dtype=torch.bool, device=self._dev))
        return obs, extras

    def step(self, latent: torch.Tensor):
        from tensordict import TensorDict

        latent = latent.to(self._dev)
        if self.clip_actions is not None:
            latent = torch.clamp(latent, -self.clip_actions, self.clip_actions)

        with torch.no_grad():
            # 1. push current (pre-step) state into history; last_action = prev decoder out
            q_sonic, qd_sonic, av, gv = self._read_state()
            self._h_jp = self._push(self._h_jp, q_sonic - self._default_sonic)
            self._h_jv = self._push(self._h_jv, qd_sonic)
            self._h_la = self._push(self._h_la, self._prev_body_29)
            self._h_av = self._push(self._h_av, av)
            self._h_gv = self._push(self._h_gv, gv)

            # 2. split latent into body-token + finger latents, decode each side
            body_latent = latent[:, :TOKEN_TOTAL_DIM]
            finger_latent = latent[:, TOKEN_TOTAL_DIM:]
            body_29 = self._decode_body_29(body_latent)
            finger_cmds = self._fingers_from_latent(finger_latent)
            # 3. assemble full env action
            env_action = self._body29_to_env_action(body_29, finger_cmds)

        # 4. step the underlying env
        obs_dict, rew, terminated, truncated, extras = self.env.step(env_action)

        self._prev_body_29 = body_29
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        reset_mask = (terminated | truncated)
        if bool(reset_mask.any()):
            self._seed_history(reset_mask)

        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras
