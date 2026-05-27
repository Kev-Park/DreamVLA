"""RL-train a custom encoder against the FROZEN SONIC decoder.

The SONIC ``UniversalTokenModule`` is an encoder → FSQ quantizer → decoder stack.
Here we *freeze the decoder + quantizer* and let an RL (PPO) policy play the role
of the encoder: the rsl_rl actor consumes the env's native observation and emits a
``token_total_dim``-D continuous latent. This wrapper then

  1. reshapes the latent to (B, max_num_tokens, token_dim) and runs the frozen FSQ
     quantizer → flat token (``token_flattened``),
  2. runs the frozen ``g1_dyn`` decoder (token + batched proprioception history) →
     29-D body command (SONIC-IsaacLab joint order),
  3. maps the 29-D command to the env's action space (drop waist roll/pitch, permute
     to JointNamesOrder, append constant default finger targets),
  4. steps the underlying Isaac Lab env and updates the decoder's rolling history.

PPO is model-free: the frozen decoder + sim are just the "augmented environment", so
gradients never flow through the decoder/quantizer. We therefore run them under
``torch.no_grad()`` and only need batched forward inference.

Contract notes (verified against gear_sonic):
- ``decode("g1_dyn", {...})`` concatenates the decoder's declared ``input_features``
  (``token_flattened`` + ``proprioception``) along the last dim, so both must carry a
  sequence dim: ``token_flattened`` (B,1,64), ``proprioception`` (B,1,930). Output
  ``["action"]`` is (B,1,29). See ``inference_helpers.export_universal_token_decoder_as_onnx``.
- proprioception order matches ``vla_sonic.planner_to_utm.build_decoder_obs`` (minus the
  token slot): his_base_angular_velocity(30), his_body_joint_positions(290),
  his_body_joint_velocities(290), his_last_actions(290), his_gravity_dir(30) = 930.
- The 29-D body command → env action transform mirrors
  ``vla_sonic.action_assembler.utm_body_29_to_env_27`` exactly (batched here).
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from .action_assembler import (
    G1_ACTION_SCALE_SONIC,
    G1_DEFAULT_ANGLES_SONIC,
    PERM_SONIC27_TO_ENV27,
    WAIST_PITCH_SONIC_IDX,
    WAIST_ROLL_SONIC_IDX,
)

# Left/right hand action-term joint order — must match the cfg term order in
# ContinuousFingersActionsCfg (left_hand_action then right_hand_action).
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

# SONIC-IsaacLab 29-joint order (no fingers). Used to gather the env's measured
# joint state into the order the decoder history was trained on. Mirrors
# eval_parquet_sonic.UTM_29_JOINT_NAMES.
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


# =========================================================================
# Frozen UTM loader
# =========================================================================

def load_frozen_utm(config_path: str, ckpt_path: str, device, *, decoder_name: str = "g1_dyn"):
    """Build the SONIC ``UniversalTokenModule`` from a released checkpoint and freeze it.

    The released config (``model_config.yaml`` produced by ``eval_agent_trl.py``'s export
    branch) carries the resolved ``env_config`` and ``algo_config``. The actor module is
    instantiated exactly as eval_agent_trl.py does:
        ``custom_instantiate(algo_config.actor, env_config=..., algo_config=..., module_dim_dict=...)``
    and the UTM is ``policy.actor_module`` (the actor is a thin policy wrapper around it).

    Returns the frozen UTM in eval mode. Decoder + quantizer + encoders are all frozen
    (we only ever call ``utm.quantizer`` and ``utm.decode`` for forward inference).
    """
    from omegaconf import OmegaConf

    from gear_sonic.trl.utils import common as trl_common

    cfg = OmegaConf.load(config_path)

    # The export dumps {env_config, algo_config}. Be tolerant of either that shape or a
    # config that already *is* the algo config.
    if "env_config" in cfg and "algo_config" in cfg:
        env_config = cfg.env_config
        algo_config = cfg.algo_config
    elif "algo" in cfg:  # raw training config fallback
        env_config = cfg.get("env", None)
        algo_config = cfg.algo.config
    else:
        raise KeyError(
            f"Unrecognized SONIC config layout in {config_path}; "
            f"top-level keys = {list(cfg.keys())}. Expected 'env_config'+'algo_config'."
        )

    actor_cfg = algo_config["actor"]
    module_dim_dict = algo_config.get("module_dim", {})

    print(f"[load_frozen_utm] actor _target_ = {actor_cfg.get('_target_', '<none>')}")
    policy = trl_common.custom_instantiate(
        actor_cfg,
        env_config=env_config,
        algo_config=algo_config,
        module_dim_dict=module_dim_dict,
        backbone_kwargs={},
        _resolve=False,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "actor_model_state_dict" in ckpt:
        state_dict = ckpt["actor_model_state_dict"]
    elif isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
        state_dict = ckpt["policy_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt  # assume raw state_dict

    # std/log_std backward-compat (mirrors eval_agent_trl.py).
    model_keys = set(policy.state_dict().keys())
    if "std" in model_keys and "log_std" in state_dict and "std" not in state_dict:
        state_dict["std"] = torch.exp(state_dict.pop("log_std"))
    elif "log_std" in model_keys and "std" in state_dict and "log_std" not in state_dict:
        state_dict["log_std"] = torch.log(state_dict.pop("std"))

    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_frozen_utm] WARNING missing state_dict keys ({len(missing)}): {missing[:8]}...")
    if unexpected:
        print(f"[load_frozen_utm] WARNING unexpected state_dict keys ({len(unexpected)}): {unexpected[:8]}...")

    utm = getattr(policy, "actor_module", policy)
    assert hasattr(utm, "decode") and hasattr(utm, "decoders"), \
        "Loaded module has no .decode/.decoders — not a UniversalTokenModule?"
    assert decoder_name in utm.decoders, \
        f"decoder '{decoder_name}' not in {list(utm.decoders.keys())}"

    for p in utm.parameters():
        p.requires_grad = False
    utm.eval()

    print(f"[load_frozen_utm] token_total_dim={utm.token_total_dim} "
          f"(max_num_tokens={utm.max_num_tokens}, token_dim={utm.token_dim})")
    print(f"[load_frozen_utm] decoders={list(utm.decoders.keys())}  "
          f"quantizer={'present' if utm.quantizer is not None else 'NONE'}")
    print(f"[load_frozen_utm] decoder_input_features[{decoder_name}]="
          f"{utm.decoder_input_features.get(decoder_name)}")
    return utm


# =========================================================================
# Batched token → action vec-env wrapper
# =========================================================================

class TokenActionDecoderVecEnvWrapper(RslRlVecEnvWrapper):
    """rsl_rl vec-env wrapper that makes the policy a SONIC *encoder*.

    The rsl_rl actor outputs a ``token_total_dim``-D continuous latent per env. This
    wrapper quantizes it, runs the frozen ``g1_dyn`` decoder, and feeds the resulting
    body command to the underlying Isaac Lab env. ``num_actions`` is overridden to the
    token dim so rsl_rl sizes the actor's output head correctly (the base
    RslRlVecEnvWrapper would otherwise read the env's 41-D action_manager dim).
    """

    def __init__(self, env, utm, device, *, decoder_name: str = "g1_dyn", clip_actions=None):
        # Base init: sets num_envs/device, num_actions(=env 41-D), modifies action space,
        # and calls env.reset() once. We override num_actions afterwards.
        super().__init__(env, clip_actions=clip_actions)

        self.utm = utm
        self.decoder_name = decoder_name
        self._dev = self.device

        # ---- token dims drive the rsl_rl actor output size ----
        self.token_total_dim = int(utm.token_total_dim)
        self.token_dim = int(utm.token_dim)
        self.max_num_tokens = int(utm.max_num_tokens)
        self.num_actions = self.token_total_dim  # OVERRIDE: actor emits the latent

        # rebuild the (cosmetic) action space to the token dim so any clipping uses it
        import gymnasium as gym
        self.unwrapped.single_action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.token_total_dim,)
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
        # 27 kept joints (drop waist roll/pitch) then permute to env JointNamesOrder.
        keep = [i for i in range(N_BODY_JOINTS) if i not in (WAIST_ROLL_SONIC_IDX, WAIST_PITCH_SONIC_IDX)]
        self._keep_idx = torch.as_tensor(keep, device=self._dev, dtype=torch.long)
        self._perm27 = torch.as_tensor(PERM_SONIC27_TO_ENV27, device=self._dev, dtype=torch.long)
        self._n_body_env = len(keep)  # 27

        # ---- finger action slots: constant default (open) targets ----
        left_ids = [name_to_idx[n] for n in _LEFT_FINGER_JOINTS if n in name_to_idx]
        right_ids = [name_to_idx[n] for n in _RIGHT_FINGER_JOINTS if n in name_to_idx]
        if len(left_ids) != 7 or len(right_ids) != 7:
            print(f"[token-wrapper] WARNING finger joints found L={len(left_ids)} R={len(right_ids)} (expected 7/7)")
        default_q = robot.data.default_joint_pos[0]  # (n_joints,)
        self._left_finger_default = default_q[left_ids].clone().to(self._dev)   # (7,)
        self._right_finger_default = default_q[right_ids].clone().to(self._dev)  # (7,)
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
        print(f"[token-wrapper] ready: num_actions={self.num_actions} (token latent), "
              f"env_action_dim={self._env_action_dim}, num_envs={N}")

    # ---------- helpers ----------

    def _gather_sonic(self, q_env: torch.Tensor) -> torch.Tensor:
        """(N, n_joints) env order → (N, 29) SONIC-IsaacLab order (zero-fill missing)."""
        out = q_env.index_select(1, self._gather_idx)  # (N,29)
        out = out * self._gather_valid.to(out.dtype)  # zero invalid slots
        return out

    def _read_state(self):
        robot = self.unwrapped.scene["robot"]
        q_sonic = self._gather_sonic(robot.data.joint_pos)
        qd_sonic = self._gather_sonic(robot.data.joint_vel)
        av = robot.data.root_ang_vel_b  # (N,3)
        gv = robot.data.projected_gravity_b  # (N,3) gravity dir in base frame
        return q_sonic, qd_sonic, av, gv

    def _seed_history(self, mask: torch.Tensor):
        """Fill all 10 history frames for masked envs from current robot state."""
        if not bool(mask.any()):
            return
        q_sonic, qd_sonic, av, gv = self._read_state()
        jp = q_sonic - self._default_sonic          # delta from default (SONIC order)
        la = jp * self._scale_inv                    # approx "action that holds this pose"
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

    def _build_proprioception(self) -> torch.Tensor:
        """(N, 930) in build_decoder_obs order (frames flattened oldest→newest)."""
        N = self.num_envs
        return torch.cat(
            [
                self._h_av.reshape(N, -1),  # 30
                self._h_jp.reshape(N, -1),  # 290
                self._h_jv.reshape(N, -1),  # 290
                self._h_la.reshape(N, -1),  # 290
                self._h_gv.reshape(N, -1),  # 30
            ],
            dim=-1,
        )

    def _decode_body_29(self, latent: torch.Tensor) -> torch.Tensor:
        """(N, token_total_dim) policy latent → (N, 29) SONIC-order body command."""
        N = self.num_envs
        # FSQ quantize: (N,64) → (N, max_num_tokens, token_dim) → codes → flat (N,64)
        if self.utm.quantizer is not None:
            z = latent.reshape(N, self.max_num_tokens, self.token_dim)
            codes, _ = self.utm.quantizer(z)
            token_flat = codes.reshape(N, self.token_total_dim)
        else:
            token_flat = latent
        decode_input = {
            "token_flattened": token_flat.unsqueeze(1),          # (N,1,64)
            "proprioception": self._build_proprioception().unsqueeze(1),  # (N,1,930)
        }
        out = self.utm.decode(self.decoder_name, decode_input)
        if "action" in out:
            body = out["action"]
        elif "body_action" in out:
            body = out["body_action"]
        else:
            body = next(iter(out.values()))
        return body.reshape(N, -1)[:, :N_BODY_JOINTS]

    def _body29_to_env_action(self, body_29: torch.Tensor) -> torch.Tensor:
        """(N,29) SONIC body command → (N, env_action_dim) full env action."""
        N = self.num_envs
        q_target = self._default_sonic + body_29 * self._scale_sonic        # (N,29)
        sonic27 = q_target.index_select(1, self._keep_idx)                  # (N,27)
        env_body = sonic27.index_select(1, self._perm27)                    # (N,27) JointNamesOrder
        act = torch.empty(N, self._env_action_dim, device=self._dev, dtype=env_body.dtype)
        act[:, : self._n_body_env] = env_body
        act[:, self._n_body_env : self._n_body_env + 7] = self._left_finger_default
        act[:, self._n_body_env + 7 : self._n_body_env + 14] = self._right_finger_default
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

            # 2. quantize + decode → body command, 3. assemble full env action
            body_29 = self._decode_body_29(latent)
            env_action = self._body29_to_env_action(body_29)

        # 4. step the underlying env
        obs_dict, rew, terminated, truncated, extras = self.env.step(env_action)

        self._prev_body_29 = body_29
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        # re-seed history for envs that just reset (obs already post-reset)
        reset_mask = (terminated | truncated)
        if bool(reset_mask.any()):
            self._seed_history(reset_mask)

        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras
