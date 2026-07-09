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
``BinaryJointPositionActionCfg`` exactly: the policy emits one scalar per hand, which
the wrapper passes straight through to the env's action manager. The env's
BinaryJointPositionActionCfg internally maps action<0 → closed pose, action≥0 → open
pose (the canonical 7-D poses live in the env cfg, not here). The left hand is held at
constantly-open (the task only needs the right hand to grasp), so the wrapper emits a
fixed positive constant for the left-hand slot and only the right-hand slot is policy-
driven. The original train.py finger reward (binary match between action_manager.action's
right-hand slot and motion_lib's is_closed) then trains the right-hand scalar.

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
    build_sonic29_to_env_perm,
)

# With BinaryFingersActionsCfg the env's action_manager owns the 1-D → 7-D finger
# expansion via open_command_expr / close_command_expr — the wrapper only writes
# the 1-D binary scalars and never sees individual finger joints.

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
# Mirrors train.py's BinaryJointPositionActionCfg exactly: policy emits a single right-hand
# open/close scalar; the wrapper passes it straight through to the env's action vector,
# where the action manager handles open↔closed pose substitution internally.
ENV_FINGER_SLOTS = 2                          # 1 left-hand binary + 1 right-hand binary
TOTAL_FINGER_DIM = 1                          # policy-side: a single right-hand scalar
POLICY_ACTION_DIM = TOKEN_TOTAL_DIM + TOTAL_FINGER_DIM  # 65

# Left hand: held open the entire episode (this task only needs the right hand to grasp).
# BinaryJointPositionActionCfg convention: action >= 0 → open, action < 0 → closed. A
# constant +1.0 puts the left hand firmly in the "open" branch with margin against any
# numerical noise.
_LEFT_HAND_OPEN_SCALAR = 1.0


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
        self._inv_perm27 = torch.argsort(self._perm27)  # env-JointNamesOrder-27 -> SONIC-27
        # SONIC-29 -> env-29 body perm for the 29-DOF env (apply_29dof_waist_override appends
        # waist_roll then waist_pitch after the 27 JointNamesOrder body joints).
        # env_body[i] = sonic29[_perm29[i]].
        self._perm29 = torch.cat([
            self._keep_idx.index_select(0, self._perm27),
            torch.as_tensor([WAIST_ROLL_SONIC_IDX, WAIST_PITCH_SONIC_IDX],
                            device=self._dev, dtype=torch.long),
        ])

        # Reference (motion_lib) joints -> SONIC-29, by NAME. motion_lib.dof_pos is in
        # motion_lib.joint_names order (= env JointNamesOrder; the 29-DOF order has waist roll/pitch
        # at idx 13,14, NOT appended at the end). ref[i] = sonic29[perm[i]] -> we SCATTER the ref into
        # SONIC-29 slots. Used by _seed_history (history) AND the encoder child (base token), so BOTH
        # inputs share one correct mapping. Handles 27- or 29-DOF refs (a 27-DOF ref leaves SONIC's
        # waist_roll/pitch slots at 0). Replaces the 27-hardcoded `dof[..., _inv_perm27]`+`_keep_idx`.
        try:
            _ref_names = list(self.unwrapped.motion_lib.joint_names)
        except AttributeError:
            from isaaclab_tasks.utils.motion_lib.motion_lib_base import JointNamesOrder
            _ref_names = list(JointNamesOrder)
        self._ref_to_sonic_perm = torch.as_tensor(
            build_sonic29_to_env_perm(_ref_names), device=self._dev, dtype=torch.long)

        # ---- finger action slots: train.py-style 1-D binary per hand ----
        # The policy emits ONE right-hand scalar; the wrapper writes that scalar into the
        # right-hand action slot, and writes a constant positive scalar into the left-hand
        # slot (left hand always open). The env's BinaryJointPositionActionCfg owns the
        # actual 1-D → 7-D joint expansion via open/close_command_expr — the wrapper never
        # touches individual finger joints. This is exactly the train.py contract.
        self._env_action_dim = self.unwrapped.action_manager.total_action_dim
        self._n_body_env = self._env_action_dim - ENV_FINGER_SLOTS  # 27 (welded) or 29 (waist-actuated)
        self._is_29dof = (self._n_body_env == N_BODY_JOINTS)        # 29
        assert self._n_body_env in (27, N_BODY_JOINTS), (
            f"env body action dim {self._n_body_env} (= total {self._env_action_dim} - "
            f"{ENV_FINGER_SLOTS} finger slots) must be 27 (BinaryFingers welded-waist) or 29 "
            f"(apply_29dof_waist_override). Got total={self._env_action_dim}."
        )
        print(f"[token-wrapper] env body DOF = {self._n_body_env} "
              f"({'29-DOF: waist actuated, no drop' if self._is_29dof else '27-DOF: waist roll/pitch dropped'})")

        # Decoder(SONIC-29) -> env body action, NAME-MATCHED to the ACTUAL action-term joint
        # order. CRITICAL: the body action term uses joint_names=JointNamesOrder (preserve_order
        # =True). That constant is 29-DOF INTERLEAVED (waist_roll/pitch at idx 13,14, arms at
        # 15..28) whenever motion_lib is 29-DOF, NOT [JointNamesOrder-27, waist_roll, waist_pitch]
        # with waist appended at 27,28. The hand-built self._perm29 assumed the appended layout,
        # so it mis-routed both arms + waist (16 joints) whenever the action term is interleaved
        # -> whole-body flailing regardless of the waist reference. Building the perm by NAME from
        # the term's real joint order is correct for 27- or 29-DOF and any Isaac-resolved order.
        try:
            _body_action_names = list(self.unwrapped.action_manager.get_term("joint_pos")._joint_names)
        except Exception:
            try:
                _body_action_names = list(self.unwrapped.cfg.actions.joint_pos.joint_names)
            except Exception:
                from isaaclab_tasks.utils.motion_lib.motion_lib_base import JointNamesOrder as _JNO
                _body_action_names = list(_JNO)
        assert len(_body_action_names) == self._n_body_env, (
            f"action-term body joints ({len(_body_action_names)}) != env body dim "
            f"({self._n_body_env}); order={_body_action_names}")
        self._sonic_to_env_body = torch.as_tensor(
            build_sonic29_to_env_perm(_body_action_names), device=self._dev, dtype=torch.long)
        if self._is_29dof:
            _agree = bool(torch.equal(self._sonic_to_env_body, self._perm29))
            print(f"[token-wrapper] name-matched body perm == hand-built _perm29: {_agree} "
                  f"{'(appended-waist layout)' if _agree else '(INTERLEAVED layout — hand _perm29 would MIS-ROUTE; using name-matched)'}")

        self._left_open_scalar = torch.tensor(_LEFT_HAND_OPEN_SCALAR, device=self._dev, dtype=torch.float32)

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
              f"{TOTAL_FINGER_DIM} right-hand binary scalar; left hand held open at +{_LEFT_HAND_OPEN_SCALAR}; "
              f"FSQ levels=[{FSQ_LEVEL_LIST[0]}]*{len(FSQ_LEVEL_LIST)}), "
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

    def _seed_history_frozen(self, mask: torch.Tensor):
        """Fallback: fill all 10 history frames from the current robot state (static)."""
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

    def _seed_history(self, mask: torch.Tensor):
        """Seed the 10 history frames PRECEDING the reset frame from motion_lib (causal,
        MOVING) so the decoder gets a real gait phase when the env resets mid-motion
        (skip_start). Joint positions/velocities — the gait signal — come from the reference;
        base angular velocity + projected gravity are computed PER-FRAME from the reference
        root orientation (forward quaternion finite-difference), matching the per-frame seed
        in eval_parquet_sonic.py rather than broadcasting the single reset-frame value. Falls
        back to the frozen-frame seed if motion_lib is unavailable.
        """
        if not bool(mask.any()):
            return
        unw = self.unwrapped
        if not (hasattr(unw, "motion_lib") and hasattr(unw, "motion_ids")
                and hasattr(unw, "start_motion_times")):
            self._seed_history_frozen(mask)
            return
        import isaaclab.utils.math as math_utils
        N = self.num_envs
        dt = float(unw.step_dt)
        mids = unw.motion_ids
        t0 = unw.start_motion_times.to(self._dev, dtype=torch.float32)   # (N,) reset frame time [s]
        # 11 reference frames f[i] = motion @ (t0 - (10-i)*dt). f[0]=t0-10dt (oldest) ...
        # f[10]=t0. History slots 0..9 use f[0..9]; velocities = forward diff f[i+1]-f[i].
        pos = []
        rot = []
        for i in range(N_HISTORY_FRAMES + 1):
            tk = torch.clamp(t0 - (N_HISTORY_FRAMES - i) * dt, min=0.0)
            res = unw.motion_lib.get_motion_state(mids, tk)
            dof = res["dof_pos"]                                         # (N, n_ref) motion_lib.joint_names order
            sonic29 = torch.zeros(N, N_BODY_JOINTS, device=self._dev, dtype=dof.dtype)
            sonic29.index_copy_(1, self._ref_to_sonic_perm, dof)         # name-matched scatter ref -> SONIC-29
            pos.append(sonic29)
            rot.append(res["root_rot"].to(self._dev))                   # (N, 4) wxyz
        pos = torch.stack(pos, dim=1)                                    # (N, 11, 29)
        jp = pos[:, :N_HISTORY_FRAMES, :] - self._default_sonic          # (N, 10, 29)
        jv = (pos[:, 1:, :] - pos[:, :N_HISTORY_FRAMES, :]) / dt          # (N, 10, 29) forward diff
        la = jp * self._scale_inv
        # Per-frame projected gravity (body) and base angular velocity (body), from root_rot.
        quats = torch.stack(rot, dim=1)                                  # (N, 11, 4) wxyz
        q_s = quats[:, :N_HISTORY_FRAMES, :].reshape(-1, 4)              # (N*10, 4) frames 0..9
        q_s1 = quats[:, 1:, :].reshape(-1, 4)                            # (N*10, 4) frames 1..10
        grav_w = torch.tensor([0.0, 0.0, -1.0], device=self._dev, dtype=q_s.dtype).expand(q_s.shape[0], 3)
        gv = math_utils.quat_apply_inverse(q_s, grav_w).reshape(N, N_HISTORY_FRAMES, 3)
        dq = math_utils.quat_mul(q_s1, math_utils.quat_conjugate(q_s))   # world-frame delta s->s+1
        ang_w = math_utils.axis_angle_from_quat(dq) / dt                 # (N*10, 3) world ang vel
        av = math_utils.quat_apply_inverse(q_s, ang_w).reshape(N, N_HISTORY_FRAMES, 3)  # body frame
        m = mask
        self._h_jp[m] = jp[m].to(self._h_jp.dtype)
        self._h_jv[m] = jv[m].to(self._h_jv.dtype)
        self._h_la[m] = la[m].to(self._h_la.dtype)
        self._h_av[m] = av[m].to(self._h_av.dtype)
        self._h_gv[m] = gv[m].to(self._h_gv.dtype)
        self._prev_body_29[m] = la[m, -1].to(self._prev_body_29.dtype)   # last seeded action (t0-dt)

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
        """(N, 1) right-hand open/close scalar → (N, 2) env-finger-action vector.

        Direct passthrough to the env's BinaryJointPositionActionCfg slots. The action
        manager itself owns the 1-D → 7-D joint expansion via open/close_command_expr —
        we just write the policy scalar into the right-hand slot and a constant positive
        scalar into the left-hand slot (left hand always open; not used for the pick
        task). BinaryJointPositionActionCfg convention: action < 0 → closed, action >= 0
        → open. Output layout: ``[1 left | 1 right]`` after the 27 body slots, matching
        BinaryFingersActionsCfg in motion_tracking_pick_env.py.
        """
        N = finger_latent.shape[0]
        right = finger_latent[:, :1]                                                  # (N, 1)
        left = self._left_open_scalar.expand(N, 1)                                    # (N, 1)
        return torch.cat([left, right], dim=1)                                        # (N, 2)

    def _body29_to_env_action(self, body_29: torch.Tensor, finger_cmds: torch.Tensor) -> torch.Tensor:
        """(N,29) SONIC body + (N,2) binary finger cmds → (N, env_action_dim) full env action."""
        N = self.num_envs
        q_target = self._default_sonic + body_29 * self._scale_sonic
        # Name-matched SONIC-29 -> env body action (handles 27- or 29-DOF, interleaved or appended
        # waist; a 27-DOF action order simply never references SONIC's waist_roll/pitch slots).
        env_body = q_target.index_select(1, self._sonic_to_env_body)
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
