"""Residual token-adapter wrapper: frozen SONIC encoder → learned residual → frozen decoder.

Residual-policy-learning in SONIC token space. Per step:

    1. Build the G1-mode encoder input from motion_lib (10 future reference frames at
       0.1 s spacing — the exact ``build_g1_encoder_obs`` recipe eval_parquet_sonic.py
       validated, batched in torch).
    2. Run the FROZEN encoder → 64-D base token (FSQ-quantized by the ONNX itself).
    3. The rsl_rl policy (the "adapter") sees [env obs | base token] and outputs a 65-D
       action: 64-D token RESIDUAL + 1-D right-hand binary scalar.
    4. token = FSQ(base + residual_scale * tanh(residual)) → frozen decoder → env action.

With the adapter's output layer zero-initialized (see ``adapter_actor_critic.py``), step 0
behavior is EXACTLY zero-shot SONIC playback of the reference motion — walking, balance,
reaching all intact before any learning. PPO then learns a *delta* for the task. The
tanh bound is a hard anchor: behavior can never leave the ``residual_scale``-neighborhood
of the frozen base, so there is nothing to catastrophically forget.

Encoder input construction notes (mirrors eval_parquet_sonic.py:1179-1356):
  - motion_joint_positions_10frame_step5 (290): 10 frames × 29 joints, SONIC-IsaacLab
    INTERLEAVED order (NOT MUJOCO-grouped — see eval's 7f comment + policy_parameters.hpp:92).
    motion_lib's dof_pos is 27 joints in env JointNamesOrder → inverse-permute to SONIC-27,
    scatter to 29 with zeros at waist_roll/waist_pitch.
  - motion_joint_velocities_10frame_step5 (290): np.gradient-style central differences over
    the 10-frame window (dt = 0.1 s).
  - motion_anchor_orientation_10frame_step5 (60): per-frame rot6d of
    (R_robot_now)^-1 · R_ref_root(t+k·0.1), ROW-major flatten of mat[:, :2]
    (eval: ``_R_rel_k[:, :2].flatten("C")``).
  - encoder_mode_4 = [0,0,0,0] (G1 mode id 0 → slot stays all-zero).
  - All other slots zero-filled (G1-mode branch ignores them).
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.utils.math as math_utils

from .planner_to_utm import ENCODER_SLICES, ENCODER_TOTAL_DIM
from .token_action_wrapper import (
    N_BODY_JOINTS,
    TOKEN_TOTAL_DIM,
    TokenActionDecoderVecEnvWrapper,
)

# SONIC G1-encoder lookahead: 10 frames at 0.1 s spacing (num_future_frames=10,
# dt_future_ref_frames=0.1 in sonic_release.yaml).
N_FUTURE_FRAMES = 10
FUTURE_DT = 0.1


def _patch_reshape_batch_dims(gm) -> int:
    """Make an onnx2torch GraphModule batch-polymorphic.

    ``model_encoder.onnx`` was exported at batch=1, which bakes the batch size into the
    shape constants of its Reshape nodes (e.g. ``[1, 1, 10, 58]``). A batched input
    (N, 1762) then fails with ``shape '[1, 1, 10, 58]' is invalid for input of size
    N*580``. Setting the LEADING dim of every such constant to -1 lets any batch size
    flow through (torch.reshape infers it), leaving the per-sample shape untouched.
    The decoder ONNX never needed this — its graph ops happen to be batch-polymorphic.

    Returns the number of patched constants.
    """
    def _resolve_constant_tensor(node):
        """Find the tensor behind a shape-constant FX node.

        onnx2torch materializes ONNX constants two ways:
          - initializers → ``get_attr`` nodes (tensor lives as a module attribute/buffer)
          - Constant ops → ``call_module`` nodes to an OnnxConstant submodule whose
            tensor is stored on the submodule (attribute ``value`` or its sole buffer).
            This is the form model_encoder.onnx uses (``constant_20 = self.Constant_20()``).
        """
        op = getattr(node, "op", None)
        if op == "get_attr":
            owner = gm
            for atom in node.target.split("."):
                owner = getattr(owner, atom)
            return owner if torch.is_tensor(owner) else None
        if op == "call_module":
            sub = gm.get_submodule(node.target)
            t = getattr(sub, "value", None)
            if torch.is_tensor(t):
                return t
            bufs = list(sub.buffers())
            if len(bufs) == 1:
                return bufs[0]
        return None

    patched = 0
    for node in gm.graph.nodes:
        if node.op != "call_module":
            continue
        submodule = gm.get_submodule(node.target)
        if "Reshape" not in type(submodule).__name__ or len(node.args) < 2:
            continue
        shape_tensor = _resolve_constant_tensor(node.args[1])
        if (
            shape_tensor is not None
            and shape_tensor.ndim == 1
            and not shape_tensor.dtype.is_floating_point
            and shape_tensor.numel() >= 1
            and int(shape_tensor[0]) == 1
        ):
            shape_tensor[0] = -1
            patched += 1
    return patched


def load_frozen_encoder(onnx_path: str, device):
    """Convert ``model_encoder.onnx`` to a frozen, BATCHED torch module.

    Same onnx2torch path as ``load_frozen_decoder``, plus the batch-dim patch above
    (the encoder export hardcodes batch=1 in its Reshape constants).
    """
    from onnx2torch import convert

    module = convert(onnx_path)
    module = module.to(device).eval()
    for p in module.parameters():
        p.requires_grad = False
    n_patched = _patch_reshape_batch_dims(module)
    n_buffers = sum(b.numel() for b in module.buffers())
    n_state = len(module.state_dict())
    print(f"[load_frozen_encoder] converted {onnx_path} → torch module "
          f"(buffer values={n_buffers}, state_dict keys={n_state}), frozen+eval; "
          f"patched {n_patched} Reshape batch dims (1 → -1) for batch-polymorphic forward")
    if n_state == 0 and n_buffers == 0:
        raise RuntimeError("Converted encoder has NO weights — conversion failed.")
    return module


class TokenAdapterVecEnvWrapper(TokenActionDecoderVecEnvWrapper):
    """Frozen-encoder + residual-adapter variant of the token-action wrapper.

    The policy action layout is unchanged (64 + 1 = 65), but the 64 body dims are now a
    RESIDUAL added to the frozen encoder's base token instead of the full token. The
    base token is appended to the policy observation (+64 obs dims) so the adapter knows
    what it is correcting.
    """

    def __init__(self, env, decoder, encoder, device, *, residual_scale: float = 0.3, clip_actions=None):
        super().__init__(env, decoder, device, clip_actions=clip_actions)

        self.encoder = encoder
        self.residual_scale = float(residual_scale)

        # ---- env27 → SONIC29 bridges (inverse of the parent's SONIC→env mapping) ----
        # Parent: env27[j] = sonic27[perm27[j]]  ⇒  sonic27[k] = env27[argsort(perm27)[k]].
        self._inv_perm27 = torch.argsort(self._perm27)
        # keep_idx (parent) holds the 27 SONIC-29 indices that survive the waist drop;
        # scatter sonic27 back into a zero-filled 29 (waist_roll/pitch stay 0 — motion_lib
        # has no waist roll/pitch reference either).

        # Encoder input slices as (start, stop) tuples for torch column assignment.
        self._sl_pos = ENCODER_SLICES["motion_joint_positions_10frame_step5"]
        self._sl_vel = ENCODER_SLICES["motion_joint_velocities_10frame_step5"]
        self._sl_anchor = ENCODER_SLICES["motion_anchor_orientation_10frame_step5"]

        # Future time offsets (1, 10) — broadcast against (N, 1) current times.
        self._future_offsets = (
            torch.arange(N_FUTURE_FRAMES, device=self._dev, dtype=torch.float32) * FUTURE_DT
        ).unsqueeze(0)

        self._base_token = torch.zeros(self.num_envs, TOKEN_TOTAL_DIM, device=self._dev)
        self._update_base_token()
        print(
            f"[token-adapter] frozen-encoder residual adapter ready: residual_scale={self.residual_scale}, "
            f"base_token appended to obs (+{TOKEN_TOTAL_DIM} dims), policy action = "
            f"{TOKEN_TOTAL_DIM} residual + 1 finger scalar"
        )

    # ---------- token quantization (lattice snap, NOT full FSQ) ----------

    def _decode_body_29(self, body_latent: torch.Tensor) -> torch.Tensor:
        """(N, 64) composed token (base + residual) → (N, 29) body command.

        OVERRIDES the parent: the parent applies the full FSQ transform
        ``round((z + shift).tanh() * 15.48 - 0.5) / 16``, which is correct for the
        from-scratch policy's UNBOUNDED latent but WRONG here — the frozen encoder's
        base token is ALREADY FSQ-quantized, and FSQ is not idempotent: the tanh
        squashes on-grid values toward zero (1.0 → 0.75, 0.5 → 0.44), distorting the
        base token and pushing the decoder off-distribution even at residual = 0.

        Instead: snap ``base + residual`` to the same FSQ lattice (step 1/16, range
        [-1, 15/16] for 32 levels) by plain rounding. At residual = 0 this is an exact
        passthrough of the encoder's token — byte-identical to the validated
        eval_parquet_sonic.py encoder→decoder path.
        """
        half_width = 16.0  # FSQ levels 32 → half_width = 32 // 2
        token = torch.clamp(
            torch.round(body_latent * half_width) / half_width,
            min=-1.0, max=(half_width - 1.0) / half_width,
        )
        obs = self._build_obs_dict(token).to(torch.float32)
        out = self.decoder(obs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.reshape(self.num_envs, -1)[:, :N_BODY_JOINTS]

    # ---------- base-token computation ----------

    def _update_base_token(self) -> None:
        """Run the frozen encoder on the current 1.0 s reference lookahead window."""
        N = self.num_envs
        unw = self.unwrapped
        with torch.no_grad():
            # (N, 10) motion times — current + k*0.1 s.
            t_now = unw.episode_length_buf * unw.step_dt + unw.start_motion_times.clone().detach().to(
                device=self._dev, dtype=torch.float32
            )
            times = (t_now.unsqueeze(1) + self._future_offsets).reshape(-1)            # (N*10,)
            ids = unw.motion_ids.repeat_interleave(N_FUTURE_FRAMES)                    # (N*10,)
            res = unw.motion_lib.get_motion_state(ids, times)

            # Reference joints: env27 order → SONIC-IsaacLab 29 (zeros at waist roll/pitch).
            dof27_env = res["dof_pos"].reshape(N, N_FUTURE_FRAMES, -1)                 # (N, 10, 27)
            sonic27 = dof27_env[..., self._inv_perm27]                                 # (N, 10, 27)
            pos29 = torch.zeros(N, N_FUTURE_FRAMES, N_BODY_JOINTS, device=self._dev)
            pos29[..., self._keep_idx] = sonic27

            # Central-difference velocities over the window (np.gradient equivalent).
            vel29 = torch.zeros_like(pos29)
            vel29[:, 1:-1] = (pos29[:, 2:] - pos29[:, :-2]) / (2.0 * FUTURE_DT)
            vel29[:, 0] = (pos29[:, 1] - pos29[:, 0]) / FUTURE_DT
            vel29[:, -1] = (pos29[:, -1] - pos29[:, -2]) / FUTURE_DT

            # Per-frame anchor rot6d: (R_robot_now)^-1 · R_ref_root(t+k·0.1), row-major
            # flatten of the first two matrix COLUMNS (eval: _R_rel_k[:, :2].flatten("C")).
            robot_quat = unw.scene["robot"].data.root_quat_w                            # (N, 4) wxyz
            robot_quat_rep = robot_quat.repeat_interleave(N_FUTURE_FRAMES, dim=0)       # (N*10, 4)
            ref_quat = res["root_rot"]                                                  # (N*10, 4) wxyz
            rel = math_utils.quat_mul(math_utils.quat_conjugate(robot_quat_rep), ref_quat)
            mat = math_utils.matrix_from_quat(rel)                                      # (N*10, 3, 3)
            rot6d = mat[:, :, :2].reshape(N, N_FUTURE_FRAMES, 6)                        # row-major ✓

            # Assemble the (N, 1762) encoder input. G1 mode id = 0 → mode slot stays zero.
            buf = torch.zeros(N, ENCODER_TOTAL_DIM, device=self._dev, dtype=torch.float32)
            buf[:, self._sl_pos] = pos29.reshape(N, -1)
            buf[:, self._sl_vel] = vel29.reshape(N, -1)
            buf[:, self._sl_anchor] = rot6d.reshape(N, -1)

            out = self.encoder(buf)
            if isinstance(out, (tuple, list)):
                out = out[0]
            self._base_token = out.reshape(N, TOKEN_TOTAL_DIM).to(torch.float32)

            # ---- one-shot self-parity diagnostic (env 0, first call) ----
            # At reset the robot is initialized AT the reference pose, so frame-0 of the
            # encoder input must match the robot's live state. Mismatches localize bugs:
            #   pos_err large (~rad)      → joint-order mapping wrong (perm/scatter)
            #   anchor far from identity  → frame/quat convention wrong (rot6d of
            #                                R_robot⁻¹·R_ref(now) ≈ I → [1,0,0,1,0,0]
            #                                row-major at reset)
            #   token degenerate (~const) → encoder input layout/batching wrong
            if not getattr(self, "_parity_printed", False):
                self._parity_printed = True
                q_sonic_live = self._gather_sonic(unw.scene["robot"].data.joint_pos)   # (N, 29)
                pos_err = (pos29[0, 0] - q_sonic_live[0]).abs()
                ident_rot6d = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=self._dev)
                anchor_err = (rot6d[0, 0] - ident_rot6d).abs().max()
                print("[token-adapter PARITY @ first call, env 0]")
                print(f"  max|enc_pos[frame0] - robot_joint_pos(sonic)| = {pos_err.max().item():.4f} rad "
                      f"(expect <~0.05 at reset; >0.5 ⇒ joint-order bug)")
                print(f"  worst joints (sonic idx): {pos_err.topk(3).indices.tolist()} "
                      f"errs {pos_err.topk(3).values.tolist()}")
                print(f"  max|anchor_rot6d[frame0] - identity(row-major)| = {anchor_err.item():.4f} "
                      f"(expect <~0.1 at reset; ~1+ ⇒ frame/flatten bug)")
                print(f"  vel[frame0] L2 = {vel29[0, 0].norm().item():.4f} (expect small at reset)")
                print(f"  base_token[0][:8] = {[round(v, 3) for v in self._base_token[0, :8].tolist()]}")
                print(f"  base_token stats: min={self._base_token.min().item():.3f} "
                      f"max={self._base_token.max().item():.3f} "
                      f"std={self._base_token.std().item():.3f} (degenerate ≈ 0-std ⇒ input layout bug)")

    def _append_base_token(self, obs):
        """Append the (fresh) base token to the policy obs group."""
        obs["policy"] = torch.cat([obs["policy"], self._base_token], dim=-1)
        return obs

    # ---------- vec-env API ----------

    def get_observations(self):
        out = super().get_observations()
        if isinstance(out, tuple):
            obs, extras = out
            return self._append_base_token(obs), extras
        return self._append_base_token(out)

    def reset(self):
        obs, extras = super().reset()
        self._update_base_token()
        return self._append_base_token(obs), extras

    def step(self, latent: torch.Tensor):
        latent = latent.to(self._dev)
        # Residual composition: hard tanh bound keeps the token within
        # residual_scale of the frozen base — structural anti-forgetting anchor.
        residual = self.residual_scale * torch.tanh(latent[:, :TOKEN_TOTAL_DIM])
        body = self._base_token + residual
        composed = torch.cat([body, latent[:, TOKEN_TOTAL_DIM:]], dim=1)

        obs, rew, dones, extras = super().step(composed)

        # Recompute the base token for the post-step state (episode_length_buf has
        # advanced; reset envs were re-seeded by the parent) so the obs the policy
        # sees next step carries the matching lookahead token.
        self._update_base_token()
        return self._append_base_token(obs), rew, dones, extras
