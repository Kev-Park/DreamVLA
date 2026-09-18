"""SONIC residual-ADAPTER data collection with ego-view camera → HDF5.

The single collection script for this project (replaces the old collect_pick_cam.py).
Rolls out a trained adapter checkpoint (frozen SONIC encoder + learned residual + frozen
decoder) and records, per step: ego-view RGB (torso d435 camera), robot/object state,
teleop wrist+torso SE(3) poses, and the motion-library reference qpos — the payload the
GR00T VLA dataset converter consumes. HDF5 schema is identical to the old producer
(RolloutRecorder), so the downstream converter is unchanged.

Key behaviors:
  * Adapter pipeline: policy outputs a 64-D token RESIDUAL + 1-D finger scalar, routed
    through TokenAdapterVecEnvWrapper (frozen-encoder base token + residual → FSQ → frozen
    decoder → env action). Requires --residual-scale matching training.
  * Checkpoint: --checkpoint-path, else auto-selects the newest model_*.pt under the newest
    dated run in logs/rsl_rl/g1_sonic_adapter.
  * SUCCESS FILTERING (physical hold, reference-independent): a trajectory is written only if
    the object rose >= --phys-lift above its rest height while within --phys-radius of the SIM
    right palm for >= --phys-steps consecutive steps (the eval_sonic_adapter.py [PHYS] criterion),
    and the episode did NOT end by a failure termination (time_out = reference exhausted is fine).
    This is exactly the eval's headline success; a toppled-but-held object is a success there too,
    so it is written here too (--reject-topple opts into the stricter held-upright filter, which
    the eval reports as "[PHYS] held-upright"). The collector sweeps every motion once
    (deterministic policy) and writes up to --num-samples successes.
  * Native SONIC .pt: pass --sonic-pt (and --residual-transform/--residual-scale/--encoder-mode
    exactly as trained). The recorded motion_token is the EXECUTED FSQ-snapped token (base + residual)
    in that model's latent space -- the GR00T flow-matching target.
  * --skip-start-frames N: episodes begin N frames into the motion (skip the refinement's
    20-frame interpolate-to-initial-pose prepend); the recorded trajectory starts there.
    GR00T SFT windows frames independently, so a mid-motion start is safe for fine-tuning.
  * Default env is Isaac-Motion-Tracking-Pick-Cam-HOI-v0: the HOI training env (same physics,
    obs, actions, rewards, terminations, mustard object) plus the HQ-kitchen visual backdrop and the
    torso d435 ego camera (1280x960, downsized to 640x480 by the converter).
"""

from __future__ import annotations

import argparse

from vla_sonic.repo_paths import gear_sonic_deploy  # sibling-repo ONNX defaults (worktree-safe)
import builtins
import os
import time
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

print = partial(builtins.print, flush=True)


# =========================================================================
# Seeding / misc helpers (inlined from the former collect_pick_cam.py)
# =========================================================================

def _seed_for_rollout(base_seed: int, object_index: int, motion_index: int, rollout_index: int, worker_index: int = 0) -> int:
    return int(base_seed + worker_index * 1_000_000_000 + object_index * 1_000_000 + motion_index * 1_000 + rollout_index)


def _set_all_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _frame_to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    frame = np.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=0.0)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.float32)
        if np.max(frame) > 1.5:
            frame = frame / 255.0
        frame = np.clip(frame, 0.0, 1.0)
        frame = np.power(frame, 1.0 / 2.2)
        frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
    return frame


def _stack_nested(values: list[Any]) -> Any:
    first_value = values[0]
    if isinstance(first_value, dict):
        return {key: _stack_nested([value[key] for value in values]) for key in first_value}
    return np.stack([np.asarray(value) for value in values], axis=0)


# =========================================================================
# Cached env-introspection helpers (inlined from collect_pick_cam.py)
# =========================================================================

def _get_wrist_body_indices(env) -> tuple[int, int, str, str]:
    cache_attr = "_collect_wrist_body_indices"
    cached = getattr(env.unwrapped, cache_attr, None)
    if cached is not None:
        return cached
    robot = env.unwrapped.scene["robot"]
    left_ids, left_names = robot.find_bodies(["left_wrist_yaw_link"])
    right_ids, right_names = robot.find_bodies(["right_wrist_yaw_link"])
    if len(left_ids) != 1 or len(right_ids) != 1:
        raise RuntimeError(f"wrist link lookup failed: left={left_names} right={right_names}")
    resolved = (int(left_ids[0]), int(right_ids[0]), "left_wrist_yaw_link", "right_wrist_yaw_link")
    setattr(env.unwrapped, cache_attr, resolved)
    return resolved


def _get_finger_joint_names(env) -> tuple[list[str], list[str]]:
    cache_attr = "_collect_finger_joint_names"
    cached = getattr(env.unwrapped, cache_attr, None)
    if cached is not None:
        return cached
    robot = env.unwrapped.scene["robot"]
    _, left_names = robot.find_joints(["left_hand.*"])
    _, right_names = robot.find_joints(["right_hand.*"])
    resolved = (list(left_names), list(right_names))
    setattr(env.unwrapped, cache_attr, resolved)
    return resolved


def _get_torso_body_index(env) -> tuple[int, str]:
    cache_attr = "_collect_torso_body_index"
    cached = getattr(env.unwrapped, cache_attr, None)
    if cached is not None:
        return cached
    robot = env.unwrapped.scene["robot"]
    ids, names = robot.find_bodies(["torso_link"])
    if len(ids) != 1:
        raise RuntimeError(f"torso_link lookup failed: {names}")
    resolved = (int(ids[0]), "torso_link")
    setattr(env.unwrapped, cache_attr, resolved)
    return resolved


def _get_hand_joint_indices(env) -> tuple[list[int], list[int]]:
    cache_attr = "_collect_hand_joint_indices"
    cached = getattr(env.unwrapped, cache_attr, None)
    if cached is not None:
        return cached
    robot = env.unwrapped.scene["robot"]
    left_ids, _ = robot.find_joints(["left_hand.*"])
    right_ids, _ = robot.find_joints(["right_hand.*"])
    resolved = (list(left_ids), list(right_ids))
    setattr(env.unwrapped, cache_attr, resolved)
    return resolved


def _build_env_args(env, *, task_name: str | None) -> dict[str, Any]:
    cache_attr = "_collect_env_args"
    cached = getattr(env.unwrapped, cache_attr, None)
    if cached is not None:
        return cached
    robot = env.unwrapped.scene["robot"]
    joint_names = list(robot.data.joint_names)
    body_names = list(robot.data.body_names)
    left_finger_names, right_finger_names = _get_finger_joint_names(env)
    num_finger = len(left_finger_names) + len(right_finger_names)
    step_dt = float(env.unwrapped.step_dt) if env.unwrapped.step_dt else 0.0
    fps = int(round(1.0 / step_dt)) if step_dt > 0 else 0
    usd_path = ""
    try:
        usd_path = str(getattr(env.unwrapped.cfg.scene.robot.spawn, "usd_path", "") or "")
    except Exception:
        usd_path = ""
    ref_joint_names: list[str] = []
    if hasattr(env.unwrapped, "motion_lib") and hasattr(env.unwrapped.motion_lib, "joint_names"):
        ref_joint_names = list(env.unwrapped.motion_lib.joint_names)
    env_args = {
        "robot_name": "unitree_g1_27dof_dex3",
        "robot_usd": Path(usd_path).name if usd_path else "",
        "task_name": task_name or "",
        "fps": fps,
        "step_dt": step_dt,
        "num_joints": len(joint_names),
        "num_body_joints": max(len(joint_names) - num_finger, 0),
        "num_finger_joints": num_finger,
        "joint_names": joint_names,
        "body_names": body_names,
        "left_finger_joint_names": list(left_finger_names),
        "right_finger_joint_names": list(right_finger_names),
        "ref_joint_names": ref_joint_names,
        "producer": "collect_sonic_adapter.py",
        "producer_version": 2,  # same v2 RolloutRecorder HDF5 schema as the old producer
    }
    setattr(env.unwrapped, cache_attr, env_args)
    return env_args


# =========================================================================
# Per-step capture (inlined from collect_pick_cam.py)
# =========================================================================

def _capture_teleop_frame(env) -> dict[str, torch.Tensor]:
    import isaaclab.utils.math as math_utils
    left_idx, right_idx, _, _ = _get_wrist_body_indices(env)
    torso_idx, _ = _get_torso_body_index(env)
    robot = env.unwrapped.scene["robot"]
    root_pos_w = robot.data.root_pos_w[0:1]
    root_quat_w = robot.data.root_quat_w[0:1]
    left_pos_w = robot.data.body_pos_w[0:1, left_idx]
    left_quat_w = robot.data.body_quat_w[0:1, left_idx]
    right_pos_w = robot.data.body_pos_w[0:1, right_idx]
    right_quat_w = robot.data.body_quat_w[0:1, right_idx]
    torso_pos_w = robot.data.body_pos_w[0:1, torso_idx]
    torso_quat_w = robot.data.body_quat_w[0:1, torso_idx]
    left_pos_p, left_quat_p = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, left_pos_w, left_quat_w)
    right_pos_p, right_quat_p = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, right_pos_w, right_quat_w)
    torso_pos_p, torso_quat_p = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, torso_pos_w, torso_quat_w)
    left_R = math_utils.matrix_from_quat(left_quat_p)[0].to(torch.float64).cpu()
    right_R = math_utils.matrix_from_quat(right_quat_p)[0].to(torch.float64).cpu()
    torso_R = math_utils.matrix_from_quat(torso_quat_p)[0].to(torch.float64).cpu()
    left_t = left_pos_p[0].to(torch.float64).cpu()
    right_t = right_pos_p[0].to(torch.float64).cpu()
    torso_t = torso_pos_p[0].to(torch.float64).cpu()

    def _se3(R_mat, t_vec):
        T = torch.eye(4, dtype=torch.float64)
        T[:3, :3] = R_mat
        T[:3, 3] = t_vec
        return T

    return {"left_wrist": _se3(left_R, left_t), "right_wrist": _se3(right_R, right_t), "torso_pose": _se3(torso_R, torso_t)}


def _capture_reference_motion(env) -> dict[str, torch.Tensor] | None:
    u = env.unwrapped
    if not hasattr(u, "motion_lib") or not hasattr(u, "motion_ids"):
        return None
    motion_times = (
        u.episode_length_buf.float() * float(u.step_dt)
        + u.start_motion_times.to(u.device, dtype=torch.float32)
    )
    motion_res = u.motion_lib.get_motion_state(u.motion_ids, motion_times)
    return {
        "root_pos_w": motion_res["root_pos"][0].detach().cpu(),
        "root_quat_w": motion_res["root_rot"][0].detach().cpu(),
        "dof_pos": motion_res["dof_pos"][0].detach().cpu(),
    }


def _capture_rollout_state(env, action: torch.Tensor | None = None) -> dict[str, Any]:
    robot = env.unwrapped.scene["robot"]
    object_asset = env.unwrapped.scene["object"]
    left_hand_joint_ids, right_hand_joint_ids = _get_hand_joint_indices(env)
    joint_pos = robot.data.joint_pos[0]
    state: dict[str, Any] = {
        "robot": {
            "root_pos_w": robot.data.root_pos_w[0].detach().cpu(),
            "root_quat_w": robot.data.root_quat_w[0].detach().cpu(),
            "joint_pos": robot.data.joint_pos[0].detach().cpu(),
            "joint_vel": robot.data.joint_vel[0].detach().cpu(),
            "left_finger_joint_pos": joint_pos[left_hand_joint_ids].detach().cpu(),
            "right_finger_joint_pos": joint_pos[right_hand_joint_ids].detach().cpu(),
        },
        "object": {
            "root_pos_w": object_asset.data.root_pos_w[0].detach().cpu(),
            "root_quat_w": object_asset.data.root_quat_w[0].detach().cpu(),
        },
    }
    if hasattr(env.unwrapped.scene, "env_origins"):
        state["robot"]["env_origin"] = env.unwrapped.scene.env_origins[0].detach().cpu()
    ref_motion = _capture_reference_motion(env)
    if ref_motion is not None:
        state["ref_motion"] = ref_motion
    if action is not None:
        state["action"] = action[0].detach().cpu()
    return state


# =========================================================================
# Adapter-specific: checkpoint discovery + rollout
# =========================================================================

def _find_latest_adapter_checkpoint(log_root: Path) -> str:
    if not log_root.exists():
        raise FileNotFoundError(f"Adapter log root not found: {log_root}. Train first or pass --checkpoint-path.")
    run_dirs = sorted([d for d in log_root.iterdir() if d.is_dir()], key=lambda d: d.name)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories under {log_root}.")
    newest_run = run_dirs[-1]  # YYYY-MM-DD_HH-MM-SS → lexicographic == chronological
    ckpts = sorted(newest_run.glob("model_*.pt"),
                   key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or "0"))
    if not ckpts:
        raise FileNotFoundError(f"No model_*.pt in newest run {newest_run}.")
    print(f"[collect_sonic_adapter] auto-selected newest checkpoint: {ckpts[-1]}")
    return str(ckpts[-1])


class DaggerDriver:
    """DAgger relabelling: a VLA drives the robot (with probability 1-beta per re-plan chunk; the
    residual expert drives otherwise) while the RESIDUAL supplies the label for EVERY visited state.

    Per step the collector computes the expert's latent a_exp = residual(obs) and records
      * obs/motion_token           = env.expert_token(a_exp): the FSQ token the expert WOULD execute
                                     at this state (not the token that was actually executed),
      * actions                    = a_exp (65-D: residual latent + finger scalar),
      * teleop/finger_joints/right = the expert's finger COMMAND as a joint pose: the demo set's mean
                                     measured closed-on-bottle pose when a_exp[64] < 0, zeros otherwise
                                     (the measured fingers stay open while the VLA drives, so they
                                     cannot serve as the label here).
    The env's own terminations end a rollout (ee_body_pos = the expert's 0.25 m validity gate);
    rollouts are kept regardless of task success (they are supposed to contain failures).
    """

    def __init__(self, env, vla_policy, obs_adapter, *, beta: float, chunk: int, close_thres: float, seed: int):
        self.env = env; self.vla = vla_policy; self.obs_adapter = obs_adapter
        self.beta = float(beta); self.chunk = int(chunk); self.close_thres = float(close_thres)
        self.rng = np.random.default_rng(seed)
        self.begin_episode()

    def begin_episode(self):
        self._chunk = None; self._chunk_step = self.chunk; self._expert_drives = True
        self.n_expert = 0; self.n_vla = 0

    def step(self, expert_latent: torch.Tensor):
        """Advance the env one step; returns (obs, rew, dones, extras, expert_drove: bool)."""
        if self._chunk_step >= self.chunk:                               # re-plan boundary
            self._expert_drives = bool(self.rng.random() < self.beta)
            self._chunk_step = 0
            if not self._expert_drives:
                out = self.vla.get_action(self.obs_adapter())
                self._chunk = out[0] if isinstance(out, tuple) else out
        t = self._chunk_step; self._chunk_step += 1
        if self._expert_drives:
            self.n_expert += 1
            return (*self.env.step(expert_latent), True)
        tok = np.asarray(self._chunk["motion_token"], np.float32)[0, t]
        rh = np.asarray(self._chunk["right_hand_joints"], np.float32)[0, t]
        composed = torch.zeros((1, 65), device=expert_latent.device, dtype=torch.float32)
        composed[0, :64] = torch.as_tensor(tok, device=expert_latent.device)
        composed[0, 64] = -1.0 if float(np.abs(rh).mean()) > self.close_thres else 1.0
        self.n_vla += 1
        return (*self.env.step_composed(composed), False)


def dagger_closed_pose(demo_root: str, max_files: int = 8) -> np.ndarray:
    """Mean measured right-finger pose (collector joint order) over closed-command frames of the demos."""
    import h5py
    acc, n = None, 0
    for f in sorted(Path(demo_root).expanduser().rglob("*.hdf5"))[:max_files]:
        with h5py.File(f, "r") as h:
            g = h["data/demo_0"]
            closed = g["actions"][()][:, 64] < 0
            if closed.any():
                fr = g["teleop/finger_joints/right"][()][closed]
                acc = fr.sum(0) if acc is None else acc + fr.sum(0); n += len(fr)
    if not n:
        raise RuntimeError(f"no closed frames found under {demo_root}")
    return (acc / n).astype(np.float64)


def _run_rollout_adapter(env, policy, *, simulation_app, max_steps, state_on, real_time,
                         reset_at_start, lift_thres, phys_lift=0.05, phys_radius=0.15, phys_steps=25,
                         dagger: "DaggerDriver | None" = None, dagger_closed_pose_arr: np.ndarray | None = None):
    camera_frames: list[np.ndarray] = []
    state_history: list[dict[str, Any]] = []
    teleop_history: list[dict[str, torch.Tensor]] = []
    # Executed SONIC token per frame (base + learned residual, FSQ-snapped), read from
    # the adapter wrapper AFTER each step. This is the exact decoder input that produced
    # the recorded motion → the correct VLA supervision target, stored directly instead
    # of being re-derived offline (which would recover only the un-adapted base token).
    token_history: list[np.ndarray] = []
    dagger_mask: list[bool] = []                                     # expert drove this step?
    if dagger is not None:
        dagger.begin_episode()
    # Eval-parity success: a frame is a real pickup when the object (bottle) clears lift_thres
    # AND the reference motion is in its closed/grasp phase. Mirrors eval_sonic_adapter.py
    # (bottle_z > lift_thres & is_closed), evaluated POST-step.
    had_any_lift = False
    # reset_object_state drops the object from z=1.0 each episode; it settles to its ~0.9
    # rest over the first several frames, so it STARTS above the 0.95 lift threshold. Only
    # count a lift once the object has been observed at/below the threshold at least once
    # (i.e., it reached rest) — otherwise the reset-drop transient is mistaken for a pickup
    # (the bug that let no-lift trajectories through, esp. with --skip-start near the grab).
    object_settled = False
    # PHYSICAL-HOLD success (authoritative; same definition as eval_sonic_adapter.py [PHYS]):
    # object risen >= phys_lift above its rest height AND within phys_radius of the SIM right
    # palm (wrist_yaw + 0.12 m along the hand x-axis) for >= phys_steps consecutive steps.
    # Reference-independent: it cannot be satisfied by tracking the plan without the bottle.
    _robot = env.unwrapped.scene["robot"]
    _rw_bid = _robot.find_bodies("right_wrist_yaw_link")[0][0]
    obj_rest_z: float | None = None
    phys_run = 0
    phys_held = False
    max_lift = 0.0
    toppled_any = False
    term_reason = "none"

    # The explicit reset must run inside inference_mode: after a prior rollout's
    # inference_mode policy/step, the env's persistent buffers (joint_acc, etc.) are
    # inference tensors, and the reset events do in-place writes to them
    # (write_joint_state_to_sim → joint_acc[...] = 0). Updating an inference tensor in-place
    # OUTSIDE inference_mode raises RuntimeError, so wrap the reset to match.
    if reset_at_start:
        with torch.inference_mode():
            obs_out = env.reset()
    else:
        obs_out = env.get_observations()
    obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out

    # Flush the RTX render pipeline after the reset so the FIRST camera read below
    # reflects the post-reset pose with a converged (not warm-up) render. Without
    # these pumps every captured frame is the noisy first-bounce warm-up render —
    # see play_sonic_adapter.py, which does the identical flush after each step.
    simulation_app.update()
    simulation_app.update()

    if not hasattr(env.unwrapped, "n_successes"):
        env.unwrapped.n_successes = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.long)
    n_successes_start = env.unwrapped.n_successes.detach().clone()

    if "camera_robot" not in list(env.unwrapped.scene.keys()):
        raise RuntimeError(f"ego camera 'camera_robot' missing. Found: {list(env.unwrapped.scene.keys())}")
    cam_robot = env.unwrapped.scene["camera_robot"]

    terminated_flag = truncated_flag = False
    step_index = 0
    if not simulation_app.is_running():
        raise RuntimeError("Simulation app is not running at rollout start.")

    while step_index < max_steps and simulation_app.is_running():
        start_time = time.time()
        step_index += 1
        with torch.inference_mode():
            actions = policy(obs).clone()
            expert_tok = env.expert_token(actions)[0].cpu().numpy().astype(np.float64) if dagger is not None else None
        camera_output = getattr(cam_robot.data, "output", None)
        if camera_output is None or "rgb" not in camera_output:
            raise RuntimeError("camera_robot.data.output missing 'rgb'.")
        camera_frames.append(_frame_to_uint8_rgb(camera_output["rgb"][0].cpu().numpy()))
        if state_on:
            st = _capture_rollout_state(env, actions)
            if dagger is not None:
                # label = the expert's finger COMMAND as a pose (measured fingers follow the driver, not the expert)
                closed = bool(actions[0, 64].item() < 0)
                st["robot"]["right_finger_joint_pos"] = torch.as_tensor(
                    dagger_closed_pose_arr if closed else np.zeros_like(dagger_closed_pose_arr), dtype=torch.float32)
            state_history.append(st)
            teleop_history.append(_capture_teleop_frame(env))
        with torch.inference_mode():
            if dagger is not None:
                *step_result, expert_drove = dagger.step(actions)
                step_result = tuple(step_result); dagger_mask.append(expert_drove)
            else:
                step_result = env.step(actions)
        if len(step_result) == 5:
            obs, _, terminated, truncated, _ = step_result
        else:
            obs, _, done, _ = step_result
            terminated = done
            truncated = torch.zeros_like(done)
        obs = obs.clone() if hasattr(obs, "clone") else obs
        # Record the executed token for THIS frame (computed inside env.step from the
        # actions applied above). Pairs index-for-index with camera_frames/state_history
        # since all three append exactly once per iteration before the break check.
        if state_on:
            if dagger is not None:
                token_history.append(expert_tok)                     # DAgger label: the expert's token
            else:
                last_token = getattr(env, "_last_token", None)
                if last_token is not None:
                    token_history.append(last_token[0].detach().cpu().numpy().astype(np.float64))
        # Eval-parity lift check — POST-step, matching eval_sonic_adapter.py
        # (bottle_z > lift_thres while the reference is_closed). The settle gate
        # (object_settled) ignores the reset-drop transient: the object is reset to z=1.0
        # and only its genuine rise back above the threshold AFTER reaching rest counts.
        # On a terminal step the env auto-resets (episode_length_buf→0 ⇒ is_closed=0), so
        # the re-clobbered pose cannot register — same immunity eval gets.
        u = env.unwrapped
        if hasattr(u, "motion_lib") and hasattr(u, "motion_ids"):
            bottle_z = float(u.scene["object"].data.root_pos_w[0, 2].item())
            motion_times = (
                u.episode_length_buf.float() * float(u.step_dt)
                + u.start_motion_times.to(u.device, dtype=torch.float32)
            )
            is_closed = bool(
                u.motion_lib.get_motion_state(u.motion_ids, motion_times)["is_closed"][0].item()
            )
            if bottle_z <= lift_thres:
                object_settled = True
            if object_settled and bottle_z > lift_thres and is_closed:
                had_any_lift = True
            # ---- physical hold / topple bookkeeping (env-local frame) ----
            _org = u.scene.env_origins[0]
            _obj_p = u.scene["object"].data.root_pos_w[0] - _org
            _hq = _robot.data.body_quat_w[0, _rw_bid]                                  # wxyz
            _w, _x, _y, _z = _hq[0], _hq[1], _hq[2], _hq[3]
            _hand_x = torch.stack([1 - 2 * (_y * _y + _z * _z), 2 * (_x * _y + _w * _z), 2 * (_x * _z - _w * _y)])
            _palm = (_robot.data.body_pos_w[0, _rw_bid] - _org) + 0.12 * _hand_x
            # Rest height = running MIN over the first 50 steps (the object is reset above its
            # rest and settles over the first ~10 steps); frozen afterwards.
            if obj_rest_z is None or step_index <= 50:
                obj_rest_z = float(_obj_p[2].item()) if obj_rest_z is None else min(obj_rest_z, float(_obj_p[2].item()))
            _dz = float(_obj_p[2].item()) - obj_rest_z
            max_lift = max(max_lift, _dz)
            _near = bool(torch.norm(_palm - _obj_p).item() < phys_radius)
            phys_run = phys_run + 1 if (_near and _dz >= phys_lift) else 0
            if phys_run >= phys_steps:
                phys_held = True
            _oq = u.scene["object"].data.root_quat_w[0]
            _up_z = float((1.0 - 2.0 * (_oq[1] ** 2 + _oq[2] ** 2)).item())
            if _up_z < 0.7071:                                                         # tilt > 45 deg
                toppled_any = True
        # Flush the RTX render pipeline so the NEXT iteration's camera read delivers
        # this step's frame (the camera annotator otherwise lags / stays on the warm-up
        # render). Two pumps match the proven play_sonic_adapter.py cadence.
        simulation_app.update()
        simulation_app.update()
        terminated_flag = bool(torch.as_tensor(terminated).any().item())
        truncated_flag = bool(torch.as_tensor(truncated).any().item())
        if terminated_flag or truncated_flag:
            # Which termination term ended the episode (read BEFORE the next step; the manager
            # keeps the per-term flags of the step that terminated). time_out = the reference
            # ran out = a completed episode, NOT a failure.
            try:
                _tm = env.unwrapped.termination_manager
                _fired = [n for n in _tm.active_terms if bool(_tm.get_term(n)[0].item())]
                term_reason = ",".join(_fired) if _fired else "unknown"
            except Exception as _e:
                term_reason = f"unavailable({type(_e).__name__})"
            print(f"[INFO] rollout ended at step {step_index}/{max_steps} "
                  f"(terminated={terminated_flag}, truncated={truncated_flag})")
            break
        sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
        if real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if step_index == 0:
        raise RuntimeError("No rollout steps executed before SimulationApp stopped.")
    raw_state = _stack_nested(state_history) if state_history else None

    # Attach the executed-token sequence as a top-level field so the recorder writes
    # obs/motion_token. Guard the frame count: if (rarely) it diverges from the state
    # count, drop it rather than write a misaligned column (converter then re-derives).
    if raw_state is not None and token_history:
        n_state = int(np.asarray(raw_state["robot"]["joint_pos"]).shape[0])
        if len(token_history) == n_state:
            raw_state["motion_token"] = np.stack(token_history, axis=0)  # (N, 64)
            if dagger is not None and len(dagger_mask) == n_state:
                raw_state["dagger_expert_mask"] = np.asarray(dagger_mask, dtype=np.uint8)
        else:
            print(f"[WARN] token_history ({len(token_history)}) != state frames ({n_state}) "
                  "— NOT writing obs/motion_token; converter will re-derive instead.")

    teleop_payload: dict[str, Any] | None = None
    if teleop_history:
        ts = _stack_nested(teleop_history)
        left_wrist_np = np.asarray(ts["left_wrist"], dtype=np.float64)
        num_frames = left_wrist_np.shape[0]
        step_dt = float(env.unwrapped.step_dt)
        left_finger_names, right_finger_names = _get_finger_joint_names(env)
        teleop_payload = {
            "left_wrist": left_wrist_np,
            "right_wrist": np.asarray(ts["right_wrist"], dtype=np.float64),
            "torso_pose": np.asarray(ts["torso_pose"], dtype=np.float64),
            "timestamps": np.arange(num_frames, dtype=np.float64) * step_dt,
            "step_dt": step_dt,
            "source_robot": "unitree_g1_27dof_dex3",
            "left_body_name": "left_wrist_yaw_link",
            "right_body_name": "right_wrist_yaw_link",
            "torso_body_name": "torso_link",
            "finger_joints": {
                "left": np.asarray(raw_state["robot"]["left_finger_joint_pos"], dtype=np.float64),
                "right": np.asarray(raw_state["robot"]["right_finger_joint_pos"], dtype=np.float64),
                "left_names": left_finger_names,
                "right_names": right_finger_names,
            },
        }

    # Authoritative success = eval-parity lift criterion (bottle above lift_thres during the
    # grasp phase, on the recorded frames). n_successes is kept only as a diagnostic — it goes
    # through the env reward's counter, which the dataset screening showed is NOT a reliable
    # filter for the bottle, so it must NOT gate what gets written.
    n_successes_delta = env.unwrapped.n_successes - n_successes_start
    success_n_successes = bool((n_successes_delta > 0).any().item())
    rollout_success = phys_held
    # "Fallen" = ended by a FAILURE termination (anchor_pos / anchor_ori_full / ee_body_pos ...),
    # not by time_out (reference exhausted) and not by reaching --rollout-length.
    ended_early = (terminated_flag or truncated_flag) and step_index < max_steps
    error_terminated = ended_early and ("time_out" not in term_reason)
    metadata = {
        "terminated": terminated_flag,
        "truncated": truncated_flag,
        "termination_terms": term_reason,
        "error_terminated": error_terminated,
        "success": rollout_success,
        "success_criterion": (f"PHYS hold: lift>={phys_lift}m above rest & obj within {phys_radius}m of sim palm "
                              f"for >={phys_steps} consecutive steps (eval_sonic_adapter [PHYS] parity)"),
        "phys_held": bool(phys_held),
        "max_lift_m": float(max_lift),
        "object_rest_z": float(obj_rest_z) if obj_rest_z is not None else float("nan"),
        "toppled_any": bool(toppled_any),
        "legacy_lift_success": bool(had_any_lift),
        "lift_thres": float(lift_thres),
        "success_n_successes": success_n_successes,
        "num_steps": len(camera_frames),
        "app_running": bool(simulation_app.is_running()),
        "camera_on": True,
        "state_on": state_on,
    }
    if dagger is not None:
        metadata["dagger"] = {"beta": dagger.beta, "chunk": dagger.chunk, "expert_steps": dagger.n_expert,
                              "vla_steps": dagger.n_vla, "labels": "expert token + expert finger command"}
    return camera_frames, raw_state, metadata, teleop_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect SONIC-adapter pick rollouts (ego camera → HDF5).")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=1, help="Only 1 supported (one file per rollout).")
    parser.add_argument("--task", type=str, default="Isaac-Motion-Tracking-Pick-Cam-HOI-v0",
                        help="Default: kitchen-visuals + ego camera ON the HOI pick env (identical physics, "
                             "obs, actions, rewards and terminations to Isaac-Motion-Tracking-Pick-HOI-v0, "
                             "the env the residual policies are trained in). Run with the same HS_REWORK "
                             "env vars as training.")
    parser.add_argument("--seed", type=int, default=0)
    state_group = parser.add_mutually_exclusive_group()
    state_group.add_argument("--state-on", dest="state_on", action="store_true")
    state_group.add_argument("--state-off", dest="state_on", action="store_false")
    parser.set_defaults(state_on=True)
    parser.add_argument("--real-time", action="store_true", default=False)
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Adapter checkpoint. If omitted, auto-selects newest model_*.pt under "
                             "logs/rsl_rl/g1_sonic_adapter/<newest run>.")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Target number of SUCCESSFUL trajectories to write (success = task grasp "
                             "per the eval criterion). The collector sweeps each motion in the library "
                             "ONCE (deterministic policy) and writes the successes, up to this many. "
                             "The ceiling is the number of motions that succeed.")
    parser.add_argument("--output-directory", type=str, default="./datasets/sonic_adapter")
    parser.add_argument("--rollout-length", type=int, default=500)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--sonic-decoder-onnx", type=str,
                        default=gear_sonic_deploy("policy/release/model_decoder.onnx"))
    parser.add_argument("--sonic-encoder-onnx", type=str,
                        default=gear_sonic_deploy("policy/release/model_encoder.onnx"))
    parser.add_argument("--sonic-pt", type=str, default=None,
                        help="Directory of a native SONIC .pt checkpoint (groot-era). REQUIRED for policies "
                             "trained with --sonic-pt: the recorded motion_token lives in THIS model's latent "
                             "space. Overrides the ONNX encoder/decoder.")
    parser.add_argument("--encoder-mode", type=str, default="g1", choices=["g1", "teleop"],
                        help="Native .pt only; must match training (g1 = full-body reference encoder).")
    parser.add_argument("--residual-scale", type=float, default=0.1,
                        help="MUST match the value train_sonic_adapter.py used (current recipe: 0.1).")
    parser.add_argument("--residual-transform", type=str, default="multiplicative_free",
                        choices=["additive", "multiplicative", "multiplicative_free", "unclamped"],
                        help="MUST match train_sonic_adapter.py (current recipe: multiplicative_free).")
    parser.add_argument("--phys-lift", type=float, default=0.02,
                        help="SUCCESS filter: object must rise >= this (m) above its rest height ... "
                             "(0.02 = eval_sonic_adapter.py's first-reported [PHYS] held line; it also reports 0.05)")
    parser.add_argument("--phys-radius", type=float, default=0.15,
                        help="... while within this distance (m) of the sim right palm ...")
    parser.add_argument("--phys-steps", type=int, default=25,
                        help="... for at least this many CONSECUTIVE steps (25 = 0.5 s). Same physical-hold "
                             "criterion as eval_sonic_adapter.py [PHYS].")
    parser.add_argument("--reject-topple", action="store_true", default=False,
                        help="Stricter filter: also reject trajectories in which the object toppled (>45 deg) "
                             "at any point (= eval's [PHYS] held-upright). Default: the eval's headline [PHYS] "
                             "held criterion, which keeps toppled-but-held grasps.")
    parser.add_argument("--lift-thres", type=float, default=0.95,
                        help="DIAGNOSTIC ONLY (legacy criterion, reported in metadata, no longer gates "
                             "writing): object root z (m) above which a frame counts as 'lifted' "
                             "during the reference grasp "
                             "(is_closed) phase. MUST match eval_sonic_adapter.py --lift-thres "
                             "(default 0.95 = object rests at 0.9, a 5 cm pickup). Only "
                             "trajectories with at least one lifted+closed frame are written.")
    parser.add_argument("--skip-start-frames", type=int, default=None,
                        help="Start episodes N frames into the motion (pass 20 to skip the refinement "
                             "prepend). The recorded trajectory begins at the skip frame.")
    parser.add_argument("--start-pregrab-margin", type=float, default=None,
                        help="Start episodes this many seconds before the grab (drops prepend + walk).")
    parser.add_argument("--dagger-vla-checkpoint", type=str, default=None,
                        help="DAgger mode: GR00T checkpoint that DRIVES the robot (with prob 1-beta per chunk) while "
                             "the residual labels every state. Rollouts are written regardless of success.")
    parser.add_argument("--dagger-demo-root", type=str, default=None,
                        help="DAgger: collector HDF5 root of the demonstrations -- sweeps only its motion ids and "
                             "takes the closed-finger label pose from it.")
    parser.add_argument("--dagger-beta", type=float, default=0.5, help="DAgger: P(expert drives) per re-plan chunk.")
    parser.add_argument("--dagger-chunk", type=int, default=8, help="DAgger: VLA re-plan cadence (eval parity).")
    parser.add_argument("--dagger-rollouts", type=int, default=2, help="DAgger: rollouts per demo motion.")
    parser.add_argument("--dagger-min-steps", type=int, default=50, help="DAgger: discard rollouts shorter than this.")
    parser.add_argument("--dagger-finger-close-thres", type=float, default=0.6,
                        help="DAgger: VLA right-hand mean|q| above which its close is executed (eval parity).")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="Resume: skip motion ids that already have an .hdf5 anywhere under --output-directory "
                             "(previously rejected motions are re-tried; the policy is deterministic).")
    parser.add_argument("--motion-range", type=int, nargs=2, default=None, metavar=("START", "END"),
                        help="Only sweep motion ids in [START, END) -- shard the deterministic sweep across "
                             "GPUs/processes (file names are per-motion, so shards can share --output-directory).")
    parser.add_argument("--ref-motions-path", type=str, default=None,
                        help="Override the env's ref_motions_path (dir of reference .pkl files) -- the SAME "
                             "reference set the checkpoint was trained on (e.g. ../TrajGen/sample/Holosoma_Pick_29_fixH60).")
    parser.add_argument("--waist-dof", type=int, default=29, choices=[27, 29],
                        help="Body DOF. 29 actuates waist_roll/pitch (29-DOF + dex-hands USD) to match "
                             "SONIC's training articulation. 27 = legacy welded-waist asset.")

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    _set_all_seeds(args_cli.seed)
    if args_cli.num_envs != 1:
        raise ValueError("collect_sonic_adapter.py supports num_envs=1 only.")
    target_successes = int(args_cli.num_samples)

    args_cli.enable_cameras = True
    args_cli.video = True  # keep viewport render product active (clean ego RTX frames)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg
    from rsl_rl.runners import OnPolicyRunner

    from vla_sonic.token_action_wrapper import load_frozen_decoder
    from vla_sonic.token_adapter_wrapper import TokenAdapterVecEnvWrapper, load_frozen_encoder
    from vla_sonic.sonic_pt import load_sonic_pt
    from vla_sonic.physics_overrides import apply_sonic_physics_overrides
    from vla_sonic.robot_29dof import apply_29dof_waist_override
    from vla_sonic.adapter_actor_critic import AdapterActorCritic
    import rsl_rl.modules
    import rsl_rl.runners.on_policy_runner as _rsl_rl_opr
    rsl_rl.modules.AdapterActorCritic = AdapterActorCritic
    _rsl_rl_opr.AdapterActorCritic = AdapterActorCritic

    from recorder import RolloutRecorder

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1,
                            use_fabric=not args_cli.disable_fabric, enable_cameras=True)
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args_cli.seed
    apply_sonic_physics_overrides(env_cfg)
    if args_cli.ref_motions_path is not None:
        env_cfg.ref_motions_path = args_cli.ref_motions_path
        print(f"[collect_sonic_adapter] ref_motions_path override -> {args_cli.ref_motions_path}")

    # 29-DOF strict-fidelity articulation (actuated waist roll/pitch) to match SONIC training.
    if args_cli.waist_dof == 29:
        apply_29dof_waist_override(env_cfg)

    if args_cli.start_pregrab_margin is not None:
        env_cfg.motion_start_pregrab_margin_s = args_cli.start_pregrab_margin
        print(f"[collect_sonic_adapter] start_pregrab_margin = {args_cli.start_pregrab_margin}s")
    if args_cli.skip_start_frames is not None:
        # +10: reset 10 frames LATER so the decoder-history seed (the 10 frames PRECEDING the
        # reset) lands on REAL motion. Recorded data starts at frame skip+10; frames skip..skip+9
        # are decoder-history context only (not recorded).
        env_cfg.motion_skip_start_frames = args_cli.skip_start_frames + 10
        print(f"[collect_sonic_adapter] skip_start_frames = {args_cli.skip_start_frames} "
              f"(+10 history warmup -> recorded data starts at frame {args_cli.skip_start_frames + 10})")

    # Inject the ego camera if the env doesn't already define it (green-box env).
    if not hasattr(env_cfg.scene, "camera_robot"):
        env_cfg.scene.camera_robot = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/Camera_robot",
            spawn=PinholeCameraCfg(focal_length=7.6, focus_distance=400.0,
                                   horizontal_aperture=20.0, clipping_range=(0.01, 100.0)),
            data_types=["rgb"], height=int(args_cli.image_height), width=int(args_cli.image_width),
            offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.36),
                                       rot=(0.568, 0.421, -0.421, -0.568), convention="opengl"),
        )

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.experiment_name = f"{agent_cfg.experiment_name}_sonic_adapter"
    agent_cfg.policy.class_name = "AdapterActorCritic"
    agent_cfg.policy.actor_hidden_dims = [256, 128]
    agent_cfg.policy.critic_hidden_dims = [512, 256, 256]

    if args_cli.checkpoint_path is not None:
        resume_path = args_cli.checkpoint_path
    else:
        log_root = Path(os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)))
        resume_path = _find_latest_adapter_checkpoint(log_root)
    print(f"[collect_sonic_adapter] checkpoint: {resume_path}")

    # render_mode="rgb_array" (matching play_sonic_adapter.py) so the RTX render
    # product is active and the camera annotators receive flushed frames.
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Pump the Omniverse event loop once at startup so RTX shaders/materials and
    # texture streaming finish loading before the first rollout. Without this the
    # opening frames render against partially-loaded assets (the "scene not loaded"
    # look). Mirrors the 60-pump warm-up in play_sonic_adapter.py.
    print("[collect_sonic_adapter] pumping Omniverse event loop to init render/materials...")
    for _ in range(60):
        simulation_app.update()

    device = agent_cfg.device
    if args_cli.sonic_pt:
        encoder, decoder = load_sonic_pt(args_cli.sonic_pt, device, encoder=args_cli.encoder_mode)
    else:
        decoder = load_frozen_decoder(args_cli.sonic_decoder_onnx, device)
        encoder = load_frozen_encoder(args_cli.sonic_encoder_onnx, device)
    env = TokenAdapterVecEnvWrapper(env, decoder, encoder, device,
                                    residual_scale=args_cli.residual_scale,
                                    residual_transform=args_cli.residual_transform,
                                    clip_actions=None,
                                    pt_mode=bool(args_cli.sonic_pt),
                                    encoder_mode=args_cli.encoder_mode)
    print(f"[collect_sonic_adapter] RECIPE: sonic_pt={args_cli.sonic_pt or 'ONNX v1.0'} encoder_mode={args_cli.encoder_mode} "
          f"residual_transform={args_cli.residual_transform} residual_scale={args_cli.residual_scale} "
          f"waist_dof={args_cli.waist_dof} task={args_cli.task} -- these MUST match the checkpoint's training run.")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dagger = None; dagger_closed = None; dagger_motions = None
    if args_cli.dagger_vla_checkpoint:
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from vla_sonic.obs_to_policy import ObsAdapterConfig, ObsToPolicyAdapter
        from vla_sonic.simple_robot_model import SimpleG1RobotModel
        from vla_sonic.eval_helpers import demo_motion_ids
        if not args_cli.dagger_demo_root:
            raise SystemExit("--dagger-demo-root is required in DAgger mode")
        dagger_motions = demo_motion_ids(args_cli.dagger_demo_root)
        dagger_closed = dagger_closed_pose(args_cli.dagger_demo_root)
        vla = Gr00tPolicy(embodiment_tag="unitree_g1_sonic", model_path=args_cli.dagger_vla_checkpoint, device=device)
        obs_adapter = ObsToPolicyAdapter(env, ObsAdapterConfig(language_instruction="pick up the mustard bottle",
                                                               robot_model=SimpleG1RobotModel.build(),
                                                               camera_scene_key="camera_robot"))
        dagger = DaggerDriver(env, vla, obs_adapter, beta=args_cli.dagger_beta, chunk=args_cli.dagger_chunk,
                              close_thres=args_cli.dagger_finger_close_thres, seed=args_cli.seed)
        print(f"[DAgger] VLA {args_cli.dagger_vla_checkpoint} drives with P={1-args_cli.dagger_beta:.2f} per "
              f"{args_cli.dagger_chunk}-step chunk; expert labels every state. {len(dagger_motions)} demo motions x "
              f"{args_cli.dagger_rollouts} rollouts. closed-pose label (mean|q| {np.abs(dagger_closed).mean():.2f}) "
              f"= {np.round(dagger_closed, 2).tolist()}")

    output_root = Path(args_cli.output_directory).resolve()
    recorder = RolloutRecorder(output_root / datetime.now().strftime("%Y-%m-%d"))
    recorder.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving rollouts to: {recorder.output_dir}")

    # DETERMINISTIC motion iteration. The inference policy is the actor MEAN (no sampling),
    # and with fixed friction + fixed start frame a rollout is fully determined by motion_id.
    # So we sweep each motion in the library EXACTLY ONCE (forcing env._forced_motion_id)
    # rather than randomly re-drawing — random retries would (a) waste rollouts re-trying
    # motions that deterministically fail and (b) write duplicate successful trajectories.
    # The unique-success ceiling is therefore the number of motions that succeed; --num-samples
    # caps how many of those to write.
    motion_tag = "Pick_sim2"
    total_motions = int(env.unwrapped.total_motions)
    print(f"[INFO] motion library = {total_motions} motions; target = {target_successes} successful "
          f"trajectories (each motion tried once, deterministic policy).")

    written = 0
    tried = 0
    rejected_fall = 0
    rejected_nograsp = 0
    rejected_topple = 0
    errored = 0
    try:
        if args_cli.motion_range is None:
            m_lo, m_hi = 0, total_motions
        else:
            m_lo = max(0, args_cli.motion_range[0])
            m_hi = min(total_motions, args_cli.motion_range[1])
        if args_cli.motion_range is not None:
            print(f"[INFO] --motion-range: sweeping motions [{m_lo}, {m_hi}) of {total_motions}")
        if dagger is not None:
            sweep = [(m, r) for m in dagger_motions if m_lo <= m < m_hi for r in range(args_cli.dagger_rollouts)]
        else:
            sweep = [(m, 0) for m in range(m_lo, m_hi)]
        for motion_id, rollout_idx in sweep:
            if written >= target_successes or not simulation_app.is_running():
                break
            suffix = f"_r{rollout_idx}" if dagger is not None else ""
            if args_cli.skip_existing and list(output_root.rglob(f"*__motion_{motion_id:03d}{suffix}.hdf5")):
                print(f"[INFO] motion {motion_id}{suffix}: already collected under {output_root} -- skipping")
                continue
            env.unwrapped._forced_motion_id = int(motion_id)  # forces the reset's motion draw
            _set_all_seeds(args_cli.seed + motion_id * 10 + rollout_idx)   # deterministic per-rollout init
            if dagger is not None:
                dagger.rng = np.random.default_rng(args_cli.seed + motion_id * 10 + rollout_idx)
            print(f"[INFO] motion {motion_id}{suffix}/{total_motions} (written {written}/{target_successes})")

            # Per-motion resilience: a single bad rollout (or a recoverable error) shouldn't
            # lose the whole run / the partial dataset. If the sim app itself died (e.g. an
            # RTX/render crash on reset), is_running() goes False and we stop cleanly with
            # whatever was written so far.
            try:
                camera_frames, raw_state, meta, teleop_payload = _run_rollout_adapter(
                    env, policy, simulation_app=simulation_app, max_steps=args_cli.rollout_length,
                    state_on=bool(args_cli.state_on), real_time=bool(args_cli.real_time),
                    reset_at_start=True,  # always reset so the forced motion takes effect
                    lift_thres=float(args_cli.lift_thres),
                    phys_lift=float(args_cli.phys_lift), phys_radius=float(args_cli.phys_radius),
                    phys_steps=int(args_cli.phys_steps), dagger=dagger, dagger_closed_pose_arr=dagger_closed,
                )
            except Exception as rollout_exc:
                errored += 1
                print(f"[WARN] motion={motion_id} rollout raised {type(rollout_exc).__name__}: "
                      f"{rollout_exc} — skipping.")
                if not simulation_app.is_running():
                    print("[WARN] simulation app is no longer running — stopping collection "
                          f"with {written} successes preserved.")
                    break
                continue
            tried += 1

            # DAgger: keep everything long enough to carry labels (failures are the point).
            if dagger is not None:
                if meta["num_steps"] < args_cli.dagger_min_steps:
                    rejected_fall += 1
                    print(f"[INFO] REJECTED motion={motion_id}{suffix} (DAgger rollout too short: {meta['num_steps']} steps)")
                    continue
                print(f"[INFO] DAgger motion={motion_id}{suffix}: {meta['num_steps']} steps "
                      f"(expert {meta['dagger']['expert_steps']} / vla {meta['dagger']['vla_steps']}), "
                      f"end={meta['termination_terms']}, held={meta['phys_held']}, max_lift={meta['max_lift_m']:.3f}")
            # SUCCESS FILTER: write only non-fallen, task-successful trajectories.
            if dagger is None and meta["error_terminated"]:
                rejected_fall += 1
                print(f"[INFO] REJECTED motion={motion_id} (failure termination: {meta['termination_terms']}) "
                      f"steps={meta['num_steps']}")
                continue
            if dagger is None and not meta["success"]:
                rejected_nograsp += 1
                print(f"[INFO] REJECTED motion={motion_id} (no physical hold; max_lift={meta['max_lift_m']:.3f} m, "
                      f"toppled={meta['toppled_any']}) steps={meta['num_steps']}")
                continue
            if dagger is None and meta["toppled_any"] and args_cli.reject_topple:
                rejected_topple += 1
                print(f"[INFO] REJECTED motion={motion_id} (held, but object toppled during the episode) "
                      f"steps={meta['num_steps']}")
                continue

            file_name = f"sonic_adapter__{motion_tag}__motion_{motion_id:03d}{suffix}.hdf5"
            metadata = {
                "motion_reference": motion_tag,
                "motion_id": int(motion_id),
                "success_index": written,
                "skip_start_frames": args_cli.skip_start_frames,
                "start_pregrab_margin_s": args_cli.start_pregrab_margin,
                "residual_scale": args_cli.residual_scale,
                "residual_transform": args_cli.residual_transform,
                "sonic_pt": str(args_cli.sonic_pt) if args_cli.sonic_pt else None,
                "encoder_mode": args_cli.encoder_mode,
                "waist_dof": int(args_cli.waist_dof),
                "checkpoint": str(resume_path),
                **meta,
            }
            env_args = _build_env_args(env, task_name=args_cli.task)
            recorder.write_rollout(
                file_name,
                frames=np.stack(camera_frames, axis=0),
                raw_state=raw_state if args_cli.state_on else None,
                metadata=metadata, teleop=teleop_payload, env_args=env_args,
            )
            written += 1
            print(f"[INFO] WROTE success {written}/{target_successes} (motion {motion_id}): "
                  f"{recorder.output_dir / file_name}")

        print(f"\n[INFO] Collection finished. Successes written: {written}/{target_successes} "
              f"(motions tried={tried}/{total_motions}, rejected_fall={rejected_fall}, "
              f"rejected_nograsp={rejected_nograsp}, rejected_topple={rejected_topple}, errored={errored})")
        if written < target_successes:
            print(f"[WARN] Wrote {written} successes from {tried} motions tried. With a DETERMINISTIC "
                  f"policy the unique-success ceiling is the number of motions that succeed — re-running "
                  f"won't add more unless the policy/env changes. Improve the policy, or add a stochastic/"
                  f"domain-randomized collection mode for augmentation, to exceed this.")
    finally:
        env.close()
        simulation_app.close()


def _main_with_error_report() -> None:
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] collect_sonic_adapter.py failed: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    _main_with_error_report()
