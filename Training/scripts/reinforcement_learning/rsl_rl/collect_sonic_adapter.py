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
  * SUCCESS FILTERING: only trajectories that meet the eval success criterion
    (env.n_successes incremented — object lifted above the reward's height_thres during the
    grasp phase) AND did not fall (error_terminated) are written. The collector keeps
    rolling out with fresh seeds until --num-samples SUCCESSFUL trajectories are written
    (capped by --max-attempts).
  * --skip-start-frames N: episodes begin N frames into the motion (skip the refinement's
    20-frame interpolate-to-initial-pose prepend); the recorded trajectory starts there.
    GR00T SFT windows frames independently, so a mid-motion start is safe for fine-tuning.
  * Default env is the kitchen-visuals BinaryFingers env (realistic ego backdrop) which
    INHERITS the 0.9 grasp physics from the training env, so the policy behaves identically.
"""

from __future__ import annotations

import argparse
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


def _run_rollout_adapter(env, policy, *, simulation_app, max_steps, state_on, real_time,
                         reset_at_start, lift_thres):
    camera_frames: list[np.ndarray] = []
    state_history: list[dict[str, Any]] = []
    teleop_history: list[dict[str, torch.Tensor]] = []
    # Eval-parity success: a frame is a real pickup when the object (bottle) clears lift_thres
    # AND the reference motion is in its closed/grasp phase. Mirrors eval_sonic_adapter.py
    # (bottle_z > lift_thres & is_closed). Evaluated per-step on the SAME frames we record.
    had_any_lift = False

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
        camera_output = getattr(cam_robot.data, "output", None)
        if camera_output is None or "rgb" not in camera_output:
            raise RuntimeError("camera_robot.data.output missing 'rgb'.")
        camera_frames.append(_frame_to_uint8_rgb(camera_output["rgb"][0].cpu().numpy()))
        if state_on:
            state_history.append(_capture_rollout_state(env, actions))
            teleop_history.append(_capture_teleop_frame(env))
        # Eval-parity lift check on THIS (pre-step) frame — the exact instant the camera frame
        # and state above were captured. Identical criterion to eval_sonic_adapter.py: the
        # bottle root z must exceed lift_thres while the reference is_closed (grasp) flag is set.
        # Read pre-step (not post-step) so it reflects the recorded frame and is immune to the
        # env's on-done auto-reset clobbering the object pose.
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
            if bottle_z > lift_thres and is_closed:
                had_any_lift = True
        with torch.inference_mode():
            step_result = env.step(actions)
        if len(step_result) == 5:
            obs, _, terminated, truncated, _ = step_result
        else:
            obs, _, done, _ = step_result
            terminated = done
            truncated = torch.zeros_like(done)
        obs = obs.clone() if hasattr(obs, "clone") else obs
        # Flush the RTX render pipeline so the NEXT iteration's camera read delivers
        # this step's frame (the camera annotator otherwise lags / stays on the warm-up
        # render). Two pumps match the proven play_sonic_adapter.py cadence.
        simulation_app.update()
        simulation_app.update()
        terminated_flag = bool(torch.as_tensor(terminated).any().item())
        truncated_flag = bool(torch.as_tensor(truncated).any().item())
        if terminated_flag or truncated_flag:
            print(f"[INFO] rollout ended at step {step_index}/{max_steps} "
                  f"(terminated={terminated_flag}, truncated={truncated_flag})")
            break
        sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
        if real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if step_index == 0:
        raise RuntimeError("No rollout steps executed before SimulationApp stopped.")
    raw_state = _stack_nested(state_history) if state_history else None

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
    rollout_success = had_any_lift
    error_terminated = terminated_flag and step_index < max_steps
    metadata = {
        "terminated": terminated_flag,
        "truncated": truncated_flag,
        "error_terminated": error_terminated,
        "success": rollout_success,
        "success_criterion": f"bottle_z>{lift_thres} & is_closed (eval_sonic_adapter parity)",
        "lift_thres": float(lift_thres),
        "success_n_successes": success_n_successes,
        "num_steps": len(camera_frames),
        "app_running": bool(simulation_app.is_running()),
        "camera_on": True,
        "state_on": state_on,
    }
    return camera_frames, raw_state, metadata, teleop_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect SONIC-adapter pick rollouts (ego camera → HDF5).")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=1, help="Only 1 supported (one file per rollout).")
    parser.add_argument("--task", type=str, default="Isaac-Motion-Tracking-Pick-Cam-BinaryFingers-v0",
                        help="Default: kitchen-visuals BinaryFingers env (0.9 grasp physics inherited "
                             "from training). Use Isaac-Motion-Tracking-Pick-BinaryFingers-v0 for the "
                             "no-kitchen green-box env (ego camera is injected either way).")
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
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx")
    parser.add_argument("--sonic-encoder-onnx", type=str,
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx")
    parser.add_argument("--residual-scale", type=float, default=0.3,
                        help="MUST match the value train_sonic_adapter.py used.")
    parser.add_argument("--lift-thres", type=float, default=0.95,
                        help="Object (bottle) root z (m) above which a frame counts as 'lifted' "
                             "for the SUCCESS filter, evaluated during the reference grasp "
                             "(is_closed) phase. MUST match eval_sonic_adapter.py --lift-thres "
                             "(default 0.95 = object rests at 0.9, a 5 cm pickup). Only "
                             "trajectories with at least one lifted+closed frame are written.")
    parser.add_argument("--skip-start-frames", type=int, default=None,
                        help="Start episodes N frames into the motion (pass 20 to skip the refinement "
                             "prepend). The recorded trajectory begins at the skip frame.")
    parser.add_argument("--start-pregrab-margin", type=float, default=None,
                        help="Start episodes this many seconds before the grab (drops prepend + walk).")

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
    from vla_sonic.physics_overrides import apply_sonic_physics_overrides
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

    if args_cli.start_pregrab_margin is not None:
        env_cfg.motion_start_pregrab_margin_s = args_cli.start_pregrab_margin
        print(f"[collect_sonic_adapter] start_pregrab_margin = {args_cli.start_pregrab_margin}s")
    if args_cli.skip_start_frames is not None:
        env_cfg.motion_skip_start_frames = args_cli.skip_start_frames
        print(f"[collect_sonic_adapter] skip_start_frames = {args_cli.skip_start_frames} "
              "(recorded data starts at the skip frame)")

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
    decoder = load_frozen_decoder(args_cli.sonic_decoder_onnx, device)
    encoder = load_frozen_encoder(args_cli.sonic_encoder_onnx, device)
    env = TokenAdapterVecEnvWrapper(env, decoder, encoder, device,
                                    residual_scale=args_cli.residual_scale, clip_actions=None)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

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
    errored = 0
    try:
        for motion_id in range(total_motions):
            if written >= target_successes or not simulation_app.is_running():
                break
            env.unwrapped._forced_motion_id = int(motion_id)  # forces the reset's motion draw
            _set_all_seeds(args_cli.seed + motion_id)          # deterministic per-motion init
            print(f"[INFO] motion {motion_id}/{total_motions} (written {written}/{target_successes})")

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

            # SUCCESS FILTER: write only non-fallen, task-successful trajectories.
            if meta["error_terminated"]:
                rejected_fall += 1
                print(f"[INFO] REJECTED motion={motion_id} (fall) steps={meta['num_steps']}")
                continue
            if not meta["success"]:
                rejected_nograsp += 1
                print(f"[INFO] REJECTED motion={motion_id} (no grasp success) steps={meta['num_steps']}")
                continue

            file_name = f"sonic_adapter__{motion_tag}__motion_{motion_id:03d}.hdf5"
            metadata = {
                "motion_reference": motion_tag,
                "motion_id": int(motion_id),
                "success_index": written,
                "skip_start_frames": args_cli.skip_start_frames,
                "start_pregrab_margin_s": args_cli.start_pregrab_margin,
                "residual_scale": args_cli.residual_scale,
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
              f"rejected_nograsp={rejected_nograsp}, errored={errored})")
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
