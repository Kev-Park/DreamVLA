"""Data collection using robot_camera for Isaac-Motion-Tracking-Pick-Cam-v0 --> HDF5 files
"""

from __future__ import annotations

import argparse
import builtins
import copy
import os
import random
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


# helper functions for getting paths, seeding, etc.

def _resolve_training_root() -> Path:
    root = os.environ.get("ISAACLAB_PATH")
    if root:
        return Path(root).resolve()
    return Path.cwd().resolve()


def _resolve_repo_root() -> Path:
    return _resolve_training_root().parent


def _resolve_assets_root() -> Path:
    return _resolve_training_root() / "assets"


def _resolve_sample_root() -> Path:
    return _resolve_repo_root() / "TrajGen" / "sample"


def _resolve_kitchen_usd() -> Path:
    return _resolve_assets_root() / "HQ Kitchen" / "Collected_kitchen_flat" / "kitchen_flat3.usd"


def _resolve_object_usd(object_name: str) -> Path:
    normalized_name = object_name[:-4] if object_name.endswith(".usd") else object_name
    return _resolve_assets_root() / f"{normalized_name}.usd"


def _discover_motion_references() -> list[Path]:
    sample_root = _resolve_sample_root()
    pick_sim2_root = sample_root / "Pick_sim2"

    if not pick_sim2_root.exists():
        raise FileNotFoundError(f"Pick_sim2 directory not found at {pick_sim2_root}")
    return [pick_sim2_root.resolve()]


def _normalize_object_list(object_list: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in object_list:
        normalized.append(item[:-4] if item.endswith(".usd") else item)
    return normalized


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    return value


def _seed_for_rollout(base_seed: int, object_index: int, motion_index: int, rollout_index: int, worker_index: int = 0) -> int:
    return int(base_seed + worker_index * 1_000_000_000 + object_index * 1_000_000 + motion_index * 1_000 + rollout_index)


def _set_all_seeds(seed: int) -> None:
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


def _debug_wrist_and_pelvis_dump(env) -> None:
    """One-shot wrist-link / pelvis-body dump for teleop schema bring-up. DELETE AFTER USE."""
    robot = env.unwrapped.scene["robot"]

    print("===== [DEBUG-0A] WRIST LINK NAMES =====")
    body_names = list(robot.data.body_names)
    print(f"BODY_COUNT: {len(body_names)}")
    print("BODY_NAMES:")
    for n in body_names:
        print(f"  {n}")
    print(f"HAS_LEFT_WRIST_YAW:  {'left_wrist_yaw_link'  in body_names}")
    print(f"HAS_RIGHT_WRIST_YAW: {'right_wrist_yaw_link' in body_names}")

    print("===== [DEBUG-0B] PELVIS / ROOT =====")
    print(f"ROOT_NAME_candidate: {body_names[0]}")
    has_pelvis = "pelvis" in body_names
    print(f"HAS_PELVIS_BODY: {has_pelvis}")
    if has_pelvis:
        pid, _ = robot.find_bodies(["pelvis"])
        print(f"PELVIS_BODY_INDEX: {pid}")
        print(f"PELVIS_POS_W:  {robot.data.body_pos_w[0, pid[0]].tolist()}")
        print(f"ROOT_POS_W:    {robot.data.root_pos_w[0].tolist()}")
        print(f"PELVIS_QUAT_W: {robot.data.body_quat_w[0, pid[0]].tolist()}")
        print(f"ROOT_QUAT_W:   {robot.data.root_quat_w[0].tolist()}")
    print("===== [DEBUG-0 END] =====")


def _get_hand_joint_indices(env) -> tuple[list[int], list[int]]:
    """Resolve and cache left/right hand joint indices for the current env."""

    cache_attr = "_collect_pick_cam_hand_joint_indices"
    cached_value = getattr(env.unwrapped, cache_attr, None)
    if cached_value is not None:
        return cached_value

    robot = env.unwrapped.scene["robot"]
    left_ids, _ = robot.find_joints(["left_hand.*"])
    right_ids, _ = robot.find_joints(["right_hand.*"])
    resolved = (list(left_ids), list(right_ids))
    setattr(env.unwrapped, cache_attr, resolved)
    return resolved


def _capture_rollout_state(env, action: torch.Tensor | None = None) -> dict[str, Any]:
    robot = env.unwrapped.scene["robot"]
    object_asset = env.unwrapped.scene["object"]
    left_hand_joint_ids, right_hand_joint_ids = _get_hand_joint_indices(env)

    joint_pos = robot.data.joint_pos[0]
    left_finger_joint_pos = joint_pos[left_hand_joint_ids].detach().cpu()
    right_finger_joint_pos = joint_pos[right_hand_joint_ids].detach().cpu()

    state: dict[str, Any] = {
        "robot": {
            "root_pos_w": robot.data.root_pos_w[0].detach().cpu(),
            "root_quat_w": robot.data.root_quat_w[0].detach().cpu(),
            "joint_pos": robot.data.joint_pos[0].detach().cpu(),
            "joint_vel": robot.data.joint_vel[0].detach().cpu(),
            "left_finger_joint_pos": left_finger_joint_pos,
            "right_finger_joint_pos": right_finger_joint_pos,
        },
        "object": {
            "root_pos_w": object_asset.data.root_pos_w[0].detach().cpu(),
            "root_quat_w": object_asset.data.root_quat_w[0].detach().cpu(),
        },
    }

    if hasattr(env.unwrapped.scene, "env_origins"):
        state["robot"]["env_origin"] = env.unwrapped.scene.env_origins[0].detach().cpu()

    if action is not None:
        state["action"] = action[0].detach().cpu()

    return state


def _update_policy_observations(env, obs: torch.Tensor) -> torch.Tensor:
    # Import lazily so omni modules are available after Isaac app initialization.
    import isaaclab.utils.math as math_utils

    robot = env.unwrapped.scene["robot"]
    object_asset = env.unwrapped.scene["object"]
    robot_pos_world = robot.data.root_pos_w
    robot_quat_world = math_utils.quat_unique(robot.data.root_quat_w)
    object_pos_world = object_asset.data.root_pos_w
    object_pos_local = math_utils.quat_apply(
        math_utils.quat_conjugate(robot_quat_world), object_pos_world - robot_pos_world
    )
    if object_pos_local.shape[0] > 0:
        # Preserve the same policy-conditioning behavior used by play_pick_cam.py.
        obs[:, 52] = object_pos_local[0, 0]
        obs[:, 53] = object_pos_local[0, 1]
    return obs


def _resolve_checkpoint_path(args_cli, agent_cfg, log_root_path: str) -> str:
    if args_cli.use_pretrained_checkpoint:
        from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            raise RuntimeError("A pre-trained checkpoint is unavailable for this task.")
        return resume_path

    if getattr(args_cli, "checkpoint", None):
        from isaaclab.utils.assets import retrieve_file_path

        return retrieve_file_path(args_cli.checkpoint)

    if getattr(args_cli, "checkpoint_path", None):
        return args_cli.checkpoint_path

    if getattr(args_cli, "path", None):
        return args_cli.path

    from isaaclab_tasks.utils import get_checkpoint_path

    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _build_env_cfg(base_env_cfg, object_name: str, motion_reference: Path, camera_on: bool):
    env_cfg = copy.deepcopy(base_env_cfg)
    env_cfg.enable_cameras_for_collection = camera_on
    env_cfg.kitchen_usd_path = str(_resolve_kitchen_usd())
    env_cfg.object_usd_path = str(_resolve_object_usd(object_name))
    env_cfg.ref_motions_path = str(motion_reference)
    return env_cfg


def _create_policy(env, agent_cfg, resume_path: str):
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
    return wrapped_env, runner, policy


def _run_rollout(
    env,
    policy,
    *,
    simulation_app,
    max_steps: int,
    state_on: bool,
    camera_on: bool,
    real_time: bool,
    reset_at_start: bool,
) -> tuple[list[np.ndarray], dict[str, Any] | None, dict[str, Any]]:
    camera_frames: list[np.ndarray] = []
    state_history: list[dict[str, Any]] = []
    rollout_success = False
    terminated_flag = False
    truncated_flag = False

    if reset_at_start:
        obs_out = env.reset()
    else:
        obs_out = env.get_observations()
    obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
    obs = obs.clone() if hasattr(obs, "clone") else obs

    scene_keys = list(env.unwrapped.scene.keys())
    if camera_on and "camera_robot" not in scene_keys:
        raise RuntimeError(
            "Required robot camera is missing. "
            f"Found: {scene_keys}."
        )
    cam_robot = env.unwrapped.scene["camera_robot"] if camera_on else None

    step_index = 0
    if not simulation_app.is_running():
        raise RuntimeError(
            "Simulation app is not running at rollout start. "
            "The app shut down before the first rollout step."
        )
    while step_index < max_steps and simulation_app.is_running():
        start_time = time.time()
        step_index += 1
        if camera_on and cam_robot is not None:
            object_pos_world = env.unwrapped.scene["object"].data.root_pos_w
            robot_pos_world = env.unwrapped.scene["robot"].data.root_pos_w
            robot_quat_world = env.unwrapped.scene["robot"].data.root_quat_w
            object_pos_local = _quat_apply(_quat_conjugate(robot_quat_world), object_pos_world - robot_pos_world)

            obs[:, 52] = object_pos_local[0, 0]
            obs[:, 53] = object_pos_local[0, 1]

        with torch.inference_mode():
            actions = policy(obs).clone()
            step_result = env.step(actions)

        if camera_on and cam_robot is not None:
            camera_output = getattr(cam_robot.data, "output", None)
            if camera_output is None:
                raise RuntimeError("camera_robot.data.output is unavailable after env.step.")
            if "rgb" not in camera_output:
                raise RuntimeError(
                    f"camera_robot output is missing 'rgb'. Available keys: {list(camera_output.keys())}"
                )

            image_robot = camera_output["rgb"]
            frame_rgb = image_robot[0].cpu().numpy()
            camera_frames.append(_frame_to_uint8_rgb(frame_rgb))

        if len(step_result) == 5:
            obs, _, terminated, truncated, info = step_result
        else:
            obs, _, done, info = step_result
            terminated = done
            truncated = torch.zeros_like(done)

        # Clone obs to allow inplace updates in the next iteration
        obs = obs.clone() if hasattr(obs, "clone") else obs

        if state_on:
            state_history.append(_capture_rollout_state(env, actions))

        terminated_flag = bool(torch.as_tensor(terminated).any().item())
        truncated_flag = bool(torch.as_tensor(truncated).any().item())
        if isinstance(info, dict) and "success" in info:
            success_value = info["success"]
            if isinstance(success_value, (np.ndarray, torch.Tensor)):
                rollout_success = bool(np.asarray(success_value).any())
            else:
                rollout_success = bool(success_value)

        if terminated_flag or truncated_flag:
            # End this rollout and move to the next trajectory rollout.
            print(
                f"[INFO] Rollout ended early at step {step_index}/{max_steps} "
                f"(terminated={terminated_flag}, truncated={truncated_flag})"
            )
            break

        print(f"[INFO] rollout step {step_index}/{max_steps}")

        sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
        if real_time and sleep_time > 0:
            time.sleep(sleep_time)
    raw_state = _stack_nested(state_history) if state_history else None
    if step_index == 0:
        raise RuntimeError(
            "No rollout steps were executed before SimulationApp stopped. "
            "Check terminal output above for the last manager/init logs before app shutdown."
        )

    metadata = {
        "terminated": terminated_flag,
        "truncated": truncated_flag,
        "success": rollout_success,
        "num_steps": len(camera_frames) if camera_on else len(state_history),
        "no_steps_executed": step_index == 0,
        "app_running": bool(simulation_app.is_running()),
        "camera_on": camera_on,
        "state_on": state_on,
    }
    return camera_frames, raw_state, metadata


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    # Local helper to avoid importing isaaclab.utils.math at runtime during rollout loop.
    result = quat.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    # Local helper equivalent to isaaclab.utils.math.quat_apply for (w, x, y, z).
    q_xyz = quat[..., 1:]
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    return vec + quat[..., 0:1] * t + torch.cross(q_xyz, t, dim=-1)


def main() -> None:
    """Collect pick rollouts as single-rollout HDF5 files."""

    parser = argparse.ArgumentParser(description="Collect pick rollouts with mandatory robot-camera output.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate. Currently only 1 is supported.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    state_group = parser.add_mutually_exclusive_group()
    state_group.add_argument("--state-on", dest="state_on", action="store_true", help="Record state tensors in the HDF5 file.")
    state_group.add_argument("--state-off", dest="state_on", action="store_false", help="Do not record state tensors.")
    parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to the checkpoint to load.")
    parser.add_argument("--object-list", nargs="*", default=["mustard_bottle"], help="List of object names to collect.")
    parser.add_argument("--num-samples", type=int, default=1, help="How many reset-and-rollout samples to collect from the motion directory.")
    parser.add_argument("--output-directory", type=str, default="./datasets/pick_cam", help="Directory where rollout HDF5 files are written.")
    parser.add_argument("--rollout-length", type=int, default=500, help="Maximum number of steps per rollout.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed used for deterministic rollout seeding.")
    parser.set_defaults(state_on=True)

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    if args_cli.task is None:
        parser.error("--task is required")

    # Seed global RNGs before any environment is created.
    _set_all_seeds(args_cli.seed)

    # TO-DO: implement support for parallelized collection here
    if args_cli.num_envs != 1:
        raise ValueError(
            "collect_pick_cam.py currently supports num_envs=1 only. "
            "Use --num_envs 1 while rollouts are saved one-file-per-rollout."
        )

    if args_cli.checkpoint_path is not None:
        args_cli.path = args_cli.checkpoint_path
        args_cli.checkpoint = args_cli.checkpoint_path

    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg

    from recorder import RolloutRecorder

    print(f"[INFO] ISAACLAB_NUCLEUS_DIR: {ISAACLAB_NUCLEUS_DIR}")

    base_env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
        enable_cameras=True,
    )
    # Ensure Isaac env creation receives an explicit seed.
    if hasattr(base_env_cfg, "seed"):
        base_env_cfg.seed = args_cli.seed
    
    # Ensure robot camera exists in scene config (required for collection).
    if not hasattr(base_env_cfg.scene, "camera_robot"):
        base_env_cfg.scene.camera_robot = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/Camera_robot",
            spawn=PinholeCameraCfg(
                focal_length=7.6,
                focus_distance=400.0,
                horizontal_aperture=20.0,
                clipping_range=(0.01, 100.0),
            ),
            data_types=["rgb"],
            height=720,
            width=1280,
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.36),
                rot=(0.568, 0.421, -0.421, -0.568),
                convention="opengl",
            ),
        )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = _resolve_checkpoint_path(args_cli, agent_cfg, log_root_path)
    print(f"[INFO] Loading checkpoint from: {resume_path}")

    object_list = _normalize_object_list(args_cli.object_list)
    motion_references = _discover_motion_references()

    # Build a single policy instance once, then reuse it across all rollouts.
    template_object = object_list[0]
    template_motion = motion_references[0]
    template_env_cfg = _build_env_cfg(base_env_cfg, template_object, template_motion, True)
    template_env = gym.make(args_cli.task, cfg=template_env_cfg, render_mode=None)
    if isinstance(template_env.unwrapped, DirectMARLEnv):
        template_env = multi_agent_to_single_agent(template_env)

    template_env, template_runner, policy = _create_policy(template_env, agent_cfg, resume_path)
    policy_device = template_env.unwrapped.device
    print(f"[INFO] Policy device: {policy_device}")

    # DEBUG: one-shot wrist-link / pelvis-body dump. DELETE AFTER USE.
    template_env.reset()
    _debug_wrist_and_pelvis_dump(template_env)

    output_root = Path(args_cli.output_directory).resolve()
    run_date = datetime.now().strftime("%Y-%m-%d")
    dated_output_dir = output_root / run_date
    recorder = RolloutRecorder(dated_output_dir)
    recorder.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving rollouts to: {recorder.output_dir}")
    written_rollouts = 0

    try:
        for object_index, object_name in enumerate(object_list):
            for motion_index, motion_reference in enumerate(motion_references):
                use_template_env = object_index == 0 and motion_index == 0
                if use_template_env:
                    env = template_env
                else:
                    env_cfg = _build_env_cfg(base_env_cfg, object_name, motion_reference, True)
                    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
                    if isinstance(env.unwrapped, DirectMARLEnv):
                        env = multi_agent_to_single_agent(env)

                prev_ended_on_done = False
                for rollout_index in range(args_cli.num_samples):

                    seed = _seed_for_rollout(args_cli.seed, object_index, motion_index, rollout_index)
                    _set_all_seeds(seed)
                    print(
                        f"[INFO] Starting rollout object={object_name} "
                        f"motion={motion_reference.name} idx={rollout_index} seed={seed}"
                    )

                    camera_frames, raw_state, rollout_metadata = _run_rollout(
                        env,
                        policy,
                        simulation_app=simulation_app,
                        max_steps=args_cli.rollout_length,
                        state_on=bool(args_cli.state_on),
                        camera_on=True,
                        real_time=bool(args_cli.real_time),
                        reset_at_start=not prev_ended_on_done,
                    )
                    print(
                        f"[INFO] Completed sample idx={rollout_index} "
                        f"steps={rollout_metadata['num_steps']} "
                        f"terminated={rollout_metadata['terminated']} "
                        f"truncated={rollout_metadata['truncated']}"
                    )
                    prev_ended_on_done = bool(rollout_metadata["terminated"] or rollout_metadata["truncated"])

                    file_name = (
                        f"{object_name}__{motion_reference.name}__"
                        f"rollout_{rollout_index:03d}_seed_{seed}.hdf5"
                    )
                    metadata = {
                        "object_name": object_name,
                        "motion_reference": motion_reference.name,
                        "motion_reference_path": motion_reference,
                        "seed": seed,
                        "rollout_index": rollout_index,
                        "object_index": object_index,
                        "motion_index": motion_index,
                        **rollout_metadata,
                    }
                    recorder.write_rollout(
                        file_name,
                        frames=np.stack(camera_frames, axis=0),
                        raw_state=raw_state if args_cli.state_on else None,
                        metadata=metadata,
                    )
                    written_rollouts += 1
                    print(f"[INFO] Wrote rollout: {recorder.output_dir / file_name}")
                if not use_template_env:
                    env.close()
        print(f"[INFO] Collection finished. Total rollouts written: {written_rollouts}")
    finally:
        template_env.close()
        simulation_app.close()


def _main_with_error_report() -> None:
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] collect_pick_cam.py failed: {exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    _main_with_error_report()