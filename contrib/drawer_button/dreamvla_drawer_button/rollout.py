#!/usr/bin/env python3
"""Record successful expert demonstrations or evaluate a fixed rollout bank."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .paths import add_external_paths, ffmpeg_path

add_external_paths()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-dir", "--arm", dest="arm", type=Path, required=True)
    p.add_argument(
        "--track", choices=("repeatability", "fresh_start"), default="repeatability"
    )
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--mode", choices=("collect", "eval", "teacher_eval"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--episodes",
        type=int,
        required=True,
        help="Accepted-episode target for quota collection; exact attempts otherwise",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        help="Collection fail-safe only; default is max(200, 20 * target)",
    )
    p.add_argument(
        "--one-success-per-reference",
        action="store_true",
        help="Collection only: retain at most one accepted episode for every reference",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume collection or evaluation while retaining completed episodes/outcomes",
    )
    p.add_argument("--server-port", type=int, default=5777)
    # Help works without starting Kit or importing simulation dependencies.
    if "--help" in sys.argv or "-h" in sys.argv:
        p.print_help()
        return
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    from .contracts import (
        read,
        scene,
        prompt,
        validate_episode,
        digest,
        validate_recording_resume,
    )
    from .io import write_json
    from .run_config import load_run

    arm, bank = load_run(a.arm)
    task = arm["task"]
    collecting = a.mode == "collect"
    teacher_drives = collecting or a.mode == "teacher_eval"
    CHECKPOINT = a.checkpoint or Path(arm["teacher_checkpoint"])
    checkpoint_hash = digest(CHECKPOINT)
    PROMPT = prompt(task)
    references = Path(arm["reference_root"] if collecting else bank["reference_root"])
    names = (
        arm["reference_names"]
        if collecting
        else [c["reference_name"] for c in bank[a.track]]
    )
    if a.episodes < 1:
        raise ValueError("Episode count must be positive")
    if not collecting and len(names) < a.episodes:
        raise ValueError(
            "Evaluation bank is smaller than the requested exact-attempt count"
        )
    if not collecting and (a.max_attempts is not None or a.one_success_per_reference):
        raise ValueError("Collection-only options were used outside collection")
    if a.one_success_per_reference and a.episodes != len(names):
        raise ValueError("One-success-per-reference requires target == reference count")
    attempt_cap = (
        (a.max_attempts if a.max_attempts is not None else max(200, 20 * a.episodes))
        if collecting
        else a.episodes
    )
    if attempt_cap < a.episodes:
        raise ValueError("Attempt cap is below the requested success quota")
    prior = None
    if a.output.exists():
        if not a.resume:
            raise FileExistsError(a.output)
        if not (a.output / "result.json").is_file():
            raise ValueError("Cannot resume without result.json")
        prior = read(a.output / "result.json")
        if prior.get("evaluation_track") != a.track:
            raise ValueError("Cannot resume using a different evaluation track")
        validate_recording_resume(
            prior, a.mode, a.episodes, checkpoint_hash, digest(arm["eval_bank"])
        )
        if prior.get("status") == "complete":
            print(
                "RECORDING_ALREADY_COMPLETE",
                json.dumps(
                    dict(
                        mode=a.mode,
                        output=str(a.output),
                        saved=prior["saved"],
                        attempts=prior["attempts"],
                    )
                ),
                flush=True,
            )
            return
        if collecting:
            recovery = a.output / "interrupted_partials"
            recovery.mkdir(exist_ok=True)
            for folder in sorted(a.output.glob(".episode_*.tmp-*")):
                target = recovery / f"{folder.name}.{int(time.time_ns())}"
                folder.rename(target)
    else:
        a.output.mkdir(parents=True, exist_ok=False)
    a.headless = True
    a.enable_cameras = True
    a.multi_gpu = False
    a.kit_args = (a.kit_args or "") + " --/ngx/enabled=false --/rtx/post/aa/op=1"
    app = AppLauncher(a).app
    import cv2
    import gymnasium as gym
    import numpy as np
    import torch
    import isaaclab_tasks  # noqa: F401 -- task registration side effects
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.io import dump_yaml
    from isaaclab.utils.math import subtract_frame_transforms
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg
    from rsl_rl.runners import OnPolicyRunner
    from .isaac.motion_tracking_button_press_standing_sonic_env import (
        TASK,
        register_task,
        G1ButtonPressStandingPPORunnerCfg,
    )
    from .isaac.sonic_residual_actor import register_residual_actor
    from .isaac.sonic_teacher_repair_contract import (
        RepairSpec,
        validate_checkpoint,
        compose,
    )
    from .isaac.evaluation_observations import disable_observation_noise
    from .robot_model import get_project_g1_robot_model
    from .contracts import accepted
    from .button_visual import ButtonTaskVisual
    from .contracts import STATE_SLICES
    from .bridge import BridgeClient
    from .actions import nearest_binary, fsq

    if task == "drawer":
        from .isaac.sonic_teacher_repair import (
            TASK,
            register_task,
            configure,
            G1DreamVlaStylePPORunnerCfg as G1ButtonPressStandingPPORunnerCfg,
        )
    register_task()
    register_residual_actor()
    spec = RepairSpec(scale=0.5)
    validate_checkpoint(CHECKPOINT, spec)
    cfg = parse_env_cfg(TASK, device=a.device, num_envs=1)
    cfg.seed = a.seed
    cfg.ref_motions_path = str(references)
    disable_observation_noise(cfg)
    if task == "drawer":
        configure(cfg, spec)
        from .scene import close_drawer_on_reset

        close_drawer_on_reset(cfg)
    for name in ("target_ref_curr", "target_ref_next", "target_ref_next_next"):
        term = getattr(cfg.observations.policy, name, None)
        if term is not None:
            term.params["visualize_markers"] = False
    cfg.scene.camera_robot = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d435_link/Camera",
        spawn=PinholeCameraCfg(
            focal_length=9.1,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        data_types=["rgb"],
        height=480,
        width=640,
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.0), rot=(0.5, 0.5, -0.5, -0.5), convention="opengl"
        ),
    )
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.render.antialiasing_mode = "TAA"
    if task == "standing_button" and (
        cfg.scene.buttons is not None or cfg.scene.lobby is not None
    ):
        raise ValueError("Unexpected physical scene")
    env_name = (
        "env.yaml"
        if prior is None
        else f"env_resume_{len(prior.get('outcomes', [])):06d}.yaml"
    )
    dump_yaml(str(a.output / env_name), cfg)
    case = dict(reference_index=0, track="fresh_start")
    old_joints = cfg.events.reset_robot_joints.func
    old_root = cfg.events.reset_base.func

    def reset_joints(env, env_ids, asset_cfg):
        old_joints(env, env_ids, asset_cfg)
        if hasattr(env, "start_motion_times"):
            mapping = {
                Path(x).name: i for i, x in enumerate(env.motion_lib._motion_data_load)
            }
            env.motion_ids[env_ids] = mapping[names[case["reference_index"]]]
            env.start_motion_times[env_ids] = 0.0
            ref = env.motion_lib.get_motion_state(
                env.motion_ids[env_ids], env.start_motion_times[env_ids]
            )
            robot = env.scene["robot"]
            q = robot.data.default_joint_pos[env_ids].clone()
            q[:, asset_cfg.joint_ids] = ref["dof_pos"]
            robot.write_joint_state_to_sim(
                q, robot.data.default_joint_vel[env_ids].clone(), env_ids=env_ids
            )

    def reset_root(
        env,
        env_ids,
        offset_z=0.02,
        xy_range=0.01,
        yaw_range=0.034906585,
        standard_env_count=0,
    ):
        if case["track"] == "repeatability":
            xy_range = yaw_range = 0.0
        old_root(
            env,
            env_ids,
            offset_z=offset_z,
            xy_range=xy_range,
            yaw_range=yaw_range,
            standard_env_count=standard_env_count,
        )

    cfg.events.reset_robot_joints.func = reset_joints
    if task == "standing_button":
        cfg.events.reset_base.func = reset_root
    sample = {}
    if task == "standing_button":
        old_touch = cfg.rewards.touch_goal.func

        def audited_touch(
            env, asset_cfg, thres=0.05, offset_x=0.0, offset_y=0.0, offset_z=0.0
        ):
            r = old_touch(
                env,
                asset_cfg,
                thres=thres,
                offset_x=offset_x,
                offset_y=offset_y,
                offset_z=offset_z,
            )
            sample["touch"] = bool((r > 0)[0])
            return r

        cfg.rewards.touch_goal.func = audited_touch
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=None)
    base = env.unwrapped
    if int(base.max_episode_length) != 500 or abs(base.step_dt - 0.02) > 1e-9:
        raise ValueError("Timing changed")
    robot = base.scene["robot"]
    camera = base.scene["camera_robot"]
    from .scene import NoVisual

    visual = ButtonTaskVisual(base) if task == "standing_button" else NoVisual()
    feet_ids = robot.find_bodies(
        ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
    )[0]
    if task == "drawer":
        drawer_id = base.scene["cabinet"].find_joints("drawer_bottom_joint")[0][0]
    original_compute = base.reward_manager.compute

    def compute(dt):
        reward = original_compute(dt)
        sample["feet"] = robot.data.body_pos_w[0, feet_ids].clone()
        sample["upright"] = bool(-robot.data.projected_gravity_b[0, 2] > 0.8191520443)
        if task == "drawer":
            sample["drawer"] = float(base.scene["cabinet"].data.joint_pos[0, drawer_id])
            sample["touch"] = sample["drawer"] > 0.05
        return reward

    base.reward_manager.compute = compute
    model = get_project_g1_robot_model()
    ids, joint_names = robot.find_joints(model.joint_names, preserve_order=True)
    if list(joint_names) != list(model.joint_names):
        raise ValueError("Dataset joint order mismatch")
    wrists = robot.find_bodies(
        ["left_wrist_yaw_link", "right_wrist_yaw_link"], preserve_order=True
    )[0]
    hand_ids = {}
    hand_poses = {}
    for h in ("left", "right"):
        hn = getattr(model.supplemental_info, h + "_hand_actuated_joints")
        hand_ids[h] = robot.find_joints(hn, preserve_order=True)[0]
        term = base.action_manager.get_term(h + "_hand_action")
        order = [list(term._joint_names).index(n) for n in hn]
        hand_poses[h] = (
            term._open_command[order].cpu().numpy(),
            term._close_command[order].cpu().numpy(),
        )
    body = base.action_manager.get_term("joint_pos")
    original = body.process_actions
    audit = dict(calls=0, max_composition_error=0.0, nonfinite=0, history_updates=0)
    original_history = body._append_history

    def history():
        audit["history_updates"] += 1
        original_history()

    body._append_history = history
    desired = None

    def process(actions):
        before = audit["history_updates"]
        if teacher_drives:
            original(actions)
            expected = compose(body._encoded_reference, actions, spec)
        else:
            # Exact SONIC execution path with a full VLA token; no reference/residual recomposition.
            body._raw_actions[:] = actions
            body._full_latent[:] = desired
            body._append_history()
            body._assemble_decoder_input()
            if body._use_cuda_ort:
                body._ort_execution.run(body._decoder_session, body._io_binding)
            else:
                decoded = body._decoder_session.run(
                    [body._decoder_output_name],
                    {body._decoder_input_name: body._decoder_input.cpu().numpy()},
                )[0]
                body._decoded_actions[:] = torch.from_numpy(decoded).to(body.device)
            body._joint_targets[:] = (
                body._asset.data.default_joint_pos[:, body._joint_ids]
                + body._decoded_actions * body._action_scale
            )
            expected = desired
        error = float((expected - body.full_latent).abs().max())
        audit["calls"] += 1
        audit["max_composition_error"] = max(audit["max_composition_error"], error)
        finite = bool(
            torch.isfinite(body.full_latent).all()
            & torch.isfinite(body._joint_targets).all()
        )
        audit["nonfinite"] += int(not finite)
        if error or not finite or audit["history_updates"] != before + 1:
            raise RuntimeError("SONIC execution/history audit failed")

    body.process_actions = process
    client = None
    if teacher_drives:
        agent = G1ButtonPressStandingPPORunnerCfg()
        agent.device = a.device
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=a.device)
        runner.load(str(CHECKPOINT))
        policy = runner.get_inference_policy(device=base.device)
    else:
        client = BridgeClient(a.server_port)
        if prior and prior.get("policy_sha256") != client.policy_sha256:
            raise ValueError(
                "Cannot resume an evaluation with different GR00T weights/settings"
            )
    outcomes = [] if prior is None else list(prior.get("outcomes", []))
    saved = 0 if prior is None else int(prior.get("saved", 0))
    elapsed_before = 0.0 if prior is None else float(prior.get("elapsed_seconds", 0.0))
    started = time.monotonic()
    contract = scene(task)
    successful_references = {
        row["reference_name"] for row in outcomes if row.get("accepted")
    }
    if a.one_success_per_reference:
        if len(successful_references) != saved:
            raise ValueError("Resume has duplicate/mismatched accepted references")
        if not successful_references.issubset(set(names)):
            raise ValueError("Resume contains an unknown reference")
    if saved != len(list(a.output.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))):
        raise ValueError("Resume episode directory count disagrees with result.json")
    if prior:
        old = prior.get("runtime_audit", {})
        audit["calls"] = int(old.get("calls", 0))
        audit["history_updates"] = int(old.get("history_updates", 0))
        audit["nonfinite"] = int(old.get("nonfinite", 0))
        audit["max_composition_error"] = float(old.get("max_composition_error", 0.0))
        for i in range(saved):
            validate_episode(
                a.output / f"episode_{i:06d}", task, checkpoint_sha256=checkpoint_hash
            )
    collection_contract = dict(
        success_quota=collecting,
        target_successes=a.episodes if collecting else None,
        attempt_cap=attempt_cap if collecting else None,
        one_success_per_reference=bool(a.one_success_per_reference),
        retry_seed_policy="SHA256(task, reference budget, reference name, global attempt)",
        acceptance="Existing task-specific accepted criterion; failures are not retained",
        teacher_qualification="Run teacher_eval and inspect the result before collection",
    )

    def progress(status="running"):
        write_json(
            a.output / "result.json",
            dict(
                status=status,
                mode=a.mode,
                saved=saved,
                target=a.episodes,
                attempted_references=len(outcomes),
                successes=saved if collecting else sum(x["accepted"] for x in outcomes),
                attempts=len(outcomes),
                frames=saved * 499,
                outcomes=outcomes,
                runtime_audit=audit,
                elapsed_seconds=elapsed_before + time.monotonic() - started,
                collection_contract=collection_contract,
                environment_contract=contract,
                checkpoint_sha256=checkpoint_hash,
                policy_sha256=None if client is None else client.policy_sha256,
                evaluation_track=a.track,
                observation_noise_enabled=False,
                evaluation_bank_sha256=digest(arm["eval_bank"]),
                tracks={
                    t: dict(
                        episodes=sum(x["track"] == t for x in outcomes),
                        successes=sum(
                            x["track"] == t and x["standing_success"] for x in outcomes
                        ),
                        touch_successes=sum(
                            x["track"] == t and x["touch_success"] for x in outcomes
                        ),
                    )
                    for t in ("repeatability", "fresh_start")
                },
            ),
            durable=True,
        )

    limit = attempt_cap
    for attempt in range(len(outcomes), limit):
        if collecting and saved == a.episodes:
            break
        if collecting and a.one_success_per_reference:
            pending = [
                i for i, name in enumerate(names) if name not in successful_references
            ]
            if not pending:
                break
            index = pending[attempt % len(pending)]
        else:
            index = attempt % len(names) if collecting else attempt
        case.update(
            reference_index=index, track="fresh_start" if collecting else a.track
        )
        seed = (
            (
                1800000000
                + int(
                    hashlib.sha256(
                        f"{task}:{arm['n']}:{names[index]}:{attempt}".encode()
                    ).hexdigest()[:7],
                    16,
                )
            )
            if collecting
            else bank[a.track][index]["seed"]
        )
        base.seed(seed)
        with torch.inference_mode():
            obs, _ = env.reset()
        if obs.shape != (1, 679 if task == "drawer" else 670):
            raise ValueError("Wrong teacher observations")
        if client:
            client.reset(seed)
        mid = int(base.motion_ids[0])
        ref = Path(base.motion_lib._motion_data_load[mid])
        if ref.name != names[index] or ref.parent.resolve() != references.resolve():
            raise ValueError("Reference split or schedule leaked")
        initial_pose = robot.data.root_state_w[0, :7].cpu().numpy().copy()
        initial_quat = initial_pose[3:].copy()
        if hasattr(base, "goal_marker"):
            base.goal_marker.set_visibility(False)
        for marker in getattr(base, "markers", []):
            marker.set_visibility(False)
        visual.reset()
        visual.update()
        for _ in range(4):
            base.sim.render()
        data = {}
        frames = []
        touch = False
        baseline = None
        foot_xy = foot_lift = 0.0
        previous = -1
        chunk = None
        drawer_trace = []
        upright_trace = []
        for step in range(500):
            visual.update()
            base.sim.render()
            camera.update(0.0, force_recompute=True)
            rgb = camera.data.output["rgb"][0, :, :, :3].cpu().numpy().copy()
            frame = int(camera.frame[0])
            if frame <= previous or rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise ValueError("Invalid/stale raw camera")
            previous = frame

            def cpu(x):
                return x.detach().cpu().numpy().astype(np.float64).copy()

            q = cpu(robot.data.joint_pos[0, ids])
            gravity = cpu(robot.data.projected_gravity_b[0])
            eef = []
            for w in wrists:
                pos, quat = subtract_frame_transforms(
                    robot.data.root_pos_w,
                    robot.data.root_quat_w,
                    robot.data.body_pos_w[:, w],
                    robot.data.body_quat_w[:, w],
                )
                eef.extend((pos, quat))
            row = {
                "observation.state": q,
                "observation.projected_gravity": gravity,
                "observation.root_orientation": cpu(robot.data.root_quat_w[0]),
                "observation.eef_state": cpu(torch.cat(eef, -1)[0]),
                "observation.cpp_rotation_offset": np.array([1.0, 0.0, 0.0, 0.0]),
                "observation.init_base_quat": initial_quat.astype(np.float64),
                "teleop.delta_heading": np.zeros(1),
                "diagnostic.camera_frame_id": np.array([frame]),
                "diagnostic.control_step": np.array([step]),
            }
            with torch.inference_mode():
                if teacher_drives:
                    action = policy(obs)
                else:
                    if step % 20 == 0:
                        request = dict(
                            kind="infer",
                            ego_view=rgb,
                            projected_gravity=gravity.astype(np.float32),
                            prompt=PROMPT,
                        )
                        request.update(
                            {
                                k: q[lo:hi].astype(np.float32)
                                for k, (lo, hi) in STATE_SLICES.items()
                            }
                        )
                        chunk = client.call(request)
                        for k, d in [
                            ("motion_token", 64),
                            ("left_hand_joints", 7),
                            ("right_hand_joints", 7),
                        ]:
                            if (
                                chunk[k].shape != (40, d)
                                or not np.isfinite(chunk[k]).all()
                            ):
                                raise ValueError("Invalid VLA chunk")
                    desired = torch.as_tensor(
                        fsq(chunk["motion_token"][step % 20]), device=base.device
                    ).reshape(1, 64)
                    action = torch.zeros((1, 66), device=base.device)
                    for j, h in enumerate(("left", "right")):
                        action[0, 64 + j] = nearest_binary(
                            chunk[h + "_hand_joints"][step % 20], *hand_poses[h]
                        )
                obs, _, done, _ = env.step(action)
            touch |= sample["touch"]
            feet = sample["feet"]
            drawer_trace.append(sample.get("drawer", 0.0))
            upright_trace.append(sample["upright"])
            if step == 24:
                baseline = feet.clone()
            elif step > 24:
                foot_xy = max(
                    foot_xy,
                    float(
                        torch.linalg.norm(feet[:, :2] - baseline[:, :2], dim=-1).max()
                    ),
                )
                foot_lift = max(
                    foot_lift, float((feet[:, 2] - baseline[:, 2]).clamp_min(0).max())
                )
            if bool(done[0]):
                break  # terminal action terms were reset: never label this boundary
            if collecting:
                row.update(
                    {
                        "action.wbc": cpu(robot.data.joint_pos_target[0, ids]),
                        "action.motion_token": cpu(body.full_latent[0]),
                        "diagnostic.reference_token": body._encoded_reference[0]
                        .cpu()
                        .numpy()
                        .copy(),
                        "diagnostic.policy_action": action[0].cpu().numpy().copy(),
                        "diagnostic.upright": np.array([sample["upright"]], np.uint8),
                        "privileged.drawer_position": np.array(
                            [sample.get("drawer", 0.0)], np.float32
                        ),
                    }
                )
                for h in ("left", "right"):
                    row["teleop." + h + "_hand_joints"] = (
                        robot.data.joint_pos_target[0, hand_ids[h]].cpu().numpy().copy()
                    )
                for k, v in row.items():
                    data.setdefault(k, []).append(v)
                frames.append(rgb)
        else:
            raise RuntimeError("Missing terminal reset")
        early = step + 1 < 500
        ok = (
            accepted(touch, early, foot_xy, foot_lift)
            if task == "standing_button"
            else bool(
                not early
                and min(drawer_trace[-50:]) > 0.05
                and all(upright_trace[-50:])
            )
        )
        paper_success = (
            bool(
                not early
                and min(drawer_trace[-50:]) > 0.15
                and all(upright_trace[-50:])
            )
            if task == "drawer"
            else ok
        )
        outcome = dict(
            attempt=attempt,
            seed=seed,
            track=case["track"],
            reference_name=ref.name,
            initial_root_pose=initial_pose.tolist(),
            accepted=ok,
            paper_success=paper_success,
            max_drawer_m=max(drawer_trace),
            touch_success=touch,
            standing_success=ok,
            early_termination=early,
            steps=step + 1,
            foot_xy_m=foot_xy,
            foot_lift_m=foot_lift,
        )
        outcomes.append(outcome)
        if (
            collecting
            and ok
            and (
                not a.one_success_per_reference or ref.name not in successful_references
            )
        ):
            if len(frames) != 499:
                raise ValueError("Incomplete successful episode")
            folder = a.output / f".episode_{saved:06d}.tmp-{os.getpid()}"
            folder.mkdir()
            np.savez_compressed(
                folder / "data.npz", **{k: np.asarray(v) for k, v in data.items()}
            )
            cv2.imwrite(
                str(folder / "first_frame.png"),
                cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR),
            )
            cmd = [
                ffmpeg_path(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "640x480",
                "-r",
                "50",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(folder / "ego_view.mp4"),
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            try:
                for rgb in frames:
                    proc.stdin.write(rgb.tobytes())
            finally:
                proc.stdin.close()
            if proc.wait(timeout=60):
                raise RuntimeError("Video encoder failed")
            cap = cv2.VideoCapture(str(folder / "ego_view.mp4"))
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if count != 499 or fps != 50:
                raise ValueError("Encoded video mismatch")
            meta = dict(
                outcome,
                task=PROMPT,
                frames=499,
                fps=50,
                environment_contract=contract,
                evaluation_only=False,
                checkpoint=str(CHECKPOINT),
                checkpoint_sha256=checkpoint_hash,
                reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),
                split="train",
                action_labels="executed full FSQ SONIC token and commanded hand targets; pre-action observations",
                collection_contract=collection_contract,
            )
            write_json(folder / "metadata.json", meta, durable=True)
            validate_episode(folder, task, checkpoint_sha256=checkpoint_hash)
            folder.rename(a.output / f"episode_{saved:06d}")
            saved += 1
            successful_references.add(ref.name)
        progress()
        print("BUTTON_EPISODE", json.dumps(outcome), "SAVED", saved, flush=True)
    if collecting and saved != a.episodes:
        progress("attempt_budget_exhausted")
        raise RuntimeError("Collection attempt budget exhausted before success quota")
    if not collecting and len(outcomes) != a.episodes:
        raise RuntimeError("Missing planned attempts")
    progress("complete")
    if client:
        client.close()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
