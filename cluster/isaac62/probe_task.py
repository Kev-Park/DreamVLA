"""Bounded real-task reset/step/camera gate on an allocated Linux GPU.

This exercises the merged project, not a substitute stock scene. It uses zero
actions and makes no claim about learned-policy behavior or task success.
"""
import argparse
from collections.abc import Mapping
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback


def main():
    if os.name != 'posix' or sys.version_info[:2] != (3, 12):
        raise SystemExit('Run on an allocated Linux GPU in the isolated Python 3.12 environment')
    version = importlib.metadata.version('isaacsim')
    if not version.startswith('6.2.'):
        raise SystemExit('This gate requires Isaac Sim 6.2, found '+version)
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--ref-motions-path', type=Path)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--num-envs', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 16 <= args.steps <= 512 or not 1 <= args.num_envs <= 4:
        parser.error('Smoke gate requires 16..512 steps and 1..4 environments')
    if not args.device.startswith('cuda:'):
        parser.error('Select one reviewed GPU explicitly with --device cuda:N')
    ffmpeg = shutil.which('ffmpeg')
    ffprobe = shutil.which('ffprobe')
    if ffmpeg is None or ffprobe is None:
        parser.error('ffmpeg and ffprobe are required for the headless video validation gate')
    args.headless = True
    args.enable_cameras = True
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'started': time.time(), 'task': args.task, 'isaacsim': version,
               'device': args.device, 'steps_requested': args.steps,
               'num_envs': args.num_envs, 'completed': False,
               'scope': 'Real task initialization/reset/zero-action stepping/rendering only',
               'policy_validated': False, 'images': []}
    receipt_path = output/'receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2))
    app = None
    env = None
    video = None
    try:
        app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app
        import gymnasium as gym
        import numpy as np
        from PIL import Image
        import torch
        from isaacsim.core.simulation_manager import SimulationManager
        import isaaclab_tasks  # noqa: F401 -- register the actual merged tasks
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab.utils.io import dump_yaml
        from isaaclab.sim.utils import bind_physics_material
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        # Exercise real USD material relationships without adding objects to the
        # simulated task. A removed deformable class used to break even scenes
        # and rigid colliders before the environment could be constructed.
        material_stage = Usd.Stage.CreateInMemory()
        material = UsdShade.Material.Define(material_stage, '/Material')
        UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        scene_prim = UsdPhysics.Scene.Define(material_stage, '/Scene').GetPrim()
        PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
        collider_prim = UsdGeom.Cube.Define(material_stage, '/Collider').GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider_prim)
        deformable_prim = UsdGeom.Xform.Define(material_stage, '/Deformable').GetPrim()
        assert deformable_prim.ApplyAPI('OmniPhysicsDeformableBodyAPI')
        plain_prim = UsdGeom.Xform.Define(material_stage, '/Plain').GetPrim()
        checked = []
        for prim in (scene_prim, collider_prim, deformable_prim, plain_prim):
            bind_physics_material(str(prim.GetPath()), '/Material', stage=material_stage)
            bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
            if prim == plain_prim:
                assert not bound
            else:
                assert bound.GetPath() == material.GetPath(), prim.GetPath()
            checked.append(str(prim.GetPath()))
        receipt['material_binding_checks'] = checked

        cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs,
                            enable_cameras=True)
        cfg.seed = 17
        if args.ref_motions_path is not None:
            if not hasattr(cfg, 'ref_motions_path'):
                raise ValueError('This task has no ref_motions_path configuration')
            cfg.ref_motions_path = str(args.ref_motions_path.expanduser().resolve(strict=True))
        receipt['ref_motions_path'] = str(getattr(cfg, 'ref_motions_path', ''))
        receipt['physics_dt'] = cfg.sim.dt
        receipt['decimation'] = cfg.decimation
        dump_yaml(str(output/'env.yaml'), cfg)
        env = gym.make(args.task, cfg=cfg, render_mode='rgb_array')
        base = env.unwrapped
        if SimulationManager.get_active_physics_engine() != 'physx':
            raise RuntimeError('Task migration must preserve the PhysX engine')
        robot = base.scene['robot']
        if not isinstance(robot.data.joint_pos, torch.Tensor):
            raise TypeError('Project asset interface must preserve Torch tensors')
        receipt['joint_names'] = list(robot.joint_names)
        receipt['body_names'] = list(robot.body_names)

        def finite(value, name):
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all():
                    raise RuntimeError('Non-finite tensor: '+name)
            elif isinstance(value, Mapping):
                for key, item in value.items():
                    finite(item, name+'/'+str(key))

        def rgb(value, name):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            pixels = np.asarray(value)
            if pixels.ndim == 4:
                pixels = pixels[0]
            if pixels.ndim != 3 or pixels.shape[-1] not in (3, 4) or min(pixels.shape[:2]) < 16:
                raise RuntimeError(f'Invalid RGB shape for {name}: {pixels.shape}')
            if pixels.dtype != np.uint8 or not np.isfinite(pixels).all():
                raise RuntimeError(f'Expected finite uint8 RGB for {name}, got {pixels.dtype}')
            pixels = pixels[..., :3]
            if pixels.max() == 0 or pixels.std() == 0:
                raise RuntimeError('Black or constant RGB output: '+name)
            return np.ascontiguousarray(pixels)

        def frame(value, name, step):
            pixels = rgb(value, name)
            filename = re.sub(r'[^a-zA-Z0-9_-]', '_', name)+f'_{step:04d}.png'
            Image.fromarray(pixels).save(output/filename)
            receipt['images'].append({'file': filename, 'shape': list(pixels.shape),
                                      'mean': float(pixels.mean()), 'std': float(pixels.std())})

        with torch.inference_mode():
            obs, _ = env.reset(seed=17)
        finite(obs, 'reset_observations')
        receipt['pid'] = os.getpid()
        inventory = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True,
                                   text=True, check=True, timeout=20)
        (output/'nvidia_after_startup.xml').write_text(inventory.stdout)
        # Warm up render products without advancing physics.
        before = SimulationManager.get_num_physics_steps()
        # rgb_array attaches its render product lazily. Attach before warming
        # the renderer; warming only existing scene sensors leaves this product
        # empty on its first capture.
        env.render()
        for _ in range(8):
            base.sim.render()
        if SimulationManager.get_num_physics_steps() != before:
            raise RuntimeError('Render warmup advanced physics')
        roots, quats, joints, velocities, counters = [], [], [], [], []
        terminated_count = truncated_count = 0
        actions = torch.zeros((base.num_envs, base.action_manager.total_action_dim),
                              device=base.device)
        with torch.inference_mode():
            for step in range(args.steps):
                obs, reward, terminated, truncated, _ = env.step(actions)
                finite(obs, 'observations')
                finite(reward, 'reward')
                for name, value in (('root_pos_w', robot.data.root_pos_w),
                                    ('root_quat_w', robot.data.root_quat_w),
                                    ('joint_pos', robot.data.joint_pos),
                                    ('joint_vel', robot.data.joint_vel)):
                    finite(value, name)
                norm = torch.linalg.vector_norm(robot.data.root_quat_w, dim=-1)
                if not torch.allclose(norm, torch.ones_like(norm), atol=1e-3, rtol=0):
                    raise RuntimeError('Robot root quaternion is not normalized')
                roots.append(robot.data.root_pos_w.detach().cpu().numpy().copy())
                quats.append(robot.data.root_quat_w.detach().cpu().numpy().copy())
                joints.append(robot.data.joint_pos.detach().cpu().numpy().copy())
                velocities.append(robot.data.joint_vel.detach().cpu().numpy().copy())
                counters.append(SimulationManager.get_num_physics_steps())
                terminated_count += int(terminated.sum().item())
                truncated_count += int(truncated.sum().item())
                before = SimulationManager.get_num_physics_steps()
                pixels = rgb(env.render(), 'third_person')
                if video is None:
                    height, width = pixels.shape[:2]
                    video = subprocess.Popen(
                        [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
                         '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                         '-framerate', str(1/(cfg.sim.dt*cfg.decimation)), '-i', 'pipe:0',
                         '-an', '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p',
                         str(output/'third_person.mp4')],
                        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                video.stdin.write(pixels.tobytes())
                if step in {15, args.steps//2, args.steps-1}:
                    frame(pixels, 'third_person', step)
                    for name, sensor in base.scene.sensors.items():
                        data = getattr(getattr(sensor, 'data', None), 'output', {})
                        if 'rgb' in data:
                            frame(data['rgb'], name, step)
                if SimulationManager.get_num_physics_steps() != before:
                    raise RuntimeError('Camera capture advanced physics')
        video.stdin.close()
        video.wait(timeout=30)
        if video.returncode:
            raise RuntimeError('Video encoding failed: '+video.stderr.read().decode(errors='replace'))
        video.stderr.close()
        if not (output/'third_person.mp4').stat().st_size:
            raise RuntimeError('Encoded video is empty')
        probe = subprocess.run(
            [ffprobe, '-v', 'error', '-select_streams', 'v:0', '-count_frames',
             '-show_entries', 'stream=nb_read_frames,width,height,avg_frame_rate,duration',
             '-of', 'json', str(output/'third_person.mp4')],
            capture_output=True, text=True, check=True, timeout=30)
        stream = json.loads(probe.stdout)['streams'][0]
        if int(stream['nb_read_frames']) != len(roots):
            raise RuntimeError('Encoded video frame count differs from stepped frames')
        if (int(stream['height']), int(stream['width'])) != pixels.shape[:2]:
            raise RuntimeError('Encoded video resolution differs from rendered RGB')
        receipt['video'] = {'file': 'third_person.mp4', 'frames_submitted': len(roots),
                            'fps': 1/(cfg.sim.dt*cfg.decimation), 'ffprobe': stream}
        np.savez(output/'trajectory.npz', root_pos_w=np.asarray(roots),
                 root_quat_w=np.asarray(quats), joint_pos=np.asarray(joints),
                 joint_vel=np.asarray(velocities), physics_step=np.asarray(counters))
        # Keep reset in the same inference context as stepping: cached simulator
        # tensors allocated there cannot legally be mutated outside that context.
        # Save the stepped trace first so a reset failure retains its evidence.
        with torch.inference_mode():
            obs, _ = env.reset(seed=17)
        finite(obs, 'final_reset_observations')
        receipt.update(completed=True, steps_completed=len(roots),
                       terminated_count=terminated_count, truncated_count=truncated_count,
                       task_success_assessed=False, finished=time.time())
    except BaseException:
        receipt.update(error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2))
        try:
            if video is not None:
                if video.stdin is not None and not video.stdin.closed:
                    try:
                        video.stdin.close()
                    except BrokenPipeError:
                        pass
                if video.poll() is None:
                    video.terminate()
                    video.wait(timeout=10)
                if video.stderr is not None and not video.stderr.closed:
                    video.stderr.close()
            if env is not None:
                env.close()
        except BaseException:
            receipt.update(completed=False, cleanup_error=traceback.format_exc())
            raise
        finally:
            try:
                if app is not None:
                    receipt_path.write_text(json.dumps(receipt, indent=2))
                    app.close(exit_code=0 if receipt.get('completed') else 1)
            except BaseException:
                receipt.update(completed=False, cleanup_error=traceback.format_exc())
                raise
            finally:
                receipt_path.write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
