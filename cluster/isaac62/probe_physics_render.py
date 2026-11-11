"""Headless Isaac 6.2 PhysX/RTX foundation probe, with no learned policy.

Run on a reviewed cluster allocation, never the local Windows workstation.
This gate does not establish robot/task/policy compatibility.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time


def simulation_config(gpu):
    if gpu < 0:
        raise ValueError('GPU must be a nonnegative physical-device index')
    return dict(headless=True, active_gpu=gpu, physics_gpu=gpu, multi_gpu=False,
                width=128, height=128, renderer='RayTracedLighting')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--frames', type=int, default=64)
    parser.add_argument('--output', type=Path,
                        default=Path.home()/'kevin/eval_videos/isaac62_foundation_probe')
    args = parser.parse_args()
    if args.frames < 16:
        parser.error('At least 16 frames are needed for the physics/render gate')
    if sys.version_info[:2] != (3, 12):
        raise SystemExit('Run in the isolated Python 3.12 migration environment')
    version = importlib.metadata.version('isaacsim')
    if not version.startswith('6.2.'):
        raise SystemExit('This gate requires Isaac Sim 6.2, found '+version)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(started=time.time(), isaacsim=version, python=sys.version,
                   config=simulation_config(args.gpu), completed=False,
                   scope='Foundation only; no robot, learned policy or study controller')
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2))

    from isaacsim import SimulationApp
    app = SimulationApp(receipt['config'])
    try:
        import numpy as np
        import omni.physics.tensors as tensors
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager
        from PIL import Image
        from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics

        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        world = UsdGeom.Xform.Define(stage, '/World')
        stage.SetDefaultPrim(world.GetPrim())
        scene = UsdPhysics.Scene.Define(stage, '/World/physicsScene')
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0., 0., -1.))
        scene.CreateGravityMagnitudeAttr(9.81)
        physics = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physics.CreateEnableGPUDynamicsAttr(True)
        physics.CreateBroadphaseTypeAttr('GPU')
        physics.CreateTimeStepsPerSecondAttr(200)

        floor = UsdGeom.Cube.Define(stage, '/World/Floor')
        floor.CreateSizeAttr(1.)
        floor.AddTranslateOp().Set(Gf.Vec3d(0., 0., -.05))
        floor.AddScaleOp().Set(Gf.Vec3f(5., 5., .1))
        floor.CreateDisplayColorAttr([Gf.Vec3f(.3, .3, .3)])
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        cube = UsdGeom.Cube.Define(stage, '/World/ProbeCube')
        cube.CreateSizeAttr(.2)
        cube.AddTranslateOp().Set(Gf.Vec3d(0., 0., 1.))
        cube.CreateDisplayColorAttr([Gf.Vec3f(.8, .15, .1)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1.)
        light = UsdLux.DomeLight.Define(stage, '/World/Light')
        light.CreateIntensityAttr(1000.)
        camera = UsdGeom.Camera.Define(stage, '/World/Camera')
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(2., 2., 1.6), Gf.Vec3d(0., 0., .4), Gf.Vec3d(0., 0., 1.)).GetInverse())
        product = rep.create.render_product('/World/Camera', (128, 128))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
        rgb.attach([product])
        stage.GetRootLayer().Export(str(output/'probe_scene.usda'))
        SimulationManager.setup_simulation(dt=.005, device=f'cuda:{args.gpu}')
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(8):
            app.update()
        sim = tensors.create_simulation_view('numpy')
        sim.set_subspace_roots('/')
        body = sim.create_rigid_body_view('/World/ProbeCube')
        assert SimulationManager.get_active_physics_engine() == 'physx'
        assert str(SimulationManager.get_device()) == f'cuda:{args.gpu}'
        scenes = SimulationManager.get_physics_scenes()
        assert len(scenes) == 1 and scenes[0].get_enabled_gpu_dynamics()
        timeline.pause()
        # Reset through the same public tensor API used by policy environments.
        initial = np.asarray(body.get_transforms()).copy()
        initial[0, 2] = 1.
        ids = np.array([0], dtype=np.int32)
        body.set_transforms(initial, ids)
        body.set_velocities(np.zeros((1, 6), dtype=np.float32), ids)
        initial_steps = SimulationManager.get_num_physics_steps()
        positions = []; images = []; camera_stats = []
        for frame in range(args.frames):
            SimulationManager.step(steps=4)
            before_render = SimulationManager.get_num_physics_steps()
            app.update()
            assert SimulationManager.get_num_physics_steps() == before_render, 'Render advanced physics while paused'
            transforms = np.asarray(body.get_transforms()).copy()
            assert transforms.shape == (1, 7) and np.isfinite(transforms).all(), transforms
            positions.append(transforms[0, :3])
            image = np.asarray(rgb.get_data())
            if image.size:
                assert image.shape[:2] == (128, 128), image.shape
                assert np.isfinite(image).all()
                camera_stats.append(dict(frame=frame, minimum=int(image.min()),
                                         maximum=int(image.max()), mean=float(image.mean())))
                images.append((frame, image[..., :3].astype(np.uint8).copy()))
        actual_steps = SimulationManager.get_num_physics_steps() - initial_steps
        assert actual_steps == args.frames * 4, actual_steps
        timeline.stop()
        trajectory = np.stack(positions)
        assert np.isfinite(trajectory).all()
        assert float(trajectory[:, 2].min()) < .99, 'Dynamic body did not fall'
        assert len(images) >= 16, 'Camera delivered too few images'
        assert any(float(image.mean()) > 0 for _, image in images), 'Camera output is entirely black'
        for i in sorted({0, len(images)//2, len(images)-1}):
            frame, image = images[i]
            Image.fromarray(image).save(output/f'frame_{frame:04d}.png')
        np.savez(output/'trajectory.npz', position=trajectory)
        (output/'nvidia_after.xml').write_text(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True))
        receipt.update(completed=True, finished=time.time(), physics_backend='PhysX',
                       gpu_dynamics_verified=True, simulation_device=str(SimulationManager.get_device()),
                       physics_steps=actual_steps, capture_frames=len(trajectory),
                       image_frames=len(images), z_min=float(trajectory[:, 2].min()),
                       z_max=float(trajectory[:, 2].max()), camera_stats=camera_stats)
        rgb.detach([product])
    finally:
        app.close()
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'camera_stats'}, indent=2))


if __name__ == '__main__':
    main()
