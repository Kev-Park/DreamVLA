"""Render the unchanged kitchen in both approved SDKs to diagnose asset errors.

Static visual fixture only: identical USD, transform, light and camera; no robot,
policy, physics evolution or task-success claim. Source layers are never saved.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--usd', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    version = importlib.metadata.version('isaacsim')
    assert version.startswith(('5.1.', '6.2.')), version
    report = dict(completed=False, started=time.time(), scope=__doc__, isaacsim=version,
        usd=str(args.usd), usd_sha256=hashlib.sha256(args.usd.read_bytes()).hexdigest(), images=[])
    app = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, active_gpu=0, physics_gpu=0,
            multi_gpu=False, fast_shutdown=True, width=640, height=480,
            renderer='RayTracedLighting', extra_args=[
                '--/app/settings/loadUserConfig=0', '--/app/settings/persistent=0']))
        import numpy as np
        import omni.usd
        import omni.replicator.core as rep
        from PIL import Image
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        root = UsdGeom.Xform.Define(stage, '/World')
        stage.SetDefaultPrim(root.GetPrim())
        kitchen = UsdGeom.Xform.Define(stage, '/World/KitchenVisual')
        kitchen.GetPrim().GetReferences().AddReference(str(args.usd))
        # Same backdrop transform as G1PickCamBinaryFingersEnvCfg.
        kitchen.AddTranslateOp().Set(Gf.Vec3d(2.07, 1., 0.))
        kitchen.AddScaleOp().Set(Gf.Vec3f(1., 1., .89*.800/.824))
        disabled = dict(collision=0, rigid_body=0, joint=0)
        dependencies = []
        for prim in Usd.PrimRange(kitchen.GetPrim()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                disabled['collision'] += 1
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
                disabled['rigid_body'] += 1
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
                disabled['joint'] += 1
            for attr in prim.GetAttributes():
                if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                    continue
                value = attr.Get()
                if value and value.path:
                    dependencies.append(dict(attribute=str(attr.GetPath()),
                        path=value.path, resolved_path=value.resolvedPath))
        report.update(disabled_physics=disabled, asset_dependencies=dependencies)
        light = UsdLux.DomeLight.Define(stage, '/World/FixtureLight')
        light.CreateIntensityAttr(1000.)
        camera = UsdGeom.Camera.Define(stage, '/World/Camera')
        camera.CreateFocalLengthAttr(18.1476)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(20.955*480/640)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.1, 10000.))
        camera.AddTranslateOp().Set(Gf.Vec3d(1.036, 3.15, 1.31))
        camera.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(
            -.11727178463934727, Gf.Vec3d(-.09523866571812599,.6232015643149337,.7673309164338117)))
        product = rep.create.render_product('/World/Camera', (640, 480))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
        rgb.attach([product])
        for index in range(3):
            rep.orchestrator.step(rt_subframes=8, pause_timeline=True)
            pixels = np.asarray(rgb.get_data())
            assert pixels.shape[:2] == (480, 640) and pixels.shape[2] in (3, 4), pixels.shape
            assert pixels.dtype == np.uint8 and pixels.max() > 0 and pixels.std() > 0
            path = args.output/f'kitchen_{index:02d}.png'
            Image.fromarray(pixels[..., :3]).save(path)
            report['images'].append(dict(file=path.name, shape=list(pixels.shape),
                mean=float(pixels[..., :3].mean()), std=float(pixels[..., :3].std())))
        rgb.detach([product])
        report.update(completed=True, source_layer_changed=False, task_success_assessed=False)
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['finished'] = time.time()
        (args.output/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
        if app is not None:
            # Static fixture has no writers or simulation buffers to flush.
            app.close(wait_for_replicator=False, skip_cleanup=True,
                exit_code=0 if report['completed'] else 1)


if __name__ == '__main__':
    main()
