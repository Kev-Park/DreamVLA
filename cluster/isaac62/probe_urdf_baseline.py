"""Convert the same original URDF in the existing Isaac 5.1 baseline profile.

Persist the result before fast shutdown; legacy exit zero alone proves nothing.
This fixture runs the original converter, not the SDK 6 compatibility bridge.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import time
import traceback
import xml.etree.ElementTree as ET


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    version = importlib.metadata.version('isaacsim')
    assert version.startswith('5.1.')
    args.headless = True
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, started=time.time(), isaacsim=version,
                  urdf_sha256=hashlib.sha256(args.urdf.read_bytes()).hexdigest(), scope=__doc__)
    app = None
    try:
        app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from pxr import Usd, UsdGeom, UsdPhysics
        report['converter_source'] = inspect.getfile(UrdfConverter)
        cfg = UrdfConverterCfg(asset_path=str(args.urdf), usd_dir=str(args.output/'asset'),
                               usd_file_name='g1_29dof_with_hand.usd', force_usd_conversion=True,
                               make_instanceable=True, fix_base=False, merge_fixed_joints=False,
                               self_collision=False, collider_type='convex_hull', collision_from_visuals=False)
        cfg.joint_drive.drive_type = 'force'
        cfg.joint_drive.target_type = 'position'
        cfg.joint_drive.gains.stiffness = 0.0
        cfg.joint_drive.gains.damping = 0.0
        path = Path(UrdfConverter(cfg).usd_path)
        stage = Usd.Stage.Open(str(path))
        assert stage and stage.GetDefaultPrim()
        bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
        report.update(usd_path=str(path), bodies=len(bodies), movable_joints=len(joints))
        # The original importer keeps even non-inertial fixed sensor links as
        # rigid bodies; the matched G1 baseline has 54 bodies, not 50.
        original = ET.parse(args.urdf).getroot()
        assert len(bodies) == len(original.findall('link')) and len(joints) == 43
        sensor = next(p for p in stage.Traverse() if p.GetName() == 'mid360_link')
        report.update(completed=True, usd_path=str(path), bodies=len(bodies), movable_joints=len(joints),
                      mid360_local=[[float(v) for v in row] for row in UsdGeom.Xformable(sensor).GetLocalTransformation()])
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['finished'] = time.time()
        (args.output/'receipt.json').write_text(json.dumps(report, indent=2))
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
