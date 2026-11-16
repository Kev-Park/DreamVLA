"""Measure original optional URDF conversion contracts on a small owned fixture.

No research robot or policy is trained. The same source can later audit the 6.2
bridge against these actual converter outputs; unsupported cases stay failures.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import traceback


URDF = """<robot name="option_fixture">
<link name="base"><inertial><mass value="2"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial></link>
<link name="arm"><inertial><origin xyz="0 0 .2"/><mass value="1"/><inertia ixx=".1" ixy="0" ixz="0" iyy=".1" iyz="0" izz=".01"/></inertial><collision><geometry><cylinder radius=".05" length=".4"/></geometry></collision></link>
<link name="finger"><inertial><origin xyz="0 0 .1"/><mass value=".2"/><inertia ixx=".01" ixy="0" ixz="0" iyy=".01" iyz="0" izz=".001"/></inertial><collision><geometry><box size=".02 .02 .2"/></geometry></collision></link>
<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/><origin xyz="0 0 .1" rpy="0 .1 0"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="2"/></joint>
<joint name="finger_hinge" type="revolute"><parent link="arm"/><child link="finger"/><origin xyz="0 0 .4"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="5" velocity="2"/><mimic joint="hinge" multiplier="-1" offset=".1"/></joint>
</robot>"""


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, isaacsim=importlib.metadata.version('isaacsim'), cases={})
    app = None
    try:
        app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from pxr import Usd, UsdGeom, UsdPhysics
        source = args.output/'option_fixture.urdf'
        source.write_text(URDF)
        cases = dict(default={}, capsule=dict(replace_cylinders_with_capsules=True),
            mimic_normal=dict(convert_mimic_joints_to_normal_joints=True),
            root_name=dict(root_link_name='base'), natural_force={}, natural_acceleration={})
        for name, options in cases.items():
            cfg = UrdfConverterCfg(asset_path=str(source), usd_dir=str(args.output/name),
                force_usd_conversion=True, fix_base=True, merge_fixed_joints=False,
                make_instanceable=False, **options)
            cfg.joint_drive.gains = UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=1., damping=.2)
            if name.startswith('natural_'):
                cfg.joint_drive.gains = UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg(
                    natural_frequency=8., damping_ratio=.7)
                cfg.joint_drive.drive_type = 'force' if name == 'natural_force' else 'acceleration'
            path = Path(UrdfConverter(cfg).usd_path)
            stage = Usd.Stage.Open(str(path))
            assert stage and stage.GetDefaultPrim()
            row = dict(usd_path=str(path), bodies=[], geometry=[], joints=[])
            for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    row['bodies'].append(prim.GetName())
                if prim.GetTypeName() in ('Cylinder', 'Capsule'):
                    shape = UsdGeom.Cylinder(prim) if prim.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(prim)
                    row['geometry'].append(dict(type=prim.GetTypeName(), path=str(prim.GetPath()),
                        height=shape.GetHeightAttr().Get(), radius=shape.GetRadiusAttr().Get(),
                        axis=shape.GetAxisAttr().Get(), extent=[list(x) for x in shape.GetExtentAttr().Get()]))
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    drive = UsdPhysics.DriveAPI.Get(prim, 'angular')
                    inertia = prim.GetAttribute('physics:JointEquivalentInertia')
                    row['joints'].append(dict(name=prim.GetName(), schemas=list(prim.GetAppliedSchemas()),
                        stiffness=drive.GetStiffnessAttr().Get(), damping=drive.GetDampingAttr().Get(),
                        drive_type=drive.GetTypeAttr().Get(), equivalent_inertia=inertia.Get() if inertia else None))
            assert len(row['bodies']) == 3 and len(row['joints']) == 2, row
            report['cases'][name] = row
        report['completed'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        (args.output/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
