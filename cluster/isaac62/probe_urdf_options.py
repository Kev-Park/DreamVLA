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
<link name="arm"><inertial><origin xyz="0 0 .2"/><mass value="1"/><inertia ixx=".1" ixy="0" ixz="0" iyy=".1" iyz="0" izz=".01"/></inertial><visual><geometry><cylinder radius=".05" length=".4"/></geometry></visual><collision><geometry><cylinder radius=".05" length=".4"/></geometry></collision></link>
<link name="finger"><inertial><origin xyz="0 0 .1"/><mass value=".2"/><inertia ixx=".01" ixy="0" ixz="0" iyy=".01" iyz="0" izz=".001"/></inertial><collision><geometry><box size=".02 .02 .2"/></geometry></collision></link>
<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/><origin xyz="0 0 .1" rpy="0 .1 0"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="2"/><dynamics damping=".27" stiffness=".31"/></joint>
<joint name="finger_hinge" type="revolute"><parent link="arm"/><child link="finger"/><origin xyz="0 0 .4"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="5" velocity="2"/><mimic joint="hinge" multiplier="-1" offset=".1"/></joint>
</robot>"""


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', help='Run selected cases independently; by default run all.')
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
            root_name=dict(root_link_name='base'), root_interior=dict(root_link_name='arm'),
            floating_default=dict(fix_base=False), floating_root=dict(fix_base=False, root_link_name='arm'),
            capsule_instanceable=dict(replace_cylinders_with_capsules=True, make_instanceable=True),
            natural_force={}, natural_acceleration={}, natural_velocity={}, natural_none={},
            natural_patterns={}, natural_partial={}, natural_mimic=dict(convert_mimic_joints_to_normal_joints=True))
        if args.cases:
            unknown = set(args.cases) - cases.keys()
            if unknown:
                raise ValueError(f'Unknown cases: {sorted(unknown)}')
            cases = {name: cases[name] for name in args.cases}
        for name, options in cases.items():
            try:
                cfg = UrdfConverterCfg(asset_path=str(source), usd_dir=str(args.output/name),
                    force_usd_conversion=True, merge_fixed_joints=False,
                    **{'make_instanceable': False, 'fix_base': True, **options})
                cfg.joint_drive.gains = UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=1., damping=.2)
                if name.startswith('natural_'):
                    cfg.joint_drive.gains = UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg(
                        natural_frequency=8., damping_ratio=.7)
                    cfg.joint_drive.drive_type = 'force' if name == 'natural_force' else 'acceleration'
                    if name != 'natural_acceleration':
                        cfg.joint_drive.drive_type = 'force'
                    if name in ('natural_velocity', 'natural_none'):
                        cfg.joint_drive.target_type = 'velocity' if name == 'natural_velocity' else 'none'
                    elif name == 'natural_patterns':
                        cfg.joint_drive.drive_type = {'hinge': 'force', '^finger': 'acceleration'}
                        cfg.joint_drive.target_type = {'hinge': 'position', '^finger': 'velocity'}
                        cfg.joint_drive.gains.natural_frequency = {'hinge': 8., '^finger': 5.}
                        cfg.joint_drive.gains.damping_ratio = {'hinge': .7, '^finger': .4}
                    elif name == 'natural_partial':
                        cfg.joint_drive.gains.natural_frequency = {'^hinge$': 8.}
                path = Path(UrdfConverter(cfg).usd_path)
                stage = Usd.Stage.Open(str(path))
                assert stage and stage.GetDefaultPrim()
                row = dict(completed=True, requested_options=options, usd_path=str(path), bodies=[], geometry=[], joints=[], articulation_roots=[], root_settings={}, fixed_joints=[])
                row['drive_request'] = dict(drive_type=cfg.joint_drive.drive_type, target_type=cfg.joint_drive.target_type,
                    natural_frequency=getattr(cfg.joint_drive.gains, 'natural_frequency', None),
                    damping_ratio=getattr(cfg.joint_drive.gains, 'damping_ratio', None))
                for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
                    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                        row['articulation_roots'].append(str(prim.GetPath()))
                        row['root_settings'][str(prim.GetPath())] = {
                            attr.GetName(): attr.Get() for attr in prim.GetAttributes()
                            if attr.GetName().startswith('physxArticulation:') and
                            ('solver' in attr.GetName() or 'SelfCollisions' in attr.GetName())}
                    if prim.IsA(UsdPhysics.FixedJoint):
                        joint = UsdPhysics.FixedJoint(prim)
                        row['fixed_joints'].append(dict(name=prim.GetName(),
                            body0=[str(p) for p in joint.GetBody0Rel().GetTargets()],
                            body1=[str(p) for p in joint.GetBody1Rel().GetTargets()],
                            pos0=list(joint.GetLocalPos0Attr().Get()), pos1=list(joint.GetLocalPos1Attr().Get()),
                            rot0=str(joint.GetLocalRot0Attr().Get()), rot1=str(joint.GetLocalRot1Attr().Get())))
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        row['bodies'].append(prim.GetName())
                    if prim.GetTypeName() in ('Cylinder', 'Capsule'):
                        shape = UsdGeom.Cylinder(prim) if prim.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(prim)
                        row['geometry'].append(dict(type=prim.GetTypeName(), path=str(prim.GetPath()),
                            collision=prim.HasAPI(UsdPhysics.CollisionAPI), instance_proxy=prim.IsInstanceProxy(),
                            height=shape.GetHeightAttr().Get(), radius=shape.GetRadiusAttr().Get(),
                            axis=shape.GetAxisAttr().Get(), extent=[list(x) for x in shape.GetExtentAttr().Get()]))
                    if prim.IsA(UsdPhysics.RevoluteJoint):
                        drive = UsdPhysics.DriveAPI.Get(prim, 'angular')
                        inertia = prim.GetAttribute('physics:JointEquivalentInertia')
                        row['joints'].append(dict(name=prim.GetName(), schemas=list(prim.GetAppliedSchemas()),
                            axis=UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get(),
                            lower_limit=UsdPhysics.RevoluteJoint(prim).GetLowerLimitAttr().Get(),
                            upper_limit=UsdPhysics.RevoluteJoint(prim).GetUpperLimitAttr().Get(),
                            max_velocity=prim.GetAttribute('physxJoint:maxJointVelocity').Get(),
                            stiffness=drive.GetStiffnessAttr().Get(), damping=drive.GetDampingAttr().Get(),
                            drive_type=drive.GetTypeAttr().Get(), equivalent_inertia=inertia.Get() if inertia else None,
                            attributes={attr.GetName(): str(attr.Get()) for attr in prim.GetAttributes()
                                if 'mimic' in attr.GetName().lower()},
                            relationships={rel.GetName(): [str(x) for x in rel.GetTargets()] for rel in prim.GetRelationships()}))
                assert len(row['bodies']) == 3 and len(row['joints']) == 2, row
                report['cases'][name] = row
            except Exception:
                report['cases'][name] = dict(completed=False, requested_options=options, error=traceback.format_exc())
        report['measurement_completed'] = True
        report['completed'] = all(row['completed'] for row in report['cases'].values())
        if not report['completed']:
            raise RuntimeError('One or more independent converter cases failed; see receipt.json')
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        (args.output/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
        if app is not None:
            # Kit can terminate the process inside close(), swallowing a pending
            # Python exception. Keep its exit status consistent with the receipt.
            import inspect
            supported = inspect.signature(app.close).parameters
            options = dict(wait_for_replicator=False, skip_cleanup=True,
                exit_code=0 if report['completed'] else 1)
            app.close(**{key: value for key, value in options.items() if key in supported})


if __name__ == '__main__':
    main()
