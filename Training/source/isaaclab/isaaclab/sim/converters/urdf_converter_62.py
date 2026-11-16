# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Translate the existing converter contract to Isaac Sim 6's public importer.

Reference: Isaac Lab 3.0's URDF converter and NVIDIA's Isaac Sim 6.2 importer API.
The requested public USD filename is retained with a relative sublayer wrapper;
the SDK owns the generated layered asset and its internal reference paths.
"""
import os
from pathlib import Path
import math
import struct
import xml.etree.ElementTree as ET


def convert_asset(converter, cfg):
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from pxr import Usd, UsdGeom
    from .urdf_converter_cfg import UrdfConverterCfg

    for name in ("root_link_name",):
        if getattr(cfg, name):
            raise NotImplementedError(f"Isaac Sim 6 URDF importer cannot preserve {name}")
    drive_type = target_type = stiffness = damping = None
    if cfg.joint_drive is not None:
        drive = cfg.joint_drive
        if not isinstance(drive.gains, UrdfConverterCfg.JointDriveCfg.PDGainsCfg):
            raise NotImplementedError("Isaac Sim 6 removed natural-frequency URDF drive conversion")
        drive_type, target_type = drive.drive_type, drive.target_type
        stiffness, damping = drive.gains.stiffness, drive.gains.damping
    collision = {"convex_hull": "Convex Hull", "convex_decomposition": "Convex Decomposition"}
    if cfg.collider_type not in collision:
        raise ValueError(f"Unknown collision simplification: {cfg.collider_type}")
    config = URDFImporterConfig(
        urdf_path=os.path.normpath(cfg.asset_path), usd_path=os.path.normpath(converter.usd_dir),
        merge_fixed_joints=cfg.merge_fixed_joints, merge_mesh=False,
        collision_from_visuals=cfg.collision_from_visuals,
        collision_type=collision[cfg.collider_type], allow_self_collision=cfg.self_collision,
        fix_base=cfg.fix_base, link_density=cfg.link_density if cfg.link_density > 0 else None,
        joint_drive_type=drive_type, joint_target_type=target_type,
        override_joint_stiffness=stiffness, override_joint_damping=damping,
        run_asset_transformer=True, run_multi_physics_conversion=True,
    )
    generated = URDFImporter(config).import_urdf()
    if not generated or not Path(generated).is_file():
        raise RuntimeError(f"URDF importer produced no asset: {generated}")
    source = Usd.Stage.Open(str(generated))
    root = source.GetDefaultPrim() if source is not None else None
    if not root or not root.IsValid():
        raise RuntimeError(f"URDF asset has no default prim: {generated}")
    if not cfg.make_instanceable:
        from ..utils import make_uninstanceable
        make_uninstanceable(root.GetPath(), stage=source)
        source.GetRootLayer().Save()
    target = Path(converter.usd_path)
    if target.resolve() != Path(generated).resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        wrapper = Usd.Stage.CreateNew(str(target))
        wrapper.GetRootLayer().subLayerPaths = [os.path.relpath(generated, target.parent)]
        wrapper.SetDefaultPrim(wrapper.GetPrimAtPath(root.GetPath()))
        UsdGeom.SetStageUpAxis(wrapper, UsdGeom.GetStageUpAxis(source))
        UsdGeom.SetStageMetersPerUnit(wrapper, UsdGeom.GetStageMetersPerUnit(source))
        physics = wrapper.GetDefaultPrim().GetVariantSet("Physics")
        if "physx" not in physics.GetVariantNames():
            raise RuntimeError("URDF importer produced no PhysX physics variant")
        physics.SetVariantSelection("physx")
        wrapper.GetRootLayer().Save()
    else:
        physics = root.GetVariantSet("Physics")
        if "physx" not in physics.GetVariantNames():
            raise RuntimeError("URDF importer produced no PhysX physics variant")
        physics.SetVariantSelection("physx")
        source.GetRootLayer().Save()
    _preserve_empty_fixed_links(target, cfg)
    _preserve_joint_velocity_limits(target, cfg)
    _preserve_collision_capsules(target, cfg)
    _preserve_mimic_joints(target, cfg)


def _preserve_mimic_joints(usd_path, cfg):
    """Preserve the original importer's observed (inverted) mimic flag.

    Original configureMimicAPI: NVIDIA urdf-importer-extension, Apache-2.0,
    commit 9908f0341ee4c375a94d594c4eeb318dea3f4cf2, UrdfImporter.cpp.
    True parses mimic constraints; false leaves independent driven joints.
    PD gain overrides do not change its default 25/.005 constraint parameters.
    Natural-frequency gain configuration is rejected before reaching here.
    """
    if not cfg.convert_mimic_joints_to_normal_joints:
        return
    from pxr import PhysxSchema, Tf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    assert stage and stage.GetDefaultPrim()
    joints = {prim.GetName(): prim for prim in stage.Traverse()
              if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)}
    elements = {Tf.MakeValidIdentifier(element.get('name')): element
                for element in ET.parse(cfg.asset_path).getroot().findall('joint')}
    visiting, completed = set(), set()
    f32 = lambda value: struct.unpack('f', struct.pack('f', value))[0]

    def preserve(name):
        if name in completed:
            return
        if name in visiting:
            raise ValueError(f'Cyclic URDF mimic dependency: {name}')
        visiting.add(name)
        element = elements[name]
        mimic = element.find('mimic')
        if mimic is not None:
            source_name = Tf.MakeValidIdentifier(mimic.get('joint'))
            if source_name not in joints or name not in joints:
                raise ValueError(f'Cannot resolve URDF mimic {name} -> {source_name}')
            preserve(source_name)
            prim, source = joints[name], joints[source_name]
            axis_element = element.find('axis')
            axis = [float(x) for x in (axis_element.get('xyz', '1 0 0') if axis_element is not None else '1 0 0').split()]
            cardinal = [i for i in range(3) if abs(axis[i]) == 1 and all(axis[j] == 0 for j in range(3) if i != j)]
            if len(cardinal) != 1:
                raise NotImplementedError(f'Cannot preserve non-cardinal legacy mimic axis for {name}: {axis}')
            # Legacy leaves referenceJointAxis at its schema default rotX,
            # even when the reference's authored axis differs. Preserve it.
            api = PhysxSchema.PhysxMimicJointAPI.Apply(prim, 'rot'+'XYZ'[cardinal[0]])
            multiplier = f32(float(mimic.get('multiplier', '1')))
            offset = f32(float(mimic.get('offset', '0')))
            if not math.isfinite(multiplier) or not math.isfinite(offset):
                raise ValueError(f'Non-finite mimic coefficients for {name}')
            api.CreateGearingAttr(-multiplier)
            api.CreateOffsetAttr(offset)
            api.CreateNaturalFrequencyAttr(25.0)
            api.CreateDampingRatioAttr(.005)
            api.CreateReferenceJointRel().SetTargets([source.GetPath()])
            instance = 'linear' if prim.IsA(UsdPhysics.PrismaticJoint) else 'angular'
            prim.RemoveAPI(UsdPhysics.DriveAPI, instance)
            for attr in prim.GetAttributes():
                if attr.GetName().startswith('drive:'+instance+':'):
                    attr.Block()
            prim.GetAttribute('physxJoint:maxJointVelocity').Block()
            joint = UsdPhysics.PrismaticJoint(prim) if instance == 'linear' else UsdPhysics.RevoluteJoint(prim)
            reference = UsdPhysics.PrismaticJoint(source) if source.IsA(UsdPhysics.PrismaticJoint) else UsdPhysics.RevoluteJoint(source)
            lower, upper = reference.GetLowerLimitAttr().Get(), reference.GetUpperLimitAttr().Get()
            margin = f32(f32(.2)*f32(upper-lower))
            bounds = [f32(multiplier*f32(lower-margin)), f32(multiplier*f32(upper+margin))]
            joint.CreateLowerLimitAttr(min(bounds))
            joint.CreateUpperLimitAttr(max(bounds))
        visiting.remove(name)
        completed.add(name)

    for name, element in elements.items():
        if element.find('mimic') is not None:
            preserve(name)
    stage.GetRootLayer().Save()


def _preserve_collision_capsules(usd_path, cfg):
    """Match the legacy collision-only cylinder replacement in our USD layer.

    The original importer retained the cylinder length as capsule height (it
    did not subtract the caps) and parsed dimensions as float32. Keep visual
    shapes, transforms, material bindings and collision APIs intact. Primitive
    geometry emitted by the modern importer is not instanced; reject an
    unexpected instance proxy rather than silently dropping this option.
    """
    if not cfg.replace_cylinders_with_capsules:
        return
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    assert stage and stage.GetDefaultPrim()
    cylinders = [prim for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
                 if prim.IsA(UsdGeom.Cylinder) and prim.HasAPI(UsdPhysics.CollisionAPI)]
    if any(prim.IsInstanceProxy() for prim in cylinders):
        raise NotImplementedError("Cannot replace an instanced collision cylinder without changing instanceability")
    for prim in cylinders:
        cylinder = UsdGeom.Cylinder(prim)
        height = struct.unpack('f', struct.pack('f', cylinder.GetHeightAttr().Get()))[0]
        radius = struct.unpack('f', struct.pack('f', cylinder.GetRadiusAttr().Get()))[0]
        axis = cylinder.GetAxisAttr().Get()
        capsule = UsdGeom.Capsule.Define(stage, prim.GetPath())
        capsule.CreateHeightAttr(height)
        capsule.CreateRadiusAttr(radius)
        capsule.CreateAxisAttr(axis)
        # Python exposes Boundable.ComputeExtent(time), not the C++ static
        # capsule overload. The analytic local half-extents are identical.
        half_extents = Gf.Vec3f(radius)
        half_extents[{'X': 0, 'Y': 1, 'Z': 2}[axis]] = height/2 + radius
        capsule.CreateExtentAttr([-half_extents, half_extents])
    stage.GetRootLayer().Save()


def _preserve_empty_fixed_links(usd_path, cfg):
    """Retain the original importer's physical empty sensor links.

    The SDK 5.1 importer kept empty, unmerged fixed links as rigid bodies.
    Its zero authored mass resolves to 1 kg in PhysX, with isotropic 1e-4
    inertia. SDK 6 instead leaves them as massless Xforms. For the original
    G1 this silently removes four bodies and 4 kg. Author the measured legacy
    properties explicitly in our wrapper; do not modify the SDK-generated
    layers or add visual/collision geometry. Merged fixed links stay merged.
    """
    if cfg.merge_fixed_joints:
        return
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    robot = ET.parse(cfg.asset_path).getroot()
    empty = {link.get("name") for link in robot.findall("link")
             if all(link.find(field) is None for field in ("inertial", "visual", "collision"))}
    if not empty:
        return
    stage = Usd.Stage.Open(str(usd_path))
    assert stage and stage.GetDefaultPrim()
    prims = list(stage.Traverse())
    root_path = stage.GetDefaultPrim().GetPath()
    restored = []
    for joint in robot.findall("joint"):
        child = joint.find("child").get("link")
        if child not in empty or joint.get("type") != "fixed":
            continue
        matches = [prim for prim in prims if prim.GetName() == child and prim.IsA(UsdGeom.Xform)]
        if len(matches) != 1:
            raise RuntimeError(f"Cannot identify empty URDF link {child}: {matches}")
        body = matches[0]
        if body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        parent_name = joint.find("parent").get("link")
        parents = [prim for prim in prims if prim.GetName() == parent_name and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(parents) != 1:
            raise RuntimeError(f"Cannot identify physical parent {parent_name} of empty link {child}")
        UsdPhysics.RigidBodyAPI.Apply(body)
        mass = UsdPhysics.MassAPI.Apply(body)
        mass.CreateMassAttr(1.0)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-4))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
        origin = joint.find("origin")
        xyz = [float(value) for value in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
        rpy = [float(value) for value in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
        sx, sy, sz = [math.sin(value/2) for value in rpy]
        cx, cy, cz = [math.cos(value/2) for value in rpy]
        rotation = Gf.Quatf(cx*cy*cz+sx*sy*sz,
                           Gf.Vec3f(sx*cy*cz-cx*sy*sz, cx*sy*cz+sx*cy*sz, cx*cy*sz-sx*sy*cz))
        fixed = UsdPhysics.FixedJoint.Define(stage, root_path.AppendPath("LegacyFixedLinks").AppendChild(joint.get("name")))
        fixed.CreateBody0Rel().SetTargets([parents[0].GetPath()])
        fixed.CreateBody1Rel().SetTargets([body.GetPath()])
        fixed.CreateLocalPos0Attr(Gf.Vec3f(*xyz))
        fixed.CreateLocalRot0Attr(rotation)
        fixed.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        fixed.CreateLocalRot1Attr(Gf.Quatf(1.0))
        restored.append(child)
    if restored:
        stage.GetRootLayer().customLayerData = {
            **stage.GetRootLayer().customLayerData,
            "dreamvla_legacy_empty_fixed_links": ",".join(sorted(restored)),
        }
        stage.GetRootLayer().Save()


def _preserve_joint_velocity_limits(usd_path, cfg):
    """Author the original importer's PhysX velocity limits in our wrapper.

    The modern pipeline writes newton:velocityLimit but does not author the
    PhysX maxJointVelocity field. The original converter writes angular limits
    in degrees/s, linear limits in m/s, and FLT_MAX for non-positive limits.
    This restores raw-import behavior even when no actuator overrides it.
    """
    from pxr import PhysxSchema, Tf, Usd, UsdPhysics

    robot = ET.parse(cfg.asset_path).getroot()
    expected = {}
    for element in robot.findall("joint"):
        if element.get("type") not in ("revolute", "continuous", "prismatic"):
            continue
        name = Tf.MakeValidIdentifier(element.get("name"))
        if name in expected:
            raise ValueError(f"URDF joint names collide after USD sanitization: {name}")
        limit = element.find("limit")
        velocity = float(limit.get("velocity", "0")) if limit is not None else 0.0
        if not math.isfinite(velocity):
            raise ValueError(f"Non-finite URDF velocity limit for {name}")
        # The original C++ parser and AddSingleJoint use float, including
        # 180.0f / static_cast<float>(M_PI). Preserve those rounding boundaries.
        f32 = lambda value: struct.unpack("f", struct.pack("f", value))[0]
        velocity = f32(velocity)
        scale = 1.0 if element.get("type") == "prismatic" else f32(180.0/f32(math.pi))
        expected[name] = f32(velocity*scale) if velocity > 0 else 3.4028234663852886e38
    stage = Usd.Stage.Open(str(usd_path))
    assert stage and stage.GetDefaultPrim()
    found = set()
    for prim in stage.Traverse():
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
            continue
        name = prim.GetName()
        if name not in expected:
            continue
        PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(expected[name])
        found.add(name)
    if found != set(expected):
        raise RuntimeError(f"Cannot preserve velocity limits for missing joints: {sorted(set(expected)-found)}")
    stage.GetRootLayer().Save()
