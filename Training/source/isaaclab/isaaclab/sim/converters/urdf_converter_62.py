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
import xml.etree.ElementTree as ET


def convert_asset(converter, cfg):
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from pxr import Usd, UsdGeom
    from .urdf_converter_cfg import UrdfConverterCfg

    for name in ("root_link_name", "convert_mimic_joints_to_normal_joints",
                 "replace_cylinders_with_capsules"):
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
