# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Translate the existing converter contract to Isaac Sim 6's public importer.

Reference: Isaac Lab 3.0's URDF converter and NVIDIA's Isaac Sim 6.2 importer API.
The requested public USD filename is retained with a relative sublayer wrapper;
the SDK owns the generated layered asset and its internal reference paths.
"""
import os
from pathlib import Path


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
