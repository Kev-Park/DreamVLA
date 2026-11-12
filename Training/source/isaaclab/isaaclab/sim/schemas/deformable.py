# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit volume-deformable USD schema ownership in the current PhysX API.

The physics and graphical meshes are separate prims in Isaac Sim 6.2. Keep the
public asset root and its visual mesh in place; author solver/cooking properties
on their documented replacement schemas rather than on removed Python classes.
"""
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
from omni.physx.scripts import deformableUtils


def _set(prim, name, value):
    if value is None:
        return
    attr = prim.GetAttribute(name)
    if not attr or not attr.Set(value):
        raise RuntimeError(f"Cannot set supported deformable attribute {name!r} on {prim.GetPath()}")


def define_volume_deformable_properties(prim_path, cfg, stage):
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise ValueError(f"Prim path {prim_path!r} is not valid")
    if not root.HasAPI('OmniPhysicsDeformableBodyAPI'):
        if root.IsA(UsdGeom.TetMesh):
            if not deformableUtils.set_physics_volume_deformable_body(stage, prim_path):
                raise RuntimeError(f"Could not configure volume deformable {prim_path}")
        else:
            if not root.IsA(UsdGeom.Imageable) or root.IsA(UsdGeom.Gprim):
                raise ValueError('A procedurally cooked volume deformable needs an Xform asset root with a Mesh child')
            meshes = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)]
            if len(meshes) != 1:
                raise ValueError(f"Expected one graphical mesh under {prim_path}, found {len(meshes)}")
            if not deformableUtils.create_auto_volume_deformable_hierarchy(
                    stage, prim_path, str(root.GetPath().AppendChild('simulationMesh')),
                    str(root.GetPath().AppendChild('collisionMesh')), str(meshes[0].GetPath()),
                    simulation_hex_mesh_enabled=cfg.simulation_hexahedral_resolution is not None,
                    cooking_src_simplification_enabled=cfg.collision_simplification,
                    set_visibility_with_guide_purpose=True):
                raise RuntimeError(f"Could not configure volume deformable hierarchy {prim_path}")
    return modify_volume_deformable_properties(prim_path, cfg, stage)


def modify_volume_deformable_properties(prim_path, cfg, stage):
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid() or not root.HasAPI('OmniPhysicsDeformableBodyAPI'):
        return False
    root.ApplyAPI('OmniPhysicsBodyAPI')
    root.ApplyAPI('PhysxBaseDeformableBodyAPI')
    _set(root, 'omniphysics:deformableBodyEnabled', cfg.deformable_enabled)
    _set(root, 'omniphysics:kinematicEnabled', cfg.kinematic_enabled)
    mapping = dict(solver_position_iteration_count='solverPositionIterationCount',
                   vertex_velocity_damping='linearDamping', sleep_damping='settlingDamping',
                   sleep_threshold='sleepThreshold', settling_threshold='settlingThreshold',
                   self_collision='selfCollision', self_collision_filter_distance='selfCollisionFilterDistance',
                   max_depenetration_velocity='maxDepenetrationVelocity')
    for old, new in mapping.items():
        _set(root, 'physxDeformableBody:'+new, getattr(cfg, old))
    # The legacy add_physx_deformable_body helper explicitly enabled CCD.
    _set(root, 'physxDeformableBody:enableSpeculativeCCD', True)
    if root.HasAPI('PhysxAutoDeformableBodyAPI'):
        if cfg.simulation_hexahedral_resolution is not None:
            root.ApplyAPI('PhysxAutoDeformableHexahedralMeshAPI')
            _set(root, 'physxDeformableBody:resolution', cfg.simulation_hexahedral_resolution)
        root.ApplyAPI('PhysxAutoDeformableMeshSimplificationAPI')
        cooking = dict(collision_simplification='autoDeformableMeshSimplificationEnabled',
                       collision_simplification_remeshing='remeshingEnabled',
                       collision_simplification_remeshing_resolution='remeshingResolution',
                       collision_simplification_target_triangle_count='targetTriangleCount',
                       collision_simplification_force_conforming='forceConforming')
        for old, new in cooking.items():
            _set(root, 'physxDeformableBody:'+new, getattr(cfg, old))
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            if cfg.contact_offset is not None:
                collision.CreateContactOffsetAttr().Set(cfg.contact_offset)
            if cfg.rest_offset is not None:
                collision.CreateRestOffsetAttr().Set(cfg.rest_offset)
    return True
