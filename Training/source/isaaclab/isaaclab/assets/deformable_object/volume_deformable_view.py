# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Preserve the asset's soft-body tensor interface with current volume views.

Nodal state stays on the simulator's Torch frontend. The SDK's public
DeformablePrim computes the removed simulation element diagnostics. Legacy
matrix buffers are column-major, even though their consumers reshape to 3x3.
"""
import torch
import warp as wp
import numpy as np
from pxr import UsdGeom
from isaacsim.core.experimental.prims import DeformablePrim


class VolumeDeformableView:
    def __init__(self, view, pattern, stage):
        self._view = view
        self._diagnostics = DeformablePrim(pattern, deformable_type='volume')
        self._stage = stage
        self._collision_rest = None

    def __getattr__(self, name):
        return getattr(self._view, name)

    @property
    def max_sim_elements_per_body(self):
        return self._view.max_simulation_elements_per_body

    @property
    def max_elements_per_body(self):
        return self._view.max_collision_elements_per_body

    @property
    def max_sim_vertices_per_body(self):
        return self._view.max_simulation_nodes_per_body

    @property
    def max_vertices_per_body(self):
        return self._view.max_collision_nodes_per_body

    def get_sim_nodal_positions(self):
        return self._view.get_simulation_nodal_positions()

    def get_sim_nodal_velocities(self):
        return self._view.get_simulation_nodal_velocities()

    def get_sim_kinematic_targets(self):
        return self._view.get_simulation_nodal_kinematic_targets()

    def set_sim_nodal_positions(self, values, indices):
        return self._view.set_simulation_nodal_positions(values, indices)

    def set_sim_nodal_velocities(self, values, indices):
        return self._view.set_simulation_nodal_velocities(values, indices)

    def set_sim_kinematic_targets(self, values, indices):
        return self._view.set_simulation_nodal_kinematic_targets(values, indices)

    @staticmethod
    def _legacy_matrix_buffer(value):
        return wp.to_torch(value).transpose(-2, -1).contiguous().flatten(-2)

    def get_sim_element_rotations(self):
        return wp.to_torch(self._diagnostics.get_nodal_rotations())

    def get_sim_element_deformation_gradients(self):
        return self._legacy_matrix_buffer(self._diagnostics.get_nodal_gradients())

    def get_sim_element_stresses(self):
        self._material_parameters()  # Never turn absent material properties into zero stresses.
        return self._legacy_matrix_buffer(self._diagnostics.get_nodal_stresses())

    def _collision_gradient(self):
        current = self._view.get_collision_nodal_positions()
        indices = self._view.get_collision_element_indices().long()
        if self._collision_rest is None:
            rows = []
            for path in self._view.collision_mesh_prim_paths:
                prim = self._stage.GetPrimAtPath(path)
                pose = prim.GetAttribute('deformablePose:default:omniphysics:points')
                points = pose.Get() if pose and pose.Get() is not None else UsdGeom.PointBased(prim).GetPointsAttr().Get()
                if points is None:
                    raise RuntimeError(f'Collision rest/bind points missing on {path}')
                row = torch.as_tensor(np.asarray(points), device=current.device, dtype=current.dtype)
                if row.shape != current.shape[1:]:
                    raise RuntimeError('Collision rest topology does not match the live nodal tensor')
                rows.append(row)
            self._collision_rest = torch.stack(rows)
        batch = torch.arange(self.count, device=current.device)[:, None, None]
        actual = current[batch, indices]
        rest = self._collision_rest[batch, indices]
        edges = (actual[..., 1:, :] - actual[..., :1, :]).transpose(-2, -1)
        rest_edges = (rest[..., 1:, :] - rest[..., :1, :]).transpose(-2, -1)
        if torch.any(torch.linalg.det(rest_edges).abs() < 1e-12):
            raise RuntimeError('Cannot recover collision element diagnostics from a degenerate rest tetrahedron')
        return edges @ torch.linalg.inv(rest_edges)

    def get_element_deformation_gradients(self):
        return self._collision_gradient().transpose(-2, -1).contiguous().flatten(-2)

    @staticmethod
    def _polar_rotation(gradient):
        u, _, vh = torch.linalg.svd(gradient)
        correction = torch.ones_like(gradient[..., 0, :])
        correction[..., -1] = torch.linalg.det(u @ vh)
        return (u * correction.unsqueeze(-2)) @ vh

    def get_element_rotations(self):
        from isaaclab.utils.math import quat_from_matrix, convert_quat
        rotation = self._polar_rotation(self._collision_gradient())
        return convert_quat(quat_from_matrix(rotation), to='xyzw')

    def _material_parameters(self):
        material = self._diagnostics.get_applied_physics_materials()
        if len(material) != self.count or any(m is None for m in material):
            raise RuntimeError('Element stress diagnostics require an explicitly bound deformable material')
        youngs = [wp.to_torch(m.get_youngs_moduli()).reshape(()) for m in material]
        poisson = [wp.to_torch(m.get_poissons_ratios()).reshape(()) for m in material]
        return torch.stack(youngs).view(-1, 1, 1, 1), torch.stack(poisson).view(-1, 1, 1, 1)

    def get_element_stresses(self):
        gradient = self._collision_gradient()
        rotation = self._polar_rotation(gradient)
        strain_gradient = rotation.transpose(-2, -1) @ gradient
        youngs, poisson = (v.to(gradient.device) for v in self._material_parameters())
        mu = youngs / (2*(1+poisson))
        lam = youngs*poisson / ((1+poisson)*(1-2*poisson))
        identity = torch.eye(3, device=gradient.device, dtype=gradient.dtype)
        strain = .5*(strain_gradient+strain_gradient.transpose(-2, -1))-identity
        trace = strain.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
        # Preserve the SDK's existing diagnostic convention, including the
        # column-major matrix layout consumed by DeformableObjectData.
        first_piola = 2*mu*(rotation @ strain)+identity*lam*trace
        stress = (first_piola @ strain_gradient.transpose(-2, -1))/torch.linalg.det(strain_gradient)[..., None, None]
        return stress.transpose(-2, -1).contiguous().flatten(-2)
