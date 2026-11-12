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
from pxr import UsdGeom, UsdShade
from isaacsim.core.experimental.prims import DeformablePrim


class VolumeDeformableView:
    def __init__(self, view, pattern, stage):
        self._view = view
        # DeformablePrim recreates non-auto Xform hierarchies in its constructor.
        # A saved cooked body must retain its existing topology/tensor view.
        roots = [stage.GetPrimAtPath(path) for path in view.prim_paths]
        can_wrap = all(p.HasAPI('PhysxAutoDeformableBodyAPI') or p.IsA(UsdGeom.TetMesh) for p in roots)
        self._diagnostics = DeformablePrim(pattern, deformable_type='volume') if can_wrap else None
        self._stage = stage
        self._collision_rest = None
        self._element_count_cache = {}

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

    def _element_counts(self, kind):
        if kind not in self._element_count_cache:
            paths = getattr(self._view, kind + "_mesh_prim_paths")
            counts = []
            for path in paths:
                prim = self._stage.GetPrimAtPath(path)
                indices = UsdGeom.TetMesh(prim).GetTetVertexIndicesAttr().Get()
                if indices is None:
                    raise RuntimeError(f"Missing cooked tetrahedron connectivity on {path}")
                counts.append(np.asarray(indices).reshape(-1, 4).shape[0])
            device = self._view.get_collision_nodal_positions().device
            self._element_count_cache[kind] = torch.tensor(counts, device=device, dtype=torch.long)
        return self._element_count_cache[kind]

    @property
    def simulation_element_counts(self):
        return self._element_counts("simulation")

    @property
    def collision_element_counts(self):
        return self._element_counts("collision")

    def _element_mask(self, kind, size):
        counts = self._element_counts(kind)
        if len(counts) != self.count or torch.any(counts > size):
            raise RuntimeError("Cooked element counts disagree with the live tensor view")
        return torch.arange(size, device=counts.device)[None, :] < counts[:, None]

    def _mask_elements(self, value, kind):
        # Padding is not a physical element. Preserve finite zero padding while
        # retaining the SDK result (including invalid values) for real elements.
        mask = self._element_mask(kind, value.shape[1])
        mask = mask.reshape(*mask.shape, *((1,) * (value.ndim - 2)))
        return torch.where(mask, value, torch.zeros_like(value))

    def get_sim_element_rotations(self):
        if self._diagnostics is not None:
            return self._mask_elements(wp.to_torch(self._diagnostics.get_nodal_rotations()), "simulation")
        from isaaclab.utils.math import quat_from_matrix, convert_quat
        rotation = self._polar_rotation(self._simulation_gradient())
        return self._mask_elements(convert_quat(quat_from_matrix(rotation), to='xyzw'), 'simulation')

    def get_sim_element_deformation_gradients(self):
        if self._diagnostics is not None:
            return self._mask_elements(self._legacy_matrix_buffer(self._diagnostics.get_nodal_gradients()), "simulation")
        gradient = self._simulation_gradient()
        stretch = self._polar_rotation(gradient).transpose(-2, -1) @ gradient
        return self._mask_elements(stretch.transpose(-2, -1).contiguous().flatten(-2), 'simulation')

    def get_sim_element_stresses(self):
        self._material_parameters()  # Never turn absent material properties into zero stresses.
        if self._diagnostics is not None:
            return self._mask_elements(self._legacy_matrix_buffer(self._diagnostics.get_nodal_stresses()), "simulation")
        stress = self._stress(self._simulation_gradient())
        return self._mask_elements(stress.transpose(-2, -1).contiguous().flatten(-2), 'simulation')

    def _simulation_gradient(self):
        current = self._view.get_simulation_nodal_positions()
        rest = self._view.get_rest_nodal_positions()
        indices = self._view.get_simulation_element_indices().long()
        rest_indices = self._view.get_rest_element_indices().long()
        valid = self._element_mask('simulation', indices.shape[1])
        indices = torch.where(valid[..., None], indices, torch.zeros_like(indices))
        rest_indices = torch.where(valid[..., None], rest_indices, torch.zeros_like(rest_indices))
        batch = torch.arange(self.count, device=current.device)[:, None, None]
        actual, original = current[batch, indices], rest[batch, rest_indices]
        edges = (actual[..., 1:, :] - actual[..., :1, :]).transpose(-2, -1)
        rest_edges = (original[..., 1:, :] - original[..., :1, :]).transpose(-2, -1)
        identity = torch.eye(3, device=current.device, dtype=current.dtype)
        edges = torch.where(valid[..., None, None], edges, identity)
        rest_edges = torch.where(valid[..., None, None], rest_edges, identity)
        if torch.any(torch.linalg.det(rest_edges).abs() < 1e-12):
            raise RuntimeError('Degenerate real simulation rest tetrahedron')
        return edges @ torch.linalg.inv(rest_edges)

    def _collision_gradient(self):
        current = self._view.get_collision_nodal_positions()
        indices = self._view.get_collision_element_indices().long()
        if self._collision_rest is None:
            rest = torch.zeros_like(current)
            for body, path in enumerate(self._view.collision_mesh_prim_paths):
                prim = self._stage.GetPrimAtPath(path)
                pose = prim.GetAttribute('deformablePose:default:omniphysics:points')
                points = pose.Get() if pose and pose.Get() is not None else UsdGeom.PointBased(prim).GetPointsAttr().Get()
                if points is None:
                    raise RuntimeError(f'Collision rest/bind points missing on {path}')
                row = torch.as_tensor(np.array(points, copy=True), device=current.device, dtype=current.dtype)
                if row.ndim != 2 or row.shape[1] != 3 or row.shape[0] > current.shape[1]:
                    raise RuntimeError('Collision rest topology exceeds the live nodal tensor')
                rest[body, :row.shape[0]] = row
            self._collision_rest = rest
        valid = self._element_mask("collision", indices.shape[1])
        indices = torch.where(valid[..., None], indices, torch.zeros_like(indices))
        if torch.any(indices < 0) or torch.any(indices >= current.shape[1]):
            raise RuntimeError("Collision element connectivity exceeds the live nodal tensor")
        batch = torch.arange(self.count, device=current.device)[:, None, None]
        actual = current[batch, indices]
        rest = self._collision_rest[batch, indices]
        edges = (actual[..., 1:, :] - actual[..., :1, :]).transpose(-2, -1)
        rest_edges = (rest[..., 1:, :] - rest[..., :1, :]).transpose(-2, -1)
        identity = torch.eye(3, device=current.device, dtype=current.dtype)
        rest_edges = torch.where(valid[..., None, None], rest_edges, identity)
        edges = torch.where(valid[..., None, None], edges, identity)
        if torch.any(torch.linalg.det(rest_edges).abs() < 1e-12):
            raise RuntimeError('Cannot recover collision element diagnostics from a degenerate rest tetrahedron')
        return edges @ torch.linalg.inv(rest_edges)

    def get_element_deformation_gradients(self):
        return self._mask_elements(self._collision_gradient().transpose(-2, -1).contiguous().flatten(-2), "collision")

    @staticmethod
    def _polar_rotation(gradient):
        u, _, vh = torch.linalg.svd(gradient)
        correction = torch.ones_like(gradient[..., 0, :])
        correction[..., -1] = torch.linalg.det(u @ vh)
        return (u * correction.unsqueeze(-2)) @ vh

    def get_element_rotations(self):
        from isaaclab.utils.math import quat_from_matrix, convert_quat
        rotation = self._polar_rotation(self._collision_gradient())
        return self._mask_elements(convert_quat(quat_from_matrix(rotation), to='xyzw'), 'collision')

    def _material_parameters(self):
        if self._diagnostics is None:
            values = []
            for path in self._view.prim_paths:
                prim = self._stage.GetPrimAtPath(path)
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
                if not material:
                    raise RuntimeError('Element stress diagnostics require an explicitly bound deformable material')
                mp = material.GetPrim()
                values.append([mp.GetAttribute('omniphysics:youngsModulus').Get(),
                               mp.GetAttribute('omniphysics:poissonsRatio').Get()])
            if any(v is None for row in values for v in row):
                raise RuntimeError('Missing bound deformable material properties')
            data = torch.tensor(values, device=self._view.get_simulation_nodal_positions().device)
            return data[:, 0].reshape(-1, 1, 1, 1), data[:, 1].reshape(-1, 1, 1, 1)
        material = self._diagnostics.get_applied_physics_materials()
        if len(material) != self.count or any(m is None for m in material):
            raise RuntimeError('Element stress diagnostics require an explicitly bound deformable material')
        youngs = [wp.to_torch(m.get_youngs_moduli()).reshape(()) for m in material]
        poisson = [wp.to_torch(m.get_poissons_ratios()).reshape(()) for m in material]
        return torch.stack(youngs).view(-1, 1, 1, 1), torch.stack(poisson).view(-1, 1, 1, 1)

    def _stress(self, gradient):
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
        return stress

    def get_element_stresses(self):
        stress = self._stress(self._collision_gradient())
        return self._mask_elements(stress.transpose(-2, -1).contiguous().flatten(-2), "collision")
