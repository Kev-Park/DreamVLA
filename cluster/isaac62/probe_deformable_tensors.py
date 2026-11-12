"""Measure actual volume-deformable tensors in either installed simulator SDK.

An explicitly authored single tetrahedron isolates tensor conventions from
procedural mesh cooking. Run on a manually reviewed Linux GPU, never locally.
"""
import argparse
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.name != 'posix' or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        parser.error('Requires the reviewed Linux physical GPU0 allocation')
    version = importlib.metadata.version('isaacsim')
    if version not in ('5.1.0.0', '6.2.0.0'):
        parser.error('Expected one of the existing legacy/target SDKs')
    legacy = version == '5.1.0.0'
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(started=time.time(), completed=False, isaacsim=version,
                   scope='Explicit tetrahedron tensor conventions, not robot behavior',
                   reservation_released=False)
    slots = [sys.executable, str(Path(__file__).resolve().parents[2]/'cluster/gpu_slots.py')]
    app = None
    reserved = False
    try:
        result = subprocess.run([*slots, 'reserve', '--owner', str(os.getpid()), '--count', '1',
                                 '--mb', '8000', '--gpu', '0'], capture_output=True,
                                text=True, check=True, timeout=20)
        assert result.stdout.strip() == '0', 'Reviewed reservation unavailable'
        reserved = True
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, active_gpu=0, physics_gpu=0, multi_gpu=False,
                                 fast_shutdown=True, width=128, height=128))
        import numpy as np
        import torch
        import omni.physics.tensors as tensors
        import omni.timeline
        import omni.usd
        from isaacsim.core.simulation_manager import SimulationManager
        from omni.physx.scripts import deformableUtils
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt
        torch.set_num_threads(1)
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.Xform.Define(stage, '/World')
        scene = UsdPhysics.Scene.Define(stage, '/World/physicsScene')
        scene.CreateGravityMagnitudeAttr(0.)
        physics = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        physics.CreateEnableGPUDynamicsAttr(True)
        physics.CreateBroadphaseTypeAttr('GPU')
        physics.CreateTimeStepsPerSecondAttr(200)
        points = [Gf.Vec3f(0, 0, 2), Gf.Vec3f(.2, 0, 2),
                  Gf.Vec3f(0, .3, 2), Gf.Vec3f(0, 0, 2.4)]
        path = '/World/Tetrahedron'
        if legacy:
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
            mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
            assert deformableUtils.add_physx_deformable_body(
                stage, path, simulation_rest_points=points, simulation_indices=[0, 1, 2, 3],
                collision_rest_points=points, collision_indices=[0, 1, 2, 3])
            UsdShade.Material.Define(stage, '/World/Material')
            material = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(stage.GetPrimAtPath('/World/Material'))
            material.CreateYoungsModulusAttr(50000.)
            material.CreatePoissonsRatioAttr(.3)
            material.CreateDensityAttr(1000.)
        else:
            mesh = UsdGeom.TetMesh.Define(stage, path)
            mesh.CreatePointsAttr(points)
            mesh.CreateTetVertexIndicesAttr(Vt.Vec4iArray([Gf.Vec4i(0, 1, 2, 3)]))
            assert deformableUtils.set_physics_volume_deformable_body(stage, path)
            assert deformableUtils.add_deformable_material(stage, '/World/Material',
                                                           youngs_modulus=50000., poissons_ratio=.3, density=1000.)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            UsdShade.Material(stage.GetPrimAtPath('/World/Material')), materialPurpose='physics')
        stage.GetRootLayer().Export(str(output/'scene.usda'))
        if legacy:
            SimulationManager.set_backend('torch')
            SimulationManager.set_physics_sim_device('cuda:0')
            SimulationManager.set_physics_dt(.005)
        else:
            SimulationManager.setup_simulation(dt=.005, device='cuda:0')
        omni.timeline.get_timeline_interface().play()
        for _ in range(8):
            app.update()
        SimulationManager.initialize_physics()
        sim = (tensors.create_simulation_view('torch') if legacy else
               tensors.create_simulation_view('torch', stage_id=context.get_stage_id(), backend='physx'))
        sim.set_subspace_roots('/')
        view = sim.create_soft_body_view(path) if legacy else sim.create_volume_deformable_body_view(path)
        assert view.count == 1
        if not legacy:
            from isaaclab.assets.deformable_object.volume_deformable_view import VolumeDeformableView
            view = VolumeDeformableView(view, path, stage)
        get_positions = view.get_sim_nodal_positions
        set_positions = view.set_sim_nodal_positions
        set_targets = view.set_sim_kinematic_targets
        rest = get_positions().clone()
        assert rest.shape == (1, 4, 3) and rest.device.type == 'cuda', rest.shape
        angle = math.radians(31)
        rotation = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                                 [math.sin(angle), math.cos(angle), 0], [0, 0, 1]], device=rest.device)
        stretch = torch.tensor([[1.12, .07, 0], [0, .91, 0], [0, 0, 1.03]], device=rest.device)
        affine = rotation @ stretch
        target = (rest-rest[:, :1]) @ affine.T + rest[:, :1] + .03
        indices = torch.tensor([0], dtype=torch.int32, device=rest.device)
        commands = torch.cat([target, torch.zeros((*target.shape[:2], 1), device=rest.device)], dim=-1)
        set_positions(target, indices)
        set_targets(commands, indices)
        for _ in range(4):
            app.update()
        actual = get_positions().clone()
        torch.testing.assert_close(actual, target, atol=1e-5, rtol=0)
        arrays = dict(rest=rest.cpu().numpy(), target=target.cpu().numpy(), actual=actual.cpu().numpy(),
                      affine=affine.cpu().numpy(), rotation=rotation.cpu().numpy(), stretch=stretch.cpu().numpy())
        for name, method in [('simulation_gradient', 'get_sim_element_deformation_gradients'),
                             ('collision_gradient', 'get_element_deformation_gradients'),
                             ('simulation_rotation', 'get_sim_element_rotations'),
                             ('collision_rotation', 'get_element_rotations'),
                             ('simulation_stress', 'get_sim_element_stresses'),
                             ('collision_stress', 'get_element_stresses')]:
            value = getattr(view, method)().clone()
            assert torch.isfinite(value).all(), name
            arrays[name] = value.cpu().numpy()
        np.savez(output/'tensors.npz', **arrays)
        receipt.update(completed=True, frames=4, array_shapes={k:list(v.shape) for k,v in arrays.items()},
                       requested_nodal_error=float((actual-target).abs().max()), finished=time.time())
    except BaseException:
        receipt.update(error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        if reserved:
            subprocess.run([*slots, 'release', '--owner', str(os.getpid()), '--gpu', '0'], check=True, timeout=20)
            receipt['reservation_released'] = True
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        if app is not None:
            kwargs = dict(wait_for_replicator=False)
            if 'exit_code' in inspect.signature(app.close).parameters:
                kwargs['exit_code'] = 0 if receipt['completed'] else 1
            app.close(**kwargs)


if __name__ == '__main__':
    main()
