"""Measure density-only saved volumes through the public file converter.

The target uses convert_copy without temporary masses or post-conversion edits.
With --mass-from-density, it instead uses the normal explicit-mass conversion
as a control, with mass computed from the identical saved simulation rest mesh.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--density', type=float, required=True)
    parser.add_argument('--candidate', action='store_true')
    parser.add_argument('--mass-from-density', action='store_true')
    parser.add_argument('--bound-material', action='store_true',
                        help='Retain the saved explicit material for density/mass controls')
    parser.add_argument('--root-scale', type=float, nargs=3,
                        help='Exercise density inference under a nonuniform body scale')
    args = parser.parse_args()
    version = importlib.metadata.version('isaacsim')
    assert version == ('6.2.0.0' if args.candidate else '5.1.0.0')
    assert os.name == 'posix' and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert args.density > 0
    source = args.source.expanduser().resolve(strict=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, started=time.time(), scope=__doc__, isaacsim=version,
                  source_sha256=digest, density=args.density, candidate=args.candidate,
                  bound_material=args.bound_material)
    app = sim = None
    try:
        from isaacsim import SimulationApp
        launch = dict(headless=True, active_gpu=0, physics_gpu=0, multi_gpu=False,
                      fast_shutdown=True, width=128, height=128)
        if 'active_cuda_gpus' in SimulationApp.DEFAULT_LAUNCHER_CONFIG:
            launch.update(active_gpu=None, active_cuda_gpus=[0], max_gpu_count=1)
        app = SimulationApp(launch)
        import numpy as np
        import torch
        from pxr import Usd, UsdGeom, UsdShade
        import isaaclab.sim as sim_utils
        from isaaclab.assets import DeformableObject, DeformableObjectCfg
        from isaacsim.core.utils.stage import open_stage
        from legacy_volume_usd import authored_apis
        torch.set_num_threads(1)
        stage = Usd.Stage.Open(Usd.Stage.Open(str(source)).Flatten())
        bodies = [p for p in stage.Traverse() if 'PhysxDeformableBodyAPI' in authored_apis(p)]
        assert len(bodies) == 1
        body = bodies[0]
        if args.root_scale:
            assert all(value > 0 for value in args.root_scale)
            UsdGeom.Xformable(body).AddScaleOp().Set(tuple(args.root_scale))
        if not args.bound_material:
            body.RemoveProperty('material:binding:physics')
        assert bool(UsdShade.MaterialBindingAPI(body).ComputeBoundMaterial('physics')[0]) == args.bound_material
        rest = np.asarray(body.GetAttribute('physxDeformable:simulationRestPoints').Get(), dtype=np.float64)
        connectivity = np.asarray(body.GetAttribute('physxDeformable:simulationIndices').Get()).reshape(-1, 4)
        tetrahedra = rest[connectivity]
        volume = float(np.abs(np.linalg.det(tetrahedra[:, 1:]-tetrahedra[:, :1])).sum()/6)
        # Density/mass use the same authored rest volume including world scale.
        from pxr import UsdPhysics
        assert UsdGeom.GetStageMetersPerUnit(stage) == UsdPhysics.GetStageKilogramsPerUnit(stage) == 1
        determinant = abs(np.linalg.det(np.asarray(
            UsdGeom.Xformable(body).ComputeLocalToWorldTransform(0))[:3, :3]))
        volume *= determinant
        report['body_world_volume_scale'] = float(determinant)
        explicit_mass = volume*args.density if args.mass_from_density else 0.
        body.GetAttribute('physics:mass').Set(explicit_mass)
        body.GetAttribute('physics:density').Set(args.density)
        report.update(simulation_rest_volume_m3=volume, explicit_mass_kg=explicit_mass)
        source_counts = dict(nodes=len(body.GetAttribute('physxDeformable:simulationPoints').Get()),
                             elements=len(body.GetAttribute('physxDeformable:simulationIndices').Get())//4)
        fixture = args.output/'native_fixture.usda'
        assert stage.GetRootLayer().Export(str(fixture))
        report['native_fixture_sha256'] = hashlib.sha256(fixture.read_bytes()).hexdigest()
        if args.candidate:
            from legacy_volume_usd import convert_copy
            path = body.GetPath()
            converted_file = args.output/'candidate.usda'
            fixture_digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
            report['candidate_conversion'] = convert_copy(fixture, converted_file)
            assert hashlib.sha256(fixture.read_bytes()).hexdigest() == fixture_digest
            stage = Usd.Stage.Open(str(converted_file))
            converted = stage.GetPrimAtPath(path)
            expected_mass = volume*args.density if args.bound_material else explicit_mass
            assert np.isclose(converted.GetAttribute('omniphysics:mass').Get(), expected_mass, rtol=1e-6, atol=1e-9)
            material = UsdShade.MaterialBindingAPI(converted).ComputeBoundMaterial('physics')[0]
            if not args.bound_material:
                assert material.GetPrim().GetAttribute('omniphysics:density').Get() == args.density
                assert material.GetPrim().GetCustomDataByKey('isaaclab:implicitLegacyDeformableMaterial')
            else:
                assert str(material.GetPath()) == '/World/SoftCube/geometry/material'
                assert not material.GetPrim().GetCustomDataByKey('isaaclab:implicitLegacyDeformableMaterial')
                assert material.GetPrim().GetAttribute('omniphysics:youngsModulus').Get() == 50000.
                # This saved material has no density override of its own.
                assert material.GetPrim().GetAttribute('omniphysics:density').Get() != args.density
            fixture = converted_file
            report['public_converter_without_post_edits'] = True
            from check_legacy_volume_rejections import run_checks
            report['atomic_rejections'] = run_checks()
        assert open_stage(str(fixture))
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device='cuda:0'))
        asset = DeformableObject(DeformableObjectCfg(prim_path='/World/SoftCube', spawn=None))
        sim.reset()
        assert asset.is_initialized
        assert (asset.material_physx_view is not None) == args.bound_material
        assert asset.max_sim_vertices_per_body == source_counts['nodes']
        assert asset.max_sim_elements_per_body == source_counts['elements']
        initial = asset.data.nodal_state_w.clone()
        assert torch.all(asset.data.nodal_kinematic_target[..., 3] == 1)
        state = initial.clone()
        center = state[..., :3].mean(1, keepdim=True)
        state[..., 0] = center[..., 0]+1.1*(state[..., 0]-center[..., 0])
        state[..., 3:] = 0
        asset.write_nodal_state_to_sim(state)
        traces = []
        for _ in range(64):
            asset.write_data_to_sim();sim.step(render=False);asset.update(sim.get_physics_dt())
            value = asset.data.nodal_state_w.clone()
            assert torch.isfinite(value).all()
            traces.append(value.cpu().numpy())
        arrays = np.asarray(traces)
        np.savez(args.output/'traces.npz', states=arrays, initial=initial.cpu().numpy(), strained=state.cpu().numpy())
        centered = arrays[..., :3]-arrays[..., :3].mean(axis=-2, keepdims=True)
        started = state[..., :3].cpu().numpy()
        started -= started.mean(axis=-2, keepdims=True)
        relaxation = float(np.abs(centered[-1]-started).max())
        assert relaxation > 1e-5, 'No informative free relaxation'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        report.update(completed=True, source_unchanged=True, source_counts=source_counts,
                      steps=64, dt=.005, relaxation_m=relaxation,
                      public_material_view_none=asset.material_physx_view is None)
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        if sim is not None:
            sim.clear_all_callbacks();sim.clear_instance()
        report['finished'] = time.time()
        (args.output/'receipt.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if app is not None:
            kwargs = dict(wait_for_replicator=False)
            if 'exit_code' in inspect.signature(app.close).parameters:
                kwargs['exit_code'] = 0 if report['completed'] else 1
            app.close(**kwargs)


if __name__ == '__main__':
    main()
