"""Measure native deformable mass/density precedence through free relaxation.

Within-SDK controls compare body mass, body density and material density. There
is no public legacy nodal-mass query, so authored metadata alone is insufficient.
This diagnostic changes no production spawning or conversion semantics.
"""
import argparse
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dt', type=float, default=.005)
    parser.add_argument('--case', help='Run one case at the origin in an isolated process')
    parser.add_argument('--rotated', action='store_true', help='Capture rest tensors under a rotated, scaled spawn')
    args = parser.parse_args()
    assert os.name == 'posix' and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    version = importlib.metadata.version('isaacsim')
    assert version in ('5.1.0.0', '6.2.0.0')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, started=time.time(), isaacsim=version, scope=__doc__)
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
        import isaaclab
        import isaaclab.sim as sim_utils
        from isaaclab.assets import DeformableObject, DeformableObjectCfg
        from pxr import Gf, UsdGeom, UsdShade
        report['isaaclab_source'] = isaaclab.__file__
        torch.set_num_threads(1)
        assert 0 < args.dt <= .005
        report['dt'] = args.dt
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.dt, device='cuda:0'))
        # The cube volume is .008 m^3: density1300 ->10.4kg,1100 ->8.8kg,100 ->.8kg.
        # Independently measured in the a850b45 native fixture. The cookers
        # produce different topologies/volumes; nominal volume is not enough
        # to compare density against an explicit mass for a stiff material.
        measured_volume = .008024893316787251 if version == '5.1.0.0' else .008048089583646978
        report['mass_reference_volume_m3'] = measured_volume
        report['mass_reference_evidence'] = 'isaac62_density_mass_a850b45'
        settings = [('mass_low', .8, 1300., 1100.), ('mass_high', 10.4, None, 1100.),
                    ('body_density', None, 1300., 1100.), ('material_density', None, None, 1100.),
                    ('mass88', 8.8, None, 1100.), ('material_low', None, None, 100.),
                    ('repeat', .8, 1300., 1100.), ('nomat_body_density', None, 1300., None),
                    ('nomat_mass_high', 10.4, None, None), ('nomat_mass_low', .8, None, None),
                    ('nomat_default', None, None, None),
                    ('nomat_mass8', 8., None, None),
                    ('nomat_density1000', None, 1000., None),
                    ('nomat_cooked_mass1000', measured_volume*1000., None, None),
                    ('nomat_cooked_mass1300', measured_volume*1300., None, None),
                    ('nomat_repeat', None, None, None)]
        if args.case:
            settings = [setting for setting in settings if setting[0] == args.case]
            if not settings:
                raise ValueError(f'Unknown diagnostic case: {args.case}')
        report['isolated_case'] = args.case
        cubes, configs = {}, {}
        for index, (name, mass, density, material_density) in enumerate(settings):
            material = None if material_density is None else sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=50000., poissons_ratio=.3, density=material_density)
            spawn = sim_utils.MeshCuboidCfg(size=(.2, .2, .2),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                    simulation_hexahedral_resolution=3, solver_position_iteration_count=4),
                physics_material=material, mass_props=sim_utils.MassPropertiesCfg(mass=mass, density=density))
            cubes[name] = DeformableObject(DeformableObjectCfg(prim_path='/World/'+name, spawn=spawn,
                init_state=DeformableObjectCfg.InitialStateCfg(pos=(index, 0, 1),
                    rot=(.9238795325, 0., 0., .3826834324) if args.rotated else (1., 0., 0., 0.))))
            if args.rotated:
                root_prim = sim.stage.GetPrimAtPath('/World/'+name)
                scale = root_prim.GetAttribute('xformOp:scale')
                if scale:
                    scale.Set(Gf.Vec3f(1.2, .9, 1.1))
                else:
                    UsdGeom.Xformable(root_prim).AddScaleOp().Set(Gf.Vec3f(1.2, .9, 1.1))
            configs[name] = dict(body_mass=mass, body_density=density, material_density=material_density)
        sim.reset()
        original = {name: cube.data.nodal_state_w.clone() for name, cube in cubes.items()}
        traces = {name: [] for name in cubes}
        topology = {}
        for name, cube in cubes.items():
            assert cube.is_initialized and torch.all(cube.data.nodal_kinematic_target[..., 3] == 1)
            assert (cube.material_physx_view is None) == name.startswith('nomat_')
            root = sim.stage.GetPrimAtPath('/World/'+name+('/geometry/mesh' if version == '5.1.0.0' else ''))
            material = UsdShade.MaterialBindingAPI(root).ComputeBoundMaterial('physics')[0]
            values = {a.GetName(): dict(value=str(a.Get()), authored=a.HasAuthoredValueOpinion())
                      for a in root.GetAttributes()
                      if a.GetName() in ('physics:mass', 'physics:density', 'omniphysics:mass')}
            configs[name]['resolved_body_values'] = values
            topology[name+'_root_local_to_world'] = np.asarray(
                UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0))
            # Measure actual cooked volume rather than assuming that the mesh
            # cooker retained the nominal cuboid volume. Use public native
            # tensor calls; the legacy wrapper need not expose new aliases.
            view = cube.root_physx_view
            simulation_indices = view.get_sim_element_indices()
            collision_indices = view.get_element_indices()
            collision_positions = view.get_nodal_positions()
            assert simulation_indices.dtype == collision_indices.dtype == torch.int32
            assert simulation_indices.shape[-1] == collision_indices.shape[-1] == 4
            assert collision_positions.shape[-1] == 3
            cooked = {}
            for kind, positions, indices in (
                    ('simulation', original[name][..., :3], simulation_indices),
                    ('collision', collision_positions, collision_indices)):
                points = positions[0].double()
                connectivity = indices[0].long().reshape(-1, 4)
                assert connectivity.min() >= 0 and connectivity.max() < len(points)
                tetrahedra = points[connectivity]
                edges = tetrahedra[:, 1:] - tetrahedra[:, :1]
                volumes = torch.linalg.det(edges).abs() / 6
                cooked[kind] = dict(volume_m3=float(volumes.sum()),
                    nodes=len(points), elements=len(connectivity),
                    degenerate_elements=int((volumes == 0).sum()),
                    index_dtype=str(indices.dtype))
                topology[name+'_'+kind+'_points'] = points.cpu().numpy()
                topology[name+'_'+kind+'_indices'] = connectivity.cpu().numpy()
                rest_method = 'get_sim_element_rest_poses' if kind == 'simulation' else 'get_element_rest_poses'
                if hasattr(view, rest_method):
                    topology[name+'_'+kind+'_native_rest_poses'] = getattr(view, rest_method)().cpu().numpy().copy()
                prefix = 'get_sim_element_' if kind == 'simulation' else 'get_element_'
                for diagnostic in ('deformation_gradients', 'stresses', 'rotations'):
                    topology[name+'_'+kind+'_'+diagnostic] = getattr(view, prefix+diagnostic)().cpu().numpy().copy()
                if version == '5.1.0.0':
                    rest_attr = root.GetAttribute('physxDeformable:'+kind+'RestPoints')
                    if rest_attr and rest_attr.Get() is not None:
                        topology[name+'_'+kind+'_authored_rest_points'] = np.asarray(rest_attr.Get())
                elif kind == 'simulation':
                    topology[name+'_'+kind+'_native_rest_points'] = view.get_rest_nodal_positions().cpu().numpy().copy()
                else:
                    mesh = sim.stage.GetPrimAtPath(view.collision_mesh_prim_paths[0])
                    rest_attr = mesh.GetAttribute('deformablePose:default:omniphysics:points')
                    if rest_attr and rest_attr.Get() is not None:
                        topology[name+'_'+kind+'_authored_rest_points'] = np.asarray(rest_attr.Get())
            configs[name]['cooked_geometry'] = cooked
            if material:
                configs[name]['bound_material_values'] = {a.GetName(): str(a.Get())
                    for a in material.GetPrim().GetAttributes() if 'density' in a.GetName().lower()}
            if args.rotated:
                assert len(cubes) == 1, 'Controlled rotated diagnostics require an isolated case'
                angle = np.pi/6
                modes = dict(stretch=np.diag([1.1, 1., 1.]),
                    shear=np.array([[1., .12, 0.], [0., 1., .07], [0., 0., 1.]]),
                    rigid=np.array([[1., 0., 0.], [0., np.cos(angle), -np.sin(angle)],
                                    [0., np.sin(angle), np.cos(angle)]]))
                for mode, matrix in modes.items():
                    state = original[name].clone()
                    center = state[..., :3].mean(1, keepdim=True)
                    targets = cube.data.nodal_kinematic_target.clone()
                    targets[..., :3] = center+(state[..., :3]-center) @ torch.as_tensor(
                        matrix.T, device=state.device, dtype=state.dtype)
                    targets[..., 3] = 0
                    cube.write_nodal_kinematic_target_to_sim(targets)
                    for _ in range(4):
                        cube.write_data_to_sim();sim.step(render=False);cube.update(sim.get_physics_dt())
                    torch.testing.assert_close(cube.data.nodal_pos_w, targets[..., :3], atol=1e-5, rtol=0)
                    for kind in ('simulation', 'collision'):
                        prefix = 'get_sim_element_' if kind == 'simulation' else 'get_element_'
                        tag = name+'_'+kind+'_'+mode+'_'
                        points = view.get_sim_nodal_positions() if kind == 'simulation' else view.get_nodal_positions()
                        topology[tag+'points'] = points.cpu().numpy().copy()
                        for diagnostic in ('deformation_gradients', 'stresses', 'rotations'):
                            topology[tag+diagnostic] = getattr(view, prefix+diagnostic)().cpu().numpy().copy()
                    topology[name+'_'+mode+'_matrix'] = matrix
                targets[..., 3] = 1
                cube.write_nodal_kinematic_target_to_sim(targets)
            state = original[name].clone()
            center = state[..., :3].mean(1, keepdim=True)
            state[..., 0] = center[..., 0]+1.1*(state[..., 0]-center[..., 0])
            state[..., 3:] = 0
            cube.write_nodal_state_to_sim(state)
        for _ in range(64):
            for cube in cubes.values():
                cube.write_data_to_sim()
            sim.step(render=False)
            for name, cube in cubes.items():
                cube.update(sim.get_physics_dt())
                p = cube.data.nodal_pos_w.clone()
                assert torch.isfinite(p).all()
                traces[name].append((p-p.mean(1, keepdim=True))[0].cpu().numpy())
        arrays = {name: np.asarray(values) for name, values in traces.items()}
        pairs = [('repeat', 'mass_low'), ('mass_high', 'mass_low'), ('mass_low', 'material_low'),
                 ('body_density', 'mass_high'), ('body_density', 'mass88'),
                 ('material_density', 'mass88'), ('material_density', 'mass_high'),
                 ('material_density', 'material_low'), ('nomat_body_density', 'nomat_mass_high'),
                 ('nomat_body_density', 'nomat_mass_low'), ('nomat_default', 'nomat_mass_high'),
                 ('nomat_default', 'nomat_mass_low'), ('nomat_mass_high', 'nomat_mass_low'),
                 ('nomat_default', 'nomat_mass8'), ('nomat_density1000', 'nomat_mass8'),
                 ('nomat_default', 'nomat_density1000'),
                 ('nomat_default', 'nomat_cooked_mass1000'),
                 ('nomat_density1000', 'nomat_cooked_mass1000'),
                 ('nomat_body_density', 'nomat_cooked_mass1300'),
                 ('nomat_default', 'nomat_repeat')]
        comparisons = {}
        for left, right in pairs:
            if left not in arrays or right not in arrays:
                continue
            delta = arrays[left]-arrays[right]
            comparisons[left+'_vs_'+right] = dict(max_error_m=float(np.abs(delta).max()),
                                                 rms_error_m=float(np.sqrt(np.mean(delta**2))))
        report.update(cases=configs, comparisons=comparisons, steps=64, measurement_completed=True,
            informative_control=(None if args.case else
                comparisons['material_density_vs_material_low']['max_error_m'] > 1e-4))
        np.savez(args.output/'traces.npz', **arrays)
        np.savez(args.output/'topology.npz', **topology)
        if not args.case:
            assert report['informative_control'], 'Material density control did not separate'
        report['completed'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        if sim is not None:
            sim.clear_all_callbacks()
            sim.clear_instance()
        report['finished'] = time.time()
        (args.output/'receipt.json').write_text(json.dumps(report, indent=2))
        if app is not None:
            kwargs = dict(wait_for_replicator=False)
            if 'exit_code' in inspect.signature(app.close).parameters:
                kwargs['exit_code'] = 0 if report['completed'] else 1
            app.close(**kwargs)


if __name__ == '__main__':
    main()
