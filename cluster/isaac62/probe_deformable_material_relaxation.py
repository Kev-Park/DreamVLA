"""Bounded native implicit/explicit material relaxation comparison.

Three isolated cubes share geometry, mass and solver settings. Material is either
unbound, the legacy 50 MPa/0.45 recipe, or a 50 Pa/0.45 control. Production
spawning defaults are not changed. Comparisons are within each SDK, not a claim
of identical meshes or trajectories across different SDKs.
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
    args = parser.parse_args()
    assert os.name == 'posix' and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    version = importlib.metadata.version('isaacsim')
    assert version in ('5.1.0.0', '6.2.0.0')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(started=time.time(), isaacsim=version, completed=False, scope=__doc__)
    app = sim = None
    try:
        from isaacsim import SimulationApp
        config = dict(headless=True, active_gpu=0, physics_gpu=0, multi_gpu=False,
                      fast_shutdown=True, width=128, height=128)
        if 'active_cuda_gpus' in SimulationApp.DEFAULT_LAUNCHER_CONFIG:
            config.update(active_gpu=None, active_cuda_gpus=[0], max_gpu_count=1)
        app = SimulationApp(config)
        import numpy as np
        import torch
        import isaaclab
        import isaaclab.sim as sim_utils
        from isaaclab.assets import DeformableObject, DeformableObjectCfg
        report['isaaclab_source'] = isaaclab.__file__
        torch.set_num_threads(1)
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device='cuda:0'))
        cubes = {}
        for index, (name, youngs) in enumerate([('unbound', None), ('legacy_explicit', 50e6),
                                               ('soft_control', 50.)]):
            material = None if youngs is None else sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=youngs, poissons_ratio=.45)
            spawn = sim_utils.MeshCuboidCfg(size=(.2, .2, .2),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                    simulation_hexahedral_resolution=3, solver_position_iteration_count=4),
                physics_material=material, mass_props=sim_utils.MassPropertiesCfg(mass=.8))
            cubes[name] = DeformableObject(DeformableObjectCfg(prim_path='/World/'+name,
                spawn=spawn, init_state=DeformableObjectCfg.InitialStateCfg(pos=(index, 0, 1))))
        sim.reset()
        original, targets, traces = {}, {}, {name: [] for name in cubes}
        for name, cube in cubes.items():
            assert cube.is_initialized
            assert (cube.material_physx_view is None) == (name == 'unbound')
            original[name] = cube.data.nodal_state_w.clone()
            target = cube.data.nodal_kinematic_target.clone()
            center = original[name][..., :3].mean(1, keepdim=True)
            target[..., :3] = original[name][..., :3]
            target[..., 0] = center[..., 0]+1.1*(target[..., 0]-center[..., 0])
            target[..., 3] = 0
            targets[name] = target
            cube.write_nodal_kinematic_target_to_sim(target)
        for _ in range(4):
            for cube in cubes.values():
                cube.write_data_to_sim()
            sim.step(render=False)
            for cube in cubes.values():
                cube.update(sim.get_physics_dt())
        report['held_strain_rms_m'] = {}
        for name, cube in cubes.items():
            # Zero nodal velocity before release in every arm.
            state = cube.data.nodal_state_w.clone()
            pos = state[..., :3]
            rest = original[name][..., :3]
            delta = (pos-pos.mean(1, keepdim=True))-(rest-rest.mean(1, keepdim=True))
            report['held_strain_rms_m'][name] = float(torch.sqrt(torch.mean(delta**2)))
            state[..., 3:] = 0
            cube.write_nodal_state_to_sim(state)
            target = targets[name].clone()
            target[..., 3] = 1
            cube.write_nodal_kinematic_target_to_sim(target)
        for _ in range(32):
            for cube in cubes.values():
                cube.write_data_to_sim()
            sim.step(render=False)
            for name, cube in cubes.items():
                cube.update(sim.get_physics_dt())
                pos = cube.data.nodal_pos_w.clone()
                assert torch.isfinite(pos).all()
                traces[name].append((pos-pos.mean(1, keepdim=True))[0].cpu().numpy())
        arrays = {name: np.asarray(values) for name, values in traces.items()}
        for name, state in original.items():
            pos = state[..., :3]
            arrays[name+'_rest'] = (pos-pos.mean(1, keepdim=True))[0].cpu().numpy()
        comparisons = {}
        for name in ('legacy_explicit', 'soft_control'):
            assert arrays[name].shape == arrays['unbound'].shape
            rest_error = float(np.max(np.abs(arrays[name+'_rest']-arrays['unbound_rest'])))
            assert rest_error <= 1e-6, 'Within-SDK source geometry differs'
            delta = arrays[name]-arrays['unbound']
            comparisons[name] = dict(rest_geometry_max_error_m=rest_error,
                maximum_centered_node_error_m=float(np.abs(delta).max()),
                rms_centered_node_error_by_step_m=np.sqrt(np.mean(delta**2, axis=(1, 2))).tolist())
        report['informative_control'] = all(x > .001 for x in report['held_strain_rms_m'].values()) and (
            comparisons['soft_control']['maximum_centered_node_error_m'] > .0001)
        report.update(completed=True, steps=36, release_steps=32,
            implicit_versus_explicit=comparisons,
            relaxation_rms_from_rest_by_step_m={name: np.sqrt(np.mean(
                (arrays[name]-arrays[name+'_rest'])**2, axis=(1, 2))).tolist() for name in cubes})
        np.savez(args.output/'traces.npz', **arrays)
        assert report['informative_control'], 'Control cannot distinguish material response'
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
