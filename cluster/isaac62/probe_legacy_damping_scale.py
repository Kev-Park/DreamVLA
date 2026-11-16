"""Measure legacy dampingScale before proposing a replacement.

Original 5.1 runtime only. Baseline repeat measures numerical noise, and a
changed elasticity-damping control must produce a measurable response. Candidate
coefficient multiplication is tested within 5.1, not assumed equivalent in 6.2.
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
    assert importlib.metadata.version('isaacsim') == '5.1.0.0'
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, started=time.time(), scope=__doc__, cases={})
    app = sim = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, active_gpu=0, physics_gpu=0,
                                 multi_gpu=False, fast_shutdown=True, width=128, height=128))
        import numpy as np
        import torch
        import isaaclab
        import isaaclab.sim as sim_utils
        from isaaclab.assets import DeformableObject, DeformableObjectCfg
        report['isaaclab_source'] = isaaclab.__file__
        torch.set_num_threads(1)
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device='cuda:0'))
        settings = [('base', .02, 1.), ('repeat', .02, 1.), ('scale0', .02, 0.),
                    ('scale05', .02, .5), ('scale2', .02, 2.), ('damping0', 0., 1.),
                    ('damping02', .2, 1.), ('candidate05', .01, 1.), ('candidate2', .04, 1.)]
        cubes = {}
        for index, (name, damping, scale) in enumerate(settings):
            material = sim_utils.DeformableBodyMaterialCfg(youngs_modulus=50000., poissons_ratio=.3,
                elasticity_damping=damping, damping_scale=scale)
            spawn = sim_utils.MeshCuboidCfg(size=(.2, .2, .2),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                    simulation_hexahedral_resolution=3, solver_position_iteration_count=4),
                physics_material=material, mass_props=sim_utils.MassPropertiesCfg(mass=.8))
            cubes[name] = DeformableObject(DeformableObjectCfg(prim_path='/World/'+name, spawn=spawn,
                init_state=DeformableObjectCfg.InitialStateCfg(pos=(index, 0, 1))))
            report['cases'][name] = dict(elasticity_damping=damping, damping_scale=scale)
        sim.reset()
        original = {name: cube.data.nodal_state_w.clone() for name, cube in cubes.items()}
        transforms = [np.array([[1.05, .1, 0.], [0., .97, .06], [0., 0., 1.02]]),
                      np.diag([1.1, .95, .95])]
        arrays = {}
        metrics = []
        for mode, transform in enumerate(transforms):
            traces = {name: [] for name in cubes}
            for name, cube in cubes.items():
                assert cube.is_initialized and cube.material_physx_view is not None
                assert torch.all(cube.data.nodal_kinematic_target[..., 3] == 1)
                state = original[name].clone()
                center = state[..., :3].mean(1, keepdim=True)
                matrix = torch.tensor(transform, dtype=state.dtype, device=state.device)
                state[..., :3] = (state[..., :3]-center) @ matrix.T + center
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
            traces = {name: np.asarray(values) for name, values in traces.items()}
            for name, values in traces.items():
                arrays[f'mode{mode}_{name}'] = values
            pairs = [(name, 'base') for name, _, _ in settings if name != 'base']
            pairs += [('scale05', 'candidate05'), ('scale2', 'candidate2'), ('scale0', 'damping0')]
            comparisons = {}
            for left, right in pairs:
                delta = traces[left]-traces[right]
                comparisons[left+'_vs_'+right] = dict(max_error_m=float(np.abs(delta).max()),
                    rms_error_m=float(np.sqrt(np.mean(delta**2))))
            metrics.append(dict(mode=mode, requested_gradient=transform.tolist(), comparisons=comparisons))
        report['modes'] = metrics
        report['informative_control'] = all(row['comparisons']['damping02_vs_base']['max_error_m'] > 1e-4
                                           for row in metrics)
        report['measurement_completed'] = True
        np.savez(args.output/'traces.npz', **arrays)
        assert report['informative_control'], 'Elasticity-damping control did not separate'
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
