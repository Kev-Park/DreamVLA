"""Measure native stress diagnostics for an unbound material under known strain.

No default material constants are supplied or inferred for production use.
Completion means measurements were collected, not that old/new stress agrees.
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
    report = dict(started=time.time(), isaacsim=version, completed=False, cases=[], scope=__doc__)
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
        import isaaclab.sim as sim_utils
        from isaaclab.assets import DeformableObject, DeformableObjectCfg
        from pxr import UsdShade
        torch.set_num_threads(1)
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device='cuda:0'))
        spawn = sim_utils.MeshCuboidCfg(size=(.2, .2, .2),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                simulation_hexahedral_resolution=3, solver_position_iteration_count=4),
            physics_material=None, mass_props=sim_utils.MassPropertiesCfg(mass=.8))
        asset = DeformableObject(DeformableObjectCfg(prim_path='/World/SoftCube', spawn=spawn,
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0, 0, 1))))
        sim.reset()
        assert asset.is_initialized and asset.material_physx_view is None
        root = sim.stage.GetPrimAtPath('/World/SoftCube/geometry/mesh' if version == '5.1.0.0'
                                      else '/World/SoftCube')
        material = UsdShade.MaterialBindingAPI(root).ComputeBoundMaterial('physics')[0]
        report.update(material_bound=bool(material), material_physx_view_is_none=True,
            private_legacy_default=bool(material) and bool(material.GetPrim().GetCustomDataByKey(
                'isaaclab:implicitLegacyDeformableMaterial')))
        original = asset.data.nodal_state_w.clone()
        assert torch.isfinite(original).all()
        center = original[..., :3].mean(1, keepdim=True)
        transforms = [np.diag([1.1, 1., 1.]), np.diag([1., .95, 1.]),
                      np.array([[1., .04, 0.], [0., 1., 0.], [0., 0., 1.]])]
        arrays = {}
        for index, transform in enumerate(transforms):
            asset.write_nodal_state_to_sim(original)
            target = asset.data.nodal_kinematic_target.clone()
            matrix = torch.tensor(transform, dtype=original.dtype, device=original.device)
            target[..., :3] = (original[..., :3]-center) @ matrix.T + center
            target[..., 3] = 0
            asset.write_nodal_kinematic_target_to_sim(target)
            for _ in range(4):
                asset.write_data_to_sim()
                sim.step(render=False)
                asset.update(sim.get_physics_dt())
            row = dict(index=index, requested_gradient=transform.tolist(), measurements={})
            for name in ('sim_element_deform_gradient_w', 'collision_element_deform_gradient_w',
                         'sim_element_stress_w', 'collision_element_stress_w'):
                try:
                    value = getattr(asset.data, name).clone()
                    assert torch.isfinite(value).all(), name
                    arrays[f'case{index}_{name}'] = value.cpu().numpy()
                    row['measurements'][name] = dict(available=True, shape=list(value.shape),
                        maximum_absolute=float(value.abs().max()), mean_matrix=value.mean(1).cpu().tolist())
                except RuntimeError as error:
                    if version != '6.2.0.0' or 'explicitly bound deformable material' not in str(error):
                        raise
                    row['measurements'][name] = dict(available=False, error=str(error))
            report['cases'].append(row)
        np.savez(args.output/'measurements.npz', **arrays)
        report.update(completed=True, steps=12,
                      stress_available=all(x['measurements']['sim_element_stress_w']['available']
                                           for x in report['cases']))
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
