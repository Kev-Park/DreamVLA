"""Exercise actual DeformableObject spawning, cooked meshes, state and diagnostics."""
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
    assert os.name == 'posix' and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    version = importlib.metadata.version('isaacsim')
    assert version in ('5.1.0.0', '6.2.0.0')
    legacy = version == '5.1.0.0'
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(started=time.time(), completed=False, reservation_released=False, isaacsim=version,
                   scope='Actual procedural volume asset; no robot or study recipe')
    app = None
    reserved = False
    slots = [sys.executable, str(Path(__file__).resolve().parents[2]/'cluster/gpu_slots.py')]
    try:
        result = subprocess.run([*slots, 'reserve', '--owner', str(os.getpid()), '--count', '1',
                                 '--mb', '8000', '--gpu', '0'], check=True, capture_output=True,
                                text=True, timeout=20)
        assert result.stdout.strip() == '0'
        reserved = True
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, active_gpu=0, physics_gpu=0, multi_gpu=False,
                                 fast_shutdown=True, width=128, height=128))
        import numpy as np
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import DeformableObject, DeformableObjectCfg
        from pxr import UsdPhysics, UsdShade
        torch.set_num_threads(1)
        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device='cuda:0'))
        asset = DeformableObject(DeformableObjectCfg(
            prim_path='/World/SoftCube',
            spawn=sim_utils.MeshCuboidCfg(size=(.2, .2, .2),
                deformable_props=sim_utils.DeformableBodyPropertiesCfg(simulation_hexahedral_resolution=3,
                                                                      solver_position_iteration_count=4),
                physics_material=sim_utils.DeformableBodyMaterialCfg(youngs_modulus=50000., poissons_ratio=.3),
                mass_props=sim_utils.MassPropertiesCfg(mass=.8, density=1300.)),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0, 0, 1))))
        sim.reset()
        assert asset.is_initialized
        root = sim.stage.GetPrimAtPath('/World/SoftCube/geometry/mesh' if legacy else '/World/SoftCube')
        if legacy:
            mass_api = UsdPhysics.MassAPI(root)
            mass = mass_api.GetMassAttr().Get()
            density = mass_api.GetDensityAttr().Get()
        else:
            assert root.HasAPI('OmniPhysicsDeformableBodyAPI')
            mass = root.GetAttribute('omniphysics:mass').Get()
            material_path = UsdShade.MaterialBindingAPI(root).GetDirectBindingRel('physics').GetTargets()[0]
            density = sim.stage.GetPrimAtPath(material_path).GetAttribute('omniphysics:density').Get()
        assert math.isclose(mass, .8, abs_tol=1e-7, rel_tol=0)
        assert density == 1300.
        original = asset.data.nodal_state_w.clone()
        assert original.shape[0] == 1 and original.shape[-1] == 6 and original.shape[1] >= 4
        for _ in range(16):
            asset.write_data_to_sim()
            sim.step(render=False)
            asset.update(sim.get_physics_dt())
        moved = asset.data.nodal_state_w.clone()
        assert torch.isfinite(moved).all()
        assert float(moved[..., 2].mean()) < float(original[..., 2].mean())
        asset.write_nodal_state_to_sim(original)
        asset.update(sim.get_physics_dt())
        torch.testing.assert_close(asset.data.nodal_state_w, original, atol=1e-6, rtol=0)
        targets = asset.data.nodal_kinematic_target.clone()
        targets[..., :3] = original[..., :3]
        targets[..., 3] = 0
        asset.write_nodal_kinematic_target_to_sim(targets)
        for _ in range(4):
            sim.step(render=False)
            asset.update(sim.get_physics_dt())
        torch.testing.assert_close(asset.data.nodal_pos_w, original[..., :3], atol=1e-5, rtol=0)
        arrays = dict(initial=original.cpu().numpy(), fallen=moved.cpu().numpy())
        for key in ('sim_element_quat_w', 'collision_element_quat_w', 'sim_element_deform_gradient_w',
                    'collision_element_deform_gradient_w', 'sim_element_stress_w', 'collision_element_stress_w'):
            value = getattr(asset.data, key)
            assert torch.isfinite(value).all(), key
            arrays[key] = value.cpu().numpy()
        np.savez(output/'asset.npz', **arrays)
        sim.stage.GetRootLayer().Export(str(output/'scene.usda'))
        receipt.update(completed=True, frames=20, mass=.8, bound_density=density,
                       simulation_nodes=asset.max_sim_vertices_per_body,
                       collision_nodes=asset.max_collision_vertices_per_body,
                       simulation_elements=asset.max_sim_elements_per_body,
                       collision_elements=asset.max_collision_elements_per_body,
                       mean_fall_m=float((original-moved)[..., 2].mean()),
                       finite_diagnostics={k:list(v.shape) for k,v in arrays.items()}, finished=time.time())
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
