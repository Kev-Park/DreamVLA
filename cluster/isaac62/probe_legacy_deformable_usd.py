"""Convert a saved legacy volume USD on an isolated layer and test its real asset."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback




def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert os.name=='posix' and os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    output=args.output.expanduser().resolve();output.mkdir(parents=True,exist_ok=False)
    source=args.source.expanduser().resolve(strict=True)
    receipt=dict(started=time.time(),completed=False,reservation_released=False)
    app=None;sim=None;reserved=False
    slots=[sys.executable,str(Path(__file__).resolve().parents[2]/'cluster/gpu_slots.py')]
    try:
        r=subprocess.run([*slots,'reserve','--owner',str(os.getpid()),'--count','1','--mb','8000','--gpu','0'],capture_output=True,text=True,check=True,timeout=20)
        assert r.stdout.strip()=='0';reserved=True
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,active_gpu=0,physics_gpu=0,multi_gpu=False,fast_shutdown=True,width=128,height=128))
        import torch
        import numpy as np
        import isaaclab.sim as sim_utils
        from isaaclab.assets import DeformableObject,DeformableObjectCfg
        from isaacsim.core.utils.stage import open_stage
        torch.set_num_threads(1)
        from check_legacy_volume_rejections import run_checks
        receipt['rejection_checks'] = run_checks()
        from legacy_volume_usd import convert_copy
        receipt['conversion']=convert_copy(source,output/'converted.usda')
        assert open_stage(str(output/'converted.usda'))
        sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device='cuda:0'))
        asset=DeformableObject(DeformableObjectCfg(prim_path='/World/SoftCube',spawn=None))
        sim.reset();assert asset.is_initialized
        old=receipt['conversion']['legacy_bodies'][0]
        assert asset.max_sim_vertices_per_body==old['simulation_nodes']
        assert asset.max_sim_elements_per_body==old['simulation_elements']
        assert asset.max_collision_vertices_per_body==old['collision_nodes']
        assert asset.max_collision_elements_per_body==old['collision_elements']
        prim=sim.stage.GetPrimAtPath(old['path'])
        assert abs(prim.GetAttribute('omniphysics:mass').Get()-.8)<1e-7
        before=asset.data.nodal_state_w.clone()
        for _ in range(16):
            asset.write_data_to_sim();sim.step(render=False);asset.update(sim.get_physics_dt())
        after=asset.data.nodal_state_w.clone()
        assert torch.isfinite(after).all()
        assert float(after[...,2].mean())<float(before[...,2].mean())
        # A known affine strain verifies actual values from the read-only
        # non-auto diagnostic path, rather than just finite array shapes.
        asset.write_nodal_state_to_sim(before)
        targets=asset.data.nodal_kinematic_target.clone()
        targets[..., :3]=before[..., :3]
        targets[..., 0]*=1.1
        targets[..., 3]=0
        asset.write_nodal_kinematic_target_to_sim(targets)
        for _ in range(4):
            sim.step(render=False);asset.update(sim.get_physics_dt())
        torch.testing.assert_close(asset.data.nodal_pos_w,targets[..., :3],atol=1e-5,rtol=0)
        arrays=dict(initial=before.cpu().numpy(),fallen=after.cpu().numpy())
        for name in ('sim_element_quat_w','collision_element_quat_w','sim_element_deform_gradient_w',
                     'collision_element_deform_gradient_w','sim_element_stress_w','collision_element_stress_w'):
            value=getattr(asset.data,name);assert torch.isfinite(value).all(),name
            arrays[name]=value.cpu().numpy()
        np.savez(output/'state.npz',**arrays)
        expected=np.diag([1.1,1.,1.])
        mu=50000./(2*1.3);lam=50000.*.3/(1.3*.4)
        strain=expected-np.eye(3)
        stress=((2*mu*strain+np.eye(3)*lam*np.trace(strain))@expected.T)/1.1
        errors={}
        for kind in ('sim','collision'):
            grad=arrays[kind+'_element_deform_gradient_w']
            sig=arrays[kind+'_element_stress_w']
            errors[kind]=dict(gradient=float(np.max(np.abs(grad-expected))),
                              stress_pa=float(np.max(np.abs(sig-stress))))
        receipt['affine_check']=dict(errors=errors,gradient_atol=1e-4,stress_atol_pa=10.)
        assert all(e['gradient']<=1e-4 and e['stress_pa']<=10. for e in errors.values()),errors
        receipt.update(completed=True,frames=20,source_unchanged=True,
                       mean_fall_m=float((before-after)[...,2].mean()),
                       finite_arrays={k:list(v.shape) for k,v in arrays.items()})
    except BaseException:
        receipt['error']=traceback.format_exc();raise
    finally:
        if sim is not None:sim.clear_all_callbacks();sim.clear_instance()
        if reserved:
            subprocess.run([*slots,'release','--owner',str(os.getpid()),'--gpu','0'],check=True,timeout=20)
            receipt['reservation_released']=True
        receipt['finished']=time.time();(output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if app is not None:
            kwargs=dict(wait_for_replicator=False)
            if 'exit_code' in inspect.signature(app.close).parameters:kwargs['exit_code']=0 if receipt['completed'] else 1
            app.close(**kwargs)

if __name__=='__main__':main()
