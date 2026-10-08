"""Headless startup/physics/render compatibility check; no task assessment."""
import argparse
import json
import os
from pathlib import Path
os.environ['OMNI_KIT_ALLOW_ROOT'] = '1'
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app
try:
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    import torch
    from PIL import Image
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/60, device='cuda:0'))
    sim_utils.GroundPlaneCfg().func('/World/Ground', sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2000)
    light.func('/World/Light', light)
    camera = Camera(CameraCfg(prim_path='/World/TestCamera', height=128, width=128,
        data_types=['rgb'], spawn=sim_utils.PinholeCameraCfg()))
    sim.reset()
    camera.set_world_poses_from_view(torch.tensor([[2., 2., 2.]], device='cuda:0'),
                                     torch.tensor([[0., 0., 0.]], device='cuda:0'))
    for _ in range(20):
        sim.step()
        camera.update(1/60, force_recompute=True)
    rgb = camera.data.output['rgb'][0, :, :, :3].cpu().numpy()
    assert rgb.max() > 0 and rgb.std() > 0, 'Camera frame is black or constant'
    Image.fromarray(rgb).save('/workspace/logs/isaac_smoke.png')
    Path('/workspace/logs/isaac_smoke.json').write_text(json.dumps({'physics_steps': 20,
        'headless': args.headless, 'enable_cameras': args.enable_cameras,
        'rgb_shape': list(rgb.shape), 'rgb_std': float(rgb.std())}))
finally:
    app.close()
