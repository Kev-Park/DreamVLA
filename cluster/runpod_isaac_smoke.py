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
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/60, device='cuda:0'))
    sim_utils.GroundPlaneCfg().func('/World/Ground', sim_utils.GroundPlaneCfg())
    sim.reset()
    for _ in range(10):
        sim.step()
    Path('/workspace/logs/isaac_smoke.json').write_text(json.dumps({'physics_steps': 10, 'headless': args.headless, 'enable_cameras': args.enable_cameras}))
finally:
    app.close()
