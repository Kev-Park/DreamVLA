"""Construct the contributed task and PPO configurations in the real SDK.

This is an import/constructor gate, not simulation, training or rendering proof.
"""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(started=time.time(), completed=False,
                  isaacsim=importlib.metadata.version('isaacsim'), tasks=[],
                  scope='Contributed task registration, task/PPO configuration constructors only')
    assert report['isaacsim'] == '6.2.0.0'
    app = None
    try:
        app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app
        import isaaclab_tasks
        from isaaclab_tasks.utils import parse_env_cfg
        report['backend_package'] = str(Path(isaaclab_tasks.__file__).resolve())
        cases = [
            ('drawer', 'sonic_teacher_repair', 'G1DreamVlaStylePPORunnerCfg'),
            ('standing_button', 'motion_tracking_button_press_standing_sonic_env',
             'G1ButtonPressStandingPPORunnerCfg'),
        ]
        for task, module, agent in cases:
            row = dict(task=task, completed=False)
            report['tasks'].append(row)
            try:
                source = importlib.import_module('dreamvla_drawer_button.isaac.'+module)
                source.register_task()
                cfg = parse_env_cfg(source.TASK, device=args.device, num_envs=2)
                policy = getattr(source, agent)()
                row.update(completed=True, task_id=source.TASK,
                           config_class=type(cfg).__name__,
                           policy_class=type(policy).__name__, decimation=cfg.decimation,
                           physics_dt=cfg.sim.dt, num_envs=cfg.scene.num_envs,
                           action_terms=[k for k,v in vars(cfg.actions).items() if v is not None],
                           config=cfg.to_dict(), policy=policy.to_dict())
            except Exception:
                row['error'] = traceback.format_exc()
        report['completed'] = all(row['completed'] for row in report['tasks'])
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['finished'] = time.time()
        (args.output/'receipt.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
        if app is not None:
            app.close(exit_code=0 if report['completed'] else 1)
    if not report['completed']:
        raise SystemExit('Contributed task constructor gate failed')


if __name__ == '__main__':
    main()
