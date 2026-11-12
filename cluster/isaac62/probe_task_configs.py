"""Construct every registered project motion-task configuration in Isaac 6.2.

This checks configuration constructors and registry targets, not simulator or
policy behavior. Representative reset/step/render gates are separate probes.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback


def main():
    assert os.name == 'posix'
    version = importlib.metadata.version('isaacsim')
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-51', action='store_true',
                        help='Run the same constructor audit against an original 5.1 source export')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    assert version.startswith('5.1.') if args.baseline_51 else version == '6.2.0.0'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    args.headless = True
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(started=time.time(), completed=False, tasks=[], isaacsim=version,
                  scope='All registered project motion-task config constructors; no environment allocation')
    app = None
    try:
        app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401: populates the actual registry
        report['task_package'] = str(Path(isaaclab_tasks.__file__).resolve())
        from isaaclab_tasks.utils import load_cfg_from_registry
        for task, spec in sorted(gym.registry.items()):
            entry = spec.kwargs.get('env_cfg_entry_point', '')
            if not isinstance(entry, str) or not any(
                    name in entry for name in ('.motion_tracking.', '.interactive_motion_tracking.')):
                continue
            row = dict(task=task, entry=entry, completed=False)
            report['tasks'].append(row)
            try:
                cfg = load_cfg_from_registry(task, 'env_cfg_entry_point')
                row.update(completed=True, config_class=type(cfg).__name__,
                           decimation=cfg.decimation, physics_dt=cfg.sim.dt,
                           default_num_envs=cfg.scene.num_envs,
                           agent_config_entries=sorted(k for k in spec.kwargs if k != 'env_cfg_entry_point'))
            except Exception:
                row['error'] = traceback.format_exc()
        assert report['tasks'], 'No project task configs discovered'
        report['completed'] = all(row['completed'] for row in report['tasks'])
        report['task_count'] = len(report['tasks'])
        report['failure_count'] = sum(not row['completed'] for row in report['tasks'])
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['finished'] = time.time()
        (args.output/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
        if app is not None:
            if args.baseline_51:
                # SDK 5.1 has no exit_code argument and its fast shutdown may
                # return zero even on failure. The persisted receipt is the
                # authoritative baseline result, never the process exit alone.
                report['legacy_shutdown_exit_is_not_a_result'] = True
                (args.output/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
                app.close()
            else:
                app.close(exit_code=0 if report['completed'] else 1)
    if not report['completed']:
        raise SystemExit('Project task configuration audit failed')


if __name__ == '__main__':
    main()
