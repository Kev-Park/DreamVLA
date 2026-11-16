"""Bounded real contributed PPO updates, checkpoint restore, and resumed update.

The small rollout/optimizer budgets are technical smoke settings. This does not
measure task success or change the contributed production training recipe.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    if os.name != 'posix' or sys.version_info[:2] != (3, 12):
        raise SystemExit('Run only in the approved Linux Isaac 6.2 environment')
    assert importlib.metadata.version('isaacsim').startswith('6.2.')
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['drawer', 'standing_button'], required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(started=time.time(), task=args.task, completed=False,
                   scientific_training_run=False, scope=__doc__)
    app = env = None
    try:
        app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app
        import gymnasium as gym
        import torch
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from dreamvla_drawer_button.isaac.sonic_residual_actor import register_residual_actor
        from dreamvla_drawer_button.isaac.sonic_teacher_repair_contract import RepairSpec
        if args.task == 'drawer':
            from dreamvla_drawer_button.isaac import sonic_teacher_repair as task
            Agent = task.G1DreamVlaStylePPORunnerCfg
        else:
            from dreamvla_drawer_button.isaac import motion_tracking_button_press_standing_sonic_env as task
            Agent = task.G1ButtonPressStandingPPORunnerCfg
        task.register_task()
        register_residual_actor()
        cfg = parse_env_cfg(task.TASK, device=args.device, num_envs=2)
        cfg.seed = 17
        cfg.ref_motions_path = str(args.references.expanduser().resolve(strict=True))
        if args.task == 'drawer':
            from dreamvla_drawer_button.scene import close_drawer_on_reset
            task.configure(cfg, RepairSpec(scale=0.5))
            close_drawer_on_reset(cfg)
        agent = Agent()
        agent.seed = 17
        agent.device = args.device
        agent.logger = 'tensorboard'
        agent.num_steps_per_env = 8
        agent.algorithm.num_mini_batches = 1
        agent.algorithm.num_learning_epochs = 1
        agent.save_interval = 1
        receipt['smoke_overrides'] = dict(num_envs=2, num_steps_per_env=8,
                                         num_mini_batches=1, num_learning_epochs=1)
        from isaaclab.utils.io import dump_yaml
        dump_yaml(str(output/'env.yaml'), cfg)
        dump_yaml(str(output/'agent.yaml'), agent)
        env = RslRlVecEnvWrapper(gym.make(task.TASK, cfg=cfg), clip_actions=None)
        assert env.unwrapped.max_episode_length_s == 10.0
        assert env.unwrapped.max_episode_length == 500
        receipt['episode_length_s'] = env.unwrapped.max_episode_length_s
        receipt['max_episode_length'] = env.unwrapped.max_episode_length
        assert env.num_actions == 66, env.num_actions
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(output/'initial'), device=args.device)
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
        first = output/'initial/model_1.pt'
        saved = torch.load(first, map_location='cpu', weights_only=False)
        assert saved['iter'] == 1

        def exact(left, right):
            if isinstance(left, torch.Tensor):
                assert isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
            elif isinstance(left, dict):
                assert left.keys() == right.keys()
                for key in left:
                    exact(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                assert len(left) == len(right)
                for a, b in zip(left, right):
                    exact(a, b)
            else:
                assert left == right

        del runner
        resumed = OnPolicyRunner(env, agent.to_dict(), log_dir=str(output/'resumed'), device=args.device)
        resumed.load(str(first))
        # Compare live restored state before another optimizer update. Calling
        # runner.save here would require a logger initialized only by learn().
        exact(saved['model_state_dict'], resumed.alg.policy.state_dict())
        exact(saved['optimizer_state_dict'], resumed.alg.optimizer.state_dict())
        assert resumed.current_learning_iteration == 1
        resumed.current_learning_iteration += 1  # Same zero-based accounting as train_ppo.
        resumed.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        final = torch.load(output/'resumed/model_2.pt', map_location='cpu', weights_only=False)
        assert final['iter'] == 2
        before, after = saved['optimizer_state_dict']['state'], final['optimizer_state_dict']['state']
        assert before and before.keys() == after.keys()
        for key in before:
            assert float(after[key]['step']) > float(before[key]['step'])
        assert any(not torch.equal(value, final['model_state_dict'][key])
                   for key, value in saved['model_state_dict'].items())
        for state in (saved, final):
            for value in state['model_state_dict'].values():
                assert torch.isfinite(value).all()
        receipt.update(completed=True, action_width=env.num_actions,
                       checkpoint_iterations=[saved['iter'], final['iter']],
                       exact_model_and_optimizer_restore=True,
                       resumed_parameters_changed=True, finished=time.time(),
                       task_success_assessed=False,
                       limitation='New runner in same process/environment; not a fresh simulator process or RNG parity test')
    except BaseException:
        receipt.update(error=traceback.format_exc(), finished=time.time())
        raise
    finally:
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2))
        try:
            if env is not None:
                env.close()
        except BaseException:
            receipt.update(completed=False, cleanup_error=traceback.format_exc())
            (output/'receipt.json').write_text(json.dumps(receipt, indent=2))
            raise
        finally:
            if app is not None:
                app.close(exit_code=0 if receipt['completed'] else 1)


if __name__ == '__main__':
    main()
