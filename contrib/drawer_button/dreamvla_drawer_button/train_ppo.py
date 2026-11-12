"""Train an isolated SONIC residual PPO expert; never alter upstream tasks."""

import argparse
from pathlib import Path
import sys

from .contracts import TASKS
from .io import write_json, read, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--iterations",
        type=int,
        default=9000,
        help="PPO iterations, not environment steps; set 3000 for that paper recipe",
    )
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path)
    if "--help" in sys.argv or "-h" in sys.argv:
        p.print_help()
        return
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    if min(a.iterations, a.num_envs) < 1:
        p.error("Iteration and environment counts must be positive")
    from .run_config import references

    root, hashes = references(a.references)
    a.output = a.output.resolve()
    contract = dict(
        task=a.task,
        references=str(root),
        reference_sha256=hashes,
        seed=a.seed,
        num_envs=a.num_envs,
        iterations=a.iterations,
        residual_mode="multiplicative_free",
        residual_scale=0.5,
        fsq=True,
    )
    if a.output.exists():
        if not a.resume or read(a.output / "training_contract.json") != contract:
            raise ValueError("Existing output requires an exact-contract resume")
        if a.resume.resolve().parent != a.output:
            raise ValueError("Resume checkpoints must belong to this run")
    elif a.resume:
        raise ValueError("Cannot resume into a different or missing run directory")
    else:
        a.output.mkdir(parents=True)
        write_json(a.output / "training_contract.json", contract, durable=True)
    a.headless = True
    app = AppLauncher(a).app
    env = None
    try:
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401 -- task registration side effects
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from .isaac.sonic_residual_actor import register_residual_actor
        from .isaac.sonic_teacher_repair_contract import RepairSpec

        if a.task == "drawer":
            from .isaac.sonic_teacher_repair import (
                TASK,
                register_task,
                configure,
                G1DreamVlaStylePPORunnerCfg as Agent,
            )
        else:
            from .isaac.motion_tracking_button_press_standing_sonic_env import (
                TASK,
                register_task,
                G1ButtonPressStandingPPORunnerCfg as Agent,
            )
        register_task()
        register_residual_actor()
        cfg = parse_env_cfg(TASK, device=a.device, num_envs=a.num_envs)
        cfg.seed = a.seed
        cfg.ref_motions_path = str(root)
        spec = RepairSpec(scale=0.5)
        if a.task == "drawer":
            configure(cfg, spec)
            from .scene import close_drawer_on_reset

            close_drawer_on_reset(cfg)
        agent = Agent()
        agent.seed = a.seed
        agent.device = a.device
        agent.max_iterations = a.iterations
        agent.save_interval = 500
        params = a.output / "params"
        params.mkdir(exist_ok=True)
        dump_yaml(str(params / "env.yaml"), cfg)
        dump_yaml(str(params / "agent.yaml"), agent)
        write_json(params / "repair_spec.json", spec.to_dict(), durable=True)
        env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=None)
        runner = OnPolicyRunner(
            env, agent.to_dict(), log_dir=str(a.output), device=a.device
        )
        completed = 0
        if a.resume:
            runner.load(str(a.resume))
            # RSL-RL 2.3 saves the zero-based index AFTER that update.
            completed = runner.current_learning_iteration + 1
            runner.current_learning_iteration = completed
        if completed >= a.iterations:
            raise ValueError(
                "Checkpoint already reached the requested total iteration budget"
            )
        runner.learn(
            num_learning_iterations=a.iterations - completed, init_at_random_ep_len=True
        )
        write_json(
            a.output / "completed.json",
            dict(
                iterations=a.iterations,
                training_contract_sha256=digest(a.output / "training_contract.json"),
            ),
            durable=True,
        )
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
