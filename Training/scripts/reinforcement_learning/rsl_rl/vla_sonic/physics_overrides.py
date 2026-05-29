"""SONIC-matching physics overrides for the IsaacLab env.

The SONIC decoder was trained against a physics setup the deploy reference / eval
scripts call out explicitly:

    eval_parquet_sonic.py:755  "500 Hz (dt=2ms, 10 substeps per 20ms control step)
                                matches MuJoCo's integration rate and is required
                                for stable contact dynamics."

The default IsaacLab env runs at 200 Hz substeps (decimation=4, sim.dt=0.005) and
randomizes friction per reset. Without these overrides, contact dynamics are noisier
than what the decoder was trained on → the robot jitters/spasms even when commanded
tokens are reasonable. The overrides bring the sim back into the decoder's training
distribution.

Apply at BOTH training and inference time — and use identical values so the encoder
trained on these dynamics evaluates against the same dynamics.
"""

from __future__ import annotations


def apply_sonic_physics_overrides(env_cfg, *, static_friction: float = 1.0, dynamic_friction: float = 1.0) -> None:
    """Override env_cfg physics in-place to match the SONIC decoder's training conditions.

    Mirrors eval_parquet_sonic.py:739-766. Wrapped in try/except per term because not
    every env variant exposes the same paths (e.g. event-term keys vary).
    """
    # 1. Substep rate: 500 Hz physics, 50 Hz control. Required for stable contacts.
    # `decimation` is on the env_cfg TOP LEVEL, not on env_cfg.sim — the env sets
    # `self.decimation = 4` in its __post_init__. (eval_parquet_sonic.py writes
    # `env_cfg.sim.decimation = 10` which silently no-ops; we fix that here so the
    # override actually lands.)
    env_cfg.sim.dt = 1.0 / 500.0
    env_cfg.decimation = 10
    # Also set the sim-side attribute defensively in case future IsaacLab versions
    # add a SimulationCfg.decimation field — harmless if it stays unrecognized.
    try:
        env_cfg.sim.decimation = 10
    except Exception:
        pass
    actual_step_ms = env_cfg.sim.dt * env_cfg.decimation * 1000.0
    actual_hz = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
    print(f"[sonic-physics] sim.dt={env_cfg.sim.dt}s, env_cfg.decimation={env_cfg.decimation} "
          f"→ control step = {actual_step_ms:.1f} ms ({actual_hz:.1f} Hz)")
    if abs(actual_hz - 50.0) > 0.5:
        raise RuntimeError(
            f"[sonic-physics] computed control rate {actual_hz:.1f} Hz != 50 Hz target — "
            f"check env_cfg.decimation path"
        )

    # 2. Fixed terrain friction (no randomization → distribution match).
    try:
        env_cfg.scene.terrain.physics_material.static_friction = static_friction
        env_cfg.scene.terrain.physics_material.dynamic_friction = dynamic_friction
        print(f"[sonic-physics] terrain friction fixed: static={static_friction} dynamic={dynamic_friction}")
    except AttributeError:
        print("[sonic-physics] terrain.physics_material not found — skipping terrain friction override")

    # 3. Fixed robot-body friction (no randomization).
    try:
        p = env_cfg.events.physics_material.params
        p["static_friction_range"] = (static_friction, static_friction)
        p["dynamic_friction_range"] = (dynamic_friction, dynamic_friction)
        print(f"[sonic-physics] body friction fixed: static={static_friction} dynamic={dynamic_friction}")
    except (AttributeError, KeyError):
        print("[sonic-physics] events.physics_material not found — skipping body friction override")

    # 4. Articulation solver position iterations.
    try:
        env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        print("[sonic-physics] articulation solver_position_iteration_count = 8")
    except AttributeError:
        print("[sonic-physics] could not override articulation solver_position_iteration_count")
