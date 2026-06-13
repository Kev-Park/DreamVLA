"""SONIC-matching physics overrides for the IsaacLab env.

CORRECTED 2026-06-13 after a deep audit of gear_sonic's actual training config:
**SONIC trains in IsaacLab, not MuJoCo.** The decoder's training-time dynamics are
defined by ``gear_sonic/config/manager_env/base_env.yaml`` and
``gear_sonic/envs/manager_env/robots/g1.py`` — NOT the MuJoCo deploy XML. The previous
override (500 Hz / decimation-10, "to match MuJoCo's 2 ms integration") was based on a
mis-reading: it pushed the decoder OFF its training distribution rather than onto it.

Verified gear_sonic training values to match:
    sim_dt      = 0.005   (200 Hz physics)        base_env.yaml:32
    decimation  = 4       (→ 50 Hz control)       base_env.yaml:30
    feet/ankle gains = 2.0 × 5020                 g1.py:281-283  (see actuator builder)
    enabled_self_collisions = True                g1.py:215
    armature    = per-motor                       g1.py  (already matched)
    action_scale = 0.25*effort/stiffness          g1.py  (already matched)

Apply at BOTH training and inference time so the encoder trains on, and is evaluated
against, the same dynamics the frozen decoder was trained for.
"""

from __future__ import annotations


def apply_sonic_physics_overrides(
    env_cfg,
    *,
    static_friction: float = 1.0,
    dynamic_friction: float = 1.0,
    enable_self_collisions: bool = True,
) -> None:
    """Override env_cfg physics in-place to match the SONIC decoder's training conditions.

    Matches gear_sonic's IsaacLab training env (base_env.yaml + g1.py). Wrapped in
    try/except per term because not every env variant exposes the same paths.
    """
    # 1. Substep rate: 200 Hz physics, 50 Hz control — matches gear_sonic base_env.yaml
    # (sim_dt=0.005, decimation=4), i.e. the dynamics the decoder was trained against.
    # `decimation` is on the env_cfg TOP LEVEL (the env sets self.decimation in
    # __post_init__); env_cfg.sim.decimation is a no-op on this IsaacLab version but set
    # defensively.
    env_cfg.sim.dt = 0.005
    env_cfg.decimation = 4
    try:
        env_cfg.sim.decimation = 4
    except Exception:
        pass
    actual_step_ms = env_cfg.sim.dt * env_cfg.decimation * 1000.0
    actual_hz = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
    print(f"[sonic-physics] sim.dt={env_cfg.sim.dt}s, env_cfg.decimation={env_cfg.decimation} "
          f"→ control step = {actual_step_ms:.1f} ms ({actual_hz:.1f} Hz) [matches gear_sonic training]")
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

    # 4. Articulation solver position iterations (gear_sonic g1.py: position=8, velocity=4).
    try:
        env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        print("[sonic-physics] articulation solver_position_iteration_count = 8")
    except AttributeError:
        print("[sonic-physics] could not override articulation solver_position_iteration_count")

    # 5. Self-collisions ON — gear_sonic trains with enabled_self_collisions=True (g1.py:215).
    # G1_MINIMAL_CFG ships False, so the decoder was trained expecting self-contact (e.g.
    # arm-vs-torso) that our env wasn't generating — and the left arm was free to clip
    # through the leg. Matching this both removes the clipping and restores the contact
    # regime the decoder saw in training.
    if enable_self_collisions:
        try:
            env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
            print("[sonic-physics] enabled_self_collisions = True (matches gear_sonic training)")
        except AttributeError:
            print("[sonic-physics] could not override enabled_self_collisions")
