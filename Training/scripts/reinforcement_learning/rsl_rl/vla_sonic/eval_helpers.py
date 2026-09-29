"""Shared helpers for the SONIC eval / play / collect scripts.

* ``flush_render(env)`` -- deliver the current camera frame WITHOUT stepping physics.
"""

from __future__ import annotations


def flush_render(env, n: int = 2) -> None:
    """Flush the RTX pipeline so camera annotators deliver the CURRENT frame, without stepping physics.

    A raw ``simulation_app.update()`` with the timeline playing advances PhysX by one sim.dt
    (measured: +2 ms at 500 Hz, +5 ms at 200 Hz per call), so the old "2 pumps after every
    env.step" ran the robot 24-30 ms per 20 ms SONIC control step. ``SimulationContext.render()``
    performs the same app update with /app/player/playSimulations disabled (SONIC audit 2026-09-29).
    """
    for _ in range(n):
        env.unwrapped.sim.render()
