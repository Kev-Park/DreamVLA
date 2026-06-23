"""Promote the SONIC eval env from 27-DOF to 29-DOF (actuated waist roll/pitch).

SONIC's decoder was trained on a 29-DOF articulation — gear_sonic's g1.py actuates
``waist_roll_joint`` / ``waist_pitch_joint``. Our motion-tracking env defaults to the
27-DOF asset (those two joints welded FIXED), which is a train/eval mismatch: the
decoder commands 29 joints but the sim can only execute 27, and a rigid torso changes
the closed-loop dynamics the controller expects.

This helper, applied to a parsed ``env_cfg``, makes the env truly 29-DOF:
  1. swaps in the 29-DOF + dex-hands USD (built by make_g1_29dof_with_hands.py),
  2. actuates waist roll/pitch with gear_sonic's gains (2x 5020, == the feet),
  3. seeds their init pose at 0, and
  4. extends the body ``joint_pos`` action term from 27 to 29 joints.

All OTHER env terms (obs joint_pos/vel, rewards, JOINTS_MASK/KEYPTS_MASK, reset) keep
operating on their existing 27-joint ``JointNamesOrder`` sets — they are independent of
the action term and of the two newly-actuated joints, so they need no change for the
playback/eval path (which reads proprioception from ``robot.data`` by name, not via the
policy obs group).

Lazy-imports isaaclab inside the function so the vla_sonic package stays importable in a
plain-numpy environment.
"""

from __future__ import annotations

WAIST_DOF_JOINTS = ["waist_roll_joint", "waist_pitch_joint"]

_USD_27 = "g1_27dof_with_hands_min_collisions_flat_white"
_USD_29 = "g1_29dof_with_hands_min_collisions_flat_white"

# gear_sonic gains (g1.py): waist roll/pitch use the SAME gains as the feet — 2x 5020.
_ARMATURE_5020 = 0.003609725
_NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535  # 10 Hz
_DAMPING_RATIO = 2.0
_STIFFNESS_5020 = _ARMATURE_5020 * _NATURAL_FREQ ** 2
_DAMPING_5020 = 2.0 * _DAMPING_RATIO * _ARMATURE_5020 * _NATURAL_FREQ


def apply_29dof_waist_override(env_cfg, *, usd_path: str | None = None) -> list[str]:
    """Mutate ``env_cfg`` in place for 29-DOF waist actuation.

    Returns the body action term's joint-name list (length 29, the order the env's
    action vector uses for its body portion) so the caller can build the
    SONIC-29 -> env-body permutation via ``action_assembler.build_sonic29_to_env_perm``.
    """
    from pathlib import Path

    from isaaclab.actuators import ImplicitActuatorCfg

    robot = env_cfg.scene.robot

    # 1. Swap to the 29-DOF + hands USD (derive from the 27-DOF filename if not given).
    cur = getattr(robot.spawn, "usd_path", "") or ""
    new = usd_path or (cur.replace(_USD_27, _USD_29) if _USD_27 in cur else None)
    if not new:
        raise ValueError(
            f"could not derive the 29-DOF USD from spawn.usd_path={cur!r}; "
            f"pass usd_path= explicitly (expected a '{_USD_29}.usd')."
        )
    if not Path(new).exists():
        # Isaac's asset resolver may still find it via a search path, so warn (don't fail).
        print(f"[29dof][warn] 29-DOF USD not found at literal path '{new}' — relying on "
              f"Isaac's asset resolver. Build it with make_g1_29dof_with_hands.py if it errors.")
    robot.spawn.usd_path = new
    print(f"[29dof] robot USD -> {new}")

    # 2. Actuate waist roll/pitch with gear_sonic's gains (2x 5020, == feet).
    robot.actuators["waist"] = ImplicitActuatorCfg(
        joint_names_expr=list(WAIST_DOF_JOINTS),
        effort_limit_sim=50.0,
        velocity_limit_sim=37.0,
        stiffness=2.0 * _STIFFNESS_5020,
        damping=2.0 * _DAMPING_5020,
        armature=2.0 * _ARMATURE_5020,
    )
    print(f"[29dof] added 'waist' actuator (k={2.0 * _STIFFNESS_5020:.2f}, "
          f"d={2.0 * _DAMPING_5020:.3f}) for {WAIST_DOF_JOINTS}")

    # 3. Seed the two new joints at 0 (default standing) if init_state.joint_pos is a dict.
    try:
        jp = robot.init_state.joint_pos
        if isinstance(jp, dict):
            for j in WAIST_DOF_JOINTS:
                jp.setdefault(j, 0.0)
    except AttributeError:
        pass

    # 4. Extend the body joint_pos action term to 29 (preserve_order is already True;
    #    the SONIC-29->env perm is name-matched downstream, so append order is irrelevant).
    body = env_cfg.actions.joint_pos
    names = list(body.joint_names)
    for j in WAIST_DOF_JOINTS:
        if j not in names:
            names.append(j)
    body.joint_names = names
    print(f"[29dof] body action term joint_names: {len(names)} joints (was {len(names) - 2})")
    return names
