#!/usr/bin/env python3
"""Produce a 29-DOF + dex-hands G1 USD for SONIC eval (strict-fidelity articulation).

The in-use pick asset ``g1_27dof_with_hands_min_collisions_flat_white.usd`` has the
torso welded: ``waist_roll_joint`` / ``waist_pitch_joint`` are FixedJoints (27 DOF).
SONIC's decoder was trained on a 29-DOF articulation, so a rigid waist changes the
closed-loop dynamics it expects. This script restores those two joints to REVOLUTE,
copying axis / limits / drive from ``g1_29dof.usd`` (which has them revolute) while
preserving the target asset's own body hierarchy + hands + collision meshes.

It is DIAGNOSTIC-FIRST and SAFE: it prints what it finds in both assets, and only
writes the output if both waist joints are present and convertible. If anything is
unexpected it aborts without writing — paste the printout back for a tailored fix.

Run on the box that has USD Python bindings (Isaac Lab / Isaac Sim python, or any
env with `usd-core`):

    cd <repo>/WBCBenchmark/Training/HumanoidVerse/humanoidverse/data/robots/g1
    python make_g1_29dof_with_hands.py            # inspect + convert
    python make_g1_29dof_with_hands.py --inspect  # inspect only, write nothing

If `from pxr import ...` fails:  pip install usd-core
"""
from __future__ import annotations

import argparse
import shutil
import sys

try:
    from pxr import Usd, UsdPhysics, Gf, Sdf  # noqa: F401
except ImportError as exc:  # pragma: no cover
    sys.exit(f"[fatal] cannot import pxr ({exc}); run with Isaac Lab python or `pip install usd-core`")

SRC_29 = "g1_29dof.usd"  # revolute waist, no hands — parameter donor
TGT_27 = "g1_27dof_with_hands_min_collisions_flat_white.usd"  # hands + collisions, welded waist
OUT_29H = "g1_29dof_with_hands_min_collisions_flat_white.usd"  # what we build
WAIST = ["waist_roll_joint", "waist_pitch_joint"]


def find_joints(stage: "Usd.Stage") -> dict:
    """Map every joint prim by its leaf name -> (prim, type-string)."""
    out = {}
    for p in stage.Traverse():
        t = str(p.GetTypeName())
        if "Joint" in t and "JointState" not in t:
            out[p.GetName()] = (p, t)
    return out


def is_revolute(prim) -> bool:
    return bool(prim.IsA(UsdPhysics.RevoluteJoint))


def is_fixed(prim) -> bool:
    return bool(prim.IsA(UsdPhysics.FixedJoint))


def describe_joint(prim) -> str:
    j = UsdPhysics.Joint(prim)
    b0 = j.GetBody0Rel().GetTargets()
    b1 = j.GetBody1Rel().GetTargets()
    extra = ""
    if is_revolute(prim):
        r = UsdPhysics.RevoluteJoint(prim)
        axis = r.GetAxisAttr().Get()
        lo = r.GetLowerLimitAttr().Get()
        hi = r.GetUpperLimitAttr().Get()
        extra = f" axis={axis} limits=[{lo},{hi}]"
    return f"body0={[str(x) for x in b0]} body1={[str(x) for x in b1]}{extra}"


def dump(stage: "Usd.Stage", label: str) -> dict:
    joints = find_joints(stage)
    n_hand = sum(1 for n in joints if any(k in n for k in ("thumb", "index", "middle", "ring", "pinky")))
    print(f"\n=== {label} ===")
    print(f"  total joints: {len(joints)} | hand/finger joints: {n_hand}")
    for w in WAIST:
        if w in joints:
            prim, t = joints[w]
            kind = "REVOLUTE" if is_revolute(prim) else ("FIXED" if is_fixed(prim) else t)
            print(f"  {w}: {kind}  @ {prim.GetPath()}")
            print(f"      {describe_joint(prim)}")
        else:
            print(f"  {w}: *** NOT FOUND ***")
    return joints


def copy_xform_attr(j_src, j_dst, getter, creator):
    """Copy a local-pose attr (localPos/localRot 0/1) if authored."""
    val = getter(j_src).Get()
    if val is not None:
        creator(j_dst).Set(val)


def convert(args) -> int:
    print("[1/4] inspecting source (revolute donor) and target (hands) assets ...")
    src_stage = Usd.Stage.Open(SRC_29)
    tgt_stage = Usd.Stage.Open(TGT_27)
    if src_stage is None or tgt_stage is None:
        sys.exit(f"[fatal] could not open {SRC_29} or {TGT_27} (run from the g1 robots dir)")

    src_joints = dump(src_stage, f"SOURCE  {SRC_29}")
    tgt_joints = dump(tgt_stage, f"TARGET  {TGT_27}")

    # ---- pre-flight checks ----
    problems = []
    for w in WAIST:
        if w not in src_joints or not is_revolute(src_joints[w][0]):
            problems.append(f"{w} is not a REVOLUTE joint in {SRC_29}")
        if w not in tgt_joints:
            problems.append(f"{w} not present as a joint in {TGT_27} "
                            "(may be welded into a link — needs a different approach)")
    if problems:
        print("\n[ABORT] preconditions not met — writing nothing:")
        for p in problems:
            print(f"   - {p}")
        print("\nPaste this printout back and I'll tailor the conversion.")
        return 2

    if args.inspect:
        print("\n[inspect-only] structure looks convertible; re-run without --inspect to build.")
        return 0

    # ---- build output = flattened copy of the hands asset ----
    print(f"\n[2/4] copying {TGT_27} -> {OUT_29H} (flattened, self-contained) ...")
    flat = tgt_stage.Flatten()                 # compose refs/payloads into one layer
    flat.Export(OUT_29H)
    out_stage = Usd.Stage.Open(OUT_29H)
    out_joints = find_joints(out_stage)

    # ---- convert each waist joint: keep target's body frame, take source's axis/limits/drive ----
    print("[3/4] converting waist joints fixed -> revolute ...")
    for w in WAIST:
        src_prim = src_joints[w][0]
        out_prim, _ = out_joints[w]
        path = out_prim.GetPath()

        # read the joint frame from the TARGET (preserves the hands-asset hierarchy)
        old = UsdPhysics.Joint(out_prim)
        b0 = old.GetBody0Rel().GetTargets()
        b1 = old.GetBody1Rel().GetTargets()
        lp0 = old.GetLocalPos0Attr().Get(); lr0 = old.GetLocalRot0Attr().Get()
        lp1 = old.GetLocalPos1Attr().Get(); lr1 = old.GetLocalRot1Attr().Get()

        # read axis / limits / drive from the SOURCE revolute joint
        src_rev = UsdPhysics.RevoluteJoint(src_prim)
        axis = src_rev.GetAxisAttr().Get()
        lo = src_rev.GetLowerLimitAttr().Get()
        hi = src_rev.GetUpperLimitAttr().Get()
        src_drive = UsdPhysics.DriveAPI.Get(src_prim, "angular")

        # replace the fixed joint with a revolute joint at the same path
        out_stage.RemovePrim(path)
        rev = UsdPhysics.RevoluteJoint.Define(out_stage, path)
        if b0: rev.CreateBody0Rel().SetTargets(b0)
        if b1: rev.CreateBody1Rel().SetTargets(b1)
        if lp0 is not None: rev.CreateLocalPos0Attr().Set(lp0)
        if lr0 is not None: rev.CreateLocalRot0Attr().Set(lr0)
        if lp1 is not None: rev.CreateLocalPos1Attr().Set(lp1)
        if lr1 is not None: rev.CreateLocalRot1Attr().Set(lr1)
        if axis is not None: rev.CreateAxisAttr().Set(axis)
        if lo is not None: rev.CreateLowerLimitAttr().Set(lo)
        if hi is not None: rev.CreateUpperLimitAttr().Set(hi)

        # drive: Isaac's ImplicitActuatorCfg overrides gains at runtime, but author one so
        # the joint is recognised as actuated. Copy source values if present.
        drv = UsdPhysics.DriveAPI.Apply(rev.GetPrim(), "angular")
        if src_drive:
            for get, create in (
                (src_drive.GetStiffnessAttr, drv.CreateStiffnessAttr),
                (src_drive.GetDampingAttr, drv.CreateDampingAttr),
                (src_drive.GetMaxForceAttr, drv.CreateMaxForceAttr),
                (src_drive.GetTargetPositionAttr, drv.CreateTargetPositionAttr),
            ):
                v = get().Get()
                if v is not None:
                    create().Set(v)
        else:
            drv.CreateStiffnessAttr().Set(0.0)   # actuator cfg sets the real gains
            drv.CreateDampingAttr().Set(0.0)
        print(f"   [ok] {w}: FIXED -> REVOLUTE  axis={axis} limits=[{lo},{hi}]")

    out_stage.GetRootLayer().Save()

    # ---- verify ----
    print("[4/4] verifying output ...")
    chk = Usd.Stage.Open(OUT_29H)
    chk_joints = dump(chk, f"OUTPUT  {OUT_29H}")
    ok = all(w in chk_joints and is_revolute(chk_joints[w][0]) for w in WAIST)
    n_hand = sum(1 for n in chk_joints if any(k in n for k in ("thumb", "index", "middle")))
    if ok and n_hand > 0:
        print(f"\n[SUCCESS] wrote {OUT_29H}  (both waist joints revolute, {n_hand} hand joints preserved)")
        print("Next: point a SONIC 29-DOF robot cfg at this USD + add waist_roll/pitch actuators "
              "(mirror gear_sonic G1_CYLINDER_MODEL_12_DEX_CFG gains) + expand the action space 27->29.")
        return 0
    print("\n[WARN] output did not verify cleanly — inspect the OUTPUT dump above.")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build a 29-DOF + dex-hands G1 USD for SONIC.")
    ap.add_argument("--inspect", action="store_true", help="inspect only; write nothing")
    raise SystemExit(convert(ap.parse_args()))
