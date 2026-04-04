from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open("g1_29dof.usd")
joint_prim = stage.GetPrimAtPath("/World/Robot/waist_roll_joint")

# Change to fixed joint
fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_prim.GetPath())
stage.RemovePrim(joint_prim.GetPath())  # Optional: Remove old revolute joint

stage.GetRootLayer().Save()
