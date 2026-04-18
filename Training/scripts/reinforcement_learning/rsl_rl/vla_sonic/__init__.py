"""VLA → SONIC glue for closed-loop eval in WBCBenchmark.

Pipeline wired up across these modules:
    env.obs ─▶ obs_to_policy ─▶ Gr00tPolicy ─▶ teleop command dict
                                                      │
                                              vla_to_planner
                                                      │
                                              PlannerWrapper (ONNX)
                                                      │
                                              planner_to_utm
                                                      │
                                              UtmWrapper (PyTorch)
                                                      │
                                              env.step(joint_targets)

Shipped now (no VLA checkpoint or SONIC weights required):
- planner_wrapper: onnxruntime session for planner_sonic.onnx.
- frame_transforms: coordinate transforms + speed_to_mode.
- obs_to_policy:   Isaac Lab env obs → Gr00tPolicy input dict.

Shipped once weights land (Stage 2c):
- utm_wrapper:     load UniversalTokenModule from sonic_release/last.pt.
- vla_to_planner:  VLA action dict → planner ONNX input dict.
- planner_to_utm:  planner output + VLA upper-body → UTM tokenizer input.
"""

# Eager re-exports: modules that have no heavy dependencies.
from .frame_transforms import (
    speed_to_mode,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    world_to_anchor_local_position,
    world_to_anchor_local_orientation,
    body_vel_to_world,
    world_vel_to_body,
    pelvis_relative_pose,
)
from .planner_wrapper import PlannerWrapper
from .utm_wrapper import UtmWrapper
from .planner_to_utm import (
    build_encoder_obs,
    build_decoder_obs,
    rot6d_to_quat_wxyz,
    ENCODER_LAYOUT,
    ENCODER_SLICES,
    ENCODER_TOTAL_DIM,
    DECODER_LAYOUT,
    DECODER_SLICES,
    DECODER_TOTAL_DIM,
)

# obs_to_policy is NOT eagerly imported: it depends on gear_sonic + a live
# Isaac Lab env. Import it explicitly at use-site:
#     from vla_sonic.obs_to_policy import ObsToPolicyAdapter, ObsAdapterConfig

__all__ = [
    "speed_to_mode",
    "quat_wxyz_to_xyzw",
    "quat_xyzw_to_wxyz",
    "world_to_anchor_local_position",
    "world_to_anchor_local_orientation",
    "body_vel_to_world",
    "world_vel_to_body",
    "pelvis_relative_pose",
    "PlannerWrapper",
    "UtmWrapper",
    "build_encoder_obs",
    "build_decoder_obs",
    "rot6d_to_quat_wxyz",
    "ENCODER_LAYOUT",
    "ENCODER_SLICES",
    "ENCODER_TOTAL_DIM",
    "DECODER_LAYOUT",
    "DECODER_SLICES",
    "DECODER_TOTAL_DIM",
]
