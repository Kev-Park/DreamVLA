"""Construct NVIDIA's G1 feature model from an explicitly configured backend."""

from .paths import BACKEND_ROOT, add_external_paths


def get_project_g1_robot_model():
    add_external_paths()
    from gear_sonic.data.robot_model.robot_model import RobotModel
    from gear_sonic.data.robot_model.supplemental_info.g1.g1_supplemental_info import (
        ElbowPose,
        G1SupplementalInfo,
        WaistLocation,
    )

    directory = BACKEND_ROOT / "Sim2Real/resources/robots/g1_description"
    path = directory / "g1_29dof_with_hand.urdf"
    if not path.is_file():
        raise FileNotFoundError(
            "Missing G1 URDF/meshes. Resolve the backend's Git-LFS robot assets."
        )
    return RobotModel(
        str(path),
        str(directory),
        supplemental_info=G1SupplementalInfo(
            waist_location=WaistLocation.LOWER_AND_UPPER_BODY, elbow_pose=ElbowPose.LOW
        ),
    )
