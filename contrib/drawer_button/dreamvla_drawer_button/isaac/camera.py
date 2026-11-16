"""Keep the production ego camera on its physical mount across USD hierarchies."""


def ego_camera_cfg(robot_usd):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg

    stage = Usd.Stage.Open(str(robot_usd))
    root = stage.GetDefaultPrim() if stage is not None else None
    if not root or not root.IsValid():
        raise ValueError(f"Robot USD has no default prim: {robot_usd}")
    mounts = [prim for prim in Usd.PrimRange(root) if prim.GetName() == 'd435_link']
    if len(mounts) != 1:
        raise ValueError(f"Expected one physical d435_link camera mount in {robot_usd}")
    mount = mounts[0]
    parent = mount
    while parent and parent.IsValid() and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
        parent = parent.GetParent()
    if not parent or not parent.IsValid():
        raise ValueError("Camera mount has no rigid-body ancestor")
    # The SDK 6 importer keeps massless fixed links as non-rigid Xforms. Fabric
    # does not propagate their camera world transforms when the torso moves.
    # Attach directly to the nearest rigid link with the same composed pose.
    relative_transform, resets = UsdGeom.XformCache().ComputeRelativeTransform(mount, parent)
    if resets:
        raise ValueError("Camera mount crosses a reset transform below its rigid parent")
    camera_transform = Gf.Matrix4d(1.0)
    camera_transform.SetRotate(Gf.Quatd(0.5, Gf.Vec3d(0.5, -0.5, -0.5)))
    camera_transform.SetTranslateOnly(Gf.Vec3d(0.05, 0.0, 0.0))
    transform = Gf.Transform(camera_transform*relative_transform)
    if any(abs(float(x)-1.0) > 1e-6 for x in transform.GetScale()):
        raise ValueError("Camera mount must preserve unit scale")
    rotation = transform.GetRotation().GetQuat()
    position = tuple(float(x) for x in transform.GetTranslation())
    quaternion = (float(rotation.GetReal()), *(float(x) for x in rotation.GetImaginary()))
    relative = parent.GetPath().MakeRelativePath(root.GetPath())
    camera_name = 'Camera' if parent == mount else 'd435_Camera'
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{relative}/{camera_name}",
        spawn=PinholeCameraCfg(
            focal_length=9.1, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 100.0),
        ),
        data_types=['rgb'], height=480, width=640,
        offset=CameraCfg.OffsetCfg(
            pos=position, rot=quaternion, convention='opengl',
        ),
    )
