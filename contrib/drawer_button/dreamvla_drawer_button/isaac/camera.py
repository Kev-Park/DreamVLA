"""Keep the production ego camera on its physical mount across USD hierarchies."""


def ego_camera_cfg(robot_usd):
    from pxr import Usd
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg

    stage = Usd.Stage.Open(str(robot_usd))
    root = stage.GetDefaultPrim() if stage is not None else None
    if not root or not root.IsValid():
        raise ValueError(f"Robot USD has no default prim: {robot_usd}")
    mounts = [prim for prim in Usd.PrimRange(root) if prim.GetName() == 'd435_link']
    if len(mounts) != 1:
        raise ValueError(f"Expected one physical d435_link camera mount in {robot_usd}")
    relative = mounts[0].GetPath().MakeRelativePath(root.GetPath())
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{relative}/Camera",
        spawn=PinholeCameraCfg(
            focal_length=9.1, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 100.0),
        ),
        data_types=['rgb'], height=480, width=640,
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.0), rot=(0.5, 0.5, -0.5, -0.5), convention='opengl',
        ),
    )
