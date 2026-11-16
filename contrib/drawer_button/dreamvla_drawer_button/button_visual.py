"""DreamVLA's original elevator panel, derived as a render-only task fixture.

Original meshes, UVs, and materials are retained. Only broken texture paths,
physics removal, and a target-alignment transform are authored in new files.
"""

import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .button_geometry import ButtonTaskVisual as BaseTaskVisual, audit_visual_only

from .paths import ASSET_ROOT

DIRECTORY = ASSET_ROOT / "button_press"
SOURCE = DIRECTORY / "source"
ASSET = DIRECTORY / "dreamvla_button_visual_v1.usda"
PANEL = DIRECTORY / "button_panel_visual.usdc"
METADATA = DIRECTORY / "visual_manifest.json"
VERSION = "dreamvla-g1-863d7095-button-visual-v1"
CAP = "/root/ElevatorButtonUpDown_dn/geo"
CAP_MESH = CAP + "/ElevatorButtonUp"
TRAVEL = 0.0113671338185668
FINGER_RADIUS = 0.012


def manifest():
    return json.loads(METADATA.read_text())


def world_vertices(stage, path):
    from pxr import UsdGeom, Gf

    prim = stage.GetPrimAtPath(path)
    mesh = UsdGeom.Mesh(prim)
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return np.array(
        [matrix.Transform(Gf.Vec3d(*map(float, p))) for p in mesh.GetPointsAttr().Get()]
    )


def mesh_signature(stage):
    from pxr import UsdGeom

    result = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        # Includes geometry, normals and original UVs; excludes physics/xforms.
        attrs = {
            a.GetName(): str(a.Get())
            for a in prim.GetAttributes()
            if a.GetName()
            in (
                "points",
                "faceVertexCounts",
                "faceVertexIndices",
                "normals",
                "orientation",
                "subdivisionScheme",
            )
            or a.GetName().startswith("primvars:")
        }
        result[str(prim.GetPath())] = hashlib.sha256(
            json.dumps(attrs, sort_keys=True).encode()
        ).hexdigest()
    return result


def audit(stage):
    from pxr import Sdf

    audit_visual_only(stage.GetDefaultPrim())
    for prim in stage.Traverse():
        if "Joint" in prim.GetTypeName():
            raise ValueError("Physics joint survived")
        if any(s.startswith(("Physics", "Physx")) for s in prim.GetAppliedSchemas()):
            raise ValueError("Physics schema survived")
        for attr in prim.GetAttributes():
            if attr.GetName().startswith(("physics:", "physx")):
                raise ValueError("Physics attribute survived")
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            if not value or not value.path or value.path == "OmniPBR.mdl":
                continue
            if not value.resolvedPath or not Path(value.resolvedPath).is_file():
                raise ValueError("Unresolved asset: " + str(attr.GetPath()))


def build():
    from pxr import Usd, UsdGeom, Sdf, Gf

    for path in (ASSET, PANEL, METADATA):
        if path.exists():
            raise FileExistsError(
                "Never overwrite versioned visual assets: " + str(path)
            )
    source_manifest = json.loads((DIRECTORY / "source_manifest.json").read_text())
    for row in source_manifest["downloaded_files"]:
        if (
            hashlib.sha256((SOURCE / row["local_path"]).read_bytes()).hexdigest()
            != row["sha256"]
        ):
            raise ValueError("Upstream source changed")
    source = Usd.Stage.Open(str(SOURCE / "buttons.usd"))
    original_meshes = mesh_signature(source)
    cap_points = world_vertices(source, CAP_MESH)
    low = cap_points.min(axis=0)
    high = cap_points.max(axis=0)
    face = np.array([low[0], (low[1] + high[1]) / 2, (low[2] + high[2]) / 2])
    radius = float(max(high[1:] - low[1:]) / 2)
    stage = Usd.Stage.Open(source.Flatten())
    removed = []
    fixed = []
    for prim in list(stage.Traverse()):
        if "Joint" in prim.GetTypeName():
            removed.append(str(prim.GetPath()))
            stage.RemovePrim(prim.GetPath())
            continue
        schemas = list(prim.GetAppliedSchemas())
        keep = [s for s in schemas if not s.startswith(("Physics", "Physx"))]
        prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(keep))
        for attr in list(prim.GetAttributes()):
            if attr.GetName().startswith(("physics:", "physx")):
                prim.RemoveProperty(attr.GetName())
                continue
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            if value and value.path and value.path != "OmniPBR.mdl":
                if Path(value.path).name != "UpDown.jpg":
                    raise ValueError("Unexpected button texture: " + value.path)
                fixed.append(
                    dict(
                        attribute=str(attr.GetPath()),
                        old=value.path,
                        new="./source/textures/UpDown.jpg",
                    )
                )
                attr.Set(Sdf.AssetPath("./source/textures/UpDown.jpg"))
        for rel in list(prim.GetRelationships()):
            if rel.GetName().startswith(("physics:", "physx")):
                prim.RemoveProperty(rel.GetName())
    if mesh_signature(stage) != original_meshes:
        raise ValueError("Original mesh/UV changed")
    stage.GetRootLayer().Export(str(PANEL))
    derived = Usd.Stage.Open(str(PANEL))
    audit(derived)
    stage = Usd.Stage.CreateNew(str(ASSET))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Button").GetPrim()
    stage.SetDefaultPrim(root)
    root.SetCustomDataByKey("version", VERSION)
    root.SetCustomDataByKey("visual_only", True)
    panel = UsdGeom.Xform.Define(stage, "/Button/Panel")
    panel.GetPrim().GetReferences().AddReference("./" + PANEL.name)
    # Replace the source root's identity scale op; actual child mesh transforms stay unchanged.
    panel.ClearXformOpOrder()
    panel.AddTranslateOp().Set(Gf.Vec3d(*(-face)))
    stage.GetRootLayer().Save()
    audit(stage)
    aligned = world_vertices(stage, "/Button/Panel" + CAP_MESH[len("/root") :])
    aligned_face = [
        aligned[:, 0].min(),
        (aligned[:, 1].min() + aligned[:, 1].max()) / 2,
        (aligned[:, 2].min() + aligned[:, 2].max()) / 2,
    ]
    if not np.allclose(aligned_face, 0, atol=1e-9):
        raise ValueError("Press surface does not align with target")
    result = dict(
        version=VERSION,
        source_repository=source_manifest["repository"],
        source_commit=source_manifest["commit"],
        asset=str(ASSET),
        asset_sha256=hashlib.sha256(ASSET.read_bytes()).hexdigest(),
        panel_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(),
        cap_path="/Panel" + CAP[len("/root") :],
        cap_mesh_path="/Panel" + CAP_MESH[len("/root") :],
        cap_radius_m=radius,
        cap_travel_m=TRAVEL,
        source_press_face_center_m=face.tolist(),
        target_local_m=[0.0, 0.0, 0.0],
        outward_axis=[-1.0, 0.0, 0.0],
        original_mesh_signatures=original_meshes,
        original_geometry_and_uvs_preserved=True,
        texture_fixes=fixed,
        removed_joints=removed,
        collision=False,
        rigid_body=False,
        physics=False,
        full_lobby_spawned=False,
        animation="Original movable upper panel button; cosmetic geometry-driven translation only.",
        placement="Move panel as a whole to align the movable button face with the original task target.",
        future_vla_contract="Use this exact asset/helper in BOTH new RGB training datasets and camera-based evaluation.",
        license="source/LICENSE; retain upstream attribution and terms",
    )
    with METADATA.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


class VisualButton:
    def __init__(self, stage, path, asset=ASSET):
        from pxr import UsdGeom, Gf

        self.Gf = Gf
        self.meta = manifest()
        if stage.GetPrimAtPath(path):
            raise ValueError("Button already exists")
        self.prim = stage.DefinePrim(path, "Xform")
        self.prim.GetReferences().AddReference(str(asset))
        audit_visual_only(self.prim)
        xf = UsdGeom.Xformable(self.prim)
        xf.SetResetXformStack(True)
        self.translate = xf.AddTranslateOp()
        cap = stage.GetPrimAtPath(path + self.meta["cap_path"])
        self.cap = cap.GetAttribute("xformOp:translate")
        self.cap_initial = np.array(self.cap.Get())
        parent = UsdGeom.XformCache().GetLocalToWorldTransform(cap.GetParent())
        self.press_axis = np.array(parent.GetInverse().TransformDir(Gf.Vec3d(1, 0, 0)))
        self.target = np.zeros(3)
        self.depth = 0.0

    def place(self, target, ground_z=0.0):
        target = np.asarray(target, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Invalid button target")
        self.target = target.copy()
        self.translate.Set(self.Gf.Vec3d(*target))

    def update(self, tip, dt=0.02):
        delta = np.asarray(tip, dtype=float) - self.target
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError("Invalid fingertip")
        x, y, z = delta
        desired = 0.0
        if math.hypot(y, z) <= self.meta["cap_radius_m"] + FINGER_RADIUS and x <= 0.03:
            desired = TRAVEL * float(np.clip((x + FINGER_RADIUS) / FINGER_RADIUS, 0, 1))
        tau = 0.035 if desired > self.depth else 0.08
        self.depth += (desired - self.depth) * (1 - math.exp(-dt / tau))
        self.cap.Set(self.Gf.Vec3d(*(self.cap_initial + self.press_axis * self.depth)))
        return self.depth

    def reset(self):
        self.depth = 0.0
        self.cap.Set(self.Gf.Vec3d(*self.cap_initial))


class ButtonTaskVisual(BaseTaskVisual):
    def __init__(self, env):
        import omni.usd
        import torch
        from isaaclab.utils.math import quat_apply

        self.env = env
        self.torch = torch
        self.quat_apply = quat_apply
        self.body = env.scene["robot"].find_bodies("right_wrist_yaw_link")[0][0]
        self.offset = torch.tensor([[0.21, 0.0, 0.03]], device=env.device).expand(
            env.num_envs, -1
        )
        stage = omni.usd.get_context().get_stage()
        self.buttons = [
            VisualButton(stage, path + "/VisualButton")
            for path in env.scene.env_prim_paths
        ]


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
