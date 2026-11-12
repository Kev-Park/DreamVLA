"""Read-only dependency/asset preflight. Does not allocate a GPU or launch Kit."""

import argparse
import importlib.util
import json

from .paths import (
    BACKEND_ROOT,
    SONIC_REPO,
    GROOT_REPO,
    ASSET_ROOT,
    add_external_paths,
    ffmpeg_path,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage", choices=("isaac", "reference", "data", "groot"), default="isaac"
    )
    a = p.parse_args()
    add_external_paths()
    dependencies = {
        "isaac": ["isaaclab", "isaaclab_tasks", "rsl_rl", "onnxruntime"],
        "reference": ["torch", "pytorch_kinematics", "pyroki", "jaxlie"],
        "data": ["gear_sonic", "cv2", "pyarrow"],
        "groot": ["gr00t", "transformers", "torch"],
    }
    assets = {
        "isaac": [
            ASSET_ROOT / "sonic_release/model_encoder.onnx",
            ASSET_ROOT / "sonic_release/model_decoder.onnx",
            ASSET_ROOT / "sonic_g1/g1_29dof_with_hand.usd",
        ],
        "reference": [
            BACKEND_ROOT
            / "Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_27dof.urdf"
        ],
        "data": [
            BACKEND_ROOT
            / "Sim2Real/resources/robots/g1_description/g1_29dof_with_hand.urdf"
        ],
        "groot": [],
    }
    modules = {
        name: importlib.util.find_spec(name) is not None
        for name in dependencies[a.stage]
    }
    files = {}
    for path in assets[a.stage]:
        ok = path.is_file() and path.stat().st_size > 256
        if ok:
            with path.open("rb") as stream:
                ok = not stream.read(80).startswith(
                    b"version https://git-lfs.github.com/spec/"
                )
        files[str(path)] = ok
    try:
        encoder = ffmpeg_path()
    except FileNotFoundError:
        encoder = None
    print(
        json.dumps(
            dict(
                stage=a.stage,
                modules=modules,
                files=files,
                ffmpeg=encoder,
                backend=str(BACKEND_ROOT),
                sonic=str(SONIC_REPO),
                groot=str(GROOT_REPO),
                note="Presence checks only; verify version locks and run an allocated-GPU smoke test",
            ),
            indent=2,
        )
    )
    if (
        not all(modules.values())
        or not all(files.values())
        or (a.stage in ("isaac", "data") and not encoder)
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
