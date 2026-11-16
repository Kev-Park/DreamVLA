"""Prepare a render-only panel from separately obtained upstream licensed assets."""

import argparse
from pathlib import Path
import shutil

from .io import digest, write_json
from .paths import BACKEND_ROOT, ASSET_ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=BACKEND_ROOT / "Training/assets/Collected_ElevatorLobby",
    )
    p.add_argument(
        "--source-commit",
        required=True,
        help="Revision from which the separately obtained panel was taken",
    )
    p.add_argument(
        "--build", action="store_true", help="Requires a working USD/pxr runtime"
    )
    a = p.parse_args()
    files = {
        "buttons.usd": a.source / "buttons.usd",
        "textures/UpDown.jpg": a.source / "textures/UpDown.jpg",
        "LICENSE": BACKEND_ROOT / "LICENSE",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError(f"Resolve this Git-LFS asset first: {path}")
    directory = ASSET_ROOT / "button_press"
    if directory.exists():
        raise FileExistsError(
            "Use a new asset root rather than overwriting a prepared panel"
        )
    rows = []
    for relative, path in files.items():
        dest = directory / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        rows.append(dict(local_path=relative, sha256=digest(dest)))
    write_json(
        directory / "source_manifest.json",
        dict(
            repository="Kev-Park/DreamVLA",
            commit=a.source_commit,
            downloaded_files=rows,
        ),
        durable=True,
    )
    if a.build:
        from .button_visual import build

        print(build())


if __name__ == "__main__":
    main()
