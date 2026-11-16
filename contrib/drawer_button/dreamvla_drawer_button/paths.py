"""Explicit external assets and backends; no user, host, or GPU assumptions."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys


def _repository_root():
    configured = os.environ.get("DREAMVLA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "Training").is_dir() and (parent / ".git").exists():
            return parent
    return Path.cwd().resolve()


REPOSITORY_ROOT = _repository_root()
BACKEND_ROOT = (
    Path(os.environ.get("DREAMVLA_BACKEND_ROOT", REPOSITORY_ROOT))
    .expanduser()
    .resolve()
)
ASSET_ROOT = (
    Path(
        os.environ.get(
            "DREAMVLA_ASSET_ROOT",
            REPOSITORY_ROOT / "contrib/drawer_button/.runtime/assets",
        )
    )
    .expanduser()
    .resolve()
)
SONIC_REPO = (
    Path(
        os.environ.get(
            "DREAMVLA_SONIC_REPO",
            REPOSITORY_ROOT / "third_party/GR00T-WholeBodyControl",
        )
    )
    .expanduser()
    .resolve()
)
GROOT_REPO = (
    Path(
        os.environ.get(
            "DREAMVLA_GROOT_REPO", REPOSITORY_ROOT / "third_party/Isaac-GR00T"
        )
    )
    .expanduser()
    .resolve()
)


def add_external_paths():
    """Use explicit existing dependency checkouts, never download or install them."""
    for path in (SONIC_REPO, GROOT_REPO):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def ffmpeg_path():
    candidate = os.environ.get("DREAMVLA_FFMPEG") or shutil.which("ffmpeg")
    if not candidate or not Path(candidate).is_file():
        raise FileNotFoundError(
            "Install ffmpeg or set DREAMVLA_FFMPEG to its executable."
        )
    return str(Path(candidate).resolve())
