# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Self-locating paths for assets that live OUTSIDE this repo.

Several call sites need files that ship with a *sibling* repo (holosoma's object meshes,
GR00T-WholeBodyControl's recorded motions) or with a shared asset drop (SONIC weights).
Those used to be absolute paths into one machine's home directory, which meant the code
only ran on that machine and silently pointed at the wrong checkout everywhere else.

The convention is that clones sit side by side::

    <parent>/
      DreamVLA/            <- this repo (a.k.a. WBCBenchmark)
      holosoma/
      GR00T-WholeBodyControl/
      sonic/               <- shared weight drop, not a repo

``sibling()`` resolves against that layout with no configuration. It searches the repo's
own parent first so a git worktree checked out under its own directory picks up whatever
sits beside *it*, then falls back to ``~/kevin`` and ``$HOME`` so a worktree can share one
copy of the big read-only assets instead of duplicating them per branch.

Nothing here needs to be set up by someone cloning the repo: put the sibling clone next to
this one (or under ``~/kevin``) and it is found.
"""

from __future__ import annotations

import os
from pathlib import Path

# .../DreamVLA/Training/source/isaaclab_tasks/isaaclab_tasks/utils/repo_paths.py -> DreamVLA
REPO_ROOT = Path(__file__).resolve().parents[5]


def search_roots() -> list[Path]:
    """Directories searched for a sibling clone / shared asset drop, in priority order."""
    roots = [REPO_ROOT.parent, Path.home() / "kevin", Path.home()]
    seen, out = set(), []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def sibling(name: str, *parts: str, env: str | None = None) -> Path:
    """Path to ``<root>/name/*parts`` for the first search root where it exists.

    Args:
        name: sibling directory name, e.g. ``"holosoma"``.
        parts: path components below it.
        env: optional environment variable holding a full override path.

    Returns:
        The first existing candidate; if none exist, the candidate under the repo's own
        parent, so the error message points at the layout the caller is expected to use.
    """
    if env and os.environ.get(env):
        return Path(os.environ[env]).expanduser()
    candidates = [root / name / Path(*parts) if parts else root / name for root in search_roots()]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]
