# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Self-locating paths for things that live OUTSIDE this repo.

Several call sites need files that ship with a *sibling* repo (holosoma's object meshes,
GR00T-WholeBodyControl's recorded motions) or with a shared data pool (retargeted datasets).
Those used to be absolute paths into one machine's home directory, which meant the code only
ran on that machine and silently pointed at the wrong checkout everywhere else.

Two resolution policies, because sibling repos and pooled data want opposite defaults:

``sibling()`` -- CODE, worktree-local first::

    <parent>/
      DreamVLA/                <- this repo (a.k.a. WBCBenchmark)
      holosoma/
      GR00T-WholeBodyControl/

It searches the repo's own parent first, so a git worktree that has its own holosoma
worktree beside it picks *that* one up -- which is what lets two branches develop different
retargeters concurrently. A worktree with no paired clone falls through to the shared
``~/kevin`` copy, so pairing is opt-in per branch and costs nothing when unused.

``pooled()`` -- DATA, shared first. Retargeted datasets are expensive to regenerate and
branches normally want the same best/most-recent one, so the shared drop wins and a
worktree-local copy is only used when nothing shared exists.

Both accept an explicit override via environment variable, named after the directory:
``HOLOSOMA_DIR``, ``GR00T_WHOLEBODYCONTROL_DIR``, ``HS_PICK_OUT_DIR``. That is the escape
hatch for pointing one worktree at a specific checkout or a variant dataset without moving
anything. Nothing here needs setting up by someone cloning the repo: put the sibling clone
next to this one (or under ``~/kevin``) and it is found.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# .../DreamVLA/Training/source/isaaclab_tasks/isaaclab_tasks/utils/repo_paths.py -> DreamVLA
REPO_ROOT = Path(__file__).resolve().parents[5]

#: Shared drop for assets and datasets that all worktrees pool.
SHARED_ROOT = Path.home() / "kevin"


def env_var_for(name: str) -> str:
    """Environment variable that overrides ``name``: ``GR00T-WholeBodyControl`` -> ``GR00T_WHOLEBODYCONTROL_DIR``."""
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_") + "_DIR"


def _resolve(name: str, parts: tuple[str, ...], roots: list[Path], env: str | None) -> Path:
    override = os.environ.get(env or "") or os.environ.get(env_var_for(name))
    if override:
        base = Path(override).expanduser()
        return base.joinpath(*parts) if parts else base
    seen, candidates = set(), []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        candidates.append(root / name / Path(*parts) if parts else root / name)
    return next((c for c in candidates if c.exists()), candidates[0])


def sibling(name: str, *parts: str, env: str | None = None) -> Path:
    """Path inside a sibling clone, preferring one beside THIS checkout (worktree-local)."""
    return _resolve(name, parts, [REPO_ROOT.parent, SHARED_ROOT, Path.home()], env)


def pooled(name: str, *parts: str, env: str | None = None) -> Path:
    """Path inside a shared data pool, preferring the drop all worktrees share."""
    return _resolve(name, parts, [SHARED_ROOT, REPO_ROOT.parent, Path.home()], env)
