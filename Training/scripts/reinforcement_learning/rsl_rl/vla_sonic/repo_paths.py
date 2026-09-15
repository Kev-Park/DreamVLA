"""Pre-AppLauncher access to ``isaaclab_tasks.utils.repo_paths``.

The scripts build their argparse defaults BEFORE the Isaac app is launched, and importing
``isaaclab_tasks`` at that point pulls in ``isaaclab.envs`` -> ``omni.*`` and fails. So load the
canonical module by file path (no package ``__init__`` side effects) and re-export it. The file
is the single source of truth; this shim only changes how it is reached.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# .../Training/scripts/reinforcement_learning/rsl_rl/vla_sonic/repo_paths.py -> Training
_CANON = Path(__file__).resolve().parents[4] / "source/isaaclab_tasks/isaaclab_tasks/utils/repo_paths.py"
_spec = importlib.util.spec_from_file_location("_wbc_repo_paths", _CANON)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

REPO_ROOT = _mod.REPO_ROOT
SHARED_ROOT = _mod.SHARED_ROOT
sibling = _mod.sibling
pooled = _mod.pooled
dataset = _mod.dataset


def gear_sonic_deploy(*parts: str) -> str:
    """Path under the sibling GR00T-WholeBodyControl clone's gear_sonic_deploy (ONNX weights,
    planner, observation config). Replaces the old CWD-relative ``../../GR00T-WholeBodyControl/...``
    defaults, which only resolved when run from the main checkout's ``Training/`` directory."""
    return str(sibling("GR00T-WholeBodyControl", "gear_sonic_deploy", *parts))
