import ast
import os
import re
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dreamvla_drawer_button"


def test_no_private_paths_or_unexported_tool_imports():
    for path in PACKAGE.rglob("*.py"):
        text = path.read_text()
        assert not re.search(r"/(?:clifford-data|home)/[A-Za-z0-9_.-]+/", text)
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("tools.")
                if node.level and node.module:
                    base = path.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    target = base.joinpath(*node.module.split("."))
                    assert (
                        target.with_suffix(".py").exists()
                        or (target / "__init__.py").exists()
                    ), (path, node.module)


@pytest.mark.parametrize(
    "command",
    [
        "doctor",
        "prepare-run",
        "train-ppo",
        "record",
        "convert",
        "finetune",
        "serve",
        "refs-button",
        "refs-drawer-stand",
        "refs-drawer-squat",
        "sample-human",
        "prepare-button-asset",
        "prepare-references",
        "prepare-robot-asset",
    ],
)
def test_help_does_not_need_isaac_or_model_assets(command):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-m", "dreamvla_drawer_button", command, "--help"],
        cwd="/tmp",
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
