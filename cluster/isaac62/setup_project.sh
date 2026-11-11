#!/usr/bin/env bash
# Manual capacity review is required before running this installer.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../.." && pwd)
owned=$(realpath "$HOME/kevin")
prefix="$owned/envs/dreamcontrol_62"
[[ -f "$prefix/dreamvla_install_receipt.json" ]] || {
    echo 'Create and inspect the isolated foundation environment first.' >&2; exit 2;
}
command -v uv >/dev/null
store=$("$prefix/bin/python" - "$prefix/dreamvla_install_receipt.json" "$owned" <<'PY'
import json,sys
with open(sys.argv[1]) as stream:
    print(json.load(stream).get('storage_root',sys.argv[2]))
PY
)
export UV_CACHE_DIR="$store/cache/uv_isaac62"
export UV_PYTHON_INSTALL_DIR="$store/envs/managed_python_isaac62"
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$prefix/bin/python" - <<'PY'
import importlib.metadata as m
import sys
assert sys.version_info[:2] == (3, 12), sys.version
assert m.version('isaacsim') == '6.2.0.0'
assert m.version('torch') == '2.11.0+cu128'
PY
uv pip install --python "$prefix/bin/python" \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --index-strategy unsafe-best-match -r "$here/requirements.project.lock" \
    -e "$repo/Training/isaac_utils" \
    -e "$repo/Training/source/isaaclab" \
    -e "$repo/Training/source/isaaclab_assets" \
    -e "$repo/Training/source/isaaclab_tasks" \
    -e "$repo/Training/source/isaaclab_rl[rsl-rl]"
uv pip check --python "$prefix/bin/python"
"$prefix/bin/python" - "$prefix" "$repo" <<'PY'
import importlib.metadata as m
import json
from pathlib import Path
import subprocess
import sys
import time
prefix, repo = map(Path, sys.argv[1:])
commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
packages = ('isaacsim', 'torch', 'torchvision', 'torchaudio', 'numpy', 'pillow',
            'isaaclab', 'isaaclab_assets', 'isaaclab_tasks', 'isaaclab_rl',
            'isaac_utils', 'rsl-rl-lib', 'onnx2torch', 'onnxruntime-gpu')
receipt = {'installed_at': time.time(), 'source_commit': commit, 'source_checkout': str(repo),
           'versions': {name: m.version(name) for name in packages},
           'legacy_environment_modified': False, 'project_runtime_validated': False,
           'gr00t_runtime_installed': False}
(prefix/'dreamvla_project_install_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
PY
