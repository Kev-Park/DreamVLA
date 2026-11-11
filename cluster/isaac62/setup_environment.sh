#!/usr/bin/env bash
# Run only after manual account CPU/GPU allocation review. No study is launched.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
owned=$(realpath "$HOME/kevin")
prefix="$owned/envs/dreamcontrol_62"
command -v uv >/dev/null || { echo 'uv is required; no existing environment was changed.' >&2; exit 2; }
if [[ -e "$prefix" ]]; then
    echo "Refusing to modify existing environment $prefix; inspect its install receipt first." >&2
    exit 2
fi
mkdir -p "$owned/envs"
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
uv venv --python 3.12 --managed-python "$prefix"
uv pip install --python "$prefix/bin/python" --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.11.0' 'torchvision==0.26.0'
uv pip install --python "$prefix/bin/python" --extra-index-url https://pypi.nvidia.com \
    -r "$here/requirements.sim.txt"
"$prefix/bin/python" - "$prefix" <<'PY'
import importlib.metadata as m
import json
from pathlib import Path
import sys
import time
prefix = Path(sys.argv[1])
assert sys.version_info[:2] == (3, 12), sys.version
versions = {name: m.version(name) for name in ('isaacsim', 'torch', 'torchvision', 'numpy', 'pillow')}
assert versions['isaacsim'] == '6.2.0.0', versions
assert versions['torch'].split('+')[0] == '2.11.0', versions
receipt = {'created': time.time(), 'prefix': str(prefix), 'python': sys.version,
           'versions': versions, 'foundation_only': True, 'project_runtime_validated': False,
           'legacy_environment_modified': False}
(prefix / 'dreamvla_install_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
PY
