#!/usr/bin/env bash
# Target-runtime CPU contracts only. No Isaac, CUDA packages or real study.
# Review account-wide CPU and disk capacity before launching.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
owned=$(realpath "$HOME/kevin")
prefix="$owned/envs/dreamcontrol_62_cpu_contracts"
command -v uv >/dev/null || { echo 'uv is required.' >&2; exit 2; }
if [[ -e "$prefix" || -L "$prefix" ]]; then
    echo "Refusing to modify existing environment $prefix; inspect its receipt first." >&2
    exit 2
fi
free_bytes=$(df -PB1 "$owned" | awk 'NR==2 {print $4}')
if (( free_bytes < 8 * 1024 * 1024 * 1024 )); then
    echo 'Less than 8 GiB free; review space before creating the CPU test environment.' >&2
    exit 2
fi
export UV_CACHE_DIR="$owned/cache/uv_isaac62_cpu_contracts"
export UV_PYTHON_INSTALL_DIR="$owned/envs/managed_python_isaac62_cpu_contracts"
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
mkdir -p "$owned/envs"
uv venv --python 3.12 --managed-python "$prefix"
uv pip install --python "$prefix/bin/python" --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match -r "$here/requirements.cpu-contracts.lock"
uv pip check --python "$prefix/bin/python"
"$prefix/bin/python" - "$prefix" <<'PY'
import importlib.metadata as m
import json
from pathlib import Path
import shutil
import sys
import time
import torch
assert sys.version_info[:2] == (3, 12)
assert m.version('torch') == '2.11.0+cpu'
assert m.version('numpy') == '2.3.1'
assert torch.version.cuda is None
receipt = {'installed_at': time.time(), 'python': sys.version, 'prefix': sys.argv[1],
           'packages': {key: m.version(key) for key in ('torch', 'numpy', 'scipy', 'tensordict', 'rsl-rl-lib', 'h5py', 'pyzmq')},
           'cuda_build': False, 'isaacsim_installed': False, 'legacy_environment_modified': False,
           'disk_free_after': shutil.disk_usage(sys.argv[1]).free}
(Path(sys.argv[1])/'dreamvla_cpu_contract_install_receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
PY
