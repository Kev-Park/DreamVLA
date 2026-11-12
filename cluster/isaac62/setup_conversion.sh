#!/usr/bin/env bash
# Run only after approval for this separate CPU-only conversion environment.
# Existing simulator/model environments and their package versions are untouched.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_root=${1:?Pass the committed GR00T-WholeBodyControl export root}
prefix="$HOME/kevin/envs/dreamvla_conversion_cpu"
sim_prefix="$HOME/kevin/envs/dreamcontrol_62"
[[ -f "$source_root/gear_sonic/pyproject.toml" ]]
[[ ! -e "$prefix" ]] || { echo 'Refusing to overwrite an existing environment'; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$HOME/kevin/cache/uv_isaac62"
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
uv venv --python "$sim_prefix/bin/python" "$prefix"
uv pip compile --python "$prefix/bin/python" "$here/requirements.conversion.txt" \
    --output-file "$prefix/requirements.lock"
uv pip install --python "$prefix/bin/python" -r "$prefix/requirements.lock"
uv pip install --python "$prefix/bin/python" -e "$source_root/gear_sonic" \
    --constraint "$prefix/requirements.lock"
uv pip check --python "$prefix/bin/python"
"$prefix/bin/python" - "$prefix" <<'PY'
import hashlib, importlib.metadata as m, json, sys, time
from pathlib import Path
import torch, h5py, datasets
from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_sonic_vla import get_g1_robot_model
p = Path(sys.argv[1])
assert torch.version.cuda is None
get_g1_robot_model()
receipt = dict(installed_at=time.time(), cpu_only=True,
               lock_sha256=hashlib.sha256((p/'requirements.lock').read_bytes()).hexdigest(),
               versions={n:m.version(n) for n in ('torch','torchvision','torchcodec','datasets','lerobot','numpy','scipy','pin','h5py','gear_sonic')})
(p/'install_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
PY
