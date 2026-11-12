#!/usr/bin/env bash
# Optional CPU reference-tool dependencies; preserve the installed simulator stack.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
prefix="$HOME/kevin/envs/dreamcontrol_62"
[[ -f "$prefix/dreamvla_project_install_receipt.json" ]]
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null
export UV_CACHE_DIR="$HOME/kevin/cache/uv_isaac62"
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$prefix/bin/python" - "$prefix/reference_packages_before.json" <<'PY'
import importlib.metadata as m,json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))
PY
uv pip install --python "$prefix/bin/python" -r "$here/requirements.references.lock" \
    --constraint "$here/requirements.project.lock"
uv pip check --python "$prefix/bin/python"
"$prefix/bin/python" - "$prefix" <<'PY'
import importlib.metadata as m,json,sys,time
from pathlib import Path
prefix=Path(sys.argv[1]);before=json.loads((prefix/'reference_packages_before.json').read_text())
assert all(m.version(name)==version for name,version in before.items()), 'Existing package versions changed'
import jax,jaxlie,pytorch_kinematics
assert jax.default_backend()=='cpu'
receipt={'installed_at':time.time(),'existing_versions_preserved':True,
         'versions':{n:m.version(n) for n in ('jax','jaxlib','jaxlie','numpy','scipy','torch')}}
(prefix/'dreamvla_reference_install_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
PY
