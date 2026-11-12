#!/usr/bin/env bash
# Install the real SONIC client package without changing the isolated runtime.
set -euo pipefail
[[ $# == 1 ]] || { echo 'Usage: setup_vla_client.sh <GR00T-WholeBodyControl-source>' >&2; exit 2; }
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../.." && pwd)
gear=$(realpath "$1")
prefix="$HOME/kevin/envs/dreamcontrol_62"
[[ -f "$prefix/dreamvla_project_install_receipt.json" ]]
[[ -f "$gear/gear_sonic/pyproject.toml" ]]
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$HOME/kevin/cache/uv_isaac62"
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$prefix/bin/python" - "$prefix/vla_client_packages_before.json" <<'PY'
import importlib.metadata as m,json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))
PY
uv pip install --python "$prefix/bin/python" --constraint "$here/requirements.project.lock" \
    -e "$gear/gear_sonic"
uv pip check --python "$prefix/bin/python"
"$prefix/bin/python" - "$prefix" "$repo" "$gear" <<'PY'
import importlib.metadata as m,json,sys,time
from pathlib import Path
prefix,repo,gear=map(Path,sys.argv[1:])
before=json.loads((prefix/'vla_client_packages_before.json').read_text())
assert all(m.version(name)==version for name,version in before.items()), 'Existing versions changed'
sys.path.insert(0,str(repo/'Training/scripts/reinforcement_learning/rsl_rl'))
from gear_sonic.data.features_sonic_vla import EGO_VIEW_HEIGHT,EGO_VIEW_WIDTH,_JOINT_GROUPS_FOR_STATE
from vla_sonic.obs_to_policy import ObsToPolicyAdapter
from vla_sonic.simple_robot_model import SimpleG1RobotModel
robot=SimpleG1RobotModel.build()
groups={name:robot.get_joint_group_indices(name) for name in _JOINT_GROUPS_FOR_STATE}
assert [len(groups[name]) for name in _JOINT_GROUPS_FOR_STATE]==[6,6,3,7,7,7,7]
assert (EGO_VIEW_HEIGHT,EGO_VIEW_WIDTH)==(480,640)
receipt={'installed_at':time.time(),'existing_versions_preserved':True,
         'gear_source':str(gear),'joint_groups':groups,'image_shape':[480,640,3],
         'versions':{name:m.version(name) for name in ('gear_sonic','numpy','scipy','torch','pin')}}
(prefix/'dreamvla_vla_client_install_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
PY
