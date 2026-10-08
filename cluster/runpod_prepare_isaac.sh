#!/bin/bash
set -euo pipefail
mkdir -p /workspace/logs /opt/kevin
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
test "${DRIVER%%.*}" -ge 580 || { echo "Isaac 5.1 requires a newer host driver: $DRIVER"; exit 1; }
cd /opt/kevin
test -d DreamVLA/.git || git clone --depth 1 -b g1 https://github.com/Kev-Park/DreamVLA.git DreamVLA
test -d holosoma/.git || git clone --depth 1 https://github.com/Kev-Park/holosoma.git holosoma
test -d GR00T-WholeBodyControl/.git || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/Kev-Park/GR00T-WholeBodyControl.git GR00T-WholeBodyControl
if [ ! -d Isaac-GR00T/.git ]; then
  mkdir -p Isaac-GR00T
  git -C Isaac-GR00T init
  git -C Isaac-GR00T remote add origin https://github.com/Kev-Park/Isaac-GR00T.git
  git -C Isaac-GR00T fetch --depth 1 origin 29c95b9cda2249b5285c84460c789d4fe3841d97
  git -C Isaac-GR00T checkout --detach FETCH_HEAD
fi
until test -f /workspace/logs/isaac_env.ready && test -f /workspace/logs/isaac_data.ready && test -f /workspace/logs/isaac_assets.ready; do sleep 10; done
test -e DreamVLA/Training/assets || ln -s /opt/isaac_assets/assets DreamVLA/Training/assets
# Resolve the original editable-install root from the copied environment.
/opt/dreamcontrol_51/bin/python - <<'PY'
import ast
from pathlib import Path
env = Path('/opt/dreamcontrol_51')
for finder in (env/'lib/python3.11/site-packages').glob('*finder.py'):
    for node in ast.walk(ast.parse(finder.read_text())):
        if isinstance(node, ast.AnnAssign) and getattr(node.target,'id',None)=='MAPPING':
            for old in ast.literal_eval(node.value).values():
                if '/kevin/' in old:
                    oldroot=Path(old.split('/kevin/')[0])
                    oldroot.mkdir(parents=True,exist_ok=True)
                    if not (oldroot/'kevin').exists():
                        (oldroot/'kevin').symlink_to('/opt/kevin')
PY
export OMNI_KIT_ALLOW_ROOT=1 ACCEPT_EULA=Y
cd DreamVLA/Training
/opt/dreamcontrol_51/bin/python -c 'import torch; print(torch.__version__,torch.cuda.get_device_name(),torch.cuda.get_arch_list())'
touch /workspace/logs/isaac_prepare.ready
