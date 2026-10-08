#!/bin/bash
# Relocate the exact working GR00T environment; no dependency upgrades.
set -euo pipefail
mkdir -p /workspace/kevin /workspace/hf/hub /workspace/logs
test -e /root/kevin || ln -s /workspace/kevin /root/kevin
export HF_HOME=/workspace/hf
export CUDA_HOME=/usr/local/cuda
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
if ! command -v python3.10 >/dev/null; then
  uv python install 3.10
  ln -s "$(uv python find 3.10)" /usr/bin/python3.10
fi
mkdir -p /opt/Isaac-GR00T
if [ ! -d /opt/Isaac-GR00T/.git ]; then
  git -C /opt/Isaac-GR00T init
  git -C /opt/Isaac-GR00T remote add origin https://github.com/Kev-Park/Isaac-GR00T.git
  git -C /opt/Isaac-GR00T fetch --depth 1 origin 29c95b9cda2249b5285c84460c789d4fe3841d97
fi
git -C /opt/Isaac-GR00T checkout --detach 29c95b9cda2249b5285c84460c789d4fe3841d97
test -e /opt/Isaac-GR00T/.venv || ln -s /opt/kevin/Isaac-GR00T/.venv /opt/Isaac-GR00T/.venv
cd /workspace/kevin
if [ ! -L Isaac-GR00T ]; then
  test ! -d Isaac-GR00T || mv Isaac-GR00T Isaac-GR00T.network_copy
  ln -s /opt/Isaac-GR00T Isaac-GR00T
fi
# Recreate interpreter links/config for the destination Python while keeping
# every installed package from the source environment.
/usr/bin/python3.10 -m venv --upgrade --without-pip Isaac-GR00T/.venv
# The editable installation stores the original repository path. Supply an
# environment-level symlink, leaving the copied package and code untouched.
old=$(Isaac-GR00T/.venv/bin/python -c 'import __editable___gr00t_0_1_0_finder as f; print(f.MAPPING["gr00t"])')
mkdir -p "$(dirname "$(dirname "$old")")"
test -e "$(dirname "$old")" || ln -s /workspace/kevin/Isaac-GR00T "$(dirname "$old")"
cd Isaac-GR00T
.venv/bin/python -c 'import torch,gr00t,torchcodec,flash_attn; print(torch.__version__,torch.cuda.get_device_name(),"ENV_READY")'
