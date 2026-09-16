#!/bin/bash
# Batch template for Adroit. Runs one DreamVLA script inside the Isaac container.
#
#   sbatch cluster/adroit/adroit.sbatch scripts/reinforcement_learning/rsl_rl/train.py --task ... --headless
#   WT=my-branch sbatch cluster/adroit/adroit.sbatch scripts/...   # run from a worktree instead of main
#
# The `isaac` shim (cluster/adroit/isaac, on PATH via ~/kevin/bin) opens the proxy tunnel and
# enters the container; isaac.rc activates conda + exports the proxy/Vulkan env inside it.
#SBATCH --job-name=dreamvla
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --gres=gpu:1 --constraint=gpu80
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
if [ -n "${WT:-}" ]; then
  cd ~/kevin/wt/$WT/DreamVLA/Training
  ACTIVATE="source ~/kevin/wt/$WT/.venv/bin/activate"
else
  cd ~/kevin/DreamVLA/Training
  ACTIVATE="conda activate dreamcontrol_51"
fi
echo "node=$(hostname) gpu=$(nvidia-smi -L | head -1) cwd=$PWD"
isaac bash -c "$ACTIVATE && python $*"
