#!/bin/bash -l
# Compatibility entrypoint: A100 is compute-only for this camera workflow.
#SBATCH --job-name=walk_late1s_ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=2-00:00:00
set -eo pipefail
TAG=${1:?experiment tag required}
DS=${2:?dataset required}
STEPS=${3:?steps required}
REPO=$HOME/kevin/wt/locomotion/DreamVLA
exec bash "$REPO/cluster/adroit/walk_late_takeover_finetune.sbatch" "$TAG" "$DS" "$STEPS"
