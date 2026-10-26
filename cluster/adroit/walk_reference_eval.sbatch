#!/bin/bash -l
# Unchanged walking evaluation of an existing checkpoint; no fine-tune.
#SBATCH --job-name=walk_reference_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=08:00:00
set -eo pipefail
TAG=${1:?control label required}
CHECKPOINT=${2:?checkpoint required}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$HOME/kevin/bin/isaac" bash -c '
  source "$HOME/kevin/bin/isaac.rc"
  conda activate dreamcontrol_51
  cd "$HOME/kevin/wt/locomotion/DreamVLA"
  export PYTHONPATH="$PWD/cluster${PYTHONPATH:+:$PYTHONPATH}"
  exec python -c '\''import sys;from walk_late_takeover_process import evaluate;evaluate(sys.argv[1],sys.argv[2],0,run_name=sys.argv[1])'\'' "$1" "$2"
' bash "$TAG" "$CHECKPOINT"
