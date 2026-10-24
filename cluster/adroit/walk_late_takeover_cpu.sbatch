#!/bin/bash -l
#SBATCH --job-name=walk_late_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -eo pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
STAGE=${1:?stage required}
TAG=${2:?experiment tag required}
BATCH=${3:-}
cd "$HOME/kevin/wt/locomotion/DreamVLA"
args=(cluster/walk_late_takeover_process.py "$STAGE" --tag "$TAG")
if [ -n "$BATCH" ]; then args+=(--batch "$BATCH"); fi
exec "$HOME/kevin/Isaac-GR00T/.venv/bin/python" "${args[@]}"
