#!/bin/bash -l
# CPU-only verification; no GPU requested or used.
#SBATCH --job-name=pick_normalization
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -eo pipefail
DS=${1:?dataset required}
WORK=${2:?work directory required}
test ! -e "$WORK/normalization.exit_code.txt"
trap 'code=$?; printf "%s\n" "$code" > "$WORK/normalization.exit_code.txt"' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
"$HOME/kevin/Isaac-GR00T/.venv/bin/python" -u \
    "$HOME/kevin/DreamVLA/cluster/verify_pick_normalization_cache.py" \
    "$DS" "$WORK" "$HOME/kevin/Isaac-GR00T" > "$WORK/normalization.log" 2>&1
