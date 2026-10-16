#!/bin/bash -l
# Read-only input equivalence check; no GPU requested or used.
#SBATCH --job-name=pick_input_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -eo pipefail
REQUEST=${1:?request required}
WORK=$(dirname "$REQUEST")
test ! -e "$WORK/input_check.exit_code.txt"
trap 'code=$?; printf "%s\n" "$code" > "$WORK/input_check.exit_code.txt"' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
"$HOME/kevin/Isaac-GR00T/.venv/bin/python" -u \
    "$HOME/kevin/DreamVLA/cluster/compare_pick_base_inputs.py" "$REQUEST" \
    > "$WORK/input_check.log" 2>&1
