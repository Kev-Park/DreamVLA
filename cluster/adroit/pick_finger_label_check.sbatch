#!/bin/bash -l
#SBATCH --job-name=pick_finger_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -eo pipefail
REQUEST=${1:?request required}
WORK=$(dirname "$REQUEST")
test ! -e "$WORK/finger_check.exit_code.txt"
trap 'code=$?; printf "%s\n" "$code" > "$WORK/finger_check.exit_code.txt"' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''
"$HOME/kevin/Isaac-GR00T/.venv/bin/python" -u \
    "$HOME/kevin/DreamVLA/cluster/audit_pick_finger_labels.py" "$REQUEST" \
    > "$WORK/finger_check.log" 2>&1
