#!/bin/bash -l
# One bounded expert-only sweep over missing unaugmented references.
#SBATCH --job-name=pick200_original_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -eo pipefail
if [ "${1:-}" = "--worker" ]; then
    set +e
    "$HOME/kevin/bin/isaac" bash -c \
      'source "$HOME/kevin/bin/isaac.rc" && conda activate dreamcontrol_51 && exec python "$HOME/kevin/DreamVLA/cluster/pick_original_base_collect.py" "$PICK_ORIGINAL_REQUEST"' \
      > "$PICK_ORIGINAL_WORK/supervisor.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "$PICK_ORIGINAL_WORK/job.exit_code.txt"
    tmux -L "$PICK_ORIGINAL_SOCKET" wait-for -S completed
    exit "$code"
fi
export PICK_ORIGINAL_REQUEST=${1:?request JSON required}
export PICK_ORIGINAL_WORK
PICK_ORIGINAL_WORK=$(dirname "$PICK_ORIGINAL_REQUEST")
export PICK_ORIGINAL_SOCKET=pick_original_${SLURM_JOB_ID:?Slurm allocation required}
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
test -f "$PICK_ORIGINAL_REQUEST"
test ! -e "$PICK_ORIGINAL_WORK/job.exit_code.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$PICK_ORIGINAL_WORK/slurm_job_id.txt"
tmux -L "$PICK_ORIGINAL_SOCKET" new-session -d -s collect bash "${BASH_SOURCE[0]}" --worker
tmux -L "$PICK_ORIGINAL_SOCKET" wait-for completed
read -r code < "$PICK_ORIGINAL_WORK/job.exit_code.txt"
tmux -L "$PICK_ORIGINAL_SOCKET" kill-session -t collect 2>/dev/null || true
exit "$code"
