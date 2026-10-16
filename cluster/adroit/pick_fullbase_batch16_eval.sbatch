#!/bin/bash -l
# Submit with --dependency=afterok:<training-job>; host tmux owns Isaac.
#SBATCH --job-name=pick200_r3_b16_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
set -eo pipefail
if [ "${1:-}" = "--worker" ]; then
    set +e
    "$HOME/kevin/bin/isaac" bash -c \
        'source "$HOME/kevin/bin/isaac.rc"; conda activate dreamcontrol_51; exec python "$HOME/kevin/DreamVLA/cluster/pick_fullbase_adroit_eval.py" "$PICK_FULLBASE_REQUEST"' \
        > "$PICK_FULLBASE_WORK/eval.supervisor.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "$PICK_FULLBASE_WORK/eval.exit_code.txt"
    tmux -L "$PICK_FULLBASE_SOCKET" wait-for -S completed
    exit "$code"
fi
export PICK_FULLBASE_REQUEST=${1:?request required}
export PICK_FULLBASE_WORK
PICK_FULLBASE_WORK=$(dirname "$PICK_FULLBASE_REQUEST")
export PICK_FULLBASE_SOCKET=pick_fullbase_${SLURM_JOB_ID:?}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
test ! -e "$PICK_FULLBASE_WORK/eval.exit_code.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$PICK_FULLBASE_WORK/eval.slurm_job_id.txt"
tmux -L "$PICK_FULLBASE_SOCKET" new-session -d -s eval bash "${BASH_SOURCE[0]}" --worker
tmux -L "$PICK_FULLBASE_SOCKET" wait-for completed
read -r code < "$PICK_FULLBASE_WORK/eval.exit_code.txt"
tmux -L "$PICK_FULLBASE_SOCKET" kill-session -t eval
exit "$code"
