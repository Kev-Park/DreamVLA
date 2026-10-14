#!/bin/bash -l
# Submit only after the final checkpoint has been transferred and verified.
# Two concurrent workers share one full A100; each executes two of the unchanged
# four 25-reference shards (seeds 100..103). Host tmux owns the Isaac container.
# Usage: sbatch pick_budget_eval.sbatch ~/kevin/dline/<run>_adroit_eval/request.json
#SBATCH --job-name=pick50_r3_eval
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
    echo "$$" > "$PICK_EVAL_WORK/worker.pid"
    set +e
    "$HOME/kevin/bin/isaac" bash -c \
        'conda activate dreamcontrol_51 && exec python "$PICK_EVAL_RUNNER" run "$PICK_EVAL_REQUEST"' \
        > "$PICK_EVAL_WORK/supervisor.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "$PICK_EVAL_WORK/exit_code.txt"
    tmux -L "$PICK_EVAL_SOCKET" wait-for -S completed
    exit "$code"
fi

export PICK_EVAL_REQUEST=${1:?usage: sbatch pick_budget_eval.sbatch request.json}
export PICK_EVAL_WORK
PICK_EVAL_WORK=$(dirname "$PICK_EVAL_REQUEST")
export PICK_EVAL_RUNNER=$HOME/kevin/wt/pick-budget-eval/cluster/pick_budget_biped_eval.py
export PICK_EVAL_SOCKET=pick_budget_${SLURM_JOB_ID:?Slurm allocation required}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
test -f "$PICK_EVAL_REQUEST"
test -f "$PICK_EVAL_RUNNER"
test ! -e "$PICK_EVAL_WORK/worker.pid"
test ! -e "$PICK_EVAL_WORK/exit_code.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$PICK_EVAL_WORK/slurm_job_id.txt"
tmux -L "$PICK_EVAL_SOCKET" new-session -d -s eval bash "${BASH_SOURCE[0]}" --worker
tmux -L "$PICK_EVAL_SOCKET" wait-for completed
read -r code < "$PICK_EVAL_WORK/exit_code.txt"
tmux -L "$PICK_EVAL_SOCKET" kill-session -t eval 2>/dev/null || true
exit "$code"
