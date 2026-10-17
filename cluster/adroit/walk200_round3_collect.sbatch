#!/bin/bash -l
# Full-A40 camera allocation: requires the user's explicit gpu80-rule exception.
# One group = 50 attempts, two native collectors, no changes to student/teacher.
# Submit four distinct groups only after the verified round-2 input handoff.
#SBATCH --job-name=walk200_r3_collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --constraint=a40
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
set -eo pipefail
walk_collection_group=${1:?group 0..3 required}
case "$walk_collection_group" in
  0|1|2|3) ;;
  *) echo 'group must be 0, 1, 2, or 3' >&2; exit 2 ;;
esac
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$HOME/kevin/bin/isaac" bash -c '
  source "$HOME/kevin/bin/isaac.rc"
  conda activate dreamcontrol_51
  exec python -u "$HOME/kevin/wt/locomotion/DreamVLA/cluster/adroit/collect_walk200_round3_group.py" --group "$1"
' walk200-round3 "$walk_collection_group"
