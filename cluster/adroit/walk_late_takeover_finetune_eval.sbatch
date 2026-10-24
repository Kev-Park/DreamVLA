#!/bin/bash -l
#SBATCH --job-name=walk_late1s_ft_eval
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
RUN=${TAG}_r3
bash "$REPO/cluster/adroit/walk_gr00t_finetune.sbatch" "$DS" "$RUN" "$STEPS"
CK=$HOME/kevin/checkpoints/$RUN/checkpoint-$STEPS
exec "$HOME/kevin/bin/isaac" bash -c '
  source "$HOME/kevin/bin/isaac.rc"
  conda activate dreamcontrol_51
  cd "$HOME/kevin/wt/locomotion/DreamVLA"
  exec python cluster/walk_late_takeover_process.py evaluate --tag "$1" --checkpoint "$2"
' bash "$TAG" "$CK"
