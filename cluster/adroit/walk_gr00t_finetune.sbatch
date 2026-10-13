#!/bin/bash -l
# The walking study keeps batch 16 and independently initializes from GR00T-N1.7.
#SBATCH --job-name=walk_gr00t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=23:59:00
#SBATCH --output=%x-%j.out
set -eo pipefail
DS=${1:?dataset directory required}
RUN=${2:?run name required}
STEPS=${3:?exact update count required}
GR=$HOME/kevin/Isaac-GR00T
module load cudatoolkit/12.8
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
if ! pgrep -f 'ssh.*-L 8899:127.0.0.1:8899' >/dev/null; then
  ssh -o BatchMode=yes -o ConnectTimeout=5 -N -f -L 8899:127.0.0.1:8899 adroit5
fi
export http_proxy=http://127.0.0.1:8899
export https_proxy=$http_proxy HTTP_PROXY=$http_proxy HTTPS_PROXY=$http_proxy
cd "$GR"
exec .venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B --dataset-path "$DS" \
  --embodiment-tag unitree_g1_sonic --num-gpus 1 --global-batch-size 16 \
  --gradient-accumulation-steps 1 --learning-rate 0.0001 --dataloader-num-workers 4 \
  --output-dir "$HOME/kevin/checkpoints" --experiment-name "$RUN" \
  --max-steps "$STEPS" --save-steps "$STEPS" --save-total-limit 1
