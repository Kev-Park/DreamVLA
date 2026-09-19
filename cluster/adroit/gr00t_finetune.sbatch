#!/bin/bash -l
# GR00T (unitree_g1_sonic) fine-tune on Adroit -- ONE A100 80GB, no DeepSpeed sharding.
#
#   sbatch cluster/adroit/gr00t_finetune.sbatch <lerobot-dataset-dir> [run-name]
#   sbatch cluster/adroit/gr00t_finetune.sbatch ~/kevin/datasets/latband60_dagger2_agg latband60_dagger2_adroit
#
# WHY ONE GPU IS ENOUGH (and simpler than biped's 2x5090): 3.15B params bf16 ~6.3 GB resident;
# 1.62B trainable -> grads 3.2 GB + AdamW fp32 states 13.0 GB + fp32 master weights 6.5 GB
# ~= 29 GB fixed, + ~10 GB activations at per-device batch 8 (measured on biped) => ~39 GB at
# batch 8 / ~49 GB at batch 16. Fits 80 GB unsharded, so NO ZeRO and therefore none of biped's
# NCCL-P2P grief. --global-batch-size 16 on 1 GPU is the SAME effective batch as biped's 16
# across 2 GPUs, so this run and biped's are step-for-step comparable (whichever lands first wins).
#
# ENV: ~/kevin/Isaac-GR00T/.venv (uv, py3.10, torch 2.7.1+cu128), NOT the container/dreamcontrol_51
# -- that conda env is the public-release Isaac env and lacks torchcodec, which LeRobot video
# decoding needs. The venv DOES contain deepspeed (a gr00t dependency); accelerate imports it
# whenever present and its op builder then demands CUDA_HOME -- that is what broke the single-GPU
# path on bluesclues, which has no CUDA toolkit. Here `module load cudatoolkit/12.8` supplies
# CUDA_HOME=/usr/local/cuda-12.8 (nvcc 12.8 exactly matches torch's cu128), so the import is safe
# and deepspeed stays installed. Do NOT delete that module load.
#
# OFFLINE: compute nodes reach the internet only through the login-node proxy tunnel that the
# `isaac` shim opens (see cluster/adroit/README.md); this job does not use the container, so it
# runs fully offline instead -- both models are pre-staged in ~/.cache/huggingface (-> scratch)
# and wandb logs locally (sync later from the login node with `wandb sync`).
#
# ONE-TIME SETUP (login node, the only place with internet -- see cluster/adroit/README.md):
#   git clone https://github.com/Kev-Park/Isaac-GR00T.git ~/kevin/Isaac-GR00T
#   cd ~/kevin/Isaac-GR00T && uv venv --python 3.10 && uv pip install -e . && uv pip install torchcodec
#   .venv/bin/huggingface-cli login     # token needs the GATED nvidia/Cosmos-Reason2-2B (the N1.7
#                                       # backbone) or the run dies at model load
#   .venv/bin/python -c "from huggingface_hub import snapshot_download as d; \
#     d('nvidia/GR00T-N1.7-3B'); d('nvidia/Cosmos-Reason2-2B')"          # ~11 GB
#   # dataset: scp the *_agg.tgz from bluesclues (via your laptop) into ~/kevin/datasets/ and untar
#
#SBATCH --job-name=gr00t_ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80          # REQUIRED: plain gpu:1 hands out a MIG slice
#SBATCH --time=23:59:00             # ~7-11 h expected for 10k steps on one A100
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=kp0374@princeton.edu

set -eo pipefail                    # NOT -u: the module function trips on unbound vars
DS=${1:?usage: sbatch gr00t_finetune.sbatch <lerobot-dataset-dir> [run-name]}
RUN=${2:-$(basename "$DS")_adroit}
OUT=$HOME/kevin/checkpoints
GR=$HOME/kevin/Isaac-GR00T

module load cudatoolkit/12.8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

echo "node=$(hostname)  gpu=$(nvidia-smi -L | head -1)"
echo "dataset=$DS  run=$RUN  out=$OUT"
echo "CUDA_HOME=$CUDA_HOME  python=$GR/.venv/bin/python"

cd "$GR"
exec .venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DS" \
  --embodiment-tag unitree_g1_sonic \
  --num-gpus 1 \
  --global-batch-size 16 \
  --dataloader-num-workers 8 \
  --output-dir "$OUT" \
  --experiment-name "$RUN" \
  --max-steps 10000 \
  --save-steps 1000 \
  --save-total-limit 2 \
  --save-only-model \
  --use-wandb \
  --wandb-project dreamvla
