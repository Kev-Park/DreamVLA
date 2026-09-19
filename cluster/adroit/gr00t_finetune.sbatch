#!/bin/bash
# GR00T (unitree_g1_sonic) fine-tune on Adroit -- ONE A100 80GB, no DeepSpeed.
#
#   sbatch cluster/adroit/gr00t_finetune.sbatch <lerobot-dataset-dir> [run-name]
#   sbatch cluster/adroit/gr00t_finetune.sbatch ~/kevin/datasets/latband60_dagger2_agg latband60_dagger2_adroit
#
# WHY ONE GPU IS ENOUGH (and simpler than biped's 2x5090): 3.15B params bf16 ~6.3 GB resident;
# 1.62B trainable -> grads 3.2 GB + AdamW fp32 states 13.0 GB + fp32 master weights 6.5 GB
# ~= 29 GB fixed, + ~10 GB activations at per-device batch 8 (measured on biped) => ~39 GB at
# batch 8 / ~49 GB at batch 16. Fits 80 GB unsharded, so NO ZeRO and therefore none of biped's
# NCCL-P2P / CUDA_HOME grief. Do NOT install deepspeed in this env: accelerate imports it
# whenever present and its op builder then demands CUDA_HOME (that is what broke the
# single-GPU path on bluesclues). flash-attn is optional -- GR00T falls back to sdpa.
#
# ONE-TIME SETUP (login node, which is the only place with internet -- see cluster/adroit/README.md):
#   isaac bash -c "conda activate dreamcontrol_51 && \
#     git clone https://github.com/Kev-Park/Isaac-GR00T.git ~/kevin/Isaac-GR00T && \
#     pip install -e ~/kevin/Isaac-GR00T --no-deps && python -c 'import gr00t; print(gr00t.__file__)'"
#   isaac bash -c "conda activate dreamcontrol_51 && huggingface-cli login"   # token needs the
#     # GATED nvidia/Cosmos-Reason2-2B (the N1.7 backbone) or the run dies at model load
#   isaac bash -c "conda activate dreamcontrol_51 && python -c \"from huggingface_hub import \
#     snapshot_download as d; d('nvidia/GR00T-N1.7-3B'); d('nvidia/Cosmos-Reason2-2B')\""   # ~11 GB
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

set -euo pipefail
DS=${1:?usage: sbatch gr00t_finetune.sbatch <lerobot-dataset-dir> [run-name]}
RUN=${2:-$(basename "$DS")_adroit}
OUT=$HOME/kevin/checkpoints

echo "node=$(hostname) gpu=$(nvidia-smi -L | head -1)"
echo "dataset=$DS  run=$RUN  out=$OUT"

# The `isaac` shim opens the login-node proxy tunnel (compute nodes have no direct internet) and
# isaac.rc exports http(s)_proxy + activates conda inside the container. The models are pre-staged
# in ~/.cache/huggingface (-> scratch), so the run works even if the tunnel is down.
isaac bash -c "conda activate dreamcontrol_51 && \
  python -c 'import importlib.util as u; print(\"deepspeed present:\", u.find_spec(\"deepspeed\") is not None, \"(want False on 1 GPU)\")' && \
  cd ~/kevin/Isaac-GR00T && \
  python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path '$DS' \
    --embodiment-tag unitree_g1_sonic \
    --num-gpus 1 \
    --global-batch-size 16 \
    --dataloader-num-workers 8 \
    --output-dir '$OUT' \
    --experiment-name '$RUN' \
    --max-steps 10000 \
    --save-steps 1000 \
    --save-total-limit 2 \
    --save-only-model"

echo "done: $OUT/$RUN"
