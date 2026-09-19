#!/bin/bash
# GR00T (unitree_g1_sonic) fine-tune on ONE A100 80GB -- Princeton Adroit.
#
# WHY ONE GPU IS ENOUGH: 3.15B params bf16 ~6.3 GB resident; 1.62B trainable ->
# grads (bf16) 3.2 GB + AdamW states (fp32 m,v) 13.0 GB + fp32 master weights 6.5 GB
# ~= 29 GB of fixed state, plus ~10 GB activations at per-device batch 8 (measured on
# biped). So ~39 GB at batch 8 / ~49 GB at batch 16 -- comfortable in 80 GB with NO
# DeepSpeed and no ZeRO sharding (which is what forces the 2-GPU launch on biped's 32 GB
# cards, and what drags in the NCCL / CUDA_HOME problems).
#
# ONE-TIME SETUP ON ADROIT (login node):
#   S=/scratch/network/$USER && mkdir -p $S && cd $S
#   git clone https://github.com/Kev-Park/Isaac-GR00T.git && cd Isaac-GR00T
#   curl -LsSf https://astral.sh/uv/install.sh | sh          # if uv is absent
#   uv venv --python 3.10 && uv pip install -e .             # do NOT install deepspeed:
#        # accelerate imports it whenever it is present, and deepspeed's op builder then
#        # demands CUDA_HOME. flash-attn is optional too -- GR00T falls back to sdpa.
#   huggingface-cli login                                    # token needs access to the
#        # GATED nvidia/Cosmos-Reason2-2B (the N1.7 backbone), else the run dies at load.
#   HF_HOME=$S/hf python -c "from huggingface_hub import snapshot_download as d; \
#        d('nvidia/GR00T-N1.7-3B'); d('nvidia/Cosmos-Reason2-2B')"     # ~11 GB, login node
#   # dataset (327 MB tgz) from the Berkeley boxes, via your laptop:
#   #   scp sastrygrp-dvij@bluesclues.ist.berkeley.edu:~/kevin/datasets/latband60_dagger2/latband60_dagger2_agg.tgz .
#   #   scp latband60_dagger2_agg.tgz adroit:/scratch/network/$USER/datasets/
#   mkdir -p $S/datasets && cd $S/datasets && tar xzf latband60_dagger2_agg.tgz
#
# SUBMIT:  sbatch job_gr00t_finetune.slurm
# WATCH :  squeue -u $USER ; tail -f g1_gr00t_ft-<jobid>.out
#
#SBATCH --job-name=g1_gr00t_ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80          # A100 80GB
#SBATCH --time=23:59:00             # ~7-11 h expected for 10k steps on one A100
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=kp0374@princeton.edu

set -euo pipefail
S=/scratch/network/$USER
module purge

export HF_HOME=$S/hf
export HF_HUB_OFFLINE=1             # models pre-staged above; drop if compute nodes have internet
export TOKENIZERS_PARALLELISM=false

DS=$S/datasets/latband60_dagger2_agg
RUN=latband60_dagger2_adroit
cd $S/Isaac-GR00T

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python -c "import torch, gr00t; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
.venv/bin/python -c "import importlib.util as u; print('deepspeed present:', u.find_spec('deepspeed') is not None, '(False is what we want on 1 GPU)')"

# --num-gpus 1 => per-device batch == --global-batch-size (no accumulation, no sharding).
# --save-only-model keeps each checkpoint at ~6.5 GB; DROP it if you need to resume a run
# that outgrows the wall clock (resume needs the optimizer state).
.venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path $DS \
  --embodiment-tag unitree_g1_sonic \
  --num-gpus 1 \
  --global-batch-size 16 \
  --dataloader-num-workers 8 \
  --output-dir $S/checkpoints \
  --experiment-name $RUN \
  --max-steps 10000 \
  --save-steps 1000 \
  --save-total-limit 2 \
  --save-only-model

echo "done: $S/checkpoints/$RUN"
