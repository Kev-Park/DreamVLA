#!/bin/bash
# One unchanged expert rollout to measure real simulator/camera/HDF5 throughput.
set -euo pipefail
export OMNI_KIT_ALLOW_ROOT=1 OMNI_KIT_ACCEPT_EULA=YES
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HS_REWORK=1 HS_REWORK_OBJ_PC=1 HS_REWORK_OBJ_GATE=1 HS_REWORK_CONTACT_LEAD=2
cd ~/kevin/DreamVLA/Training
date -u +%FT%TZ
/opt/dreamcontrol_51/bin/python scripts/reinforcement_learning/rsl_rl/collect_sonic_adapter.py \
  --headless --task Isaac-Motion-Tracking-Pick-Cam-HOI-v0 --waist-dof 29 \
  --sonic-pt ~/kevin/sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000 \
  --encoder-mode g1 --residual-transform additive_free --residual-scale 0.1 \
  --checkpoint-path ~/kevin/residuals/af60v8c_model_8998.pt \
  --ref-motions-path Holosoma_Pick_29_latband60max --rollout-length 500 \
  --motion-range 0 1 --seed 3 --num-samples 1 --output-directory /workspace/collection_smoke
date -u +%FT%TZ
