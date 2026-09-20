#!/bin/bash -l
# VLA (unitree_g1_sonic) eval + render on Adroit -- ONE A100 80GB inside the Isaac container.
#
#   sbatch cluster/adroit/eval_vla_sonic.sbatch <vla-checkpoint> [tag]
#   sbatch cluster/adroit/eval_vla_sonic.sbatch \
#     ~/kevin/checkpoints/latband60_dagger2_adroit/checkpoint-10000 dagger2_adroit
#
# WHY THIS RUNS HERE rather than on bluesclues: the 12 GB checkpoint is already on Adroit scratch,
# and the home link between them measures 1.7-3.4 MB/s, so shipping it to bluesclues and back would
# cost hours. Everything else the eval needs was already staged (sonic .pt, dreamcontrol_51, the
# container); only ref_motions had to be copied over, and that is 2.9 MB.
#
# Isaac Sim needs the Apptainer container (RHEL 9 glibc is too old), so both scripts run through the
# `isaac` shim, which also opens the login-node proxy tunnel that streams the kitchen USD assets.
# --constraint=gpu80 is REQUIRED: a plain gpu:1 hands out a MIG slice with no Vulkan/PhysX-GPU, and
# the Cam-HOI env renders an ego camera.
#
# Settings are pinned to match the dagger1 eval so the numbers are comparable:
#   HS_EVAL_NO_EE_TERM=1     drop the ee_body_pos termination (dagger1's 45% was measured this way)
#   --motions-from <ids>     sweep ONLY the 54 filtered demo motions, never the rejected clips
#   HS_REWORK* as in collection.
#
#SBATCH --job-name=vla_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=07:59:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=kp0374@princeton.edu

set -eo pipefail
VLA=${1:?usage: sbatch eval_vla_sonic.sbatch <vla-checkpoint> [tag]}
TAG=${2:-$(basename "$(dirname "$VLA")")}
PT=$HOME/kevin/sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000
REFS=Holosoma_Pick_29_latband60
IDS=$HOME/kevin/ref_motions/$REFS/_motion_ids.txt
OUT=$HOME/kevin/eval_videos
mkdir -p "$OUT"

echo "node=$(hostname)  gpu=$(nvidia-smi -L | head -1)"
echo "vla=$VLA  tag=$TAG  refs=$REFS  ids=$(wc -l < "$IDS")"

ENVV="HS_REWORK=1 HS_REWORK_OBJ_PC=1 HS_REWORK_OBJ_GATE=1 HS_REWORK_CONTACT_LEAD=2 HS_EVAL_NO_EE_TERM=1"
COMMON="--headless --sonic-pt $PT --ref-motions-path $REFS --vla-checkpoint $VLA \
  --motions-from $IDS --waist-dof 29"

cd "$HOME/kevin/DreamVLA/Training"

# ---- 1. metrics sweep (no video; prints the per-episode [PHYS] held lines) ----
isaac bash -c "conda activate dreamcontrol_51 && export $ENVV && \
  python scripts/reinforcement_learning/rsl_rl/eval_vla_sonic.py $COMMON \
    --num-episodes 20" 2>&1 | tee "$OUT/${TAG}_eval.log"

# ---- 2. render a handful of rollouts (third-person + ego mp4 per episode) ----
isaac bash -c "conda activate dreamcontrol_51 && export $ENVV && \
  python scripts/reinforcement_learning/rsl_rl/play_vla_sonic.py $COMMON \
    --num-episodes 4 --record-video $OUT/${TAG}" 2>&1 | tee "$OUT/${TAG}_render.log"

echo "done: $OUT/${TAG}_eval.log ; videos $OUT/${TAG}_ep*.mp4"
ls -1 "$OUT/${TAG}"_ep*.mp4 2>/dev/null | head
