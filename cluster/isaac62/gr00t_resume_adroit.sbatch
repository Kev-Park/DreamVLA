#!/bin/bash -l
# Two real GR00T updates and a fresh-process resume; not a research fine-tune.
#SBATCH --job-name=isaac62_gr00t_resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:25:00
#SBATCH --output=isaac62_gr00t_resume-%j.log
set -eo pipefail
CHECKPOINT=${1:?checkpoint required}
DATASET=${2:?dataset required}
OUTPUT=${3:?new output required}
PROBE=${4:?absolute path to the exported validation probe required}
test -f "$PROBE"
PYTHON=$HOME/kevin/Isaac-GR00T/.venv/bin/python
module load cudatoolkit/12.8
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# Use a job-specific tunnel; never borrow or terminate another job's SSH client.
proxy_port=$((20000 + SLURM_JOB_ID % 30000))
ssh -o BatchMode=yes -o ConnectTimeout=10 -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -N \
  -L "127.0.0.1:$proxy_port:127.0.0.1:8899" adroit5 &
tunnel_pid=$!
trap 'kill "$tunnel_pid" 2>/dev/null || true; wait "$tunnel_pid" 2>/dev/null || true' EXIT
export http_proxy="http://127.0.0.1:$proxy_port"
export https_proxy=$http_proxy HTTP_PROXY=$http_proxy HTTPS_PROXY=$http_proxy
# Wait for this tunnel, not a fixed multi-second startup sleep.
"$PYTHON" -c 'import socket,sys,time
port=int(sys.argv[1]);end=time.monotonic()+15
while True:
 try:
  with socket.create_connection(("127.0.0.1",port),timeout=1):break
 except OSError:
  if time.monotonic()>end:raise
  time.sleep(.2)' "$proxy_port"
test ! -e "$OUTPUT"
cd "$HOME/kevin/Isaac-GR00T"
for phase in initial resume; do
  "$PYTHON" "$PROBE" \
    --checkpoint "$CHECKPOINT" --dataset "$DATASET" --output "$OUTPUT" --phase "$phase"
done
