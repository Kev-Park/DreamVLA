#!/bin/bash -l
# One checkpointed slice; max_steps remains 158000 in every slice.
#SBATCH --job-name=pick200_r3_b16_slice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -eo pipefail
DS=${1:?dataset required}
RUN=${2:?run required}
SEGMENT=${3:?segment required}
WORK=$HOME/kevin/dline/$RUN
trap 'code=$?; printf "%s\n" "$code" > "$WORK/segment-${SEGMENT}.exit_code.txt"' EXIT
module load cudatoolkit/12.8
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset CUDA_LAUNCH_BLOCKING
if ! pgrep -f 'ssh.*-L 8899:127.0.0.1:8899' >/dev/null; then
    ssh -o BatchMode=yes -o ConnectTimeout=5 -N -f -L 8899:127.0.0.1:8899 adroit5
fi
export http_proxy=http://127.0.0.1:8899
export https_proxy=$http_proxy HTTP_PROXY=$http_proxy HTTPS_PROXY=$http_proxy
export SLURM_JOB_END_TIME
SLURM_JOB_END_TIME=$(squeue -h -j "$SLURM_JOB_ID" -o %e | xargs -I{} date -d '{}' +%s)
echo "segment=$SEGMENT node=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES"
cd "$HOME/kevin/Isaac-GR00T"
.venv/bin/python -u "$HOME/kevin/DreamVLA/cluster/pick_fullbase_segment.py" "$DS" "$RUN" "$SEGMENT" \
    >> "$WORK/train.log" 2>&1
