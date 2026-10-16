#!/bin/bash -l
# Historical 892-demo Pick DAgger-3, batch 16 and matched data exposure.
#SBATCH --job-name=pick200_r3_b16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=23:59:00
#SBATCH --output=%x-%j.out
set -eo pipefail
DS=${1:?dataset required}
RUN=${2:?run required}
WORK=$HOME/kevin/dline/$RUN
GR=$HOME/kevin/Isaac-GR00T
mkdir -p "$WORK"
test ! -e "$WORK/train.exit_code.txt"
trap 'code=$?; printf "%s\n" "$code" > "$WORK/train.exit_code.txt"' EXIT
printf '%s\n' "$SLURM_JOB_ID" > "$WORK/train.slurm_job_id.txt"
python3 - "$DS" "$WORK/request.json" "$GR" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
ds,request,repo=map(Path,sys.argv[1:])
spec=json.loads(request.read_text());info=json.loads((ds/'meta/info.json').read_text())
assert (info['total_episodes'],info['total_frames'])==(892,443502)
for relative,digest in spec['dataset_meta_hashes'].items():
    assert hashlib.sha256((ds/relative).read_bytes()).hexdigest()==digest,relative
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==spec['gr00t_commit']
PY
module load cudatoolkit/12.8
export XLA_PYTHON_CLIENT_PREALLOCATE=false TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
unset CUDA_LAUNCH_BLOCKING
if ! pgrep -f 'ssh.*-L 8899:127.0.0.1:8899' >/dev/null; then
    ssh -o BatchMode=yes -o ConnectTimeout=5 -N -f -L 8899:127.0.0.1:8899 adroit5
fi
export http_proxy=http://127.0.0.1:8899
export https_proxy=$http_proxy HTTP_PROXY=$http_proxy HTTPS_PROXY=$http_proxy
echo "node=$(hostname) job=$SLURM_JOB_ID gpu=$CUDA_VISIBLE_DEVICES"
cd "$GR"
.venv/bin/python -u gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B --dataset-path "$DS" \
    --embodiment-tag unitree_g1_sonic --num-gpus 1 --global-batch-size 16 \
    --gradient-accumulation-steps 1 --learning-rate 0.0001 \
    --weight-decay 0.00001 --warmup-ratio 0.05 --dataloader-num-workers 8 \
    --output-dir "$HOME/kevin/checkpoints" --experiment-name "$RUN" \
    --max-steps 158000 --save-steps 20000 --save-total-limit 1 \
    > "$WORK/train.log" 2>&1
python3 - "$HOME/kevin/checkpoints/$RUN/checkpoint-158000" "$WORK" <<'PY'
import json,sys,time
from pathlib import Path
ck,work=map(Path,sys.argv[1:])
assert json.loads((ck/'trainer_state.json').read_text())['global_step']==158000
index=json.loads((ck/'model.safetensors.index.json').read_text())['weight_map']
assert all((ck/name).stat().st_size>8 for name in set(index.values()))
assert (ck/'optimizer.pt').stat().st_size>0
(work/'trained.json').write_text(json.dumps(dict(time=time.time(),checkpoint=str(ck),steps=158000,batch_size=16),indent=2)+'\n')
PY
