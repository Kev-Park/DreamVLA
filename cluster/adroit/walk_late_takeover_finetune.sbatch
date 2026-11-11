#!/bin/bash -l
# Camera evaluation runs on RTX hardware after this compute-only job exits.
#SBATCH --job-name=walk_late1s_ft
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
python3 "$REPO/cluster/walk_adroit_quota.py"
bash "$REPO/cluster/adroit/walk_gr00t_finetune.sbatch" "$DS" "$RUN" "$STEPS"
python3 - "$TAG" "$STEPS" <<'PY'
import json
from pathlib import Path
import sys
import time

tag, steps = sys.argv[1], int(sys.argv[2])
base = Path.home() / 'kevin'
relative = f'checkpoints/{tag}_r3/checkpoint-{steps}'
checkpoint = base / relative
sys.path.insert(0, str(base / 'wt/locomotion/DreamVLA/cluster'))
from walk_checkpoint_integrity import validate_checkpoint
integrity = validate_checkpoint(checkpoint, steps)
value = {'completed': time.time(), 'steps': steps, 'checkpoint_relative': relative,
         'training_host': 'adroit', 'evaluation_host': 'biped',
         'evaluation_requires_rtx': True, 'checkpoint_integrity': integrity}
target = base / 'dline' / tag / 'finetune.ready.json'
temporary = target.with_suffix('.json.tmp')
temporary.write_text(json.dumps(value, indent=2) + '\n')
temporary.replace(target)
print(json.dumps(value))
PY
