#!/bin/bash
set -euo pipefail
cd /workspace/kevin/DreamVLA_runtime
until test -f /workspace/logs/environment.ready; do sleep 5; done
bash cluster/runpod_prepare.sh
until test -f /workspace/logs/models.verified && test -f /workspace/logs/datasets.verified; do sleep 5; done
bash cluster/runpod_prepare_baseline.sh
touch /workspace/logs/prepare.ready
exec python3 cluster/runpod_queue.py
