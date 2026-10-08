#!/bin/bash
set -euo pipefail
cd /workspace/kevin/DreamVLA_runtime
# Staging readiness is a verified file, not GPU process startup.
until test -f /workspace/logs/environment.ready; do sleep 10; done
bash cluster/runpod_prepare.sh
touch /workspace/logs/prepare.ready
until test -f /workspace/logs/models.verified && test -f /workspace/logs/datasets.verified; do sleep 10; done
exec python3 cluster/runpod_queue.py
