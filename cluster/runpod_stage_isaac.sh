#!/bin/bash
# Source-side staging: data and a frozen environment, with code obtained by git.
set -euo pipefail
HOST=${1:?}; PORT=${2:?}
SSH=(ssh -i "$HOME/kevin/runpod_transport/key" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p "$PORT" "root@$HOST")
"${SSH[@]}" 'mkdir -p /opt/dreamcontrol_51 /opt/kevin /workspace/logs; ln -s /opt/kevin /root/kevin'
(
  tar -C "$HOME/miniconda3/envs/dreamcontrol_51" -cf - . |
    "${SSH[@]}" 'tar --no-same-owner -C /opt/dreamcontrol_51 -xf - && touch /workspace/logs/isaac_env.ready'
) & ep=$!
(
  tar -C ~/kevin -cf - ref_motions/Holosoma_Pick_29_latband60max ref_motions/Holosoma_Pick_29_latband60max_eval sonic/sonic_release_3pt_heading_wrist_81-20260415_051436_model_step_100000 residuals/af60v8c_model_8998.pt |
    "${SSH[@]}" 'tar --no-same-owner -C /opt/kevin -xf - && touch /workspace/logs/isaac_data.ready'
) & dp=$!
(
  tar -C ~/kevin/DreamVLA/Training -cf - assets |
    "${SSH[@]}" 'mkdir -p /opt/isaac_assets; tar --no-same-owner -C /opt/isaac_assets -xf - && touch /workspace/logs/isaac_assets.ready'
) & ap=$!
rc=0
for pid in "$ep" "$dp" "$ap"; do wait "$pid" || rc=1; done
exit "$rc"
