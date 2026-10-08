#!/bin/bash
# Clone verified inputs from an existing Pod; no cluster CPU or local relay.
set -euo pipefail
SOURCE=${1:?source host}; PORT=${2:?source port}
SSH=(ssh -i /root/.runpod_transport_key -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p "$PORT" "root@$SOURCE")
mkdir -p /workspace/logs /workspace/kevin/datasets /workspace/hf/hub /opt/kevin
"${SSH[@]}" 'cat /workspace/logs/models.sha256' > /workspace/logs/models.sha256
"${SSH[@]}" 'grep "af60v8f" /workspace/logs/datasets.sha256' > /workspace/logs/datasets.sha256
(
  "${SSH[@]}" 'tar -C /opt/kevin -cf - Isaac-GR00T/.venv' |
    tar --no-same-owner -C /opt/kevin -xf -
  touch /workspace/logs/environment.ready
) & envpid=$!
(
  "${SSH[@]}" 'tar -C /workspace/hf/hub -cf - models--nvidia--GR00T-N1.7-3B models--nvidia--Cosmos-Reason2-2B' |
    tar --no-same-owner -C /workspace/hf/hub -xf -
  (cd /workspace/hf/hub; sha256sum --quiet -c /workspace/logs/models.sha256)
  touch /workspace/logs/models.verified
) & modelpid=$!
(
  "${SSH[@]}" 'tar -C /workspace/kevin/datasets -cf - af60v8f_base af60v8f_ds_dagger1 af60v8f_ds_dagger2' |
    tar --no-same-owner -C /workspace/kevin/datasets -xf -
  (cd /workspace/kevin/datasets; sha256sum --quiet -c /workspace/logs/datasets.sha256)
  touch /workspace/logs/datasets.verified
) & datapid=$!
umask 077
"${SSH[@]}" 'cat /workspace/hf/token' > /workspace/hf/token
rc=0
for pid in "$envpid" "$modelpid" "$datapid"; do wait "$pid" || rc=1; done
exit "$rc"
