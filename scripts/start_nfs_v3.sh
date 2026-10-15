#!/bin/bash
set -uo pipefail
MODEL_DIR=$HOME/models/DeepSeek-V4.1-Flash
# 默认值（逗号分隔）+ 我们集群的网段
CLIENTS="WORKER_NETS"

echo "=== 启动 NFS（逗号分隔 clients）==="
docker rm -f dsv41-nfs >/dev/null 2>&1
docker run -d --name dsv41-nfs --restart unless-stopped \
  --privileged --network host \
  -v "$MODEL_DIR:/export:ro" \
  -e "NFS_CLIENTS=$CLIENTS" \
  dsv41-nfs:local
sleep 6

docker ps --format '{{.Names}} {{.Status}}' | grep dsv41-nfs | sed 's/^/  /'
for i in 1 2 3 4 5 6; do
  if ss -tl 2>/dev/null | grep -q ':2049'; then
    echo "  port 2049: LISTENING"
    break
  fi
  sleep 2
done
ss -tl 2>/dev/null | grep ':2049' >/dev/null || docker logs dsv41-nfs 2>&1 | tail -4
