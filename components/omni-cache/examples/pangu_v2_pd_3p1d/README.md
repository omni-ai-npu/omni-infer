# Pangu V2 Hybrid — 3P-1D Disaggregated Serving

3 Prefill + 1 Decode deployment across 2 Huawei NPU A3 nodes (16 cards each),
with an omni-proxy for APC-aware request scheduling.

## Topology

```
Node 1 (prefill-only, e.g. 10.0.0.1):
  P0  container omnicache_pangu_p0  cards 0-7   TP=8  port=8000  KV_PORT=14579
  P1  container omnicache_pangu_p1  cards 8-15  TP=8  port=8001  KV_PORT=14580

Node 2 (PD + proxy, e.g. 10.0.0.2):
  P2  container omnicache_pangu_p2  cards 0-7   TP=8  port=8000  KV_PORT=14579
  D0  container omnicache_pangu_d0  cards 8-15  DP=8  TP=1  ports=8082-8089
  Proxy  runs in omnicache_pangu_p2           port=7150
```

## Quick Start

Two steps: create containers first, then launch services.

### Step 1: Create containers

```bash
# On Node 1 — create P0 and P1 containers
DOCKER_IMAGE_ID=<image> \
  LAUNCH_MODE=prefill_2x \
  bash examples/pangu_v2_pd_3p1d/launch_containers.sh

# On Node 2 — create P2 and D0 containers (proxy reuses P2)
DOCKER_IMAGE_ID=<image> \
  LAUNCH_MODE=all \
  bash examples/pangu_v2_pd_3p1d/launch_containers.sh
```

### Step 2: Launch services

```bash
# On Node 1 — start P0 and P1
CONFIG_PROFILE=high-throughput \
  LOCAL_NODE_IP=10.0.0.1 \
  P_NODE_LIST="10.0.0.1;10.0.0.1;10.0.0.2" \
  P_NODE_PORT_LIST="10.0.0.1:16077;10.0.0.1:16078;10.0.0.2:16077" \
  LAUNCH_MODE=prefill_2x \
  bash examples/pangu_v2_pd_3p1d/launch_pd.sh

# On Node 2 — start P2, D0, and proxy
CONFIG_PROFILE=high-throughput \
  LOCAL_NODE_IP=10.0.0.2 \
  P_NODE_LIST="10.0.0.1;10.0.0.1;10.0.0.2" \
  P_NODE_PORT_LIST="10.0.0.1:16077;10.0.0.1:16078;10.0.0.2:16077" \
  LAUNCH_MODE=all \
  bash examples/pangu_v2_pd_3p1d/launch_pd.sh
```

## Logs

All logs are under `examples/pangu_v2_pd_3p1d/logs/`:

```
logs/
  prefill_0/   # P0 (Node 1)
  prefill_1/   # P1 (Node 1)
  prefill_2/   # P2 (Node 2)
  decode/      # D0 (Node 2)
  proxy/       # omni-proxy (Node 2)
    nginx_error.log
    nginx_access.log
    launch_config.log
    proxy_cmd.log
```
