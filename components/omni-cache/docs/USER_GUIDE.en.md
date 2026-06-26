# OmniCache User Guide

## 1. Overview

OmniCache is a PD-disaggregated KV Cache management plugin for vLLM. It establishes an efficient KV Cache transfer channel between Prefill and Decode nodes: after prefill completes, KV Cache is offloaded from HBM to host memory and sent via OX; Decode receives it and loads it from host memory into HBM for inference.

Key advantages: by using a host memory pool (hugetlbfs) as an intermediate cache layer, it significantly reduces KV Cache pressure on HBM on both the Prefill and Decode sides, enabling much longer sequence lengths and higher concurrency. KV Cache persistence also dramatically improves APC hit rates in multi-turn conversation scenarios.

---

## 2. Environment Setup

### 2.1 Clone the Repository

```bash
git clone https://gitee.com/omniai/omni-cache.git
cd omni-cache
```

### 2.2 Install

```bash
pip install -e . --no-build-isolation
```

### 2.3 System Requirements

- Linux (openEuler / Ubuntu)
- Python 3.11+ / PyTorch 2.5.1+
- Huawei Ascend NPU + CANN Toolkit
- Docker (for multi-node deployment)

### 2.4 HugePage Configuration

OmniCache uses 2MB HugePages to manage the host-side KV Cache memory pool (default 500 GiB).

```bash
# Auto-calculate (recommended)
sudo bash tools/setup/set_hugepage_limit.sh

# Manual page count
sudo bash tools/setup/set_hugepage_limit.sh --target-pages 1048576
```

Verify:

```bash
grep HugePages_ /proc/meminfo
```

---

## 3. Service Launch Flow

### 3.1 Set Environment Variables

**Prefill Node:**

```bash
export ENABLE_OMNI_CACHE=1
export ENABLE_HOST_MAPPING=0
```

**Decode Node:**

```bash
export ENABLE_OMNI_CACHE=1
export ENABLE_HOST_MAPPING=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
```

For other variables (memory size, HBM layout, DSA Split, etc.), see [Configuration Reference - Core Switches](CONFIG_REFERENCE.en.md#1-core-switches).

### 3.2 Configure kv-transfer-config

`--kv-transfer-config` is a vLLM launch parameter that specifies the connector type, role, and network topology as a JSON string. See [Configuration Reference - kv-transfer-config](CONFIG_REFERENCE.en.md#2-kv-transfer-config) for field descriptions.

**Prefill Node:**

```
--kv-transfer-config '{"kv_connector":"OmniCacheConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["<prefill_ip>"],"kv_producer_dp_size":1}}'
```

**Decode Node (DECODE_DP_SIZE=8):**

```
--kv-transfer-config '{"kv_connector":"OmniCacheConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["<prefill_ip>"],"kv_producer_dp_size":1}}'
```

For multiple DPs, increment `kv_rank` per instance (2, 3, ..., DECODE_DP_SIZE).

### 3.3 Launch Servers

**Prefill:**

```bash
vllm serve /path/to/model \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 8 \
    --kv-transfer-config '{"kv_connector":"OmniCacheConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["<prefill_ip>"],"kv_producer_dp_size":1}}'
```

**Decode (DP=8, launch 8 instances on ports 8082~8089):**

```bash
for rank in $(seq 0 7); do
    vllm serve /path/to/model \
        --host 0.0.0.0 --port $((8082 + rank)) \
        --tensor-parallel-size 1 \
        --data-parallel-size 8 --data-parallel-rank $rank \
        --kv-transfer-config '{"kv_connector":"OmniCacheConnector","kv_role":"kv_consumer","kv_rank":'"$((rank + 1))"',"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["<prefill_ip>"],"kv_producer_dp_size":1}}' &
done
```

### 3.4 Verify

```bash
# Prefill
curl http://127.0.0.1:8000/health

# Decode (first DP instance)
curl http://127.0.0.1:8082/health
```

---

## 4. Reference Scripts

Reference launch scripts for Pangu V2 hybrid are provided. Use directly or as templates for custom scripts. All `vllm serve` parameters are overridable via environment variables:

| Script | Purpose |
|--------|---------|
| `examples/pangu_v2_pd/launch_pd.sh` | Host-side one-shot launcher for Prefill + Decode |
| `examples/pangu_v2_pd/launch_prefill.sh` | Prefill node launcher |
| `examples/pangu_v2_pd/launch_decode.sh` | Decode multi-DP launcher |

Usage:

```bash
MODEL_PATH=/path/to/model ENABLE_OMNI_CACHE=1 bash examples/pangu_v2_pd/launch_pd.sh
```

---

## 5. Troubleshooting

### libhccl.so Not Found

CANN environment not loaded:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### EJ0003 Port Binding Failed

HCCL ports stuck from a previous unclean exit:

```bash
docker restart <container>
sleep 30
```

### Decode OOM

Reduce `NUM_GPU_BLOCKS_OVERRIDE` or `OMNI_CACHE_LAYER_BYTES`. See [Configuration Reference - Per-Layer HBM Budget](CONFIG_REFERENCE.en.md#22-per-layer-hbm-budget).

### Prefill Hidden State Divergence

`enable_moe_agrs` must be `false` in the prefill config, otherwise hidden states diverge from decoder layer 2 onwards.

---

## 6. Configuration Index

For complete documentation of all configuration parameters (environment variables, kv-transfer-config fields, HBM layout, DSA Split, networking ports, etc.), see the **[Configuration Reference](CONFIG_REFERENCE.en.md)**.
