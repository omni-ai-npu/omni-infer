# OmniCache

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2B-orange)](https://pytorch.org/)
[![Ascend NPU](https://img.shields.io/badge/Ascend-NPU-red)](https://www.hiascend.com/)

> PD-disaggregated KV Cache management plugin for vLLM

[English](README.en.md) | [中文](README.md)

---

## Overview

OmniCache is a PD-disaggregated KV Cache management plugin for vLLM. It establishes an efficient KV Cache transfer channel between Prefill and Decode nodes: after prefill completes, KV Cache is offloaded from HBM to host memory and sent via OX; Decode receives it and loads it from host memory into HBM for inference.

Key advantages: by using a host memory pool (hugetlbfs) as an intermediate cache layer, it significantly reduces KV Cache pressure on HBM on both the Prefill and Decode sides, enabling much longer sequence lengths and higher concurrency. KV Cache persistence also dramatically improves APC hit rates in multi-turn conversation scenarios.

---

## Installation

### Prerequisites

- Linux (openEuler / Ubuntu)
- Python 3.11+ / PyTorch 2.5.1+
- Huawei Ascend NPU + CANN Toolkit
- Docker (for multi-node deployment)

### Install

```bash
git clone https://gitee.com/omniai/omni-cache.git
cd omni-cache
pip install -e . --no-build-isolation
```

### HugePage Configuration

```bash
sudo bash tools/setup/set_hugepage_limit.sh
```

---

## Quick Start

OmniCache uses the `--kv-transfer-config` parameter for PD-disaggregated KV Cache transfer configuration.

**Prefill Node:**

```bash
export ENABLE_OMNI_CACHE=1
export ENABLE_HOST_MAPPING=0

vllm serve /path/to/model \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 8 \
    --kv-transfer-config '{"kv_connector":"OmniCacheConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["<prefill_ip>"],"kv_producer_dp_size":1}}'
```

**Decode Node:**

```bash
export ENABLE_OMNI_CACHE=1
export ENABLE_HOST_MAPPING=1
export VLLM_WORKER_MULTIPROC_METHOD=fork

for rank in $(seq 0 7); do
    vllm serve /path/to/model \
        --host 0.0.0.0 --port $((8082 + rank)) \
        --tensor-parallel-size 1 \
        --data-parallel-size 8 --data-parallel-rank $rank \
        --kv-transfer-config '{"kv_connector":"OmniCacheConnector","kv_role":"kv_consumer","kv_rank":'"$((rank + 1))"',"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["<prefill_ip>"],"kv_producer_dp_size":1}}' &
done
```

---

## Documentation

- [User Guide](docs/USER_GUIDE.en.md) — environment setup, service launch flow, troubleshooting
- [Configuration Reference](docs/CONFIG_REFERENCE.en.md) — complete environment variables and kv-transfer-config parameters

---
