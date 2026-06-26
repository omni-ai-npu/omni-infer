# OmniCache

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2B-orange)](https://pytorch.org/)
[![Ascend NPU](https://img.shields.io/badge/Ascend-NPU-red)](https://www.hiascend.com/)

> 面向 vLLM 的 PD 分离 KV Cache 管理插件

[English](README.en.md) | [中文](README.md)

---

## 项目简介

OmniCache 是面向 vLLM 的 PD 分离 KV Cache 管理插件。它在 Prefill 节点与 Decode 节点之间建立高效的 KV Cache 传输通道：Prefill 完成后，KV Cache 从 HBM 卸载到主机内存并通过 OX 发送；Decode 接收后从主机内存加载到 HBM 完成推理。

核心优势：以主机内存池（hugetlbfs）作为中间缓存层，显著降低 P/D 两侧 KV Cache 对 HBM 的内存压力，大幅提升推理的序列长度与并发数；同时 KV Cache 持久化可大幅提升多轮对话场景下的 APC 命中率。

---

## 安装

### 环境要求

- Linux (openEuler / Ubuntu)
- Python 3.11+ / PyTorch 2.5.1+
- 华为昇腾 NPU + CANN Toolkit
- Docker（多节点部署时）

### 安装步骤

```bash
git clone https://gitee.com/omniai/omni-cache.git
cd omni-cache
pip install -e . --no-build-isolation
```

### HugePage 配置

```bash
sudo bash tools/setup/set_hugepage_limit.sh
```

---

## 快速开始

OmniCache 通过 `--kv-transfer-config` 参数配置 PD 分离的 KV Cache 传输。

**Prefill 节点：**

```bash
export ENABLE_OMNI_CACHE=1
export ENABLE_HOST_MAPPING=0

vllm serve /path/to/model \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 8 \
    --kv-transfer-config '{"kv_connector":"OmniCacheConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["<prefill_ip>"],"kv_producer_dp_size":1}}'
```

**Decode 节点：**

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

## 文档

- [用户指南](docs/USER_GUIDE.md) — 环境准备、服务启动流程、常见问题
- [配置参考](docs/CONFIG_REFERENCE.md) — 全部环境变量与 kv-transfer-config 参数说明

---

## 开源协议

本项目基于 MIT 协议开源 - 详见 [LICENSE](LICENSE) 文件。
