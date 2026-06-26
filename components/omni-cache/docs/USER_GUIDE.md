# OmniCache 用户指南

## 一、概述

OmniCache 是面向 vLLM 的 PD 分离 KV Cache 管理插件。它在 Prefill 节点与 Decode 节点之间建立高效的 KV Cache 传输通道：Prefill 完成后，KV Cache 从 HBM 卸载到主机内存并通过 OX 发送；Decode 接收后从主机内存加载到 HBM 完成推理。

核心优势：以主机内存池（hugetlbfs）作为中间缓存层，显著降低 P/D 两侧 KV Cache 对 HBM 的内存压力，大幅提升推理的序列长度与并发数；同时 KV Cache 持久化可大幅提升多轮对话场景下的 APC 命中率。

---

## 二、环境准备

### 2.1 拉取代码

```bash
git clone https://gitee.com/omniai/omni-cache.git
cd omni-cache
```

### 2.2 安装

```bash
pip install -e . --no-build-isolation
```

### 2.3 系统要求

- Linux (openEuler / Ubuntu)
- Python 3.11+ / PyTorch 2.5.1+
- 华为昇腾 NPU + CANN Toolkit
- Docker（多节点部署时）

### 2.4 HugePage 配置

OmniCache 使用 2MB HugePages 管理主机端 KV Cache 内存池（默认 500 GiB）。

```bash
# 自动计算（推荐）
sudo bash tools/setup/set_hugepage_limit.sh

# 手动指定页数
sudo bash tools/setup/set_hugepage_limit.sh --target-pages 1048576
```

验证：

```bash
grep HugePages_ /proc/meminfo
```

---

## 三、服务启动流程

### 3.1 设置环境变量

**Prefill 节点：**

```bash
export ENABLE_OMNI_CACHE=1
export ENABLE_HOST_MAPPING=0
```

**Decode 节点：**

```bash
export ENABLE_OMNI_CACHE=1
export ENABLE_HOST_MAPPING=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
```

其他环境变量（内存大小、HBM 布局、DSA Split 等）按需设置，完整列表见 [配置参考 - 核心开关](CONFIG_REFERENCE.md#一核心开关)。

### 3.2 配置 kv-transfer-config

`--kv-transfer-config` 是 vLLM 的启动参数，以 JSON 字符串指定连接器类型、角色和网络拓扑。字段说明详见 [配置参考 - kv-transfer-config 配置](CONFIG_REFERENCE.md#二kv-transfer-config-配置)。

**Prefill 节点：**

```
--kv-transfer-config '{"kv_connector":"OmniCacheConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["<prefill_ip>"],"kv_producer_dp_size":1}}'
```

**Decode 节点（DECODE_DP_SIZE=8）：**

```
--kv-transfer-config '{"kv_connector":"OmniCacheConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["<prefill_ip>"],"kv_producer_dp_size":1}}'
```

多 DP 时每个实例 `kv_rank` 递增（2, 3, ..., DECODE_DP_SIZE）。

### 3.3 启动服务

**Prefill：**

```bash
vllm serve /path/to/model \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 8 \
    --kv-transfer-config '{"kv_connector":"OmniCacheConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["<prefill_ip>"],"kv_producer_dp_size":1}}'
```

**Decode（DP=8，启动 8 个实例，端口 8082~8089）：**

```bash
for rank in $(seq 0 7); do
    vllm serve /path/to/model \
        --host 0.0.0.0 --port $((8082 + rank)) \
        --tensor-parallel-size 1 \
        --data-parallel-size 8 --data-parallel-rank $rank \
        --kv-transfer-config '{"kv_connector":"OmniCacheConnector","kv_role":"kv_consumer","kv_rank":'"$((rank + 1))"',"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["<prefill_ip>"],"kv_producer_dp_size":1}}' &
done
```

### 3.4 验证服务

```bash
# Prefill
curl http://127.0.0.1:8000/health

# Decode（第一个 DP 实例）
curl http://127.0.0.1:8082/health
```

---

## 四、参考脚本

仓库提供了 Pangu V2 hybrid 模型的参考启动脚本，可直接使用或作为自定义脚本的模板。脚本通过环境变量覆盖所有 `vllm serve` 参数：

| 脚本 | 用途 |
|------|------|
| `examples/pangu_v2_pd/launch_pd.sh` | 宿主机一键启动 Prefill + Decode |
| `examples/pangu_v2_pd/launch_prefill.sh` | Prefill 节点启动 |
| `examples/pangu_v2_pd/launch_decode.sh` | Decode 多 DP 启动 |

使用方式：

```bash
MODEL_PATH=/path/to/model ENABLE_OMNI_CACHE=1 bash examples/pangu_v2_pd/launch_pd.sh
```

---

## 五、常见问题

### libhccl.so 找不到

CANN 环境未加载：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### EJ0003 端口绑定失败

HCCL 端口被上一轮未正常退出的进程占用：

```bash
docker restart <container>
sleep 30
```

### Decode OOM

降低 `NUM_GPU_BLOCKS_OVERRIDE` 或 `OMNI_CACHE_LAYER_BYTES`，详见 [配置参考 - HBM 层内配置](CONFIG_REFERENCE.md#22-hbm-层内配置)。

### Prefill hidden state 不一致

Prefill 配置中 `enable_moe_agrs` 须设为 `false`，否则 decoder layer 2 起 hidden state 出现差异。

---

## 六、配置参数索引

所有配置参数（环境变量、kv-transfer-config 字段、HBM 布局、DSA Split、网络端口等）的完整说明见 **[配置参考](CONFIG_REFERENCE.md)**。
