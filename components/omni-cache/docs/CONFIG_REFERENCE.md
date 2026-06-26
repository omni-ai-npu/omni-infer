# OmniCache 配置参考

## 一、核心开关

| 变量 | 取值 | 默认值 | 适用端 | 说明 |
|------|------|--------|--------|------|
| `ENABLE_OMNI_CACHE` | `0`, `1` | `1` | P / D | 总开关。`1` 启用 OmniCacheConnector + hugetlbfs 主机内存池；`0` 回退到 vLLM 原生 LLMDataDistConnector（仅 HBM，无 hugepage） |
| `ENABLE_HOST_MAPPING` | `0`, `1` | P: `0`, D: `1` | P / D | 主机内存 mmap 别名映射。`1` 时 DSA indexer 在 HBM 上，其余 KV 数据通过 NPU MMU 从主机读取。Prefill 不设置此映射 |
| `P_NODE_LIST` | string | P: `192.168.0.148`<br>D: `192.168.0.148` | P / D | Prefill 节点 IP 列表。单机逗号分隔，多机分号分隔。Decode 通过此 IP 与 Prefill 建立 ZMQ/OX 连接 |
| `OMNI_CACHE_LOCAL_DP_SIZE` | int | `8` | P / D | 单机本地 DP 并行度，即每台机器上参与当前角色的 NPU die 数量。用于计算每个 DP rank 分配到的 host KV Cache 大小和 block 数量 |

**IP 格式：**

```bash
# 单机多实例
P_NODE_LIST="192.168.1.10,192.168.1.11"

# 多机多实例
P_NODE_LIST="192.168.1.10,192.168.1.11;192.168.2.10,192.168.2.11"
```

---

## 二、kv-transfer-config 配置

OmniCache 通过 vLLM 的 `--kv-transfer-config` 命令行参数接收 JSON 配置，指定连接器类型、角色和网络拓扑。

### 参数结构

```json
{
    "kv_connector": "<connector_name>",
    "kv_role": "<role>",
    "kv_rank": <rank>,
    "kv_parallel_size": <parallel_size>,
    "kv_connector_extra_config": {
        "p_node_list": ["<ip>", ...],
        "kv_producer_dp_size": <dp_size>
    }
}
```

### 顶层字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `kv_connector` | string | 连接器名称。OmniCache 模式：`OmniCacheConnector`；基线模式：`LLMDataDistConnector` |
| `kv_role` | string | 节点角色：`kv_producer`（Prefill）或 `kv_consumer`（Decode） |
| `kv_rank` | int | 当前节点在 `kv_parallel_size` 中的编号。Prefill 固定为 `0`，Decode DP_i 为 `i + 1` |
| `kv_parallel_size` | int | KV 并行组总大小。OmniCache 模式为 `1`，基线模式配置为TP并行场景下kv cache切分份数 |
| `kv_connector_extra_config` | object | 连接器扩展配置，传递到 OmniCacheConnector 内部 |

### kv_connector_extra_config 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `p_node_list` | list[string] | Prefill 节点 IP 列表。环境变量 `P_NODE_LIST` 中分号/逗号分隔的字符串展开为列表。例：`["192.168.1.10", "192.168.1.11"]` |
| `kv_producer_dp_size` | int | Prefill 端 DP 数量，通常为 `1` |

### Prefill 节点示例（OmniCache 模式）

```
--kv-transfer-config '{"kv_connector":"OmniCacheConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["192.168.0.148"],"kv_producer_dp_size":1}}'
```

### Decode 节点示例（OmniCache 模式，DECODE_DP_SIZE=8）

```
--kv-transfer-config '{"kv_connector":"OmniCacheConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["127.0.0.1"],"kv_producer_dp_size":1}}'
```

多 DP 时每个实例 `kv_rank` 递增（2, 3, ..., 8），其余字段相同。

### 基线模式

```
--kv-transfer-config '{"kv_connector":"LLMDataDistConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":1,"kv_connector_extra_config":{"p_node_list":["<prefill_ip>"],"kv_producer_dp_size":1}}'
```

---

## 三、内存与 HugePage

### 2.1 HugePage 基础配置

| 变量 | 默认值 | 适用端 | 说明 |
|------|--------|--------|------|
| `OMNI_CACHE_MMAP_FILE` | P: `omni_cache_p`<br>D: `omni_cache_d` | P / D | `/dev/hugepages/` 下的内存映射文件名 |
| `OMNI_CACHE_MMAP_PATH` | `/dev/hugepages/${OMNI_CACHE_MMAP_FILE}` | P / D | 完整的 hugepage 文件路径，可按实际挂载点覆盖 |
| `MAP_SIZE_BYTES` | `536870912000` (500 GiB) | P / D | hugepage 文件大小（字节），决定主机 KV 池总容量 |


### 2.2 HBM 层内配置

| 变量 | 默认值 | 适用端 | 说明 |
|------|--------|--------|------|
| `OMNI_CACHE_LAYER_BYTES` | `17179869184` (16 GiB) | P / D | 每层 HOST KV Cache buffer 预算（字节），影响每 die 的 `num_blocks`。decode 侧可适当调小以节省 HBM |
| `NUM_GPU_BLOCKS_OVERRIDE` | P: `50000`, D: `11800` | P / D | vLLM 调度器 block 上限，必须小于 `OMNI_CACHE_LAYER_BYTES / (DP_SIZE_LOCAL * nybtes(kv_cache_block))` |

---

## 四、HBM 布局模式（仅 Prefill）

| 变量 | 取值 | 默认值 | 适用端 | 说明 |
|------|------|--------|--------|------|
| `OMNI_CACHE_PACKED_HBM` | `0`, `1` | `1` | Prefill | PACKED_HBM 布局模式。`1` 时 HBM 上的 device KV cache block 独立管理，与 Host 的 block id 解耦；`0` 时 host 和 device 上的 block 数必须严格一致 |

---

## 五、DSA Split 二级主机池（仅 Decode）

Pangu V2 hybrid 模型中 DSA 注意力 KV 每 block 包含两部分：

```
[ block_size * head_size ]       bf16  kv (kv_lora + k_pe, 576 bytes/token)
[ block_size * indexer_head_dim ] bf16  indexer (128 bytes/token)
```

开启 DSA Split 后，decode 额外创建一个二级 hugepage 文件，仅存储 DSA KV 数据。每次 OX pull 完成后，通过 `aclrtMemcpyAsync` 异步将 KV 段从主池复制到二级池，attention kernel 从二级池读取更窄的 KV 数据。

| 变量 | 取值 | 默认值 | 说明 |
|------|------|--------|------|
| `ENABLE_OMNI_CACHE_DSA_SPLIT` | `0`, `1` | `0` | 主开关。`1` 启用二级 DSA 池 + 异步 post-pull 拷贝 |
| `OMNI_CACHE_DSA_MMAP_FILE` | 字符串 | `omni_cache_decode_dsa` | `/dev/hugepages/` 下二级池文件名 |
| `OMNI_CACHE_DSA_MMAP_PATH` | 路径 | `/dev/hugepages/${OMNI_CACHE_DSA_MMAP_FILE}` | 二级池完整路径 |
| `OMNI_CACHE_DSA_MAP_SIZE_BYTES` | 字节 | `MAP_SIZE_BYTES * 80 / 100` | 二级池 hugepage 预留大小（自动按主池 80% 计算） |


---

## 六、网络与通信

| 变量 | 默认值 | 适用端 | 说明 |
|------|--------|--------|------|
| `BASE_PORT` | `16077` | P / D | 节点间通信基础端口 |
| `ZMQ_BASE_PORT` | `16555` | P / D | ZMQ 传输基础端口 |

---

## 七、诊断与调试

以下变量仅用于开发调试和正确性验证，生产环境不需要设置。

| 变量 | 取值 | 默认值 | 说明 |
|------|------|--------|------|
| `OMNI_KV_DUMP_GEAR` | `off`, `transfer`, `step` | `off` | KV dump 模式。`transfer` 在 4 个传输节点 dump；`step` 在每步 decode dump |
| `OMNI_KV_DUMP_BRANCH` | 字符串 | 空 | dump 文件的 branch 标识（如 `baseline` / `omnicache`） |
| `OMNI_KV_DUMP_MAX` | 整数 | `0` | 最大 dump 次数，`0` 不限制 |
| `OMNI_KV_DUMP_TARGET_REQ` | 请求 ID 前缀 | 空 | 仅 dump 匹配前缀的请求 |
| `OMNI_MOCK_SCHEDULE` | `0`, `1`, `2` | `0` | Mock 调度器：`0` 正常调度，`1` batch gate + sort，`2` 额外在第 5 步交换 InputBatch 行 |
| `OMNI_CACHE_VERIFY_TRANSFER` | `0`, `1` | `0` | 校验 KV 传输字节一致性 |
| `OMNI_CACHE_SKIP_OX_PULL` | `0`, `1` | `0` | 跳过 OX pull（仅用于无 prefill 的 decode 测试） |
| `OMNI_CACHE_MOME_DEBUG` | `0`, `1` | `0` | MoME state 传输调试日志 |
| `ENABLE_MOCK_P` | `0`, `1` | `0` | Mock prefill 模式，模拟 prefill 端行为 |
| `OMNI_CACHE_TEST_BLOCK_STATUS` | `0`, `1` | `0` | 测试模式：初始化 block 状态映射 |
| `OMNI_CACHE_DEBUG` | `0`, `1` | `0` | 调试日志开关，输出详细的模块加载和执行信息 |

---
