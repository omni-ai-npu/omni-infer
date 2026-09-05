# usefull_patch

Pangu V2 MoE 505B int8 + EP 离线精度测试所需的最小 patch 集合。

目录划分：`common/` 为通用 patch；`models/pangu_v2_base/` 为共享补丁；`models/high_throughout/` 与 `models/low_latency/` 为性能路径补丁。

## 目录结构

```
usefull_patch/
├── common/
├── models/
│   ├── pangu_v2_base/
│   ├── high_throughout/
│   └── low_latency/
└── README.md
```

## common/

| 文件 | 作用 |
|------|------|
| `patch_torch_accelerator.py` | 把 `torch.accelerator` 的内存 API（`empty_cache` / `memory_stats` / `memory_reserved` / `memory_allocated` / `reset_peak_memory_stats` / `get_memory_info`）重定向到 `torch.npu`。上游 0.25.1 把内存统计从 `current_platform.*` 迁到了 `torch.accelerator.*`，而后者不会转发到 NPU，`get_memory_info` 还会直接抛 "Allocator for npu is not a DeviceAllocator" |
| `patch_hccl_set_comm_name.py` | HCCL 分布式初始化 |
| `patch_parallel_state.py` | NPU TP/EP 通信组 |
| `patch_attention.py` | NPU attention backend 注册 |
| `patch_backends_utils.py` | CommonAttentionMetadata 扩展 |
| `patch_serving_apc.py` | PD 分离下把 APC 命中率上报改对：D 侧原生恒报 100%，改为转发 P 的真实命中；并补 `cached_rate` 字段 |
| `patch_health.py` | 卡死检测（OMNI-WATCHDOG）：引擎有在途请求却超 `OMNI_HEALTH_HANG_SEC`不推进时，`/health` 与 `/ping` 转 503，不杀进程、恢复后自动回 200。心跳由 `omni_npu_metrics` 插件（实现在 `omni/diagnostics/watchdog/`）驱动，**该插件名必须列进 `VLLM_PLUGINS`，否则不生效** |
| `patch_dump.py` | OMNI-DUMP 退出取证的三个挂载点（`AsyncLLM.__init__` / `EngineCoreProc.run_busy_loop` + `DPEngineCoreProc.run_busy_loop` / `NPUWorker.init_device`）。实现在 `omni/diagnostics/dump/`；0.25.1 接口零改动，engine 挂载点从 `EngineCore.__init__` 挪走是修一个与版本无关的缺陷（spawn 下静默失效，见 commit message）；`OMNI_DUMP_ENABLE` 未设置时默认开启 |
| `patch_kv_output_aggregator.py` | 拆分 `KVOutputAggregator` send/recv 计数策略 |
| `patch_multi_connector.py` | MultiConnector 分向透传 `get_finished_send_count` / `get_finished_recv_count`（PD 释块 send=1，Offloading load recv=world_size） |
| `patch_mrv2_sampler.py` | V2 采样算子 `gumbel_sample` / `apply_top_k_top_p` / `rejection_sample` 换 NPU 实现或绑定（上游随机数、`log1p`、`None` optional tensor 和 top-k/top-p Triton kernel 在 triton-ascend 3.2.2 上不适配）。只换 V2 消费方，MRv1 用的定义模块 `v1/sample/ops/topk_topp_sampler.py` 不动 |
| `patch_mrv2_buffer_utils.py` | `gpu/buffer_utils.UvaBuffer` → `NPUUvaBuffer`：默认 pinned host + H2D 拷贝，`OMNI_NPU_V2_UVA=1` 走 npu_uva 真视图 |
| `patch_mrv2_attn_utils.py` | `gpu/attn_utils._reshape_kv_cache`：把 KV 布局交给 NPU attention backend |
| `patch_mrv2_dp_utils.py` | V2 侧 DP：每步发布 LM head 的 all_gather pad 目标 + eager 下为 MoE EP 补齐各 rank token 数。V1 的对应实现是 `patch_dp_utils.py`，改一个要想另一个 |

> MRv2 四个 patch 的实现在 omni/worker/npu/，采样相关 Triton helper 放在 omni/worker/npu/ops/；绑定按消费方逐个枚举（应用晚于消费方 import，只改定义模块盖不住），目标模块 import 失败只记 error 不注册。

## models/pangu_v2_base/

共享补丁，由 `high_throughout` / `low_latency` 自动带上。

| 文件 | 作用 |
|------|------|
| `patch_kv_cache_interface.py` | 注入 `MomeSpec` / `DSAAttentionSpec` / `ShareKVSlidingWindowSpec` |
| `patch_single_type_kv_cache_manager.py` | 注册 `MomeManager` / `ShareKVSlidingWindowManager` 并为 Mome 注入 admission cap |
| `patch_kv_cache_utils.py` | `HYBRID_ATTN_GROUP_SIZE` 环境变量 override hybrid KV group 分组 |
| `patch_kv_cache_dtype.py` | 支持 int8/hif8 等 KV cache dtype |
| `patch_hybrid_kv_cache_coordinator.py` | hybrid APC connector：`find_longest_cache_hit_per_group` 把公共命中长度按 group 重复 |
| `patch_scheduler.py` | PD / reasoning `max_tokens` 排除 thinking |
| `patch_speculative.py` | MTP / speculative config |
| `patch_model_arch_config_convertor.py` | Pangu MLA 架构识别 |
| `patch_process_weights_after_loading.py` | NPU 权重后处理：loader 在 quant packing 之后补调 `NPUPanguSparseAttention` / `NPUmHC` / `NPUmHCRL` / `NPURMSNorm` |

## models/high_throughout/

| 文件 | 作用 |
|------|------|
| `patch_sink_attention_spec.py` | 注入 `SinkMLAAttentionSpec` |
| `patch_static_sink_attention.py` | StaticSink attention |
| `patch_hybrid_kv_cache_coordinator.py` | hybrid APC `find_longest_cache_hit`：禁止 simple-hybrid 提前退出，并把 FA 命中长度限制在实际持有的 block 上 |
| `patch_mome_hybrid.py` | Pangu V2 hybrid MoME attention |

## models/low_latency/

当前无额外 patch 文件。设置 `OMNI_VLLM_PATCHES_DIR=low_latency` 仍会加载 `pangu_v2_base`。

## 加载方式

由 `omni_npu.vllm_patches.apply_patches()` 加载：

- `common/` 始终导入
- `models/<dir>/` 仅在 `OMNI_VLLM_PATCHES_DIR` 中点名时导入；支持逗号分隔多个目录，按顺序加载
- `high_throughout` / `low_latency` 会自动附带 `pangu_v2_base`（先加载共享目录，再加载对应性能目录）
- 旧写法 `pangu_v2_hybrid` / `pangu_v2_moe` 分别等价于 `high_throughout` / `low_latency`

```bash
export OMNI_VLLM_PATCHES_DIR="high_throughout"
# 加载 common/ + models/pangu_v2_base/ + models/high_throughout/

export OMNI_VLLM_PATCHES_DIR="low_latency"
# 加载 common/ + models/pangu_v2_base/ + models/low_latency/

export OMNI_VLLM_PATCHES_DIR="pangu_v2_hybrid, pangu_v2_moe"
# 兼容旧 playbook：加载 pangu_v2_base + high_throughout + low_latency
```

未设置 `OMNI_VLLM_PATCHES_DIR` 时只加载 `common/`。每个目录内的文件按文件名排序。
