# usefull_patch

Pangu V2 MoE 505B int8 + EP 离线精度测试所需的最小 patch 集合。

## 文件说明

| 文件 | 来源 | 作用 |
|------|------|------|
| `patch_mamba_utils.py` | pangu_v2_moe | 注入 `mome_state_shape` / `mome_state_dtype` |
| `patch_kv_cache_dtype.py` | pangu_v2_moe | 支持 int8/hif8 等 KV cache dtype |
| `patch_kv_cache_interface.py` | pangu_v2_hybrid | 注入 `MomeSpec` / `DSAAttentionSpec` 等 KV cache spec |
| `patch_single_type_kv_cache_manager.py` | pangu_v2_hybrid | 注册 `MomeManager` / `ShareKVSlidingWindowManager` 并为 Mome 注入 admission cap
| `patch_kv_cache_utils.py` | pangu_v2_hybrid | `HYBRID_ATTN_GROUP_SIZE` 环境变量 override hybrid KV group 分组 |
| `patch_model_arch_config_convertor.py` | pangu_v2_moe | Pangu MLA 架构识别 |
| `patch_process_weights_after_loading.py` | pangu_v2_moe | NPU 权重后处理 |
| `patch_mla.py` | pangu_v2_base | StaticSink MLA wrapper |
| `patch_static_sink_attention.py` | pangu_sink_swa_mla | StaticSink attention |
| `patch_modelconfig.py` | pangu_sink_swa_mla | ModelArchConfigConvertor 注册 |
| `patch_torch_accelerator.py` | common | 把 `torch.accelerator` 的内存 API（`empty_cache` / `memory_stats` / `memory_reserved` / `memory_allocated` / `reset_peak_memory_stats` / `get_memory_info`）重定向到 `torch.npu`。上游 0.25.1 把内存统计从 `current_platform.*` 迁到了 `torch.accelerator.*`，而后者不会转发到 NPU，`get_memory_info` 还会直接抛 "Allocator for npu is not a DeviceAllocator" |
| `patch_hccl_set_comm_name.py` | common | HCCL 分布式初始化 |
| `patch_parallel_state.py` | common | NPU TP/EP 通信组 |
| `patch_attention.py` | common | NPU attention backend 注册 |
| `patch_backends_utils.py` | common | CommonAttentionMetadata 扩展 |
| `patch_eplb_parallel.py` | common | EP / EPLB 支持 |
| `patch_serving_apc.py` | common（已迁入本目录） | PD 分离下把 APC 命中率上报改对：D 侧原生恒报 100%，改为转发 P 的真实命中；并补 `cached_rate` 字段 |
| `patch_health.py` | common | 卡死检测（OMNI-WATCHDOG）：引擎有在途请求却超 `OMNI_HEALTH_HANG_SEC`不推进时，`/health` 与 `/ping` 转 503，不杀进程、恢复后自动回 200。心跳由 `omni_npu_metrics` 插件（实现在 `omni/diagnostics/watchdog/`）驱动，**该插件名必须列进 `VLLM_PLUGINS`，否则不生效** |
| `patch_dump.py` | common（原文件保留未动） | OMNI-DUMP 退出取证的三个挂载点（`AsyncLLM.__init__` / `EngineCoreProc.run_busy_loop` + `DPEngineCoreProc.run_busy_loop` / `NPUWorker.init_device`）。实现在 `omni/diagnostics/dump/`；0.25.1 接口零改动，engine 挂载点从 `EngineCore.__init__` 挪走是修一个与版本无关的缺陷（spawn 下静默失效，见 commit message）；`OMNI_DUMP_ENABLE` 未设置时默认开启 |
| `patch_kv_output_aggregator.py` | — | 拆分 `KVOutputAggregator` send/recv 计数策略 |
| `patch_multi_connector.py` | — | MultiConnector 分向透传 `get_finished_send_count` / `get_finished_recv_count`（PD 释块 send=1，Offloading load recv=world_size） |

## 加载方式

由 `omni_npu.vllm_patches.apply_patches()` 自动加载本目录下全部 patch（按文件名排序）。
