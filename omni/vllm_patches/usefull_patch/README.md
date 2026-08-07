# usefull_patch

Pangu V2 MoE 505B int8 + EP 离线精度测试所需的最小 patch 集合。

## 文件说明

| 文件 | 来源 | 作用 |
|------|------|------|
| `patch_mamba_utils.py` | pangu_v2_moe | 注入 `mome_state_shape` / `mome_state_dtype` |
| `patch_kv_cache_dtype.py` | pangu_v2_moe | 支持 int8/hif8 等 KV cache dtype |
| `patch_kv_cache_interface.py` | pangu_v2_hybrid | 注入 `MomeSpec` / `DSAAttentionSpec` 等 KV cache spec |
| `patch_kv_cache_utils.py` | pangu_v2_hybrid | `HYBRID_ATTN_GROUP_SIZE` 环境变量 override hybrid KV group 分组 |
| `patch_model_arch_config_convertor.py` | pangu_v2_moe | Pangu MLA 架构识别 |
| `patch_process_weights_after_loading.py` | pangu_v2_moe | NPU 权重后处理 |
| `patch_mla.py` | pangu_v2_base | StaticSink MLA wrapper |
| `patch_static_sink_attention.py` | pangu_sink_swa_mla | StaticSink attention |
| `patch_modelconfig.py` | pangu_sink_swa_mla | ModelArchConfigConvertor 注册 |
| `patch_hccl_set_comm_name.py` | common | HCCL 分布式初始化 |
| `patch_parallel_state.py` | common | NPU TP/EP 通信组 |
| `patch_attention.py` | common | NPU attention backend 注册 |
| `patch_backends_utils.py` | common | CommonAttentionMetadata 扩展 |
| `patch_eplb_parallel.py` | common | EP / EPLB 支持 |
| `patch_serving_apc.py` | common（已迁入本目录） | PD 分离下把 APC 命中率上报改对：D 侧原生恒报 100%，改为转发 P 的真实命中；并补 `cached_rate` 字段 |

## 加载方式

由 `omni_npu.vllm_patches.apply_patches()` 自动加载本目录下全部 patch（按文件名排序）。
