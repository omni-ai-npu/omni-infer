# 一、OmniCache 单元测试与源码对应关系

## 对应关系总览

> 所有源码路径相对于 `omni_cache/`，所有测试路径相对于 `tests/unit_tests/`。

```
omni_cache/                                         tests/unit_tests/
├── plugin.py ◄──────────────────────────────────── test_plugin.py
├── __init__.py
│
├── attention/
│   ├── kv_cache_interface.py ◄──────────────────── test_kv_cache_interface.py
│   ├── backends/
│   │   ├── attention_ext.py ◄───────────────────── attention/backends/test_attention_ext.py
│   │   ├── dsa_ext.py ◄────────────────────────── attention/backends/test_dsa_ext.py
│   │   ├── stride_compress_ext.py ◄────────────── attention/backends/test_stride_compress_ext.py
│   │   └── mla_ext.py                              (无测试)
│   ├── kv_managers/
│   │   ├── hd_kv_manager.py ◄──────────────────── test_hd_kv_manager.py
│   │   ├── blocks.py ◄─────────────────────────── test_kv_cache_manager.py
│   │   ├── omni_manager.py ◄───────────────────── test_kv_cache_manager.py
│   │   ├── attention_manager.py ◄──────────────── test_kv_cache_manager.py
│   │   ├── factory.py ◄────────────────────────── test_kv_cache_manager.py
│   │   └── allocator.py                            (无测试)
│   └── metadata/
│       ├── block_table.py ◄─────────────────────── test_gather_selection_utils.py
│       ├── attention.py                             (无测试)
│       ├── compress.py                              (无测试)
│       ├── dsa.py                                   (无测试)
│       ├── hybrid_attention.py                      (无测试)
│       ├── sliding_window_attention.py              (无测试)
│       └── utils.py                                 (无测试)
│
├── attn_plugins/
│   ├── base.py ◄────────────────────────────────── test_attn_plugins/test_implementations.py
│   ├── implementations.py ◄─────────────────────── test_attn_plugins/test_implementations.py
│   └── __init__.py ◄───────────────────────────── test_attn_plugins/test_implementations.py
│
├── cache/
│   ├── __init__.py ◄────────────────────────────── test_cache_init.py
│   ├── core/
│   │   ├── base.py ◄───────────────────────────── test_omni_cache_base.py, test_decode_omni_cache.py, test_prefill_omni_cache.py
│   │   └── constants.py                             (无测试)
│   ├── decode/
│   │   ├── decode_omni_cache.py ◄───────────────── test_decode_omni_cache.py, test_omni_cache_base.py
│   │   ├── hbm_buffer_utils.py ◄───────────────── test_hbm_buffer_utils.py
│   │   ├── host_kv_cache_utils.py ◄────────────── test_host_kv_cache_utils.py
│   │   └── static_utils.py ◄───────────────────── test_decode_omni_cache.py, test_gather_selection_utils.py
│   ├── prefill/
│   │   ├── prefill_omni_cache.py ◄──────────────── test_prefill_omni_cache.py, test_omni_cache_base.py
│   │   ├── prefix_copy_ops.py ◄─────────────────── test_prefix_copy_ops.py
│   │   └── tensor_utils.py ◄───────────────────── test_tensor_utils.py
│   ├── memory/
│   │   ├── memory_pool.py ◄─────────────────────── test_kv_mem_pool.py
│   │   ├── copy_ops.py                              (无测试)
│   │   ├── hugepage_ops.py                          (无测试)
│   │   ├── shape_utils.py                           (无测试)
│   │   └── constants.py                             (无测试)
│   ├── device_backend/
│   │   └── ascend/
│   │       ├── tensor_register.py ◄─────────────── test_acl_tensor_register.py
│   │       ├── tensor_register_lib/
│   │       │   ├── __init__.py ◄────────────────── test_acl_tensor_register.py (secondary)
│   │       │   ├── setup.py                         (无测试)
│   │       │   └── zero.py                          (无测试)
│   │       ├── streams.py ◄─────────────────────── test_ascend_acl.py
│   │       ├── memcopy.py ◄─────────────────────── test_ascend_acl.py
│   │       └── ops/
│   │           └── triton_ops.py                    (无测试)
│   ├── omni_attention/
│   │   ├── runtime_config.py ◄──────────────────── test_runtime_config.py, test_cache_init.py
│   │   ├── runtime_patch.py ◄───────────────────── test_runtime_patch.py, test_cache_init.py
│   │   ├── utils.py                                 (无测试)
│   │   └── pd.py                                    (无测试)
│   ├── transfer_engine/
│   │   ├── buffers.py ◄─────────────────────────── test_transfer_engine_buffers.py
│   │   ├── shapes.py ◄─────────────────────────── test_decode_omni_cache.py, test_prefill_omni_cache.py, test_kv_cache_manager.py (secondary)
│   │   ├── manager.py                               (无测试, 仅被 mock)
│   │   ├── decode.py                                (无测试)
│   │   ├── prefill.py                               (无测试)
│   │   └── synchronize.py                           (无测试)
│   └── utils/
│       ├── ops.py ◄─────────────────────────────── test_omni_cache_base.py
│       └── support.py ◄────────────────────────── test_prefill_omni_cache.py (secondary)
│
├── connector/
│   ├── register.py ◄────────────────────────────── test_connector_register.py, test_register_connector.py
│   ├── connector.py                                 (无测试, 仅被引用)
│   ├── zmq_transport.py                             (无测试)
│   ├── utils/
│   │   ├── helpers.py ◄─────────────────────────── test_connector_helpers.py
│   │   ├── settings.py ◄───────────────────────── test_connector_internal.py
│   │   ├── metadata.py ◄───────────────────────── test_connector_internal.py
│   │   ├── process_utils.py                         (无测试)
│   │   └── __init__.py                              (无测试)
│   ├── decode/
│   │   ├── kv_loader.py                             (无测试)
│   │   ├── process_manager.py                       (无测试)
│   │   └── worker.py                                (无测试)
│   ├── prefill/
│   │   └── worker.py                                (无测试)
│   └── scheduler/
│       ├── decode.py                                (无测试)
│       └── prefill.py                               (无测试)
│
└── gather_selection/
    ├── core/
    │   ├── gather_selection.py ◄────────────────── test_gather_selection_core.py
    │   └── buffers.py ◄─────────────────────────── test_selection_buffers.py
    └── status_updater/
        └── updater.py ◄─────────────────────────── test_gather_selection_updater.py
```

---

## 交叉引用矩阵 (源码文件 vs 测试文件)

下表中 **P** = 主要测试目标, **S** = 次要涉及 (被 mock/patch 或间接导入)

| 源码文件 (omni_cache/) | test_acl_tensor_register | test_ascend_acl | test_cache_init | test_connector_helpers | test_connector_internal | test_connector_register | test_register_connector | test_decode_omni_cache | test_gather_selection_core | test_gather_selection_updater | test_gather_selection_utils | test_hbm_buffer_utils | test_hd_kv_manager | test_host_kv_cache_utils | test_kv_cache_interface | test_kv_cache_manager | test_kv_mem_pool | test_omni_cache_base | test_ox_integration | test_plugin | test_prefill_omni_cache | test_prefix_copy_ops | test_runtime_config | test_runtime_patch | test_selection_buffers | test_tensor_utils | test_transfer_engine_buffers | attention/ test_attention_ext | attention/ test_dsa_ext | attention/ test_stride_compress_ext | test_attn_plugins/ test_implementations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plugin.py | | | | | | | | | | | | | | | | | | | | **P** | | | | | | | | | | | |
| attention/kv_cache_interface.py | | | S | | | | | | | | | | | | **P** | S | | | | | | | S | S | | | | | | | |
| attention/backends/attention_ext.py | | | | | | | | | | | | | | | | | | | | | | | | | | | | **P** | | | |
| attention/backends/dsa_ext.py | | | | | | | | | | | | | | | | | | | | | | | | | | | | | **P** | | |
| attention/backends/stride_compress_ext.py | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | **P** | |
| attention/kv_managers/hd_kv_manager.py | | | | | | | | | | | | | **P** | | | | | | | | | | | | | | | | | | |
| attention/kv_managers/blocks.py | | | | | | | | | | | | | | | | **P** | | | | | | | | | | | | | | | |
| attention/kv_managers/omni_manager.py | | | | | | | | | | | | | | | | **P** | | | | | | | | | | | | | | | |
| attention/kv_managers/attention_manager.py | | | | | | | | | | | | | | | | **P** | | | | | | | | | | | | | | | |
| attention/kv_managers/factory.py | | | | | | | | | | | | | S | | | **P** | | | | | | | | | | | | | | | |
| attention/metadata/block_table.py | | | | | | | | | | | **P** | | | | | | | | | | | | | | | | | | | | |
| attn_plugins/base.py | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | **P** |
| attn_plugins/implementations.py | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | **P** |
| attn_plugins/__init__.py | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | **P** |
| cache/__init__.py | | | **P** | | | | | | | | | | | | | | | | | | | | | | | | | | | | S |
| cache/core/base.py | | | | | | | | S | | | | | | | | | | **P** | | | S | | | | | | | | | | |
| cache/decode/decode_omni_cache.py | | | | | | | | **P** | | | | | S | | | | | **P** | | S | | | | | | | | S | | S | |
| cache/decode/hbm_buffer_utils.py | | | | | | | | | | | | **P** | | | | | | | | | | | | | | | | | | | |
| cache/decode/host_kv_cache_utils.py | | | | | | | | | | | | | | **P** | | | | | | | | | | | | | | | | | |
| cache/decode/static_utils.py | | | | | | | | **P** | | | S | | | | | | | | | | | | | | | | | | | | |
| cache/prefill/prefill_omni_cache.py | | | | | | | | | | | | | S | | | | | **P** | | | **P** | | | | | | | | S | | |
| cache/prefill/prefix_copy_ops.py | | | | | | | | | | | | | | | | | | | | | | **P** | | | | | | | | | |
| cache/prefill/tensor_utils.py | | | | | | | | | | | | | | | | | | | | | | | | | | **P** | | | | | |
| cache/memory/memory_pool.py | | | | | | | | | | | | | | | | | **P** | S | | | S | | | | | | | | | | |
| cache/device_backend/ascend/tensor_register.py | **P** | | | | | | | | | | | | | | | | S | | | | | | | | | | | | | | |
| cache/device_backend/ascend/tensor_register_lib/ | S | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| cache/device_backend/ascend/streams.py | | **P** | | | | | | | | | | | | | | | S | | | | | | | | | | | | | | |
| cache/device_backend/ascend/memcopy.py | | **P** | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| cache/omni_attention/runtime_config.py | | | **P** | | | | | | | | | | | | | | | | | | | | **P** | | | | | | | | |
| cache/omni_attention/runtime_patch.py | | | **P** | | | | | | | | | | | | | | | | | | | | | **P** | | | | | | | |
| cache/transfer_engine/buffers.py | | | | | | | | | | | | | | | | | | | | | | | | | | | **P** | | | | |
| cache/transfer_engine/shapes.py | | | | | | | | S | | | | | | | | S | | | | | S | | | | | | | | | | |
| cache/transfer_engine/manager.py | | | | | | | | | | | | | | | | | | S | | | S | | | | | | | | | | |
| cache/utils/ops.py | | | | | | | | | | | S | | | | | | | **P** | | | | | | | | | | | | | |
| cache/utils/support.py | | | S | | | | | | | | | | | | | | | | | | S | | | | | | | | | | |
| connector/register.py | | | | | | **P** | **P** | | | | | | | | | | | | | | | | | | | | | | | | |
| connector/connector.py | | | | | | | S | | | | | | | | | | | | | | | | | | | | | | | | |
| connector/utils/helpers.py | | | | **P** | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| connector/utils/settings.py | | | | | **P** | | | | | | | | | | | | | | | | | | | | | | | | | | |
| connector/utils/metadata.py | | | | | **P** | | | | | | | | | | | | | | | | | | | | | | | | | | |
| gather_selection/core/gather_selection.py | | | | | | | | | **P** | | | | | | | | | | | | | | | | | | | | | | |
| gather_selection/core/buffers.py | | | | | | | | | | | | | | | | | | | | | | | | | **P** | | | | | | |
| gather_selection/status_updater/updater.py | | | | | | | | | | **P** | | | | | | | | | | | | | | | | | | | | | |

---

## 各测试文件详细说明

### conftest.py
- **类型**: 测试基础设施 (非测试文件)
- **作用**: 为所有测试注入依赖桩模块 (vllm, torch_npu, triton, torchair, zmq, llm_datadist 等), 使测试可在无硬件/无完整依赖环境下运行
- **涉及源码**:
  - `gather_selection/core/gather_selection.py` (注入 logger 桩)
  - `gather_selection/core/buffers.py` (注入 logger 桩)
  - `attention/metadata/block_table.py` (注入 logger 桩)
  - `cache/device_backend/` (注入 ACL/streams/memcopy 桩)
- **关键 fixtures**: `install_stub_modules`

---

### test_acl_tensor_register.py
- **主要测试**: `cache/device_backend/ascend/tensor_register.py`
- **测试类/函数**: `NPUTensorRegister` (初始化、`host_tensor_register`), 内存对齐检查, 大页操作, ctypes 接口
- **Mock 目标**:
  - `cache/device_backend/ascend/tensor_register._lib`
  - `cache/device_backend/ascend/tensor_register_lib.zero_copy_npu`
  - `cache/device_backend/ascend/tensor_register._ZERO_COPY_AVAILABLE`
- **次要涉及**: `cache/device_backend/ascend/tensor_register_lib/__init__.py`

---

### test_ascend_acl.py
- **主要测试**:
  - `cache/device_backend/ascend/streams.py` — `AscendCLStream` (create, sync, memcpy_async, memcpy_batch, memcpy_batch_async, `__del__`,完整生命周期)
  - `cache/device_backend/ascend/memcopy.py` — `aclrtMemLocation`, `aclrtMemcpyBatchAttr` ctypes 结构体, ACL 常量
- **Mock 目标**:
  - `cache/device_backend/ascend/streams.aclrtCreateStream`
  - `cache/device_backend/ascend/streams.aclrtSynchronizeStream`
  - `cache/device_backend/ascend/streams.aclrtDestroyStream`
  - `cache/device_backend/ascend/streams.aclrtMemcpyAsync`
  - `cache/device_backend/ascend/streams.aclrtMemcpyBatch`
  - `cache/device_backend/ascend/streams.aclrtMemcpyBatchAsync`
- **次要涉及**: `cache/device_backend/ascend/__init__.py` (re-exports)

---

### test_cache_init.py
- **主要测试**:
  - `cache/__init__.py` — `apply_omni_attn_patch`, `set_omni_cache`, `omni_cache` 全局变量, `__all__`
  - `cache/omni_attention/runtime_config.py` — `check_omni_attn_cmd_arg`, `_apply_sink_config`, `_apply_beta_config`, `_apply_pattern_config`, `_apply_recent_config`, `_apply_runtime_config`
  - `cache/omni_attention/runtime_patch.py` — `_apply_attention_runtime_patch`, `_apply_cache_manager_runtime_patch`, `_load_patch_dependencies`
- **次要涉及**:
  - `cache/utils/support.py` — `to_bool_or_raise` (桩)
  - `attention/kv_cache_interface.py` — SINK, RECENT, BETA, PATTERN (桩)
  - `attention/kv_managers/__init__.py` — OmniKVCacheBlocks, OmniKVCacheManager (桩)

---

### test_connector_helpers.py
- **主要测试**: `connector/utils/helpers.py`
- **测试类/函数**: `should_round_prompt_tokens`, `align_remote_block_ids`, `resolve_prefill_endpoint`, `PrefillEndpoint`
- **Mock 目标**: 无 (使用 lambda 回调)
- **次要涉及**: 无

---

### test_connector_internal.py
- **主要测试**:
  - `connector/utils/settings.py` — 全部常量 (BASE_DIR, OX_PATH, OX_LOG_PATH, BLOCK_RELEASE_DELAY, PER_REQUEST_CONNECTION, P_NODE_LIST, CLUSTER_LIST, CLUSTER_SIZE, NODE_IP_SPECS, BASE_PORT, ZMQ_BASE_PORT, P_NODE_PORT_LIST)
  - `connector/utils/metadata.py` — ReqMeta, ReqMetaPrefill, `_build_req_meta`, DatadistConnectorMetadata, DatadistConnectorMetadataPrefill, DTypeUtils, _SendItem, PendingReq
- **Mock 目标**: `os.environ` 环境变量字典 (多种覆盖)
- **次要涉及**: 无

---

### test_connector_register.py
- **主要测试**: `connector/register.py` — `_safe_register`, `register_connectors`
- **Mock 目标**:
  - `register_module.logger.warning`
  - `register_module._safe_register`
  - `builtins.__import__` (模拟 ImportError)
  - `install_stub_modules` 注入 vllm factory 桩
- **次要涉及**: 无

---

### test_register_connector.py
- **主要测试**: `connector/register.py` — `_safe_register`, `register_connectors`
- **Mock 目标**:
  - `sys.modules` 注入 vllm 桩
  - `connector/register.logger`
  - `connector/register._safe_register`
- **次要涉及**: `connector/connector.py` (作为注册目标字符串被引用)
- **说明**: 与 `test_connector_register.py` 测试同一源文件，但采用不同的 mock 策略

---

### test_decode_omni_cache.py
- **主要测试**:
  - `cache/decode/decode_omni_cache.py` — `DecodeOmniCache` (init, `build_h2d_ops`, `build_h2d_ops_hybrid`, `build_h2d_ops_omni_attn`, `calc_cache_shape`, `calculate_kv_xsfer_params`, `initialize_decode_omni_cache`, `synchronize_d2h`)
  - `cache/transfer_engine/shapes.py` — `calc_cache_shape_for_decode`
  - `cache/decode/static_utils.py` — `get_block_table_np`
- **Mock 目标**:
  - `cache/core/base.get_tp_group`, `cache/core/base.get_dp_group`
  - `cache/core/base.KVCacheMemoryPool._map_hugepage_memory`
  - `cache/decode/decode_omni_cache.isinstance`
  - `cache/core/base.create_omni_cache`
- **次要涉及**: `cache/core/base.py`, `cache/memory/memory_pool.py`

---

### test_gather_selection_core.py
- **主要测试**: `gather_selection/core/gather_selection.py` — `gather_selection` 函数
- **测试场景**: layer_idx=0/1, kwargs 修改, 不同 compress_ratios (1,2,4), selection_kwargs key 验证, 最小 batch 边界
- **Mock 目标**: `torch_npu.npu_gather_selection_kv_cache`, `vllm.logger.init_logger` (均为外部依赖)
- **次要涉及**: 无

---

### test_gather_selection_updater.py
- **主要测试**: `gather_selection/status_updater/updater.py` — `GatherSelectionUpdater`
- **测试方法**: `__init__`, `updater`, `record_current_batch_order`, `_update_status_buffered`, `_build_table_perm`, `_reorder_block_table_only`, `maybe_update_selection_kv_block_status`
- **Mock 目标**: `gather_selection/status_updater/updater.time`
- **次要涉及**: 无

---

### test_gather_selection_utils.py
- **主要测试**: `attention/metadata/block_table.py` — `get_block_table_np`
- **测试场景**: 基本调用, 不同 kv_cache_gid, buffer 复用, ValueError (oversized num_tokens_padded), slot_mapping 填充 -1, 零请求边界
- **Mock 目标**: `cache/decode/static_utils.torch_to_numpy_zero_copy`
- **次要涉及**: `cache/decode/static_utils.py`, `cache/utils/ops.py`

---

### test_hbm_buffer_utils.py
- **主要测试**: `cache/decode/hbm_buffer_utils.py` — `construct_hbm_buffer`
- **测试场景**: 单组, 多组 (SWA, indexer, c4, c128), separate KV scale, 空 layers, block table tensor 创建, 零初始化, 连续性, 边界情况
- **Mock 目标**: 无 (仅使用 Mock 对象)
- **次要涉及**: 无

---

### test_hd_kv_manager.py
- **主要测试**: `attention/kv_managers/hd_kv_manager.py` — `HostDeviceKVCacheManager`
- **测试方法**: `__init__` (prefill/decode 角色, 多 KV 组断言, DSA 启用, head_size 整除检查), caching 行为, decode block 处理
- **Mock 目标**:
  - `attention/kv_managers/hd_kv_manager.PrefillOmniCache`
  - `attention/kv_managers/hd_kv_manager.BlockPool`
  - `attention/kv_managers/hd_kv_manager.get_manager_for_kv_cache_spec`
  - `attention/kv_managers/hd_kv_manager.DecodeOmniCache`
- **次要涉及**: `cache/prefill/prefill_omni_cache.py`, `cache/decode/decode_omni_cache.py`, `attention/kv_managers/factory.py`

---

### test_host_kv_cache_utils.py
- **主要测试**: `cache/decode/host_kv_cache_utils.py` — `parse_pa_kv_cache`
- **测试场景**: 基本解析, dtype 转换, 多组, separate KV scale, 空 layers, view 操作, 单 host tensor, 零 num_blocks, 大 block_size, 缺失 kvscale 属性
- **Mock 目标**: 无 (仅使用 Mock 对象)
- **次要涉及**: 无

---

### test_kv_cache_interface.py
- **主要测试**: `attention/kv_cache_interface.py`
- **测试类/函数**:
  - 常量: `SINK`, `RECENT`, `BETA`, `PATTERN`
  - `OmniMultiGroupBlockTable.__init__` (验证错误: 无 block_size, 无效 block_size, 位置参数)
  - `OmniAttentionSpec` (`__post_init__`, `type_id`, `max_memory_usage_bytes`)
  - `get_kv_cache_config_omni_type` (不支持的 spec, 多 page size, 基本配置, 滑动窗口, BETA 计算)
  - `get_omni_hybrid_kv_cache_spec` (基本 spec, MLA, non-decoder 断言, pattern 分布)
- **Mock 目标**:
  - `cache/kv_cache_interface.create_kv_cache_group_specs`
  - `cache/kv_cache_interface.logger`
  - `cache/kv_cache_interface.KVCacheTensor`
  - `cache/kv_cache_interface.BlockTable`
  - `cache/kv_cache_interface.get_layers_from_vllm_config`
  - `cache/kv_cache_interface.AttentionType`

---

### test_kv_cache_manager.py
- **主要测试**:
  - `attention/kv_managers/blocks.py` — `OmniKVCacheBlocks` (`create_empty`, `get_block_ids`, `get_unhashed_block_ids`, `__add__`)
  - `attention/kv_managers/omni_manager.py` — `OmniKVCacheManager` (`__init__`, `usage`, `single_type_manager`, `block_pool`, `allocate_slots`, `free`, `get_block_ids`)
  - `attention/kv_managers/attention_manager.py` — `OmniAttentionManager` (`__init__`, `get_num_blocks_to_allocate`, `allocate_new_blocks`, `save_new_computed_blocks`, `cache_blocks`, `find_longest_cache_hit`, `remove_skipped_blocks`, `get_num_common_prefix_blocks`)
  - `attention/kv_managers/factory.py` — `get_manager_for_kv_cache_spec`
- **Mock 目标**:
  - `cache/prefill.calc_cache_shape_for_prefill`
  - `cache/decode.calc_cache_shape_for_decode`
  - `attention/kv_managers/omni_manager.PrefixCacheStats`
  - `attention/kv_managers/omni_manager.get_manager_for_kv_cache_spec`
  - `attention/kv_managers/omni_manager.os.getenv`
- **次要涉及**: `attention/kv_cache_interface.py`, `cache/transfer_engine/shapes.py`

---

### test_kv_mem_pool.py
- **主要测试**: `cache/memory/memory_pool.py` — `KVCacheMemoryPool`
- **测试方法**: `__init__` (prefill 4D/decode 5D/无效 3D/DSA), `get_block`/`__getitem__`, `set_block`/`__setitem__`, `batch_layer_copy_to_npu`, `memcpy_async`, `close`/`__del__`, `info`, get/set cycle
- **Mock 目标**:
  - `cache/memory/memory_pool.get_tp_group`
  - `cache/memory/memory_pool.NPUTensorRegister` → MockNPUTensorRegister
  - `cache/memory/memory_pool.AscendCLStream` → MockAscendCLStream
- **次要涉及**: `cache/device_backend/ascend/streams.py`, `cache/device_backend/ascend/tensor_register.py`

---

### test_omni_cache_base.py
- **主要测试**:
  - `cache/core/base.py` — `BaseOmniCache`, `create_omni_cache`, `PrefixCopyMeta`, `divide_or_raise`
  - `cache/utils/ops.py` — `_is_hybrid_attention_enabled`, `generate_full_block_slot`, `pad_inputs`, `pad_tensor`
  - `cache/prefill/prefill_omni_cache.py` — `PrefillOmniCache` (init, calc_cache_shape)
  - `cache/decode/decode_omni_cache.py` — `DecodeOmniCache` (init, calc_cache_shape, DSA)
- **Mock 目标**:
  - `cache/transfer_engine/manager.TransferManager.initialize_prefill`
  - `cache/core/base.get_tp_group`, `get_dp_group`, `get_world_group`
  - `cache/core/base.KVCacheMemoryPool`, `bind_kv_cache`, `isinstance`
  - `cache/memory/memory_pool.open_hugepage_file`, `get_tp_group`, `_map_hugepage_memory`
- **次要涉及**: `cache/transfer_engine/manager.py`, `cache/memory/memory_pool.py`

---

### test_ox_integration.py
- **主要测试**: **无 Python 源码** — 这是 ox 二进制 (C++ 编译产物) 的集成/冒烟测试
- **测试内容**: CLI 参数解析, 地址/分片解析, block size/共享内存计算, 消息结构验证, TCP 连接逻辑, buffer 管理, metrics/统计, 线程管理, ZMQ 通信模式
- **说明**: 不涉及 `omni_cache/` 下任何 Python 文件

---

### test_plugin.py
- **主要测试**: `plugin.py`
- **测试类/函数**:
  - `LoadModelPlugin` (`pre_load`, `post_load`)
  - `InitConfigPlugin` (`pre_init_config`, `post_init_config`, `_initialize_omni_kv_cache`)
  - `PrepareInputsPlugin` (实例化)
  - `register` (调用所有初始化器)
  - `_register_kv_connectors`, `_init_cache`, `_init_ox`, `_register_attn_plugins` (含 ImportError 处理)
  - `get_decorators`
- **Mock 目标**: plugin 内部各初始化函数, `DecodeOmniCache` (create=True), `sys.modules`, `os.environ`
- **次要涉及**: `cache/decode/decode_omni_cache.py`

---

### test_prefill_omni_cache.py
- **主要测试**: `cache/prefill/prefill_omni_cache.py` — `PrefillOmniCache`
- **测试方法**: `__init__` (基本/混合注意力/投机 tokens), `calc_cache_shape`, `calculate_kv_xsfer_params`, `initialize_device_cache`, `_construct_device_cache`, `_shard_kv_cache_by_dp_rank`, `get_prefill_prefix_copy_meta`, `init_batch_token_indices`/`_hybrid`, `_layer_name_to_group_and_layer_idx`, `_nd_to_nz`, `get_volatile_metadata`, `synchronize_h2d`
- **Mock 目标**:
  - `cache/core/base.get_tp_group`, `get_dp_group`, `KVCacheMemoryPool`, `resolve_kv_spec_state`
- **次要涉及**: `cache/core/base.py`, `cache/memory/memory_pool.py`, `cache/transfer_engine/shapes.py`, `cache/utils/support.py`, `cache/transfer_engine/manager.py`

---

### test_prefix_copy_ops.py
- **主要测试**: `cache/prefill/prefix_copy_ops.py` — `PrefixCopyMeta`, `compute_prefix_segments`, `get_current_rank_host_data`
- **加载方式**: 通过 `importlib.util.spec_from_file_location` 直接加载源文件 (非常规 import)
- **Mock 目标**: `sys.modules['omni_cache.cache.core.base']` (Mock LOCAL_DP_SIZE=4/1)
- **次要涉及**: `cache/core/base.py` (仅常量 LOCAL_DP_SIZE)

---

### test_runtime_config.py
- **主要测试**: `cache/omni_attention/runtime_config.py`
- **测试函数**: `check_omni_attn_cmd_arg`, `_apply_sink_config`, `_apply_recent_config`, `_apply_beta_config`, `_apply_pattern_config`, `_apply_runtime_config`
- **Mock 目标**: 无 (使用 Mock 对象作为参数传入)
- **次要涉及**: `attention/kv_cache_interface.py` (导入作为类型引用)

---

### test_runtime_patch.py
- **主要测试**: `cache/omni_attention/runtime_patch.py` — `PatchDependencies`, `_load_patch_dependencies`, `_apply_attention_runtime_patch`, `_apply_cache_manager_runtime_patch`
- **Mock 目标**:
  - `attention/kv_managers.OmniKVCacheManager`
  - `attention/kv_managers.OmniKVCacheBlocks`
  - `attention/kv_cache_interface.OmniMultiGroupBlockTable`
  - `attention/kv_cache_interface.get_kv_cache_config_omni_type`
  - `sys.modules` 注入 vllm 桩
- **次要涉及**: `attention/kv_managers/__init__.py`, `attention/kv_cache_interface.py`

---

### test_selection_buffers.py
- **主要测试**: `gather_selection/core/buffers.py` — `SelectionBuffers`, `initialize_selection_buffers`
- **测试方法**: `SelectionBuffers.__init__`, `_is_dsa_enabled`, `initialize_selection_buffers`
- **Mock 目标**:
  - `gather_selection/core/buffers.torch`
  - `gather_selection/core/buffers.os`
  - `SelectionBuffers._is_dsa_enabled`
- **次要涉及**: 无

---

### test_tensor_utils.py
- **主要测试**: `cache/prefill/tensor_utils.py` — `padding_kv_cache`, `nd_to_nz`
- **加载方式**: 通过 lazy import 函数
- **Mock 目标**: 无
- **次要涉及**: 无

---

### test_transfer_engine_buffers.py
- **主要测试**: `cache/transfer_engine/buffers.py`
- **测试类/函数**:
  - `TransferBuffers` (init, `_init_basic_attrs`, `_init_arange_tensors`, `_get_head_size_config`, `_create_cpu_tensor`, `_create_single_buffer`, `_clone_speculation_buffers`, `_attach_to_cache`, `initialize_cpu_buffers`, `initialize_token_indices`)
  - `StreamManager` (init, `initialize_prefill_streams`, `initialize_decode_streams`)
  - `ThreadPoolManager` (init, `initialize`)
- **Mock 目标**: `torch.npu.Stream`
- **次要涉及**: 无

---

### test_utils.py
- **说明**: 空文件/桩文件, 仅包含 import 样板代码, 无实际测试

---

### attention/backends/test_attention_ext.py
- **主要测试**: `attention/backends/attention_ext.py` — `NPUAttentionMetadataBuilderExt` (build, build_for_drafting)
- **加载方式**: 通过 `sys.modules` 注入所有依赖桩后 import (非直接 import)
- **Mock 目标**: `cache` 模块 (omni_cache 属性), `DecodeOmniCache`, `PrefillOmniCache`, omni_npu 相关模块
- **环境变量**: `ENABLE_OMNI_CACHE`, `DISABLE_SWA_MAPPING`, `ENABLE_HOST_MAPPING`
- **次要涉及**: `cache/decode/decode_omni_cache.py`, `cache/prefill/prefill_omni_cache.py`

---

### attention/backends/test_dsa_ext.py
- **主要测试**: `attention/backends/dsa_ext.py` — `NPUDSABackendExt` (reshape_kv_cache), `NPUDSAPrefillMetadataExt`, `NPUDSAMetadataBuilderExt` (build, `_add_prefix_meta`)
- **加载方式**: 通过 `sys.modules` 注入桩
- **Mock 目标**: `cache` 模块, `PrefillOmniCache`, omni_npu/vllm 相关模块
- **环境变量**: `ENABLE_OMNI_CACHE`
- **次要涉及**: `cache/prefill/prefill_omni_cache.py`

---

### attention/backends/test_stride_compress_ext.py
- **主要测试**: `attention/backends/stride_compress_ext.py` — `StridedCompressBuilderExt` (build), `StridedCompressAttentionBackendExt`
- **加载方式**: 通过 `sys.modules` 注入桩
- **Mock 目标**: `cache` 模块, `DecodeOmniCache`, `PrefillOmniCache`, omni_npu/vllm 相关模块
- **环境变量**: `ENABLE_OMNI_CACHE`, `ENABLE_HOST_MAPPING`, `DISABLE_SWA_MAPPING`
- **次要涉及**: `cache/decode/decode_omni_cache.py`, `cache/prefill/prefill_omni_cache.py`

---

### test_attn_plugins/test_implementations.py
- **主要测试**:
  - `attn_plugins/base.py` — `AttentionPlugin`, `AttentionPluginRegistry`, `create_attn_decorator`, 预定义装饰器
  - `attn_plugins/implementations.py` — `CompressMQAAttnPlugin`, `DSAAttnPlugin`
  - `attn_plugins/__init__.py` — `register_omni_cache_plugins`
- **测试内容**:
  - Registry: register, has, get, list_plugins
  - Decorator 包装: pre_attn/post_attn hooks
  - CompressMQAAttnPlugin: omni_cache 属性, enabled, pre_attn (prefill/decode/compression/indexer), post_attn
  - DSAAttnPlugin: omni_cache 属性, enabled, pre_attn (prefill/decode), post_attn (layer_idx, prefix_meta)
- **Mock 目标**:
  - `cache.omni_cache` (create=True)
  - `vllm.forward_context.get_forward_context`
  - `torch.npu.current_stream`, `torch.npu.Event`
  - `attn_plugins.logger`
- **次要涉及**: `cache/__init__.py` (omni_cache 全局变量)

---

## 无测试覆盖的源码文件

以下源码文件目前没有对应的单元测试:

### attention/
- `attention/backends/mla_ext.py`
- `attention/kv_managers/allocator.py`
- `attention/metadata/attention.py`
- `attention/metadata/compress.py`
- `attention/metadata/dsa.py`
- `attention/metadata/hybrid_attention.py`
- `attention/metadata/sliding_window_attention.py`
- `attention/metadata/utils.py`

### cache/
- `cache/core/constants.py`
- `cache/memory/copy_ops.py`
- `cache/memory/hugepage_ops.py`
- `cache/memory/shape_utils.py`
- `cache/memory/constants.py`
- `cache/device_backend/ascend/tensor_register_lib/setup.py`
- `cache/device_backend/ascend/tensor_register_lib/zero.py`
- `cache/device_backend/ascend/ops/triton_ops.py`
- `cache/omni_attention/utils.py`
- `cache/omni_attention/pd.py`
- `cache/transfer_engine/manager.py` (仅被 mock, 未直接测试)
- `cache/transfer_engine/decode.py`
- `cache/transfer_engine/prefill.py`
- `cache/transfer_engine/synchronize.py`

### connector/
- `connector/connector.py` (仅被引用, 未直接测试)
- `connector/zmq_transport.py`
- `connector/utils/process_utils.py`
- `connector/decode/kv_loader.py`
- `connector/decode/process_manager.py`
- `connector/decode/worker.py`
- `connector/prefill/worker.py`
- `connector/scheduler/decode.py`
- `connector/scheduler/prefill.py`

---

# 二、现有测试问题分析

## 问题一览

| # | 类别 | 严重度 | 问题概述 |
|---|------|--------|----------|
| 1 | 重复测试 | 高 | `test_connector_register.py` 与 `test_register_connector.py` 几乎完全重复 |
| 2 | 重复测试 | 高 | `test_cache_init.py` 与 `test_runtime_config.py` + `test_runtime_patch.py` 大面积重叠 |
| 3 | 重复测试 | 中 | `test_omni_cache_base.py` 与 `test_decode_omni_cache.py` + `test_prefill_omni_cache.py` 重叠 |
| 4 | 命名混乱 | 高 | 多个测试文件名与实际测试的源码文件不对应 |
| 5 | 结构扁平 | 高 | 绝大多数测试平铺在 `unit_tests/` 下，未镜像源码目录结构 |
| 6 | 巨石文件 | 中 | 单个测试文件测试 4 个以上源码模块 |
| 7 | 空文件 | 低 | `test_utils.py` 仅 16 行，无任何测试 |
| 8 | 分类错误 | 中 | `test_ox_integration.py` 是集成测试，不应放在 `unit_tests/` 下 |
| 9 | conftest 臃肿 | 高 | conftest.py 1677 行，承担了过多职责 |
| 10 | Mock 对象重复定义 | 中 | `MockAttentionSpec` 等 Mock 类在 3+ 个文件中重复定义，应提取到就近的 utils.py |
| 11 | 脆弱导入 | 低 | `test_prefix_copy_ops.py` 使用硬编码相对路径 importlib 加载 |
| 12 | 跨模块泄漏 | 中 | 测试文件测试了不属于它职责范围的源码 |

---

## 问题 1: 重复测试 — connector/register.py 有两个测试文件

`test_connector_register.py` 和 `test_register_connector.py` 测试**同一源文件** `connector/register.py`，用例几乎完全重复：

| 测试场景 | test_connector_register.py | test_register_connector.py |
|----------|---------------------------|---------------------------|
| import 失败 → 记录 warning | `test_logs_warning_when_factory_import_fails` | `test_safe_register_import_failure` |
| 重复注册 → 跳过 | `test_skips_duplicate_registry_entry` | `test_safe_register_duplicate_skip` |
| 正常注册 | `test_registers_connector_when_missing` | `test_safe_register_success` |
| 注册异常 → 记录 warning | `test_logs_warning_when_registration_raises` | `test_safe_register_register_failure` |
| `register_connectors` 调用 | `test_register_connectors_registers_omni_connector` | `test_register_connectors_calls_safe_register` |

此外 `test_register_connector.py` 第 1 行注释写的是 `# test_register.py`，连文件名都自相矛盾。

---

## 问题 2: 重复测试 — test_cache_init.py 与 runtime_config/runtime_patch 大面积重叠

`test_cache_init.py` 声称测试 `cache/__init__.py`，但实际上大量测试函数与 `test_runtime_config.py` 和 `test_runtime_patch.py` 重复：

| 被测函数 | test_cache_init.py | test_runtime_config.py | test_runtime_patch.py |
|----------|-------------------|----------------------|---------------------|
| `check_omni_attn_cmd_arg` | 有 (3 个用例) | 有 (7 个用例, 更完整) | — |
| `_apply_sink_config` | 有 | 有 | — |
| `_apply_beta_config` | 有 | 有 | — |
| `_apply_pattern_config` | 有 | 有 | — |
| `_apply_runtime_config` | 有 | 有 | — |
| `_apply_attention_runtime_patch` | 有 | — | 有 |
| `_apply_cache_manager_runtime_patch` | 有 | — | 有 |
| `_load_patch_dependencies` | 有 | — | 有 |

`test_cache_init.py` 中只有 `apply_omni_attn_patch` (集成测试) 和 `set_omni_cache` 是独有的，其余全是重复。

---

## 问题 3: 重复测试 — test_omni_cache_base.py 跨模块重叠

`test_omni_cache_base.py` 同时测试了 4 个源码模块，其中 `PrefillOmniCache` 和 `DecodeOmniCache` 的测试与独立测试文件重叠：

- `TestPrefillOmniCache` (test_omni_cache_base.py) ↔ `TestPrefillOmniCacheInit` (test_prefill_omni_cache.py) — 两者都测试 `__init__` 和 `calc_cache_shape`
- `TestDecodeOmniCache` (test_omni_cache_base.py) ↔ `TestDecodeOmniCacheH2D` (test_decode_omni_cache.py) — 两者都测试 `calc_cache_shape`
- `MockAttentionSpec`, `MockKVCacheGroup`, `MockKVCacheConfig`, `MockVllmConfig` 等 Mock 类在**三个文件**中重复定义

---

## 问题 4: 命名混乱

| 当前测试文件名 | 实际测试源码 | 问题 |
|---------------|------------|------|
| `test_acl_tensor_register.py` | `cache/device_backend/ascend/tensor_register.py` | "acl" 前缀自创，路径信息丢失 |
| `test_ascend_acl.py` | `cache/device_backend/ascend/streams.py` + `memcopy.py` | 笼统的 "acl"，看不出测哪个子模块 |
| `test_connector_internal.py` | `connector/utils/settings.py` + `metadata.py` | "internal" 含义不明，且将两个独立模块合并 |
| `test_kv_mem_pool.py` | `cache/memory/memory_pool.py` | 名称不一致 (mem_pool vs memory_pool) |
| `test_gather_selection_utils.py` | `attention/metadata/block_table.py` | **完全误导** — 实际测试的是 attention 模块，与 gather_selection 无关 |
| `test_tensor_utils.py` | `cache/prefill/tensor_utils.py` | 有多个 `*utils.py` 模块，名称有歧义 |
| `test_selection_buffers.py` | `gather_selection/core/buffers.py` | 与兄弟文件命名风格不一致 (应为 `test_gather_selection_buffers.py`) |
| `test_hbm_buffer_utils.py` | `cache/decode/hbm_buffer_utils.py` | 缺少 `decode_` 前缀，可能与其他 buffer 模块混淆 |
| `test_register_connector.py` | `connector/register.py` | 与 `test_connector_register.py` 词序颠倒 |

---

## 问题 5: 目录结构扁平

源码有 5 个顶层包、多层嵌套 (`cache/decode/`, `cache/memory/`, `connector/utils/` 等)，但测试几乎全部平铺：

- **仅 2 个测试用了子目录**: `attention/backends/` 和 `test_attn_plugins/`
- 这两个子目录本身也不一致: `attention/backends/` 镜像了源码路径 (好)，`test_attn_plugins/` 给目录加了 `test_` 前缀 (不一致)
- 其余全部靠 ad-hoc 前缀伪造命名空间 (`gather_selection_`, `connector_`, `transfer_engine_` 等)
- 当两个源文件同名时 (如 `buffers.py` 同时存在于 `gather_selection/core/` 和 `cache/transfer_engine/`)，扁平结构导致必须发明不同的前缀

---

## 问题 6: 巨石测试文件

| 测试文件 | 覆盖的源码模块数 | 涉及模块 |
|----------|-----------------|---------|
| `test_kv_cache_manager.py` | 4 | `blocks.py`, `omni_manager.py`, `attention_manager.py`, `factory.py` |
| `test_omni_cache_base.py` | 4 | `core/base.py`, `utils/ops.py`, `prefill_omni_cache.py`, `decode_omni_cache.py` |
| `test_cache_init.py` | 3 | `cache/__init__.py`, `runtime_config.py`, `runtime_patch.py` |
| `test_connector_internal.py` | 2 | `settings.py`, `metadata.py` |
| `test_ascend_acl.py` | 2 | `streams.py`, `memcopy.py` |

---

## 问题 7: 空文件

`test_utils.py` 仅 16 行，包含 import 样板和一个注释 `# ==================== Global patches ====================`，**无任何测试函数或类**。是死代码。

---

## 问题 8: 集成测试混入单元测试

`test_ox_integration.py` 自述为 `# Integration tests for ox/ module components`，使用 `subprocess`、`tempfile`、`signal` 等，测试的是 ox 二进制 (C++ 编译产物) 的 CLI 行为。它不测试任何 `omni_cache/` Python 源码，不应放在 `unit_tests/` 目录下。此外它引用的路径 `connector/backends/ox/` 在当前源码树中不存在，可能已过时。

---

## 问题 9: conftest.py 过于臃肿 (1677 行)

conftest.py 承担了所有测试的全局 mock 注入，包括：
- `torch.npu` / `torch_npu` 桩 (~100 行)
- `triton` 完整包树 (~130 行)
- `torchair` 桩 (~10 行)
- Ascend ACL (streams, memcopy, tensor_register) 桩 (~185 行)
- `zmq` 桩 (~140 行)
- `torch` 缺失时的完整桩 (~330 行)
- `numpy` 桩 (~16 行)
- `vllm` 大量桩模块 (~470 行) — 包含真实业务逻辑如 `KVCacheSpec.page_size_bytes`、`SingleTypeKVCacheManager` 完整方法、`BlockTable` numpy/torch 实现
- `omni_npu` / `numba` / `zero_copy_npu` 等桩

问题:
- 混合了测试基础设施和类生产级别的类实现
- 所有测试共享同一份全局 mock，修改一处可能影响所有测试
- 难以理解某个测试到底依赖了 conftest 中的哪些部分

---

## 问题 10: Mock 对象重复定义

以下 Mock 类在多个测试文件中近乎相同地重复定义：

- `MockAttentionSpec` — 出现在 `test_omni_cache_base.py`, `test_decode_omni_cache.py`, `test_prefill_omni_cache.py`
- `MockKVCacheGroup` — 同上
- `MockKVCacheConfig` — 同上
- `MockVllmConfig` — 同上
- `MockNPUModelRunner` — 同上

**解决方式**: 提取到就近的 `utils.py` 中 (如 `unit_tests/cache/utils.py`)，测试文件直接 import。

---

## 问题 11: 脆弱的 importlib 导入

`test_prefix_copy_ops.py` 使用硬编码相对路径加载源文件：
```python
importlib.util.spec_from_file_location("prefix_copy_ops", "../cache/prefill/prefix_copy_ops.py")
```
测试文件一旦移动目录就会失败。

---

## 问题 12: 跨模块测试泄漏

- `test_omni_cache_base.py` 包含 `pad_tensor`, `generate_full_block_slot`, `pad_inputs` 的测试 — 这些函数来自 `cache/utils/ops.py`，不属于 `cache/core/base.py`
- `test_gather_selection_utils.py` 实际测试的是 `attention/metadata/block_table.py` 的 `get_block_table_np`，与 gather_selection 毫无关系
- `test_decode_omni_cache.py` 也测试了 `get_block_table_np` (来自 `cache/decode/static_utils.py`)，与 `test_gather_selection_utils.py` 形成跨文件重叠

---

# 三、重构方案

## 核心原则

1. **目录结构镜像源码**: `tests/unit_tests/` 下的目录树与 `omni_cache/` 一一对应
2. **一个源码文件对应一个测试文件**: 消除重复、巨石、跨模块泄漏
3. **就近提供 mock/patch 工具**: 在需要的子目录下放一个 `utils.py` 提供该目录测试所需的公共 mock 和 patch 工具，测试脚本直接 `from .utils import ...` 即可
4. **conftest 分层**: 全局 conftest 只做最基础的桩注入，子目录按需提供自己的 conftest 或 utils

## 目标目录结构

```
tests/
├── conftest.py                          # 全局 pytest 配置: 外部依赖桩注入 (torch_npu, triton, vllm, zmq 等)
│                                        # 从当前 1677 行精简为纯桩注入, 不含业务逻辑
│
├── unit_tests/
│   ├── conftest.py                      # unit_tests 级 fixtures (install_stub_modules 等)
│   ├── test_plugin.py                   # ← plugin.py
│   │
│   ├── attention/
│   │   ├── test_kv_cache_interface.py   # ← attention/kv_cache_interface.py
│   │   ├── backends/
│   │   │   ├── utils.py                 # 该目录公共 mock 工具 (isolated_env fixture, _mock_modules 等)
│   │   │   ├── test_attention_ext.py    # ← (保留)
│   │   │   ├── test_dsa_ext.py          # ← (保留)
│   │   │   └── test_stride_compress_ext.py # ← (保留)
│   │   │   # 待补: test_mla_ext.py
│   │   ├── kv_managers/
│   │   │   ├── utils.py                 # 该目录公共 mock 工具 (MockAttentionSpec, MockKVCacheGroup 等)
│   │   │   ├── test_blocks.py           # ← test_kv_cache_manager.py 中 OmniKVCacheBlocks 部分
│   │   │   ├── test_omni_manager.py     # ← test_kv_cache_manager.py 中 OmniKVCacheManager 部分
│   │   │   ├── test_attention_manager.py # ← test_kv_cache_manager.py 中 OmniAttentionManager 部分
│   │   │   ├── test_factory.py          # ← test_kv_cache_manager.py 中 get_manager_for_kv_cache_spec 部分
│   │   │   └── test_hd_kv_manager.py    # ← (保留, 改路径)
│   │   │   # 待补: test_allocator.py
│   │   └── metadata/
│   │       └── test_block_table.py      # ← test_gather_selection_utils.py (重命名+迁移)
│   │       # 待补: test_attention.py, test_compress.py, test_dsa.py,
│   │       #       test_hybrid_attention.py, test_sliding_window_attention.py, test_utils.py
│   │
│   ├── attn_plugins/
│   │   └── test_implementations.py      # ← test_attn_plugins/test_implementations.py (保留)
│   │
│   ├── cache/
│   │   ├── utils.py                     # cache 级公共 mock 工具 (mock_tp_group, MockVllmConfig 等)
│   │   ├── test_init.py                 # ← test_cache_init.py (仅保留 apply_omni_attn_patch + set_omni_cache)
│   │   ├── core/
│   │   │   ├── test_base.py             # ← test_omni_cache_base.py (仅 BaseOmniCache, create_omni_cache,
│   │   │   │                            #    PrefixCopyMeta, divide_or_raise)
│   │   │   # 待补: test_constants.py
│   │   ├── decode/
│   │   │   ├── test_decode_omni_cache.py # ← (保留, 移除与 test_omni_cache_base 重复的用例)
│   │   │   ├── test_hbm_buffer_utils.py  # ← (保留, 改路径)
│   │   │   ├── test_host_kv_cache_utils.py # ← (保留, 改路径)
│   │   │   └── test_static_utils.py      # ← test_decode_omni_cache.py 中 get_block_table_np 部分独立出来
│   │   ├── prefill/
│   │   │   ├── test_prefill_omni_cache.py # ← (保留, 移除与 test_omni_cache_base 重复的用例)
│   │   │   ├── test_prefix_copy_ops.py    # ← (保留, 修复 importlib 为标准 import)
│   │   │   └── test_tensor_utils.py       # ← (保留, 改路径)
│   │   ├── memory/
│   │   │   └── test_memory_pool.py        # ← test_kv_mem_pool.py (重命名+迁移)
│   │   │   # 待补: test_copy_ops.py, test_hugepage_ops.py, test_shape_utils.py
│   │   ├── device_backend/
│   │   │   └── ascend/
│   │   │       ├── test_tensor_register.py # ← test_acl_tensor_register.py (重命名+迁移)
│   │   │       ├── test_streams.py         # ← test_ascend_acl.py 中 AscendCLStream 部分
│   │   │       └── test_memcopy.py         # ← test_ascend_acl.py 中 aclrtMemLocation/aclrtMemcpyBatchAttr 部分
│   │   │       # 待补: test_triton_ops.py
│   │   ├── omni_attention/
│   │   │   ├── test_runtime_config.py     # ← (保留)
│   │   │   └── test_runtime_patch.py      # ← (保留)
│   │   │   # 待补: test_utils.py, test_pd.py
│   │   ├── transfer_engine/
│   │   │   └── test_buffers.py            # ← test_transfer_engine_buffers.py (重命名+迁移)
│   │   │   # 待补: test_manager.py, test_decode.py, test_prefill.py,
│   │   │   #       test_shapes.py, test_synchronize.py
│   │   └── utils/
│   │       └── test_ops.py                # ← test_omni_cache_base.py 中 pad_tensor/pad_inputs 等拆出
│   │       # 待补: test_support.py
│   │
│   ├── connector/
│   │   ├── test_register.py              # ← 合并 test_connector_register.py + test_register_connector.py
│   │   ├── utils/
│   │   │   ├── test_helpers.py           # ← test_connector_helpers.py
│   │   │   ├── test_settings.py          # ← test_connector_internal.py 中 settings 部分
│   │   │   └── test_metadata.py          # ← test_connector_internal.py 中 metadata 部分
│   │   │   # 待补: test_process_utils.py
│   │   # 待补: test_connector.py, test_zmq_transport.py,
│   │   #       decode/test_kv_loader.py, decode/test_worker.py, decode/test_process_manager.py,
│   │   #       prefill/test_worker.py, scheduler/test_decode.py, scheduler/test_prefill.py
│   │
│   └── gather_selection/
│       ├── core/
│       │   ├── test_gather_selection.py   # ← test_gather_selection_core.py
│       │   └── test_buffers.py            # ← test_selection_buffers.py (重命名+迁移)
│       └── status_updater/
│           └── test_updater.py            # ← test_gather_selection_updater.py
│
├── integration_tests/
│   └── test_ox_integration.py             # ← 从 unit_tests/ 迁出
│
└── benchmarks/                            # (保留现有)
```

## 具体操作步骤

### 第一步: 消除重复 (优先级最高)

| 操作 | 说明 |
|------|------|
| 删除 `test_register_connector.py` | 与 `test_connector_register.py` 完全重复。后者用例更规范 (使用 conftest fixture) |
| 删除 `test_utils.py` | 空文件, 无任何测试 |
| 精简 `test_cache_init.py` | 移除所有与 `test_runtime_config.py` / `test_runtime_patch.py` 重复的用例，仅保留 `apply_omni_attn_patch` (集成) 和 `set_omni_cache` |
| 精简 `test_omni_cache_base.py` | 移除 `TestPrefillOmniCache` 和 `TestDecodeOmniCache` 类 (已在独立文件中测试)；将 `pad_tensor`/`pad_inputs`/`generate_full_block_slot` 等提取到新的 `test_ops.py` |

### 第二步: 拆分巨石文件

| 当前文件 | 拆分为 |
|----------|--------|
| `test_kv_cache_manager.py` | `test_blocks.py`, `test_omni_manager.py`, `test_attention_manager.py`, `test_factory.py` |
| `test_omni_cache_base.py` (精简后) | `cache/core/test_base.py` + `cache/utils/test_ops.py` |
| `test_connector_internal.py` | `connector/utils/test_settings.py` + `connector/utils/test_metadata.py` |
| `test_ascend_acl.py` | `device_backend/ascend/test_streams.py` + `device_backend/ascend/test_memcopy.py` |

### 第三步: 重命名 + 迁移到镜像目录

| 当前路径 | 目标路径 |
|----------|---------|
| `test_acl_tensor_register.py` | `cache/device_backend/ascend/test_tensor_register.py` |
| `test_kv_mem_pool.py` | `cache/memory/test_memory_pool.py` |
| `test_gather_selection_utils.py` | `attention/metadata/test_block_table.py` |
| `test_hbm_buffer_utils.py` | `cache/decode/test_hbm_buffer_utils.py` |
| `test_host_kv_cache_utils.py` | `cache/decode/test_host_kv_cache_utils.py` |
| `test_transfer_engine_buffers.py` | `cache/transfer_engine/test_buffers.py` |
| `test_tensor_utils.py` | `cache/prefill/test_tensor_utils.py` |
| `test_selection_buffers.py` | `gather_selection/core/test_buffers.py` |
| `test_gather_selection_core.py` | `gather_selection/core/test_gather_selection.py` |
| `test_gather_selection_updater.py` | `gather_selection/status_updater/test_updater.py` |
| `test_connector_helpers.py` | `connector/utils/test_helpers.py` |
| `test_hd_kv_manager.py` | `attention/kv_managers/test_hd_kv_manager.py` |
| `test_kv_cache_interface.py` | `attention/test_kv_cache_interface.py` |
| `test_runtime_config.py` | `cache/omni_attention/test_runtime_config.py` |
| `test_runtime_patch.py` | `cache/omni_attention/test_runtime_patch.py` |
| `test_decode_omni_cache.py` | `cache/decode/test_decode_omni_cache.py` |
| `test_prefill_omni_cache.py` | `cache/prefill/test_prefill_omni_cache.py` |
| `test_prefix_copy_ops.py` | `cache/prefill/test_prefix_copy_ops.py` (同时修复 importlib 为标准 import) |
| `test_ox_integration.py` | `../integration_tests/test_ox_integration.py` |
| `test_attn_plugins/test_implementations.py` | `attn_plugins/test_implementations.py` (去掉目录 `test_` 前缀) |

### 第四步: 精简 conftest.py + 就近提供 utils.py

**精简 conftest.py**:
- 当前 1677 行，包含大量类生产级实现 (如 `KVCacheSpec.page_size_bytes`、`SingleTypeKVCacheManager` 完整方法等)
- 精简后只保留外部依赖桩注入 (torch_npu, triton, vllm, zmq, torchair, numba 等)
- 移除其中混入的业务逻辑，改为让测试自行构造

**就近放置 utils.py**:
在需要的子目录下提供 `utils.py`，供该目录下的测试脚本 import:

| utils.py 位置 | 内容 |
|--------------|------|
| `unit_tests/cache/utils.py` | `MockVllmConfig`, `MockKVCacheConfig`, `MockNPUModelRunner`, `mock_tp_group` 等 cache 测试通用 mock (从 test_omni_cache_base / test_decode_omni_cache / test_prefill_omni_cache 三个文件中去重提取) |
| `unit_tests/attention/kv_managers/utils.py` | `MockAttentionSpec`, `MockKVCacheGroup` 等 kv_managers 测试通用 mock |
| `unit_tests/attention/backends/utils.py` | `isolated_env` fixture, `_mock_modules` 字典, `setup_mocks`/`apply_mocks`/`clear_mocks` 工具 (从 3 个 attention backend 测试文件中提取去重) |

测试文件使用方式:
```python
# tests/unit_tests/cache/decode/test_decode_omni_cache.py
from tests.unit_tests.cache.utils import MockVllmConfig, mock_tp_group
```

### 第五步: 修复脆弱导入

| 文件 | 问题 | 修复 |
|------|------|------|
| `test_prefix_copy_ops.py` | `importlib.util.spec_from_file_location` 硬编码相对路径 | 改为标准 `from omni_cache.cache.prefill.prefix_copy_ops import ...`，依赖 conftest 桩解决依赖问题 |

### 第六步: 后期补齐测试 (标记 TODO)

以下源码文件需要新增测试，按优先级排列：

**高优先级** (核心逻辑):
- `cache/transfer_engine/manager.py` — TransferManager 是核心组件, 目前仅被 mock
- `connector/connector.py` — OmniCacheConnector 主连接器, 目前仅被引用
- `cache/utils/support.py` — KVSpecState 等被多处使用
- `cache/decode/static_utils.py` — 目前仅间接测试, 需独立用例
- `attention/kv_managers/allocator.py` — 分配器逻辑

**中优先级** (功能模块):
- `attention/backends/mla_ext.py`
- `cache/omni_attention/utils.py`
- `cache/omni_attention/pd.py`
- `cache/transfer_engine/shapes.py` (目前仅作为 secondary 被测)
- `cache/transfer_engine/decode.py`
- `cache/transfer_engine/prefill.py`
- `cache/transfer_engine/synchronize.py`
- `connector/zmq_transport.py`
- `connector/utils/process_utils.py`

**低优先级** (常量/工具/worker):
- `cache/core/constants.py`
- `cache/memory/copy_ops.py`, `hugepage_ops.py`, `shape_utils.py`, `constants.py`
- `cache/device_backend/ascend/ops/triton_ops.py`
- `cache/device_backend/ascend/tensor_register_lib/setup.py`, `zero.py`
- `connector/decode/kv_loader.py`, `process_manager.py`, `worker.py`
- `connector/prefill/worker.py`
- `connector/scheduler/decode.py`, `prefill.py`
- `attention/metadata/attention.py`, `compress.py`, `dsa.py`, `hybrid_attention.py`, `sliding_window_attention.py`, `utils.py`
