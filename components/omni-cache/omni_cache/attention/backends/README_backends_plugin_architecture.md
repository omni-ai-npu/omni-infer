# Attention Backends 插件化架构设计

## 概述

当前 `omni-npu` 的 attention backends（如 `dsa.py`, `mla.py`）中有直接对 `omni_cache` 的依赖。本方案设计一种插件化架构，让 `omni_cache` 可以实现自己的 attention backends 来扩展或替换 `omni-npu` 的基础实现。

## 当前问题

### 1. dsa.py 中的 omni_cache 依赖

```python
# 直接的环境变量检查和导入
ENABLE_OMNI_CACHE = int(os.getenv("ENABLE_OMNI_CACHE", "0")) == 1

# 在 reshape_kv_cache 中
if ENABLE_OMNI_CACHE and not is_prefill:
    shapes = [(num_blocks, block_size, 1, 128)]  # omni cache 特殊形状

# 在 build 方法中
if ENABLE_OMNI_CACHE:
    from omni_cache.cache import omni_cache
    # ... 大量 omni_cache 相关逻辑
```

### 2. 装饰器的局限

- 装饰器在函数执行前后调用，无法修改函数内部的逻辑分支
- 对于 `reshape_kv_cache` 这样的静态方法，装饰器难以介入
- 在执行过程中进行检查会增加运行时开销

## 设计方案

### 核心思路

**在 `omni-npu` 侧保留基础实现，`omni_cache` 侧提供扩展实现，通过注册机制动态选择使用哪个实现。**

```
┌─────────────────────────────────────────────────────────────┐
│                     vLLM Platform Selector                    │
│                           ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │          Backend Registry (按名称查找 Backend)            │ │
│  │                        ↓                                  │ │
│  │   优先: omni_cache 的扩展 Backend (如果注册)             │ │
│  │   兜底: omni-npu 的基础 Backend                          │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 架构图

```
omni-npu/attention/backends/          omni_cache/attention/backends/
├── __init__.py                       ├── __init__.py
├── dsa.py                           │   # entry point 注册
│   └── NPUDSABackend (基础)         ├── dsa_ext.py
├── mla.py                           │   └── NPUDSABackendExt (扩展，继承基础)
│   └── NPUMLABackend (基础)         └── mla_ext.py
└── utils.py                             └── NPUMLABackendExt (扩展)
    └── register_attention_backend()

                    Entry Points
                        ↓
    [omni.dsa_backend] → omni_cache.attention.backends.dsa_ext:NPUDSABackendExt
    [omni.mla_backend] → omni_cache.attention.backends.mla_ext:NPUMLABackendExt
```

## omni-npu 侧需要做的事情（最小改动）

### 1. 修改 `utils.py` 的注册函数

**原有代码：**
```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.logger import init_logger

logger = init_logger(__name__)
NPU_ATTENTION_BACKEND = {}


def register_attention_backend(backend: str):

    def decorator(cls: type) -> type:
        attn_module = f"{cls.__module__}.{cls.__qualname__}"
        logger.debug("Register attention %s with module %s", backend,
                     attn_module)
        NPU_ATTENTION_BACKEND[backend] = attn_module
        return cls

    return decorator
```

**修改后代码（保持原有变量名和日志格式，仅做最小改动）：**
```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from importlib.metadata import entry_points
from vllm.logger import init_logger

logger = init_logger(__name__)
NPU_ATTENTION_BACKEND = {}  # 保持原有变量名，改为存储类对象以支持插件覆盖


def register_attention_backend(backend: str):
    """保持原有的装饰器接口不变"""
    def decorator(cls: type) -> type:
        attn_module = f"{cls.__module__}.{cls.__qualname__}"
        logger.debug("Register attention %s with module %s", backend,
                     attn_module)
        NPU_ATTENTION_BACKEND[backend] = cls  # 存储类对象，便于后续直接使用
        return cls

    return decorator


def get_attention_backend(name: str) -> type | None:
    """Get a registered attention backend by name."""
    return NPU_ATTENTION_BACKEND.get(name)


def load_plugin_backends():
    """
    Load attention backend plugins from entry points.

    This should be called during module initialization.
    Plugins can override the base backends by registering with the same name.
    """
    eps = entry_points(group="omni.attention_backends")
    for ep in eps:
        try:
            backend_cls = ep.load()
            name = backend_cls.get_name()  # Assume each backend has get_name() method
            attn_module = f"{backend_cls.__module__}.{backend_cls.__qualname__}"
            logger.debug("Register attention plugin %s with module %s", name,
                         attn_module)
            NPU_ATTENTION_BACKEND[name] = backend_cls
            logger.info("Loaded attention backend plugin: %s from %s", name, ep.name)
        except Exception as e:
            logger.warning("Failed to load backend plugin %s: %s", ep.name, e)


def get_available_backends() -> list[str]:
    """List all registered backend names."""
    return list(NPU_ATTENTION_BACKEND.keys())
```

**改动说明：**
1. 保持 `NPU_ATTENTION_BACKEND` 变量名不变
2. 保持原有 `register_attention_backend` 装饰器的接口和日志格式不变
3. 仅将存储值从 `attn_module` 字符串改为 `cls` 类对象（一行改动）
4. 新增 `get_attention_backend()`、`load_plugin_backends()`、`get_available_backends()` 三个函数
5. `load_plugin_backends()` 中使用相同的日志格式记录插件注册信息

### 2. 修改 `__init__.py` 加载插件

```python
# omni_npu/attention/backends/__init__.py

# SPDX-License-Identifier: Apache-2.0
# NPU attention backend shims for vLLM

# 首先导入基础 backends（会被注册）
from omni_npu.attention.backends.attention import (
    NPUAttentionBackendImpl,
    NPUMetadata,
    NPUAttentionBackend,
    NPUAttentionMetadataBuilder,
)
from omni_npu.attention.backends.dsa import NPUDSABackend, NPUDSAMetadata, NPUDSAMetadataBuilder, NPUDSAImpl
from omni_npu.attention.backends.mla import NPUMLABackend, NPUMLAMetadata, NPUMLAMetadataBuilder, NPUMLAImpl
from omni_npu.attention.backends.pangu_hybrid import NPUPanguMomeBackend

# 然后加载插件 backends（可以覆盖基础实现）
from omni_npu.attention.backends.utils import load_plugin_backends
load_plugin_backends()

__all__ = [
    "NPUAttentionBackendImpl",
    "NPUMetadata",
    "NPUAttentionBackend",
    "NPUAttentionMetadataBuilder",
    "NPUPanguMomeBackend",
    "NPUDSABackend",
    "NPUDSAMetadata",
    "NPUDSAMetadataBuilder",
    "NPUDSAImpl",
    "NPUMLABackend",
    "NPUMLAMetadata",
    "NPUMLAMetadataBuilder",
    "NPUMLAImpl",
]
```

### 3. 保留基础实现，去除 ENABLE_OMNI_CACHE 分支

`dsa.py` 和 `mla.py` 中的基础实现去除 `omni_cache` 的特殊逻辑：

```python
# omni_npu/attention/backends/dsa.py

# 去除环境变量检查
# ENABLE_OMNI_CACHE = int(os.getenv("ENABLE_OMNI_CACHE", "0")) == 1  # 删除

@register_attention_backend(NPUDSA)
class NPUDSABackend(MLACommonBackend):
    # ... 基础实现，不含 omni_cache 逻辑 ...

    @staticmethod
    def reshape_kv_cache(...):
        # 基础实现，不含 ENABLE_OMNI_CACHE 分支
        shapes = [(num_blocks, block_size, 1, 512),
                  (num_blocks, block_size, 1, 64),
                  (num_blocks, block_size, 1, 128)]
        # ...


class NPUDSAMetadataBuilder(MLACommonMetadataBuilder[NPUDSAMetadata]):
    def build(self, ...):
        metadata = super().build(...)
        # ... 基础逻辑 ...

        # 去除 ENABLE_OMNI_CACHE 分支，统一设置
        metadata.prefix_meta = None
        return metadata
```

## omni_cache 侧需要做的事情

### 1. 文件结构

```
omni_cache/
├── attention/
│   ├── __init__.py
│   └── backends/
│       ├── __init__.py
│       ├── dsa_ext.py      # DSA 扩展实现
│       └── mla_ext.py      # MLA 扩展实现
├── plugin.py              # 前面已有的装饰器插件
└── pyproject.toml
```

### 2. 创建 `attention/backends/__init__.py`

```python
# omni_cache/attention/backends/__init__.py

import os

# 开关控制：只有当 VLLM_PLUGINS 包含 "omni_cache" 时才启用扩展 backend
# 如果不包含，则不导入扩展实现，omni-npu 会使用基础 backend
if int(os.getenv("ENABLE_OMNI_CACHE", "0")):
    from omni_cache.attention.backends.dsa_ext import NPUDSABackendExt
    from omni_cache.attention.backends.mla_ext import NPUMLABackendExt

    __all__ = [
        "NPUDSABackendExt",
        "NPUMLABackendExt",
    ]
else:
    # 不启用扩展时，不导出任何内容，让 omni-npu 使用基础实现
    __all__ = []
```

### 3. 创建 `dsa_ext.py` 扩展实现

```python
# omni_cache/attention/backends/dsa_ext.py

"""
Extended DSA backend for omni_cache.

This backend extends the base NPUDSABackend to add omni_cache-specific
functionality for APC (Attention Prefix Copy) and other optimizations.

IMPORTANT: This extension is only active when VLLM_PLUGINS env var contains "omni_cache".
Otherwise, the base NPUDSABackend from omni-npu is used.

需要重写的类：
- NPUDSABackend: 重写 reshape_kv_cache 方法
- NPUDSAMetadataBuilder: 重写 build 方法添加 prefix_meta 逻辑
- NPUDSAPrefillMetadata: 添加 prefix_meta 字段

不需要重写的类（直接使用基础实现）：
- NPUDSADecodeMetadata: 扩展不涉及 decode 阶段的元数据修改
- NPUDSAMetadata: 只是容器类，无需修改
- NPUDSAImpl: 实现类，无需修改
"""

import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonPrefillMetadata
from vllm.v1.kv_cache_interface import AttentionSpec

# 导入基础实现
from omni_npu.attention.backends.dsa import (
    NPUDSABackend,
    NPUDSAMetadata,
    NPUDSAMetadataBuilder,
)

from omni_npu.attention.backends.utils import register_attention_backend

from omni_cache.cache import omni_cache
from omni_cache.cache.prefill.prefill_omni_cache import PrefillOmniCache

logger = init_logger(__name__)

# 开关控制：检查是否启用 omni_cache 扩展
ENABLE_OMNI_CACHE = int(os.getenv("ENABLE_OMNI_CACHE", "0"))

if ENABLE_OMNI_CACHE:
    # ========== 1. 重写 Backend 类 ==========
    @register_attention_backend("NPUDSA")  # 使用相同名称覆盖基础实现
    class NPUDSABackendExt(NPUDSABackend):
        """
        Extended DSA backend that adds omni_cache functionality.

        Inherits from NPUDSABackend and overrides:
        - reshape_kv_cache: Special shape for omni_cache decode mode
        """

        @staticmethod
        def reshape_kv_cache(
            raw_tensor: torch.Tensor,
            num_blocks: int,
            kv_cache_spec: AttentionSpec,
        ) -> Tuple[torch.Tensor, ...]:
            """
            Reshape KV cache with omni_cache-specific shapes for decode mode.
            """
            is_prefill = (get_current_vllm_config().kv_transfer_config.kv_role != "kv_consumer")
            block_size = kv_cache_spec.block_size
            dtype = kv_cache_spec.dtype
            raw_tensor = raw_tensor.view(dtype=dtype)

            # omni_cache decode mode: only indexer is registered on device
            if not is_prefill:
                shapes = [(num_blocks, block_size, 1, 128)]
            else:
                # Prefill mode: use standard shapes
                shapes = [
                    (num_blocks, block_size, 1, 512),
                    (num_blocks, block_size, 1, 64),
                    (num_blocks, block_size, 1, 128)
                ]

            sizes = [math.prod(shape) for shape in shapes]
            if raw_tensor.numel() != sum(sizes):
                raise RuntimeError(
                    f"Raw tensor has {raw_tensor.numel()} elements, while "
                    f"the expected sizes for KV cache are {sizes}."
                )
            tensors = torch.split(raw_tensor, sizes)
            return tuple(t.view(shape) for t, shape in zip(tensors, shapes))

    # ========== 2. 重写 PrefillMetadata 类（添加新字段）==========
    @dataclass
    class NPUDSAPrefillMetadataExt(MLACommonPrefillMetadata):
        """Extended prefill metadata with prefix_meta field for APC."""
        query_cumlens: torch.Tensor = None
        seq_lens: torch.Tensor = None
        prefix_meta: Optional[Any] = None  # omni_cache 新增字段

    # ========== 3. 重写 MetadataBuilder 类 ==========
    class NPUDSAMetadataBuilderExt(NPUDSAMetadataBuilder):
        """
        Extended metadata builder that adds omni_cache APC functionality.

        Overrides the build() method to add prefix_meta handling.
        """

        def __init__(
            self,
            kv_cache_spec: AttentionSpec,
            layer_names: list[str],
            vllm_config: VllmConfig,
            device: torch.device,
        ):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            # 使用扩展的 prefill metadata 类
            self.prefill_metadata_cls = NPUDSAPrefillMetadataExt

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata,
            fast_build: bool = False,
        ):
            # 调用基础实现
            metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

            # 添加 omni_cache 特有逻辑：设置 prefix_meta
            if metadata.prefill is not None and self.vllm_config.kv_transfer_config is not None:
                self._add_prefix_meta(metadata, common_attn_metadata)

            return metadata

        def _add_prefix_meta(self, metadata, common_attn_metadata):
            """Add prefix metadata for APC operations."""
            num_reqs = metadata.num_reqs
            query_start_loc = common_attn_metadata.query_start_loc_cpu

            query_seq_lens_cpu = query_start_loc[1:] - query_start_loc[:-1]
            query_lens = metadata.prefill.query_start_loc[1:] - metadata.prefill.query_start_loc[:-1]
            query_lens_list = query_lens.tolist()
            num_computed_tokens_cpu = common_attn_metadata.seq_lens_cpu - query_seq_lens_cpu

            prefix_meta = omni_cache.get_prefill_prefix_copy_meta(
                self.vllm_config,
                kv_lens=num_computed_tokens_cpu[:num_reqs],
                query_lens_list=query_lens_list,
                block_tables=common_attn_metadata.block_table_tensor.cpu().numpy()[:num_reqs],
            )
            omni_cache.synchronize_h2d(
                prefix_meta=prefix_meta,
                layer_idx=0,
            )
            metadata.prefix_meta = prefix_meta

            # Initialize batch token indices if needed
            if isinstance(omni_cache, PrefillOmniCache):
                omni_cache.init_batch_token_indices(common_attn_metadata.slot_mapping)

    logger.info("omni_cache extension enabled, NPUDSABackendExt registered")

else:
    # 不启用扩展时，导出基础实现（确保模块可导入）
    NPUDSABackendExt = NPUDSABackend
    NPUDSAMetadataBuilderExt = NPUDSAMetadataBuilder
    logger.info("omni_cache extension disabled, using base NPUDSABackend")
```

**重写说明：**
- 只重写需要修改的类和方法
- `NPUDSADecodeMetadata`、`NPUDSAMetadata`、`NPUDSAImpl` 不需要重写，直接使用基础实现
- 通过继承机制，未重写的方法自动使用基础实现

### 4. 创建 `mla_ext.py` 扩展实现

```python
# omni_cache/attention/backends/mla_ext.py

"""
Extended MLA backend for omni_cache.

Currently just inherits from the base implementation.
Can be extended in the future for omni_cache-specific optimizations.

Otherwise, the base NPUMLABackend from omni-npu is used.

不需要重写的类（直接使用基础实现）：
- NPUMLAMetadata: 容器类
- NPUMLAMetadataBuilder: 无需修改
- NPUMLAImpl: 实现类
"""

import os

from vllm.logger import init_logger

from omni_npu.attention.backends.mla import NPUMLABackend
from omni_npu.attention.backends.utils import register_attention_backend

logger = init_logger(__name__)

# 开关控制：检查是否启用 omni_cache 扩展
ENABLE_OMNI_CACHE = int(os.getenv("ENABLE_OMNI_CACHE", "0"))

if ENABLE_OMNI_CACHE:
    @register_attention_backend("NPUMLA")  # 使用相同名称覆盖基础实现
    class NPUMLABackendExt(NPUMLABackend):
        """Extended MLA backend. Currently same as base."""
        pass

    logger.info("omni_cache extension enabled, NPUMLABackendExt registered")
else:
    # 不启用扩展时，导出基础实现（确保模块可导入）
    NPUMLABackendExt = NPUMLABackend
    logger.info("omni_cache extension disabled, using base NPUMLABackend")
```

## Entry Points 注册

在 `omni_cache/pyproject.toml` 中：

```toml
[project.entry-points."omni.attention_backends"]
# 注意：使用 backend 名称而非模块名
# 扩展 backend 使用与基础 backend 相同的名称，会覆盖基础实现
dsa-ext = "omni_cache.attention.backends.dsa_ext:NPUDSABackendExt"
mla-ext = "omni_cache.attention.backends.mla_ext:NPUMLABackendExt"

# 同时保留装饰器插件的注册
[project.entry-points."omni.load_model_decorators"]
omni-cache = "omni_cache.plugin:LoadModelPlugin"

[project.entry-points."omni.init_config_decorators"]
omni-cache = "omni_cache.plugin:InitConfigPlugin"

[project.entry-points."omni.build_metadata_decorators"]
omni-cache = "omni_cache.plugin:BuildMetadataPlugin"
```

## 工作流程

### 初始化阶段

```
omni_npu.attention.backends 模块加载
    ↓
导入基础 backends (dsa.py, mla.py)
    ↓
基础 backends 注册到 NPU_ATTENTION_BACKEND
    ↓
调用 load_plugin_backends()
    ↓
从 entry points 加载 omni_cache 的扩展 backends
    ↓
检查 VLLM_PLUGINS 环境变量是否包含 "omni_cache"
    ↓
┌─────────────────────────────────────────────────────────────┐
│  VLLM_PLUGINS 包含 "omni_cache"    │  VLLM_PLUGINS 不包含   │
│              ↓                      │         ↓             │
│  扩展 backends 注册并覆盖基础实现    │  使用基础 backends     │
│  (NPUDSABackendExt, NPUMLABackendExt)│  (NPUDSABackend, etc) │
└─────────────────────────────────────────────────────────────┘
    ↓
vLLM 使用注册表中的 backend
```

### 运行时

**启用 omni_cache 扩展时：**
```
vLLM 需要 DSA backend
    ↓
从注册表查找 "NPUDSA"
    ↓
返回 NPUDSABackendExt (omni_cache 的扩展实现)
    ↓
调用 reshape_kv_cache: 使用 omni_cache 特殊形状
    ↓
调用 build: 包含 prefix_meta 逻辑
```

**禁用 omni_cache 扩展时：**
```
vLLM 需要 DSA backend
    ↓
从注册表查找 "NPUDSA"
    ↓
返回 NPUDSABackend (omni-npu 的基础实现)
    ↓
调用 reshape_kv_cache: 使用标准形状
    ↓
调用 build: 不包含 prefix_meta 逻辑
```

## 优势

1. **完全解耦**: `omni-npu` 不包含任何 `omni_cache` 特殊逻辑
2. **最小侵入**: `omni-npu` 只需要修改 `__init__.py` 和 `utils.py`
3. **开关控制**: 通过 `ENABLE_OMNI_CACHE` 环境变量控制是否启用扩展，便于调试和回退
4. **类型安全**: 继承机制保证接口一致性
5. **灵活扩展**: `omni_cache` 可以选择性覆盖任意方法
6. **无运行时开销**: 加载时确定使用哪个 backend，运行时无分支判断
7. **环境变量无关**: 扩展 backend 内部决定何时使用特殊逻辑

## 与装饰器方案的比较

| 方面 | 装饰器方案 | Backend 继承方案 |
|------|-----------|------------------|
| 解耦程度 | 部分解耦（需保留环境变量检查） | 完全解耦 |
| 运行时开销 | 每次调用检查环境变量 | 无 |
| 静态方法处理 | 困难 | 可覆盖 |
| 修改范围 | 需修改多处代码 | 只修改 utils.py 和 __init__.py |
| 类型安全 | 一般 | 好（继承保证） |
| 扩展灵活性 | 钩子函数 | 可覆盖任意方法 |
| 开关控制 | 环境变量控制 | `ENABLE_OMNI_CACHE` 环境变量控制 |

## 两种方案共存

建议两种方案同时使用：

1. **Backend 继承方案**: 处理 `reshape_kv_cache` 等静态方法、核心逻辑
2. **装饰器方案**: 处理 `load_model`、`initialize_from_config` 等非 backend 相关的扩展

## 故障排查

### 检查开关状态

```python
import os
ENABLE_OMNI_CACHE = int(os.getenv("ENABLE_OMNI_CACHE", "0"))
```

### Backend 未被覆盖

```python
from omni_npu.attention.backends.utils import get_attention_backend, get_available_backends

print(get_available_backends())  # 查看所有注册的 backends
backend = get_attention_backend("NPUDSA")
print(backend)  # 启用扩展时应显示 NPUDSABackendExt，禁用时显示 NPUDSABackend
```

### 导入错误

```bash
# 检查 entry point
python -c "from importlib.metadata import entry_points; eps = entry_points(group='omni.attention_backends'); print([ep.name for ep in eps])"
```

## 总结

通过 Backend 继承方案：

1. `omni-npu` 保留纯粹的基础实现
2. `omni_cache` 通过继承扩展功能并注册覆盖
3. 运行时无分支判断开销
4. 类型安全、易于维护
5. `omni_cache` 侧改动集中在一个目录

---

## NPU Attention Backend (通用) 扩展方案

### 概述

`attention.py` 中的 `NPUAttentionBackend` 是一个通用的 attention backend 实现，当前存在对 `omni_cache` 的直接依赖，主要体现在 `NPUAttentionMetadataBuilder` 的 `build()` 和 `build_for_drafting()` 方法中。

### 当前问题

#### 1. NPUAttentionMetadataBuilder 中的 omni_cache 依赖

```python
# 在 build 方法中
def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False) -> NPUMetadata:
    from omni_npu.v1.models.config_loader.loader import model_extra_config
    if model_extra_config.operator_opt_config.use_omni_cache:
        from omni_cache.cache import omni_cache
        from omni_cache.cache.decode.decode_omni_cache import DecodeOmniCache
        if isinstance(omni_cache, DecodeOmniCache) and (not int(os.getenv("DISABLE_SWA_MAPPING", "0"))) and int(os.getenv("ENABLE_HOST_MAPPING", "1")):
            return omni_cache._construct_fake_attn_metatata(self, common_attn_metadata)
    # ... 基础实现 ...
    if model_extra_config.operator_opt_config.use_omni_cache:
        from omni_cache.cache import omni_cache
        from omni_cache.cache.prefill.prefill_omni_cache import PrefillOmniCache
        if isinstance(omni_cache, PrefillOmniCache):
            omni_cache.init_batch_token_indices_hybrid(slot_mapping)
    return attn_metadata

# 在 build_for_drafting 方法中
def build_for_drafting(self, common_attn_metadata: CommonAttentionMetadata, draft_index: int) -> M:
    from omni_npu.v1.models.config_loader.loader import model_extra_config
    if model_extra_config.operator_opt_config.use_omni_cache:
        from omni_cache.cache import omni_cache
        from omni_cache.cache.decode.decode_omni_cache import DecodeOmniCache
        if isinstance(omni_cache, DecodeOmniCache) and (not int(os.getenv("DISABLE_SWA_MAPPING", "0"))) and int(os.getenv("ENABLE_HOST_MAPPING", "1")):
            fake_meta = omni_cache._construct_fake_attn_metatata(self, common_attn_metadata, draft_index)
            return fake_meta
    return self.build(...)
```

#### 2. 类结构分析

```
┌─────────────────────────────────────────────────────────────────┐
│  NPUAttentionBackend 类结构                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NPUAttentionBackend (核心 Backend 类)                          │
│  ├── get_name()           → 返回 "VLLM_NPU_ATTN"               │
│  ├── get_metadata_cls()   → 返回 NPUMetadata                   │
│  ├── get_builder_cls()    → 返回 NPUAttentionMetadataBuilder   │
│  ├── get_impl_cls()       → 返回 NPUAttentionBackendImpl       │
│  └── reshape_kv_cache()   → KV cache 形状定义 [静态方法]        │
│                                                                 │
│  NPUMetadata (元数据容器)                                        │
│  └── 存储 attention 计算所需的元数据信息                         │
│                                                                 │
│  NPUAttentionMetadataBuilder (元数据构建器) ← 需要重写           │
│  ├── build()              → 构建通用 attention 元数据           │
│  └── build_for_drafting() → 构建 draft 模式的元数据             │
│                                                                 │
│  NPUAttentionBackendImpl (实际实现类)                            │
│  └── forward() → 执行 attention 计算                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 解决方案设计

#### omni-npu 侧改动（最小化）

##### 1. 修改 `attention.py` 去除 omni_cache 依赖

**修改 `build()` 方法：**

```python
# omni_npu/attention/backends/attention.py

def build(self,
          common_prefix_len: int,
          common_attn_metadata: CommonAttentionMetadata,
          fast_build: bool = False) -> NPUMetadata:
    # 去除 omni_cache 分支，保留纯基础实现
    num_actual_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc
    seq_lens = common_attn_metadata.seq_lens
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping
    max_query_len = common_attn_metadata.max_query_len
    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold,
        )
    )
    attn_metadata = NPUMetadata(
        num_actual_tokens=num_actual_tokens,
        block_tables=block_table,
        query_start_loc=query_start_loc.tolist(),
        seq_lens=seq_lens.tolist(),
        max_query_len=max_query_len,
        slot_mapping=slot_mapping,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        decode_threshold=self.reorder_batch_threshold
    )
    return attn_metadata
```

**修改 `build_for_drafting()` 方法：**

```python
def build_for_drafting(
    self,
    common_attn_metadata: CommonAttentionMetadata,
    draft_index: int,
) -> M:
    """Build attention metadata for draft model. Uses build by default."""
    return self.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
        fast_build=True,
    )
```

##### 2. 修改 `utils.py` 添加插件加载函数

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from importlib.metadata import entry_points
from vllm.logger import init_logger

logger = init_logger(__name__)
NPU_ATTENTION_BACKEND = {}  # 存储类对象


def register_attention_backend(backend: str):
    """装饰器：注册 attention backend"""
    def decorator(cls: type) -> type:
        attn_module = f"{cls.__module__}.{cls.__qualname__}"
        logger.debug("Register attention %s with module %s", backend, attn_module)
        NPU_ATTENTION_BACKEND[backend] = cls  # 存储类对象
        return cls
    return decorator


def get_attention_backend(name: str) -> type | None:
    """Get a registered attention backend by name."""
    return NPU_ATTENTION_BACKEND.get(name)


def load_plugin_backends():
    """
    Load attention backend plugins from entry points.
    Plugins can override the base backends by registering with the same name.
    """
    eps = entry_points(group="omni.attention_backends")
    for ep in eps:
        try:
            backend_cls = ep.load()
            name = backend_cls.get_name()
            attn_module = f"{backend_cls.__module__}.{backend_cls.__qualname__}"
            logger.debug("Register attention plugin %s with module %s", name, attn_module)
            NPU_ATTENTION_BACKEND[name] = backend_cls
            logger.info("Loaded attention backend plugin: %s from %s", name, ep.name)
        except Exception as e:
            logger.warning("Failed to load backend plugin %s: %s", ep.name, e)


def get_available_backends() -> list[str]:
    """List all registered backend names."""
    return list(NPU_ATTENTION_BACKEND.keys())
```

##### 3. 修改 `__init__.py` 加载插件

```python
# omni_npu/attention/backends/__init__.py

# SPDX-License-Identifier: Apache-2.0
# NPU attention backend shims for vLLM

# 首先导入基础 backends
from omni_npu.attention.backends.attention import (
    NPUAttentionBackendImpl,
    NPUMetadata,
    NPUAttentionBackend,
    NPUAttentionMetadataBuilder,
)
from omni_npu.attention.backends.pangu_hybrid import NPUPanguMomeBackend

# 然后加载插件 backends（可以覆盖基础实现）
from omni_npu.attention.backends.utils import load_plugin_backends
load_plugin_backends()

__all__ = [
    "NPUAttentionBackendImpl",
    "NPUMetadata",
    "NPUAttentionBackend",
    "NPUAttentionMetadataBuilder",
    "NPUPanguMomeBackend",
]
```

#### omni_cache 侧扩展实现

##### 1. 文件结构

```
omni_cache/
├── attention/
│   ├── __init__.py
│   └── backends/
│       ├── __init__.py
│       ├── attention_ext.py   # NPUAttentionBackend 扩展实现
│       ├── dsa_ext.py         # DSA 扩展实现（已存在）
│       └── mla_ext.py         # MLA 扩展实现（已存在）
├── plugin.py
└── pyproject.toml
```

##### 2. 创建 `attention_ext.py` 扩展实现

```python
# omni_cache/attention/backends/attention_ext.py

"""
Extended NPUAttentionBackend for omni_cache.

This backend extends the base NPUAttentionBackend to add omni_cache-specific
functionality for metadata construction in decode and drafting modes.

IMPORTANT: This extension is only active when VLLM_PLUGINS env var contains "omni_cache".
Otherwise, the base NPUAttentionBackend from omni-npu is used.

需要重写的类：
- NPUAttentionMetadataBuilder: 重写 build 和 build_for_drafting 方法
- NPUMetadata: 可能需要添加新字段（如果 omni_cache 需要）

不需要重写的类（直接使用基础实现）：
- NPUAttentionBackend: Backend 类本身无需修改
- NPUAttentionBackendImpl: 实现类无需修改
"""

import os
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata

# 导入基础实现
from omni_npu.attention.backends.attention import (
    NPUAttentionMetadataBuilder,
    NPUMetadata,
)
from omni_npu.attention.backends.utils import register_attention_backend

logger = init_logger(__name__)

# 开关控制：检查是否启用 omni_cache 扩展
ENABLE_OMNI_CACHE = int(os.getenv("ENABLE_OMNI_CACHE", "0"))

VLLM_NPU_ATTN = "VLLM_NPU_ATTN"

if ENABLE_OMNI_CACHE:
    # ========== 重写 MetadataBuilder 类 ==========
    class NPUAttentionMetadataBuilderExt(NPUAttentionMetadataBuilder):
        """
        Extended metadata builder that adds omni_cache functionality.

        Overrides:
        - build(): Adds omni_cache metadata construction for decode mode
        - build_for_drafting(): Adds omni_cache metadata construction for drafting
        """

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> NPUMetadata:
            """
            Build attention metadata with omni_cache support.

            When omni_cache is enabled and in decode mode with appropriate
            settings, returns the fake attention metadata from omni_cache.
            Otherwise, falls back to base implementation.
            """
            from omni_npu.v1.models.config_loader.loader import model_extra_config

            # omni_cache 特殊处理：decode 模式下的 fake metadata
            if model_extra_config.operator_opt_config.use_omni_cache:
                from omni_cache.cache import omni_cache
                from omni_cache.cache.decode.decode_omni_cache import DecodeOmniCache

                if (isinstance(omni_cache, DecodeOmniCache) and
                    (not int(os.getenv("DISABLE_SWA_MAPPING", "0"))) and
                    int(os.getenv("ENABLE_HOST_MAPPING", "1"))):
                    return omni_cache._construct_fake_attn_metatata(
                        self, common_attn_metadata
                    )

            # 调用基础实现
            metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

            # omni_cache prefill 模式下的初始化
            if model_extra_config.operator_opt_config.use_omni_cache:
                from omni_cache.cache import omni_cache
                from omni_cache.cache.prefill.prefill_omni_cache import PrefillOmniCache

                if isinstance(omni_cache, PrefillOmniCache):
                    omni_cache.init_batch_token_indices_hybrid(metadata.slot_mapping)

            return metadata

        def build_for_drafting(
            self,
            common_attn_metadata: CommonAttentionMetadata,
            draft_index: int,
        ):
            """
            Build attention metadata for draft model with omni_cache support.

            When omni_cache is enabled in decode mode, returns the fake
            metadata from omni_cache for drafting.
            Otherwise, falls back to base implementation.
            """
            from omni_npu.v1.models.config_loader.loader import model_extra_config

            # omni_cache 特殊处理
            if model_extra_config.operator_opt_config.use_omni_cache:
                from omni_cache.cache import omni_cache
                from omni_cache.cache.decode.decode_omni_cache import DecodeOmniCache

                if (isinstance(omni_cache, DecodeOmniCache) and
                    (not int(os.getenv("DISABLE_SWA_MAPPING", "0"))) and
                    int(os.getenv("ENABLE_HOST_MAPPING", "1"))):

                    import time
                    st = time.time()
                    fake_meta = omni_cache._construct_fake_attn_metatata(
                        self, common_attn_metadata, draft_index
                    )
                    duration = time.time() - st
                    logger.debug(f"<<< DEBUG metadata in attention_ext.py cost: {duration}")
                    return fake_meta

            # 调用基础实现
            return super().build_for_drafting(common_attn_metadata, draft_index)

    logger.info("omni_cache extension enabled, NPUAttentionMetadataBuilderExt registered")

else:
    # 不启用扩展时，导出基础实现（确保模块可导入）
    NPUAttentionMetadataBuilderExt = NPUAttentionMetadataBuilder
    logger.info("omni_cache extension disabled, using base NPUAttentionMetadataBuilder")
```

##### 3. 更新 `attention/backends/__init__.py`

```python
# omni_cache/attention/backends/__init__.py

import os

if int(os.getenv("ENABLE_OMNI_CACHE", "0")):
    from omni_cache.attention.backends.attention_ext import NPUAttentionMetadataBuilderExt
    from omni_cache.attention.backends.dsa_ext import NPUDSABackendExt
    from omni_cache.attention.backends.mla_ext import NPUMLABackendExt

    __all__ = [
        "NPUAttentionMetadataBuilderExt",
        "NPUDSABackendExt",
        "NPUMLABackendExt",
    ]
else:
    # 不启用扩展时，不导出任何内容，让 omni-npu 使用基础实现
    __all__ = []
```

##### 4. 配置 Entry Points

```toml
# omni_cache/pyproject.toml

[project.entry-points."omni.attention_backends"]
# NPU Attention Backend 扩展
attention-ext = "omni_cache.attention.backends.attention_ext:NPUAttentionMetadataBuilderExt"
# DSA 扩展（已存在）
dsa-ext = "omni_cache.attention.backends.dsa_ext:NPUDSABackendExt"
# MLA 扩展（已存在）
mla-ext = "omni_cache.attention.backends.mla_ext:NPUMLABackendExt"

# 同时保留装饰器插件的注册
[project.entry-points."omni.load_model_decorators"]
omni-cache = "omni_cache.plugin:LoadModelPlugin"

[project.entry-points."omni.init_config_decorators"]
omni-cache = "omni_cache.plugin:InitConfigPlugin"
```

### 工作流程

#### 初始化阶段

```
omni_npu.attention.backends 模块加载
    ↓
导入基础 backends (attention.py)
    ↓
NPUAttentionBackend 注册到 NPU_ATTENTION_BACKEND
    ↓
调用 load_plugin_backends()
    ↓
从 entry points 加载 omni_cache 的扩展
    ↓
检查 ENABLE_OMNI_CACHE 环境变量
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  ENABLE_OMNI_CACHE=1                │  ENABLE_OMNI_CACHE=0      │
│              ↓                      │         ↓                 │
│  NPUAttentionMetadataBuilderExt    │  使用基础 MetadataBuilder │
│  覆盖基础实现                       │                           │
└─────────────────────────────────────────────────────────────────┘
    ↓
vLLM 使用注册表中的 backend
```

#### 运行时

**启用 omni_cache 扩展时：**
```
vLLM 需要 attention metadata
    ↓
使用 NPUAttentionMetadataBuilderExt
    ↓
调用 build():
    - 检查 use_omni_cache 和 DecodeOmniCache
    - 满足条件时返回 omni_cache 的 fake metadata
    - 否则调用 super().build() 获取基础结果
```

**禁用 omni_cache 扩展时：**
```
vLLM 需要 attention metadata
    ↓
使用 NPUAttentionMetadataBuilder (基础实现)
    ↓
调用 build(): 纯基础逻辑，无 omni_cache 分支
```

### 重写方法对照表

| 需求场景 | 需要重写的类/方法 | 说明 |
|---------|------------------|------|
| 修改 metadata 构建逻辑 | `MetadataBuilder.build()` | 先检查 omni_cache 条件，再调用 `super().build()` |
| 修改 draft 模式 metadata | `MetadataBuilder.build_for_drafting()` | 类似 build() 的处理方式 |
| 添加新元数据字段 | `NPUMetadata` | 可选，使用 `@dataclass` 继承 |
| 修改 KV cache 形状 | `Backend.reshape_kv_cache()` | 如果需要，直接覆盖静态方法 |

### 与 DSA/MLA Backend 的区别

| 方面 | DSA/MLA Backend | NPU Attention Backend |
|------|-----------------|----------------------|
| 主要重写 | `reshape_kv_cache()`, `MetadataBuilder.build()` | `MetadataBuilder.build()`, `build_for_drafting()` |
| 重写层级 | Backend 类 + MetadataBuilder | 仅 MetadataBuilder |
| 特殊逻辑 | KV cache 形状、prefix_meta | fake metadata 构造、drafting 支持 |
| omni_cache 功能 | APC (Attention Prefix Copy) | Decode/Prefill 模式下的特殊 metadata |

---

## 开发指南：如何实现新的 Attention Backend 扩展

### 1. 理解 Backend 类结构

一个完整的 attention backend 通常包含以下类：

```
┌─────────────────────────────────────────────────────────────────┐
│  Backend 类结构                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NPUDSABackend (核心 Backend 类)                                │
│  ├── get_name()           → 返回 backend 名称                   │
│  ├── get_metadata_cls()   → 返回 Metadata 类                    │
│  ├── get_builder_cls()    → 返回 MetadataBuilder 类             │
│  ├── get_impl_cls()       → 返回 Impl 类                        │
│  └── reshape_kv_cache()   → KV cache 形状定义 [静态方法]         │
│                                                                 │
│  NPUDSAPrefillMetadata (Prefill 阶段元数据)                      │
│  └── 存储预填充阶段的元数据信息                                   │
│                                                                 │
│  NPUDSADecodeMetadata (Decode 阶段元数据)                        │
│  └── 存储解码阶段的元数据信息                                     │
│                                                                 │
│  NPUDSAMetadata (总元数据容器)                                   │
│  └── 包含 prefill 和 decode 元数据                               │
│                                                                 │
│  NPUDSAMetadataBuilder (元数据构建器)                            │
│  └── build() → 构建完整的 attention 元数据                       │
│                                                                 │
│  NPUDSAImpl (实际实现类)                                         │
│  └── forward() → 执行 attention 计算                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 确定需要重写的类

| 需求场景 | 需要重写的类 | 说明 |
|---------|-------------|------|
| 修改 KV cache 形状 | `Backend.reshape_kv_cache()` | 静态方法，直接覆盖 |
| 添加新的元数据字段 | `PrefillMetadata` 或 `DecodeMetadata` | 使用 `@dataclass` 继承 |
| 修改元数据构建逻辑 | `MetadataBuilder.build()` | 调用 `super().build()` 后添加逻辑 |
| 修改 attention 计算 | `Impl.forward()` | 覆盖整个 forward 方法 |
| 仅注册覆盖（无修改） | `Backend` 类 | 空类继承即可 |

**不需要重写的类：**
- `Metadata`: 只是容器类，除非需要更改泛型类型
- `DecodeMetadata`: 如果不涉及 decode 阶段的修改
- `Impl`: 如果不涉及 attention 计算逻辑的修改

### 3. 实现步骤

#### 步骤 1：创建扩展文件

```python
# omni_cache/attention/backends/xxx_ext.py

import os
from vllm.logger import init_logger

# 导入基础实现（只导入需要的类）
from omni_npu.attention.backends.xxx import (
    NPUXXXBackend,
    NPUXXXMetadataBuilder,  # 如果需要重写
)
from omni_npu.attention.backends.utils import register_attention_backend

logger = init_logger(__name__)
ENABLE_OMNI_CACHE = "omni_cache" in os.environ.get("VLLM_PLUGINS", "")
```

#### 步骤 2：实现开关控制

```python
if ENABLE_OMNI_CACHE:
    # 扩展实现
    @register_attention_backend("NPUXXX")  # 使用相同名称覆盖
    class NPUXXXBackendExt(NPUXXXBackend):
        # 重写需要的方法
        pass

    logger.info("omni_cache extension enabled, NPUXXXBackendExt registered")
else:
    # 退化为基础实现
    NPUXXXBackendExt = NPUXXXBackend
    logger.info("omni_cache extension disabled, using base NPUXXXBackend")
```

#### 步骤 3：重写特定方法

**重写静态方法：**
```python
class NPUXXXBackendExt(NPUXXXBackend):
    @staticmethod
    def reshape_kv_cache(raw_tensor, num_blocks, kv_cache_spec):
        # 自定义实现
        ...
        return result
```

**重写实例方法：**
```python
class NPUXXXMetadataBuilderExt(NPUXXXMetadataBuilder):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        # 先调用基础实现
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

        # 添加扩展逻辑
        metadata.new_field = self._compute_new_field(metadata)

        return metadata

    def _compute_new_field(self, metadata):
        # 辅助方法
        ...
```

**添加新的元数据字段：**
```python
from dataclasses import dataclass
from vllm.v1.attention.backends.mla.common import MLACommonPrefillMetadata

@dataclass
class NPUDSAPrefillMetadataExt(MLACommonPrefillMetadata):
    # 新增字段
    new_field: Optional[Any] = None
    another_field: torch.Tensor = None
```

#### 步骤 4：注册到 `__init__.py`

```python
# omni_cache/attention/backends/__init__.py

import os

if int(os.getenv("ENABLE_OMNI_CACHE", "0")):
    from omni_cache.attention.backends.xxx_ext import NPUXXXBackendExt
    __all__ = ["NPUXXXBackendExt"]
else:
    __all__ = []
```

#### 步骤 5：配置 Entry Points

```toml
# omni_cache/pyproject.toml

[project.entry-points."omni.attention_backends"]
xxx-ext = "omni_cache.attention.backends.xxx_ext:NPUXXXBackendExt"
```

### 4. 最佳实践

1. **最小化重写**：只重写真正需要修改的类和方法
2. **调用 super()**：在重写方法中先调用 `super().method()` 获取基础结果
3. **保持接口一致**：重写的方法签名应与基础类保持一致
4. **添加日志**：在扩展启用/禁用时记录日志，便于调试
5. **文档注释**：说明重写了哪些方法、为什么重写

### 5. 常见问题

**Q: 为什么我的扩展没有生效？**

A: 检查以下几点：
1. `ENABLE_OMNI_CACHE` 环境变量
2. entry points 是否正确配置
3. `@register_attention_backend()` 装饰器中的名称是否与基础 backend 一致

**Q: 如何只重写 decode 阶段的逻辑？**

A: 在 `MetadataBuilder.build()` 中检查 `metadata.decode` 是否为 `None`：
```python
def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
    metadata = super().build(...)
    if metadata.decode is not None:
        # decode 阶段的扩展逻辑
        ...
    return metadata
```

**Q: 如何在扩展中访问 vllm_config？**

A: `MetadataBuilder` 的 `__init__` 方法接收 `vllm_config` 参数，可以保存为实例属性：
```python
def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
    super().__init__(...)
    self.vllm_config = vllm_config  # 已在父类中保存
```