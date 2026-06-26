# Omni-Cache Attention Plugins

This directory contains the attention plugin system for Omni-Cache, which provides a non-invasive way to hook into attention layers with pre/post attention callbacks for KV cache offload operations.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Usage](#usage)
- [Configuration](#configuration)
- [Core APIs](#core-apis)
- [Developing a New Plugin](#developing-a-new-plugin)
- [Available Plugins](#available-plugins)
- [File Structure](#file-structure)

## Overview

The attention plugin system uses a **plugin-based architecture** where plugin classes are loaded dynamically via Python entry_points and called externally through decorators. This design enables:

- **Non-invasive**: Add hooks to attention layers without modifying source code
- **Type-specific**: Different plugins for different attention types (compressed MQA, MLA, DSA, etc.)
- **Flexible interface**: Plugins receive all arguments via `*args, **kwargs`
- **Multiple plugins**: Support multiple plugins per attention type with loop execution
- **Extensible**: Third-party packages can register their own plugins via entry_points

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           External Code (e.g., omni-npu)                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  # Load decorator at module top                                         │
│  def compressed_mqa_attn_decorator(func):                               │
│      def wrapper(*args, **kwargs):                                      │
│          # Load plugin classes from entry_points                        │
│          eps = entry_points(group="omni.compress_mqa_attn_decorators")  │
│          plugin_list = []                                               │
│          for ep in eps:                                                 │
│              plugin_class = ep.load()                                   │
│              plugin_list.append(plugin_class())                         │
│                                                                         │
│          # Pre-attention hooks: loop through all plugins                │
│          for plugin in plugin_list:                                     │
│              plugin.pre_attn(*args, **kwargs)                           │
│                                                                         │
│          # Execute original function                                    │
│          ret = func(*args, **kwargs)                                    │
│                                                                         │
│          # Post-attention hooks: loop through all plugins               │
│          for plugin in plugin_list:                                     │
│              plugin.post_attn(*args, result=ret, **kwargs)              │
│                                                                         │
│          return ret                                                     │
│      return wrapper                                                     │
│                                                                         │
│  # Use decorator                                                        │
│  @compressed_mqa_attn_decorator                                         │
│  def _apply_attention(self, ...):                                       │
│      ...                                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Entry Points
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         pyproject.toml                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [project.entry-points."omni.compress_mqa_attn_decorators"]             │
│  compressed_mqa = "omni_cache.attn_plugins.implementations:...          │
│                    :CompressMQAAttnPlugin"                              │
│                                                                         │
│  [project.entry-points."omni.dsa_attn_decorators"]                      │
│  dsa = "omni_cache.attn_plugins.implementations:...                     │
│         :DSAAttnPlugin"                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Load Classes
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    omni_cache/attn_plugins/                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  base.py - Base Interface                                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ class AttentionPlugin:                                           │   │
│  │     def pre_attn(self, *args, **kwargs) -> None:                 │   │
│  │         """Called before attention computation"""                │   │
│  │         pass                                                     │   │
│  │                                                                  │   │
│  │     def post_attn(self, *args, **kwargs) -> None:                │   │
│  │         """Called after attention computation"""                 │   │
│  │         pass                                                     │   │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  implementations.py - Plugin Implementations                            │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ class CompressMQAAttnPlugin(AttentionPlugin):                    │   │
│  │     def pre_attn(self, *args, **kwargs):                         │   │
│  │         # Extract instance from args[0]                          │   │
│  │         instance = args[0] if args else kwargs.get('instance')   │   │
│  │         # Extract layer_name from kwargs                         │   │
│  │         layer_name = kwargs.get('layer_name')                    │   │
│  │         # Call omni_cache API                                    │   │
│  │         omni_cache.synchronize_d2h_hybrid(...)                   │   │
│  │                                                                  │   │
│  │ class DSAAttnPlugin(AttentionPlugin):                            │   │
│  │     # Similar pattern with DSA-specific logic                    │   │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                    │
└────────────────────────────────────────┼───────────────────────────────┘
                                     │
                                     │ Calls
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      omni_cache/cache/                                  │
│                                                                         │
│  - synchronize_d2h(attn_names, attn_metadatas, kv_event)                │
│  - synchronize_d2h_hybrid(attn_names, attn_metadatas, kv_event)         │
│  - synchronize_h2d(prefix_meta, layer_idx)                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Usage

### Step 1: Define Decorator in omni-npu

In omni-npu attention layer file (e.g., `npu_compressed_mqa.py`), define the decorator at module top:

```python
import os
from importlib.metadata import entry_points
try:
    _eps = entry_points(group="omni.compress_mqa_attn_decorators")
    _PLUGIN_CLASSES = [ep.load() for ep in _eps]
except Exception as e:
    logger.warning(f"Failed to load plugins: {e}")
    _PLUGIN_CLASSES = []
    
def compressed_mqa_attn_decorator(func):
    """Decorator that applies plugin pre/post hooks."""
    def wrapper(*args, **kwargs):
        # Load plugin classes from entry points
        plugin_list = []
        for cls in _PLUGIN_CLASSES:
            plugin_list.append(cls())

        # Pre-attention hooks: loop through all plugins
        for plugin in plugin_list:
            plugin.pre_attn(*args, **kwargs)

        # Execute original function
        ret = func(*args, **kwargs)

        # Post-attention hooks: loop through all plugins
        for plugin in plugin_list:
            plugin.post_attn(*args, **kwargs)

        return ret
    return wrapper
```

### Step 2: Apply Decorator Using @ Syntax

Add the decorator to the specific attention method:

```python
class NPUDeepseekCompressedMultiQueryAttention(torch.nn.Module):
    def __init__(self, ...):
        # ... initialization code ...

    # Apply the decorator using standard @ syntax
    @compressed_mqa_attn_decorator
    def _apply_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        win_kvcache: torch.Tensor,
        cmp_kv: Optional[torch.Tensor] = None,
        cmp_kvcache: Optional[torch.Tensor] = None,
        win_metadata: Optional['NPUMetadata'] = None,
        cmp_metadata: Optional['StridedCompressAttentionMetadata'] = None,
    ) -> torch.Tensor:
        # omni-npu existing code - no changes needed
        if win_metadata is None:
            return torch.zeros(...)

        # ... rest of attn implementation ...
```

### Entry Point Groups

Available entry point groups for different attention types:

| Attention Type | Entry Point Group | Example Usage |
|----------------|-------------------|---------------|
| Compressed MQA | `omni.compress_mqa_attn_decorators` | `npu_compressed_mqa.py` |
| Standard MQA | `omni.mqa_attn_decorators` | Standard GQA/MQA attention |
| MLA | `omni.mla_attn_decorators` | DeepSeek-MLA style attention |
| DSA | `omni.dsa_attn_decorators` | Dynamic Sparse Attention |

## Configuration

### Selective Plugin Registration

For plugins defined within omni-cache, the `OMNI_CACHE_ATTN_PLUGINS` environment variable controls which plugins are available for entry_points:

| OMNI_CACHE_ATTN_PLUGINS | Available in Entry Points |
|-------------------------|---------------------------|
| Not set or empty | All plugins available |
| `"compressed_mqa"` | Only compressed_mqa plugin |
| `"compressed_mqa,dsa"` | compressed_mqa and dsa plugins |

### Examples

```bash
# All plugins available (default)
export VLLM_PLUGINS="omni_cache"

# Specific plugin (for internal omni-cache registration)
export OMNI_CACHE_ATTN_PLUGINS="compressed_mqa"
```

## Core APIs

### Attention Plugin Interface

All plugins must inherit from `AttentionPlugin` and implement:

```python
from omni_cache.attn_plugins.base import AttentionPlugin

class MyPlugin(AttentionPlugin):
    def pre_attn(self, *args, **kwargs) -> None:
        """Called before attention computation."""
        pass

    def post_attn(self, *args, **kwargs) -> None:
        """Called after attention computation."""
        pass
```

**Parameter Extraction:**

Plugins receive all arguments via `*args, **kwargs`. example pattern:

```python
def pre_attn(self, *args, **kwargs) -> None:
    # Extract instance (first positional arg)
    instance = args[0] if args else kwargs.get('instance')

    # Extract layer_name from kwargs or instance
    layer_name = kwargs.get('layer_name')
    if layer_name is None and instance is not None:
        layer_name = getattr(instance, 'prefix', None)

    # Extract other needed parameters
    attn_metadata = kwargs.get('attn_metadata')
    win_metadata = kwargs.get('win_metadata')
```

### synchronize_d2h

Core API for Device-to-Host (D2H) KV cache transfer:

```python
def synchronize_d2h(
    self,
    attn_names: list[str],
    attn_metadatas: list[Any],
    kv_event: torch.npu.Event,
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `attn_names` | `list[str]` | Layer names for KV cache transfer |
| `attn_metadatas` | `list[Any]` | Metadata objects for each layer |
| `kv_event` | `torch.npu.Event` | Synchronization event |

## Developing a New Plugin

### Step 1: Implement Plugin Class in `implementations.py`

```python
from .base import AttentionPlugin

class MyAttentionPlugin(AttentionPlugin):
    """Plugin for MyAttention type."""

    def __init__(self):
        super().__init__()
        self._omni_cache = None
        self._enabled = False

    @property
    def omni_cache(self):
        if self._omni_cache is None:
            try:
                from omni_cache.cache import omni_cache
                self._omni_cache = omni_cache
            except Exception:
                self._omni_cache = None
        return self._omni_cache

    @property
    def enabled(self) -> bool:
        if self.enabled:
            return True
        if self.omni_cache is None:
            return False
        self._enabled = getattr(self.omni_cache, 'enable', True)
        return self._enabled

    def pre_attn(self, *args, **kwargs) -> None:
        if not self.enabled:
            return

        # Extract parameters
        instance = args[0] if args else kwargs.get('instance')

        # Specific operations to get attn_names and attn_metadatas from instance, *args, **kwargs
        xxx_attn_names = ...
        xxx_attn_metadatas = ...

        # Import current_stream from omni_cache and get kv_event
        try:
            from omni_cache.cache.utils.ops import current_stream
        except Exception as e:
            # Fallback to torch.npu.current_stream if omni_cache import fails
            from torch import npu
            current_stream = npu.current_stream
            logger.warning(
                f"Failed to import current_stream from omni_cache.cache.utils.ops: {e}, "
                "using torch.npu.current_stream as fallback"
            )

        main_stream = current_stream()
        kv_event = torch.npu.Event(blocking=False, enable_timing=False)
        kv_event.record(main_stream)

        # call d2h or h2d in omni-cache
        self.omni_cache.synchronize_d2h(
            attn_names=xxx_attn_names,
            attn_metadatas=xxx_attn_metadatas,
            kv_event=kv_event
        )

    def post_attn(self, *args, **kwargs) -> None:
        if not self.enabled:
            return
        # Post-attention logic if needed
        pass
```

### Step 2: Add Entry Point in `pyproject.toml`

```toml
[project.entry-points."omni.my_attention_decorators"]
my_attention = "omni_cache.attn_plugins.implementations:MyAttentionPlugin"
```

### Step 3: Define Decorator in specific_attn.py

```python
def xxx_attention_attn_decorator(func):
    def wrapper(*args, **kwargs):
        # Load plugins
        try:
            eps = entry_points(group="omni.my_attention_decorators")
        except TypeError:
            eps = entry_points().get("omni.my_attention_decorators", [])

        plugin_list = [ep.load()() for ep in eps]

        # Pre hooks
        for plugin in plugin_list:
            plugin.pre_attn(*args, **kwargs)

        # Execute
        ret = func(*args, **kwargs)

        # Post hooks
        for plugin in plugin_list:
            plugin.post_attn(*args, result=ret, **kwargs)

        return ret
    return wrapper
```

## Available Plugins

### CompressMQAAttnPlugin

- **Entry Point**: `omni.compress_mqa_attn_decorators`
- **Purpose**: Compressed multi-query attention (DeepSeek-V3 style)
- **Features**:
  - Handles window attention prefix
  - Handles compressed attention prefix
  - Handles indexer prefix if present
  - D2H transfer in pre_attn hook

### DSAAttnPlugin

- **Entry Point**: `omni.dsa_attn_decorators`
- **Purpose**: Dynamic Sparse Attention
- **Features**:
  - Single-layer D2H transfer in pre_attn
  - H2D transfer in post_attn for APC
  - Handles sparse indexer cache

### MLAAttnPlugin

- **Entry Point**: `omni.mla_attn_decorators`
- **Purpose**: Multi-head latent attention (DeepSeek-MLA style)
- **Status**: Placeholder, implementation needed

### GeneralMQAAttnPlugin

- **Entry Point**: `omni.mqa_attn_decorators`
- **Purpose**: Standard multi-query attention
- **Status**: Placeholder, implementation needed

## File Structure

```
attn_plugins/
├── __init__.py              # Package exports and registration
├── base.py                  # AttentionPlugin base class interface
├── implementations.py        # Plugin implementations
└── README.md                 # This file

# In parent directory:
pyproject.toml                # Entry points configuration
```

### Entry Points Configuration

Entry points are defined in `pyproject.toml`:

```toml
# Plugin classes for external loading
[project.entry-points."omni.compress_mqa_attn_decorators"]
compressed_mqa = "omni_cache.attn_plugins.implementations:CompressMQAAttnPlugin"

[project.entry-points."omni.dsa_attn_decorators"]
dsa = "omni_cache.attn_plugins.implementations:DSAAttnPlugin"

[project.entry-points."omni.mla_attn_decorators"]
mla = "omni_cache.attn_plugins.implementations:MLAAttnPlugin"

[project.entry-points."omni.mqa_attn_decorators"]
mqa = "omni_cache.attn_plugins.implementations:GeneralMQAAttnPlugin"
```

## Troubleshooting

### Plugin Not Triggered

1. Verify `ENABLE_OMNI_CACHE=1`
2. Check entry point group name matches code
3. Verify decorator is applied with `@` syntax
4. Check logs for plugin loading messages

### Entry Point Not Found

1. Ensure omni-cache package is installed (`pip install -e .`)
2. Verify entry_points are correctly defined in `pyproject.toml`
3. Check Python version compatibility for `entry_points` API

### D2H Transfer Not Working

1. Verify `omni_cache` is enabled
2. Check if `device_cache` is available
3. Ensure `num_prefills > 0` for prefill phase
4. Verify plugin receives correct parameters from args/kwargs

## License

See LICENSE file for details