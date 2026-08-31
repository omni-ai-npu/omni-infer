# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU OffloadingConnector with tuple/list KV cache registration support."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from omni_npu.connector.npu_offloading_scheduler import (
    NPUOffloadingConnectorScheduler,
)

logger = init_logger(__name__)

KVCacheValue = torch.Tensor | Sequence[torch.Tensor]


def _layer_kv_cache_spec(
    layer_name: str, kv_cache_config: KVCacheConfig
) -> KVCacheSpec:
    for group in kv_cache_config.kv_cache_groups:
        if layer_name not in group.layer_names:
            continue
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            return spec.kv_cache_specs[layer_name]
        return spec
    raise KeyError(f"Layer {layer_name!r} not found in kv_cache_config")


def _full_page_int8_view(
    first: torch.Tensor,
    num_blocks: int,
    page_size_bytes: int,
) -> torch.Tensor:
    """Build a (num_blocks, page_size_bytes) int8 view over shared raw storage."""
    storage = first.untyped_storage()
    expected_bytes = num_blocks * page_size_bytes
    if storage.nbytes() < expected_bytes:
        raise RuntimeError(
            f"KV storage has {storage.nbytes()} bytes, expected at least "
            f"{expected_bytes} for {num_blocks} blocks x {page_size_bytes} bytes"
        )
    return (
        torch.tensor([], dtype=torch.int8, device=first.device)
        .set_(storage, 0, (num_blocks, page_size_bytes), (page_size_bytes, 1))
    )


def prepare_kv_caches_for_offloading_registration(
    kv_caches: dict[str, KVCacheValue],
    kv_cache_config: KVCacheConfig,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Present NPU KV caches in the shape OffloadingConnector expects.

    NPU backends keep inference views as tuple/list strided sub-tensors over one
    raw allocation per layer. OffloadingConnector registration expects either a
    single full-page tensor (Attention) or a list (Mamba/Mome). We only reshape
    the registration view; underlying storage is unchanged.
    """
    num_blocks = kv_cache_config.num_blocks
    prepared: dict[str, torch.Tensor | list[torch.Tensor]] = {}

    for layer_name, value in kv_caches.items():
        if isinstance(value, torch.Tensor):
            prepared[layer_name] = value
            continue

        if not isinstance(value, (tuple, list)):
            raise TypeError(
                f"Unsupported KV cache type for layer {layer_name!r}: {type(value)!r}"
            )
        if len(value) == 0:
            raise ValueError(f"Empty KV cache sequence for layer {layer_name!r}")

        first = value[0]
        if not isinstance(first, torch.Tensor):
            raise TypeError(
                f"Expected tensor in KV cache sequence for {layer_name!r}, "
                f"got {type(first)!r}"
            )

        layer_spec = _layer_kv_cache_spec(layer_name, kv_cache_config)
        page_size_bytes = layer_spec.page_size_bytes

        if isinstance(layer_spec, MambaSpec):
            if first.storage_offset() != 0:
                logger.warning(
                    "Layer %s first Mamba/Mome state tensor storage_offset=%d "
                    "(expected 0 for page-packed layout)",
                    layer_name,
                    first.storage_offset(),
                )
            prepared[layer_name] = list(value)
            continue

        if isinstance(layer_spec, AttentionSpec):
            if first.storage_offset() != 0:
                logger.warning(
                    "Layer %s first attention KV sub-tensor storage_offset=%d "
                    "(expected 0 for page-packed layout)",
                    layer_name,
                    first.storage_offset(),
                )
            prepared[layer_name] = _full_page_int8_view(
                first, num_blocks, page_size_bytes
            )
            continue

        raise NotImplementedError(
            f"Unsupported KV cache spec {type(layer_spec)!r} for layer {layer_name!r}"
        )

    return prepared


class NPUOffloadingConnector(OffloadingConnector):
    """OffloadingConnector that accepts NPU tuple/list KV views at registration."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        """Forked from ``OffloadingConnector.__init__`` (vLLM v0.25.1).

        The base class builds the scheduler itself and does not keep the spec,
        so there is no way to swap in the NPU scheduler afterwards without
        creating a second spec (and with it a second offloading manager).
        Deviations from upstream are bracketed by ``omni-npu diff`` markers.
        """
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)

        spec = OffloadingSpecFactory.create_spec(vllm_config, kv_cache_config)

        self.connector_scheduler: NPUOffloadingConnectorScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            # --- omni-npu diff start: NPU scheduler instead of
            # OffloadingConnectorScheduler ---
            self.connector_scheduler = NPUOffloadingConnectorScheduler(spec)
            # --- omni-npu diff end ---
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = OffloadingConnectorWorker(spec)

    @override
    def register_kv_caches(self, kv_caches: dict[str, KVCacheValue]) -> None:
        if self._kv_cache_config is None:
            raise RuntimeError(
                "register_kv_caches requires _kv_cache_config to be set"
            )
        prepared = prepare_kv_caches_for_offloading_registration(
            kv_caches,
            self._kv_cache_config,
        )
        super().register_kv_caches(prepared)

    @override
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        toks, load_async = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        if toks is None:
            self._omni_offload_defer_none = (
                getattr(self, "_omni_offload_defer_none", 0) + 1
            )
        elif load_async:
            self._omni_offload_async_load = (
                getattr(self, "_omni_offload_async_load", 0) + 1
            )
        return toks, load_async

    @override
    def build_connector_meta(self, scheduler_output):
        defer_none = getattr(self, "_omni_offload_defer_none", 0)
        async_load = getattr(self, "_omni_offload_async_load", 0)
        self._omni_offload_defer_none = 0
        self._omni_offload_async_load = 0
        scheduled = int(
            getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0
        )
        if defer_none or (scheduled == 0 and async_load):
            logger.info(
                "kv_offload_schedule scheduled_tokens=%s defer_none=%s "
                "async_load=%s",
                scheduled,
                defer_none,
                async_load,
            )
        return super().build_connector_meta(scheduler_output)
