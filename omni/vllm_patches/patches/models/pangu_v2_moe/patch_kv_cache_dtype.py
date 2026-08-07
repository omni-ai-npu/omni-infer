# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import logging
from typing import Literal

import torch
import vllm.utils.torch_utils as torch_utils_module
import vllm.model_executor.models.config as models_config_module
from vllm.v1.attention import selector
from vllm.config import cache
from vllm.config.cache import CacheConfig

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = logging.getLogger(__name__)

# New KV cache dtype types to add: maps dtype string -> torch dtype
NEW_CACHE_DTYPES: dict[str, "torch.dtype"] = {
    "hif8_ds_mla": torch.uint8,
    "int8_ds_mla": torch.int8,
    "li_int8_ds_mla": torch.bfloat16,
}

# A valid upstream cache_dtype used as a placeholder during Pydantic validation
# (must pass the Literal check in CacheConfig).
_PLACEHOLDER_DTYPE = "fp8_e4m3"

_EXTENDED_CACHE_DTYPE = Literal[
    "auto",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
    "hif8_ds_mla",
    "int8_ds_mla",
    "li_int8_ds_mla",
]


@register_patch("CacheDTypePatch", cache)
class CacheDTypePatch(VLLMPatch):
    _attr_names_to_apply = ["CacheDType"]
    CacheDType = _EXTENDED_CACHE_DTYPE


@register_patch("CacheDTypeSelectorPatch", selector)
class CacheDTypeSelectorPatch(VLLMPatch):
    _attr_names_to_apply = ["CacheDType"]
    CacheDType = _EXTENDED_CACHE_DTYPE


@register_patch("CacheConfigPatch", CacheConfig)
class CacheConfigPatch(VLLMPatch):
    _attr_names_to_apply = ["__init__"]
    _original_cache_config_init = CacheConfig.__init__

    def __init__(self, *args, **kwargs):
        cache_dtype = kwargs.get("cache_dtype")
        needs_restore = cache_dtype in NEW_CACHE_DTYPES

        if needs_restore:
            kwargs["cache_dtype"] = _PLACEHOLDER_DTYPE

        CacheConfigPatch._original_cache_config_init(self, *args, **kwargs)

        if needs_restore:
            self.cache_dtype = cache_dtype


# `vllm serve` builds argparse choices from the dataclass field's type
# annotation via vllm.engine.arg_utils._compute_kwargs(CacheConfig). That
# introspection reads CacheConfig.__dataclass_fields__["cache_dtype"].type,
# which still references the original Literal even after we replace the
# module-level CacheDType symbol. Without this patch, the CLI rejects
# hif8_ds_mla/int8_ds_mla before plugins get a chance to consume them.
@register_patch("CacheConfigFieldTypePatch", CacheConfig)
class CacheConfigFieldTypePatch(VLLMPatch):
    _attr_names_to_apply = []

    @classmethod
    def apply(cls):
        CacheConfig.__annotations__["cache_dtype"] = _EXTENDED_CACHE_DTYPE
        CacheConfig.__dataclass_fields__["cache_dtype"].type = _EXTENDED_CACHE_DTYPE


@register_patch("ExtendStrDtypeToTorchDtype", torch_utils_module)
class ExtendStrDtypeToTorchDtypePatch(VLLMPatch):
    _attr_names_to_apply = ["STR_DTYPE_TO_TORCH_DTYPE"]
    STR_DTYPE_TO_TORCH_DTYPE = {
        **torch_utils_module.STR_DTYPE_TO_TORCH_DTYPE,
        **NEW_CACHE_DTYPES,
    }


@register_patch("ExtendModelsConfigStrDtypeToTorchDtype", models_config_module)
class ExtendModelsConfigStrDtypeToTorchDtypePatch(VLLMPatch):
    _attr_names_to_apply = ["STR_DTYPE_TO_TORCH_DTYPE"]
    STR_DTYPE_TO_TORCH_DTYPE = {
        **models_config_module.STR_DTYPE_TO_TORCH_DTYPE,
        **NEW_CACHE_DTYPES,
    }