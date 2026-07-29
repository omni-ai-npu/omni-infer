# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Redact oversized routed experts payloads from vLLM request logs."""

import copy
from typing import Any

import torch
from vllm.entrypoints.logger import RequestLogger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import BeamSearchParams, SamplingParams

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

_ORIGINAL_LOG_INPUTS = RequestLogger.log_inputs
_ROUTED_EXPERTS_STR_KEY = "routed_experts_str"
_REMOTE_BLOCK_IDS_KEY = "remote_block_ids"
_REMOTE_BLOCK_IDS_LOG_LIMIT = 10
_KV_TRANSFER_PARAMS_KEY = "kv_transfer_params"


def _truncate_remote_block_ids_for_logging(remote_block_ids: list[Any]) -> list[Any]:
    if len(remote_block_ids) <= _REMOTE_BLOCK_IDS_LOG_LIMIT:
        return remote_block_ids

    return [
        *remote_block_ids[:_REMOTE_BLOCK_IDS_LOG_LIMIT],
        f"... total={len(remote_block_ids)}",
    ]


def _sanitize_kv_transfer_params_for_logging(
    kv_transfer_params: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    sanitized_kv = kv_transfer_params
    changed = False

    if _ROUTED_EXPERTS_STR_KEY in kv_transfer_params:
        sanitized_kv = dict(kv_transfer_params)
        sanitized_kv.pop(_ROUTED_EXPERTS_STR_KEY, None)
        changed = True

    remote_block_ids = kv_transfer_params.get(_REMOTE_BLOCK_IDS_KEY)
    if isinstance(remote_block_ids, list):
        truncated_remote_block_ids = _truncate_remote_block_ids_for_logging(
            remote_block_ids
        )
        if truncated_remote_block_ids is not remote_block_ids:
            if not changed:
                sanitized_kv = dict(kv_transfer_params)
            sanitized_kv[_REMOTE_BLOCK_IDS_KEY] = truncated_remote_block_ids
            changed = True

    return sanitized_kv, changed


def _sanitize_sampling_params_for_logging(params: SamplingParams) -> SamplingParams:
    if params.extra_args is None:
        return params

    kv_transfer_params = params.extra_args.get(_KV_TRANSFER_PARAMS_KEY)
    if not isinstance(kv_transfer_params, dict):
        return params

    sanitized_kv_transfer_params, changed = _sanitize_kv_transfer_params_for_logging(
        kv_transfer_params
    )
    if not changed:
        return params

    log_params = copy.copy(params)
    log_params.extra_args = dict(params.extra_args)
    log_params.extra_args[_KV_TRANSFER_PARAMS_KEY] = sanitized_kv_transfer_params
    return log_params


@register_patch("RequestLoggerLogInputsPatch", RequestLogger)
class RequestLoggerLogInputsPatch(VLLMPatch):
    _attr_names_to_apply = ["log_inputs"]

    def log_inputs(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
        params: SamplingParams | PoolingParams | BeamSearchParams | None,
        lora_request: LoRARequest | None,
    ) -> None:
        if isinstance(params, SamplingParams):
            params = _sanitize_sampling_params_for_logging(params)

        return _ORIGINAL_LOG_INPUTS(
            self,
            request_id,
            prompt,
            prompt_token_ids,
            prompt_embeds,
            params,
            lora_request,
        )
