# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
import vllm.config as vllm_config_module
from vllm import EngineArgs, SamplingParams
from vllm.config import VllmConfig
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.logger import logger
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import (
    apply_bad_words,
    apply_bad_words_with_drafts,
)
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from omni_npu.v1.config import ReasoningConfig
from omni_npu.v1.sample.thinking_ban_state import (
    maybe_create_thinking_ban_state_holder,
)
from omni_npu.v1.sample.thinking_budget_state import (
    _get_thinking_token_budget,
    _normalize_thinking_token_budget,
    maybe_create_thinking_budget_state_holder,
)
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.vllm_patches.patches.common.patch_args_utils import (
    EngineArgsPatch as _EngineArgsPatchUpstream,
    VllmConfigPatch as _VllmConfigPatchUpstream,
)
from omni_npu.vllm_patches.patches.common.patch_routed_experts import (
    GPUModelRunnerInitRoutedExpertsPatch as _GPUModelRunnerPatchUpstream,
)
from omni_npu.vllm_patches.patches.common.patch_prefilled_token_skip_tokenize import (
    enforce_speculative_generation_budget,
)

_THINKING_TOKEN_BUDGET_KEY = "thinking_token_budget"


def _relay_patch_attrs(
        dst_cls: type,
        src_cls: type,
        *,
        exclude: tuple[str, ...] = (),
) -> None:
    for name in src_cls._attr_names_to_apply:
        if name in exclude:
            continue
        if name in src_cls.__dict__:
            setattr(dst_cls, name, src_cls.__dict__[name])


def _coerce_reasoning_config(value: Any) -> ReasoningConfig | None:
    if value is None or isinstance(value, ReasoningConfig):
        return value
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        return ReasoningConfig(**value)
    return value


def _set_thinking_token_budget(
        params: SamplingParams,
        thinking_token_budget: int | None,
) -> SamplingParams:
    budget = _normalize_thinking_token_budget(thinking_token_budget)
    if budget is None:
        return params
    if params.extra_args is None:
        params.extra_args = {}
    params.extra_args[_THINKING_TOKEN_BUDGET_KEY] = budget
    return params


def _is_reasoning_enabled(reasoning_config: Any) -> bool:
    return bool(
        reasoning_config is not None
        and (
                getattr(reasoning_config, "enabled", False)
                or getattr(reasoning_config, "is_thinking_enabled", False)
        )
    )


def _get_request_thinking_token_budget(
        request: ChatCompletionRequest,
) -> int | None:
    if hasattr(request, _THINKING_TOKEN_BUDGET_KEY):
        value = getattr(request, _THINKING_TOKEN_BUDGET_KEY)
        if value is not None:
            return _normalize_thinking_token_budget(value)
    model_extra = getattr(request, "model_extra", None)
    if model_extra:
        return _normalize_thinking_token_budget(
            model_extra.get(_THINKING_TOKEN_BUDGET_KEY)
        )
    return None


@register_patch("ReasoningConfigModulePatch", vllm_config_module)
class ReasoningConfigModulePatch(VLLMPatch):
    """Expose the out-of-tree reasoning config through ``vllm.config``."""

    _attr_names_to_apply = ["ReasoningConfig"]

    ReasoningConfig = ReasoningConfig


@register_patch("VllmConfigPatch", VllmConfig)
class VllmConfigPatch(VLLMPatch):
    """Relay ``patch_args_utils.VllmConfigPatch`` and add reasoning config."""

    _attr_names_to_apply = list(_VllmConfigPatchUpstream._attr_names_to_apply) + [
        "reasoning_config",
    ]

    reasoning_config: ReasoningConfig | None = None


_relay_patch_attrs(VllmConfigPatch, _VllmConfigPatchUpstream)

_orig_ea_add_cli_args = _EngineArgsPatchUpstream.add_cli_args
_orig_ea_from_cli_args = _EngineArgsPatchUpstream.from_cli_args.__func__
_orig_ea_create_engine_config = _EngineArgsPatchUpstream.create_engine_config


@register_patch("EngineArgsPatch", EngineArgs)
class EngineArgsPatch(VLLMPatch):
    """Relay ``patch_args_utils.EngineArgsPatch`` and add reasoning config."""

    _attr_names_to_apply = list(_EngineArgsPatchUpstream._attr_names_to_apply) + [
        "reasoning_config",
    ]

    reasoning_config: ReasoningConfig | None = None

    @staticmethod
    def add_cli_args(parser):
        parser = _orig_ea_add_cli_args(parser)
        vllm_group = parser.add_argument_group(
            title="VllmConfig",
            description=VllmConfig.__doc__,
        )
        vllm_group.add_argument(
            "--reasoning-config",
            **ReasoningConfig.as_argparse_dict(),
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        instance = _orig_ea_from_cli_args(cls, args)
        raw_reasoning = getattr(args, "reasoning_config", None)
        try:
            instance.reasoning_config = _coerce_reasoning_config(raw_reasoning)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error parsing reasoning_config: %s", exc)
            instance.reasoning_config = None
        return instance

    def create_engine_config(
            self,
            usage_context=None,
            headless: bool = False,
    ) -> VllmConfig:
        vllm_config = _orig_ea_create_engine_config(self, usage_context, headless)
        reasoning_config = _coerce_reasoning_config(
            getattr(self, "reasoning_config", None)
        )
        if reasoning_config is not None and vllm_config.model_config is not None:
            reasoning_config.initialize_token_ids(vllm_config.model_config)
        vllm_config.reasoning_config = reasoning_config
        return vllm_config


_relay_patch_attrs(
    EngineArgsPatch,
    _EngineArgsPatchUpstream,
    exclude=("add_cli_args", "from_cli_args", "create_engine_config"),
)

_original_sampling_params_from_optional = SamplingParams.from_optional
_original_chat_to_sampling_params = ChatCompletionRequest.to_sampling_params


@register_patch("ReasoningSamplingParamsPatch", SamplingParams)
class SamplingParamsPatch(VLLMPatch):
    """Carry request-level thinking budgets through ``extra_args``."""

    _attr_names_to_apply = ["from_optional"]

    @staticmethod
    def from_optional(
            thinking_token_budget: int | None = None,
            **kwargs,
    ) -> SamplingParams:
        budget = kwargs.pop(_THINKING_TOKEN_BUDGET_KEY, thinking_token_budget)
        params = _original_sampling_params_from_optional(**kwargs)
        return _set_thinking_token_budget(params, budget)


@register_patch("ChatCompletionRequestPatch", ChatCompletionRequest)
class ChatCompletionRequestPatch(VLLMPatch):
    """Relay EPD ZMQ chat params, then add top-level thinking budget support."""

    _attr_names_to_apply = ["to_sampling_params"]

    def to_sampling_params(
            self,
            max_tokens: int,
            logits_processor_pattern: str | None,
            default_sampling_params: dict,
    ) -> SamplingParams:
        sampling_params = _original_chat_to_sampling_params(
            self,
            max_tokens,
            logits_processor_pattern,
            default_sampling_params,
        )
        add_ec_transfer_params = getattr(
            self,
            "_omni_add_ec_transfer_params_to_sampling_params",
            None,
        )
        if add_ec_transfer_params is not None:
            sampling_params = add_ec_transfer_params(sampling_params)
        return _set_thinking_token_budget(
            sampling_params,
            _get_request_thinking_token_budget(self),
        )


_original_validate_sampling_params = InputProcessor._validate_sampling_params
_original_InputProcessor_init = InputProcessor.__init__
_original_InputProcessor_process_inputs = InputProcessor.process_inputs


@register_patch("ReasoningInputProcessorPatch", InputProcessor)
class InputProcessorPatch(VLLMPatch):
    """Validate per-request budgets only when reasoning is configured."""

    _attr_names_to_apply = ["__init__", "_validate_sampling_params", "process_inputs"]
    
    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        _original_InputProcessor_init(self, *args, **kwargs)
        self.reasoning_config = getattr(self.vllm_config, "reasoning_config", None)
        
    def process_inputs(
        self,
        *args,
        **kwargs
    ):
        EngineCoreRequest = _original_InputProcessor_process_inputs(self, *args, **kwargs)

        enforce_speculative_generation_budget(self.vllm_config, EngineCoreRequest)

        if self.reasoning_config and self.reasoning_config.thinking_token_budget:
            thinking_token_budget = self.reasoning_config.thinking_token_budget
            request_sampling_params = EngineCoreRequest.sampling_params
            
            request_budget = None
            if hasattr(request_sampling_params, "extra_args"):
                extra_args = request_sampling_params.extra_args
                if isinstance(extra_args, dict):
                    request_budget = extra_args.get("thinking_token_budget", None)

            if not request_budget:
                sampling_params = _set_thinking_token_budget(request_sampling_params, thinking_token_budget)
                EngineCoreRequest.sampling_params = sampling_params
        return EngineCoreRequest

    def _validate_sampling_params(self, params: SamplingParams) -> None:
        _original_validate_sampling_params(self, params)
        if _get_thinking_token_budget(params) is None:
            return
        reasoning_config = getattr(self.vllm_config, "reasoning_config", None)
        if not _is_reasoning_enabled(reasoning_config):
            raise ValueError(
                "thinking_token_budget is set but reasoning_config is not "
                "configured. Please set --reasoning-config to use "
                "thinking_token_budget."
            )


def _attach_reasoning_state_holder(runner: GPUModelRunner) -> None:
    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return
    reasoning_config = getattr(runner.vllm_config, "reasoning_config", None)
    input_batch.thinking_budget_state_holder = (
        maybe_create_thinking_budget_state_holder(
            reasoning_config,
            runner.max_num_reqs,
            getattr(runner, "num_spec_tokens", 0),
            runner.device,
            runner.pin_memory,
        )
    )
    input_batch.thinking_ban_state_holder = (
        maybe_create_thinking_ban_state_holder(
            reasoning_config,
            runner.max_num_reqs,
            getattr(runner, "num_spec_tokens", 0),
            runner.device,
            runner.pin_memory,
        )
    )
    if not hasattr(input_batch, "thinking_token_budget_reqs"):
        input_batch.thinking_token_budget_reqs = set()


_orig_gpu_model_runner_init = _GPUModelRunnerPatchUpstream.__dict__["__init__"]
_original_may_reinitialize_input_batch = GPUModelRunner.may_reinitialize_input_batch


@register_patch("GPUModelRunnerInitRoutedExpertsPatch", GPUModelRunner)
class GPUModelRunnerPatch(VLLMPatch):
    """Relay routed-experts runner init and attach reasoning state."""

    _attr_names_to_apply = list(_GPUModelRunnerPatchUpstream._attr_names_to_apply) + [
        "may_reinitialize_input_batch",
    ]

    def __init__(self, *args, **kwargs):
        _orig_gpu_model_runner_init(self, *args, **kwargs)
        _attach_reasoning_state_holder(self)

    def may_reinitialize_input_batch(self, *args, **kwargs) -> None:
        _original_may_reinitialize_input_batch(self, *args, **kwargs)
        _attach_reasoning_state_holder(self)


_relay_patch_attrs(GPUModelRunnerPatch, _GPUModelRunnerPatchUpstream, exclude=("__init__",))

_original_input_batch_init = InputBatch.__init__
_original_add_request = InputBatch.add_request
_original_remove_request = InputBatch.remove_request
_original_refresh_metadata = InputBatch.refresh_metadata
_original_make_sampling_metadata = InputBatch._make_sampling_metadata


@register_patch("ReasoningInputBatchPatch", InputBatch)
class InputBatchPatch(VLLMPatch):
    """Track budgeted requests and expose the holder through sampling metadata."""

    _attr_names_to_apply = [
        "__init__",
        "add_request",
        "remove_request",
        "refresh_metadata",
        "_make_sampling_metadata",
        "no_thinking_budget",
        "no_thinking_ban",
    ]

    def __init__(
            self,
            *args,
            reasoning_config: ReasoningConfig | None = None,
            **kwargs,
    ):
        num_spec_tokens = kwargs.get("num_speculative_tokens", 0)
        _original_input_batch_init(self, *args, **kwargs)
        self.thinking_token_budget_reqs: set[str] = set()
        self.thinking_budget_state_holder = maybe_create_thinking_budget_state_holder(
            reasoning_config,
            self.max_num_reqs,
            num_spec_tokens,
            self.device,
            self.pin_memory,
        )
        self.thinking_ban_state_holder = maybe_create_thinking_ban_state_holder(
            reasoning_config,
            self.max_num_reqs,
            num_spec_tokens,
            self.device,
            self.pin_memory,
        )

    def add_request(self, request) -> int:
        req_index = _original_add_request(self, request)
        sampling_params = getattr(request, "sampling_params", None)
        if _get_thinking_token_budget(sampling_params) is not None:
            self.thinking_token_budget_reqs.add(request.req_id)
        return req_index

    def remove_request(self, req_id: str) -> int | None:
        req_index = _original_remove_request(self, req_id)
        self.thinking_token_budget_reqs.discard(req_id)
        return req_index

    def refresh_metadata(self):
        if self.is_pooling_model:
            return _original_refresh_metadata(self)

        batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
        holder = getattr(self, "thinking_budget_state_holder", None)
        if holder is not None and batch_update:
            holder.sync_batch(batch_update)
        ban_holder = getattr(self, "thinking_ban_state_holder", None)
        if ban_holder is not None and batch_update:
            ban_holder.sync_batch(batch_update)
        for logit_proc in self.logitsprocs.all:
            logit_proc.update_state(batch_update)
        if batch_update:
            self.sampling_metadata = self._make_sampling_metadata()
        return None

    def _make_sampling_metadata(self):
        original_need_output_token_ids = self.logitsprocs_need_output_token_ids
        # Either thinking feature (budget or ban) requires output_token_ids
        # so the holder can derive the running committed history.
        if not (self.no_thinking_budget and self.no_thinking_ban):
            self.logitsprocs_need_output_token_ids = True
        try:
            metadata = _original_make_sampling_metadata(self)
        finally:
            self.logitsprocs_need_output_token_ids = original_need_output_token_ids
        metadata.thinking_budget_state_holder = getattr(
            self,
            "thinking_budget_state_holder",
            None,
        )
        metadata.thinking_ban_state_holder = getattr(
            self,
            "thinking_ban_state_holder",
            None,
        )
        return metadata

    @property
    def no_thinking_budget(self) -> bool:
        return (
                getattr(self, "thinking_budget_state_holder", None) is None
                or len(getattr(self, "thinking_token_budget_reqs", set())) == 0
        )

    @property
    def no_thinking_ban(self) -> bool:
        ban_holder = getattr(self, "thinking_ban_state_holder", None)
        return ban_holder is None or not ban_holder.has_tracked_requests()


@register_patch("ReasoningSamplerPatch", Sampler)
class SamplerPatch(VLLMPatch):
    """Apply thinking-budget forcing after regular sampling processors."""

    _attr_names_to_apply = ["apply_logits_processors"]

    def apply_logits_processors(
            self,
            logits: torch.Tensor,
            sampling_metadata,
            predict_bonus_token: bool,
    ) -> torch.Tensor:
        bad_words_token_ids = sampling_metadata.bad_words_token_ids
        any_penalties_or_bad_words = (
                bool(bad_words_token_ids) or not sampling_metadata.no_penalties
        )
        holder = getattr(sampling_metadata, "thinking_budget_state_holder", None)
        needs_thinking = holder is not None and holder.has_tracked_requests()
        ban_holder = getattr(sampling_metadata, "thinking_ban_state_holder", None)
        needs_ban = ban_holder is not None and ban_holder.has_tracked_requests()

        output_token_ids = sampling_metadata.output_token_ids
        if predict_bonus_token and (
                any_penalties_or_bad_words or needs_thinking or needs_ban
        ):
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))

        if bad_words_token_ids:
            apply_bad_words(logits, bad_words_token_ids, output_token_ids)

        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)

        logits = self.apply_penalties(logits, sampling_metadata, output_token_ids)
        if needs_thinking:
            holder.update_state(
                output_token_ids,
                sampling_metadata.spec_token_ids,
                repeat_indices=None,
            )
            logits = holder.apply_to_logits(
                logits,
                predict_bonus_token,
                sampling_metadata.spec_token_ids,
            )
        if needs_ban:
            ban_holder.update_state(
                output_token_ids,
                sampling_metadata.spec_token_ids,
                repeat_indices=None,
            )
            logits = ban_holder.apply_to_logits(
                logits,
                predict_bonus_token,
                sampling_metadata.spec_token_ids,
            )
        return logits


@register_patch("ReasoningRejectionSamplerPatch", RejectionSampler)
class RejectionSamplerPatch(VLLMPatch):
    """Apply thinking-budget forcing for speculative target logits."""

    _attr_names_to_apply = ["apply_logits_processors"]

    def apply_logits_processors(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
            metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        has_penalties = not sampling_metadata.no_penalties
        any_penalties_or_bad_words = (
                sampling_metadata.bad_words_token_ids or has_penalties
        )
        holder = sampling_metadata.thinking_budget_state_holder
        needs_thinking = holder is not None and holder.has_tracked_requests()
        ban_holder = getattr(sampling_metadata, "thinking_ban_state_holder", None)
        needs_ban = ban_holder is not None and ban_holder.has_tracked_requests()

        output_token_ids = sampling_metadata.output_token_ids
        if any_penalties_or_bad_words or needs_thinking or needs_ban:
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )

        # Calculate indices of target logits.
        repeat_indices: torch.Tensor | None = None
        need_repeat_indices = (
                sampling_metadata.allowed_token_ids_mask is not None
                or has_penalties
                or needs_thinking
        )
        if need_repeat_indices:
            num_requests = len(metadata.num_draft_tokens)
            num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")
            original_indices = torch.arange(num_requests, device="cpu")
            repeat_indices_cpu = original_indices.repeat_interleave(num_draft_tokens)
            repeat_indices = repeat_indices_cpu.to(
                device=logits.device, non_blocking=True
            )
            if has_penalties:
                logits = self.apply_penalties(
                    logits,
                    sampling_metadata,
                    metadata,
                    repeat_indices,
                    output_token_ids,
                )

            # Apply allowed token ids.
            if sampling_metadata.allowed_token_ids_mask is not None:
                token_mask = sampling_metadata.allowed_token_ids_mask[repeat_indices]
                logits.masked_fill_(token_mask, float("-inf"))

        # Apply bad words exclusion.
        if bad_words_token_ids := sampling_metadata.bad_words_token_ids:
            apply_bad_words_with_drafts(
                logits, bad_words_token_ids, output_token_ids, metadata.num_draft_tokens
            )

        if holder is not None and holder.has_tracked_requests():
            assert repeat_indices is not None
            logits = holder.apply_to_logits(
                logits,
                predict_bonus_token=False,
                spec_token_ids=sampling_metadata.spec_token_ids,
            )
        if needs_ban:
            logits = ban_holder.apply_to_logits(
                logits,
                predict_bonus_token=False,
                spec_token_ids=sampling_metadata.spec_token_ids,
            )
        return logits


from dataclasses import dataclass
from typing import Any


@register_patch("SamplingMetadataPatch", SamplingMetadata)
class SamplingMetadataPatch(VLLMPatch):
    _attr_names_to_apply = []

    @classmethod
    def apply(cls):
        target = cls._target
        # Add both thinking-feature holder fields onto the SamplingMetadata
        # dataclass. Idempotent: each field is skipped if already present.
        added_any = False
        for name in ("thinking_budget_state_holder", "thinking_ban_state_holder"):
            if name in getattr(target, "__dataclass_fields__", {}):
                continue
            target.__annotations__ = dict(getattr(target, "__annotations__", {}))
            target.__annotations__[name] = Any | None
            setattr(target, name, None)
            added_any = True
        if not added_any:
            return
        # dataclass() 不会覆盖已有 __init__，所以要先删除旧的 dataclass __init__。
        if "__init__" in target.__dict__:
            delattr(target, "__init__")
        dataclass(target)
