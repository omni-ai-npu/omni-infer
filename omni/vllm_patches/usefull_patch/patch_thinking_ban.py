# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Add Omni's thinking-ban feature beside vLLM's native thinking budget.

vLLM 0.25.1 owns all thinking-token-budget request fields, validation, state,
and sampler wiring. This module deliberately does not patch any of that logic.
It only extends ReasoningConfig and grafts ThinkingBanStateHolder into the same
InputBatch/SamplingMetadata/Sampler lifecycle used by the native budget holder.
"""

from __future__ import annotations

import inspect

import torch
import vllm.config as vllm_config_module
import vllm.config.reasoning as reasoning_config_module
import vllm.config.vllm as vllm_config_impl_module
import vllm.engine.arg_utils as engine_arg_utils_module
import vllm.v1.worker.gpu_input_batch as gpu_input_batch_module
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import InputBatch

from omni_npu.v1.config import ReasoningConfig
from omni_npu.v1.sample.thinking_ban_state import (
    ThinkingBanStateHolder,
    maybe_create_thinking_ban_state_holder,
)
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)


@register_patch("ReasoningConfigSupersetPatch", vllm_config_module)
class ReasoningConfigSupersetPatch(VLLMPatch):
    """Make all runtime entry points use Omni's native-compatible superset."""

    _attr_names_to_apply: list[str] = []

    @classmethod
    def apply(cls):
        # These modules imported the native class by value. Replace every
        # runtime reference used by CLI construction and worker setup.
        for module in (
            vllm_config_module,
            reasoning_config_module,
            vllm_config_impl_module,
            engine_arg_utils_module,
            gpu_input_batch_module,
        ):
            module.ReasoningConfig = ReasoningConfig

        optional_reasoning_config = ReasoningConfig | None
        VllmConfig.__annotations__["reasoning_config"] = optional_reasoning_config
        VllmConfig.__dataclass_fields__["reasoning_config"].type = (
            optional_reasoning_config
        )
        EngineArgs.__annotations__["reasoning_config"] = optional_reasoning_config
        EngineArgs.__dataclass_fields__["reasoning_config"].type = (
            optional_reasoning_config
        )

        # argparse caches dataclass-derived JSON parsers.
        engine_arg_utils_module._compute_kwargs.cache_clear()


_original_vllm_config_init = VllmConfig.__init__


@register_patch("ReasoningConfigCoercionPatch", VllmConfig)
class ReasoningConfigCoercionPatch(VLLMPatch):
    """Accept dict configs without falling back to the native narrow schema."""

    _attr_names_to_apply = ["__init__"]

    def __init__(self, *args, **kwargs):
        reasoning_config = kwargs.get("reasoning_config")
        if isinstance(reasoning_config, dict):
            kwargs["reasoning_config"] = ReasoningConfig(**reasoning_config)
        _original_vllm_config_init(self, *args, **kwargs)


@register_patch("ThinkingBanSamplingMetadataPatch", SamplingMetadata)
class SamplingMetadataPatch(VLLMPatch):
    """Register ``thinking_ban_state_holder`` as a real dataclass field.

    A plain class attribute is **not** copied by ``dataclasses.replace``.
    ``NPURejectionSampler`` / ``RejectionSampler`` do
    ``replace(sampling_metadata, max_num_logprobs=-1)`` before bonus
    sampling; losing the holder there means the bonus token is unbanned and
    MTP can emit ``tool_call_start`` while still inside ``<think>``.
    """

    _attr_names_to_apply: list[str] = []

    @classmethod
    def apply(cls):
        from dataclasses import dataclass

        target = cls._target
        name = "thinking_ban_state_holder"
        fields = getattr(target, "__dataclass_fields__", {})
        if name in fields:
            return

        target.__annotations__ = dict(getattr(target, "__annotations__", {}))
        target.__annotations__[name] = ThinkingBanStateHolder | None
        setattr(target, name, None)
        # dataclass() will not replace an existing __init__; drop it first.
        if "__init__" in target.__dict__:
            delattr(target, "__init__")
        dataclass(target)
        target._omni_npu_applied_patches = getattr(
            target, "_omni_npu_applied_patches", {}
        )
        target._omni_npu_applied_patches[name] = cls.__name__


_original_input_batch_init = InputBatch.__init__
_original_refresh_metadata = InputBatch.refresh_metadata
_original_make_sampling_metadata = InputBatch._make_sampling_metadata
_input_batch_init_signature = inspect.signature(_original_input_batch_init)


@register_patch("ThinkingBanInputBatchPatch", InputBatch)
class InputBatchPatch(VLLMPatch):
    """Graft ban state beside the native budget holder in InputBatch."""

    _attr_names_to_apply = [
        "__init__",
        "refresh_metadata",
        "_make_sampling_metadata",
        "no_thinking_ban",
    ]

    def __init__(self, *args, **kwargs):
        bound = _input_batch_init_signature.bind_partial(self, *args, **kwargs)
        reasoning_config = bound.arguments.get("reasoning_config")
        num_spec_tokens = bound.arguments.get("num_spec_tokens", 0)

        _original_input_batch_init(self, *args, **kwargs)
        self.thinking_ban_state_holder = maybe_create_thinking_ban_state_holder(
            reasoning_config=reasoning_config,
            max_num_seqs=self.max_num_reqs,
            num_spec_tokens=num_spec_tokens,
            device=self.device,
        )

    def refresh_metadata(self):
        if self.is_pooling_model:
            return _original_refresh_metadata(self)

        # BatchUpdate is consumable, so native budget, Omni ban, and native
        # logits processors must all receive this same instance.
        batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
        budget_holder = self.thinking_budget_state_holder
        if budget_holder is not None and batch_update:
            budget_holder.sync_batch(batch_update)
        ban_holder = self.thinking_ban_state_holder
        if ban_holder is not None and batch_update:
            ban_holder.sync_batch(batch_update)
        for logit_proc in self.logitsprocs.all:
            logit_proc.update_state(batch_update)
        if batch_update:
            self.sampling_metadata = self._make_sampling_metadata()

    def _make_sampling_metadata(self) -> SamplingMetadata:
        ban_holder = getattr(self, "thinking_ban_state_holder", None)
        ban_tracks_requests = (
            ban_holder is not None and ban_holder.has_tracked_requests()
        )

        # Native metadata only includes output ids when a native consumer asks
        # for them. The ban FSM is one more such consumer.
        original_need_output_ids = self.logitsprocs_need_output_token_ids
        if ban_tracks_requests:
            self.logitsprocs_need_output_token_ids = True
        try:
            metadata = _original_make_sampling_metadata(self)
        finally:
            self.logitsprocs_need_output_token_ids = original_need_output_ids

        metadata.thinking_ban_state_holder = ban_holder
        return metadata

    @property
    def no_thinking_ban(self) -> bool:
        holder = getattr(self, "thinking_ban_state_holder", None)
        return holder is None or not holder.has_tracked_requests()


_original_sampler_apply_logits_processors = Sampler.apply_logits_processors


@register_patch("ThinkingBanSamplerPatch", Sampler)
class SamplerPatch(VLLMPatch):
    """Run native processors first, then apply only Omni's ban mask."""

    _attr_names_to_apply = ["apply_logits_processors"]

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        logits = _original_sampler_apply_logits_processors(
            self,
            logits,
            sampling_metadata,
            predict_bonus_token,
        )

        holder = getattr(sampling_metadata, "thinking_ban_state_holder", None)
        if holder is None or not holder.has_tracked_requests():
            return logits

        output_token_ids = sampling_metadata.output_token_ids
        if predict_bonus_token:
            output_token_ids = self._combine_outputs_with_spec_tokens(
                output_token_ids,
                sampling_metadata.spec_token_ids,
            )
        holder.update_state(
            output_token_ids,
            sampling_metadata.spec_token_ids,
            repeat_indices=None,
        )
        return holder.apply_to_logits(
            logits,
            predict_bonus_token,
            sampling_metadata.spec_token_ids,
        )


_original_rejection_apply_logits_processors = (
    RejectionSampler.apply_logits_processors
)


@register_patch("ThinkingBanRejectionSamplerPatch", RejectionSampler)
class RejectionSamplerPatch(VLLMPatch):
    """Apply the ban mask after native speculative-decoding processors."""

    _attr_names_to_apply = ["apply_logits_processors"]

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
    ) -> torch.Tensor:
        logits = _original_rejection_apply_logits_processors(
            self,
            logits,
            sampling_metadata,
            metadata,
        )

        holder = getattr(sampling_metadata, "thinking_ban_state_holder", None)
        if holder is None or not holder.has_tracked_requests():
            return logits

        output_token_ids = Sampler._combine_outputs_with_spec_tokens(
            sampling_metadata.output_token_ids,
            sampling_metadata.spec_token_ids,
        )
        holder.update_state(
            output_token_ids,
            sampling_metadata.spec_token_ids,
            repeat_indices=None,
        )
        return holder.apply_to_logits(
            logits,
            predict_bonus_token=False,
            spec_token_ids=sampling_metadata.spec_token_ids,
        )


@register_patch("ThinkingBanEagleDraftSamplePatch", EagleProposer)
class ThinkingBanEagleDraftSamplePatch(VLLMPatch):
    """Wrap Eagle ``_sample_draft_tokens`` to ban tool markers while drafting.

    Applied after ``TorchEagleProposer`` so the Omni Eagle path (incl.
    ``spec_step_idx``) is preserved; this only injects ``apply_draft_ban``
    around ``compute_logits``.
    """

    # Cannot list ``_sample_draft_tokens`` in ``_attr_names_to_apply`` — Eagle
    # already owns that attribute. Custom ``apply`` chains instead.
    _attr_names_to_apply: list[str] = []

    @classmethod
    def apply(cls):
        target = cls._target
        original = target._sample_draft_tokens

        def _sample_draft_tokens(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
            spec_step_idx: int | None = None,
        ):
            holder = getattr(sampling_metadata, "thinking_ban_state_holder", None)
            if holder is None or not holder.has_tracked_requests():
                return original(
                    self, hidden_states, sampling_metadata, spec_step_idx
                )

            model = self.model
            orig_compute_logits = model.compute_logits

            def compute_logits_with_ban(*args, **kwargs):
                logits = orig_compute_logits(*args, **kwargs)
                return holder.apply_draft_ban(logits, sampling_metadata)

            model.compute_logits = compute_logits_with_ban
            try:
                return original(
                    self, hidden_states, sampling_metadata, spec_step_idx
                )
            finally:
                model.compute_logits = orig_compute_logits

        target._sample_draft_tokens = _sample_draft_tokens
        if not hasattr(target, "_omni_npu_applied_patches"):
            target._omni_npu_applied_patches = {}
        target._omni_npu_applied_patches[
            "_sample_draft_tokens+thinking_ban"
        ] = cls.__name__
        logger.info(
            "patch applied: %s => %s._sample_draft_tokens (wrap)",
            cls.__name__,
            target.__name__,
        )
