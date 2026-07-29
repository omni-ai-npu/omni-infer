# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from copy import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from vllm.config import (
    CUDAGraphMode,
    get_layers_from_vllm_config,
)
from vllm.forward_context import set_forward_context, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata
from vllm.v1.spec_decode.eagle import EagleProposer, PADDING_SLOT_ID

from omni_npu.attention.backends.mome import NPUMomeAttentionMetadataBuilder
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.model_config.config_loader.loader import model_extra_config

logger = init_logger(__name__)

EagleProposer_original_init = EagleProposer.__init__


@dataclass
class DraftAttnGroup:
    """A group of draft layers sharing the same KV cache group and attention
    backend. All layers in a DraftAttnGroup share one block table and one metadata
    builder."""
    kv_cache_group_id: int
    layer_names: list[str] = field(default_factory=list)
    builder: AttentionMetadataBuilder | None = None
    is_base: bool = False  # True for group containing attn_layer_names[0]


@register_patch("TorchEagleProposer", EagleProposer)
class EagleProposerPatch(VLLMPatch):
    """Patch for vLLM's EagleProposer to support omni-npu compilation and execution.
    """

    _attr_names_to_apply = [
        '__init__',
        'prepare_next_token_ids_padded',
        'prepare_inputs_padded',
        'dummy_run',
        'load_model',
        'propose',
        'propose_multi_mtp', # new
        'validate_same_kv_cache_group',
        '_build_common_attn_metadata_for_group',
        '_build_per_group_metadata',
        '_rebuild_per_group_metadata_for_step',
        "_save_and_change_target_input", # new
    ]

    def __init__(self, *args, **kwargs):
        EagleProposer_original_init(self, *args, **kwargs)
        self.use_cuda_graph = True
        self.n_predict = getattr(
            self.draft_model_config.hf_config, "n_predict", 1
        )
        logger.info(f"n_predict = {self.n_predict}, num_speculative_tokens = {self.num_speculative_tokens}.")

        self.fix_multi_mtp_kvcache = (
            self.n_predict > 1 and model_extra_config.operator_opt_config.fix_multi_mtp_kvcache)

    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead.
        It also accounts for the rejected tokens in `sampled_token_ids`.
        This function must use device functions to operate on the inputs, and
        should not introduce any blocking CPU-GPU synchronization.
        """
        # NOTE(Ben): Combine this into a custom fused kernel

        # Precompute get_token_id for when there is no valid next token
        num_reqs = gpu_input_batch.num_reqs
        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(
                    common_attn_metadata.seq_lens_cpu[i].item()
                )
                for i in range(num_reqs)
            ]
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)

        # Mask out sampled tokens for requests that should not be sampled.
        valid_sampled_token_ids_gpu = sampled_token_ids.clone()
        valid_sampled_token_ids_gpu = valid_sampled_token_ids_gpu.masked_fill(
            discard_request_mask[:num_reqs].unsqueeze(1),
            -1,
        )

        # Generate a mask for all valid tokens within those requests
        valid_mask = (valid_sampled_token_ids_gpu != -1) & (
            valid_sampled_token_ids_gpu < gpu_input_batch.vocab_size
        )

        # Count the number of valid tokens in each request
        valid_sampled_tokens_count = valid_mask.sum(dim=1)

        # Get the rightmost valid index per row
        last_valid_indices = valid_sampled_tokens_count - 1
        last_valid_indices_safe = torch.clamp(last_valid_indices, min=0)

        # Get last valid token from each row
        # (assume undefined state where there is no valid token)
        selected_tokens = torch.gather(
            valid_sampled_token_ids_gpu, 1, last_valid_indices_safe.unsqueeze(1)
        ).squeeze(1)

        # Use last token if valid, pre-computed backup if not
        batch_size = valid_sampled_token_ids_gpu.shape[0]
        next_token_ids = torch.where(
            last_valid_indices != -1,
            selected_tokens,
            self.backup_next_token_ids.gpu[:batch_size],
        )

        return next_token_ids, valid_sampled_tokens_count


    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        No blocking CPU operations should be introduced in this function.
        """
        num_draft_tokens_gpu = torch.cat(
            [
                spec_decode_metadata.cu_num_draft_tokens[0:1],
                spec_decode_metadata.cu_num_draft_tokens[1:]
                - spec_decode_metadata.cu_num_draft_tokens[:-1],
            ]
        )

        num_rejected_tokens_gpu = torch.where(
            num_draft_tokens_gpu > 0,
            num_draft_tokens_gpu + 1 - valid_sampled_tokens_count,
            torch.zeros_like(num_draft_tokens_gpu),
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()
        token_indices = self.arange[:total_num_tokens]

        # Adapt: used padded num_actual_tokens and slot_mapping across dp
        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=True,
        )
        # End Adapt

        actual_num_reqs = len(num_draft_tokens_gpu)
        token_indices_to_sample = common_attn_metadata.query_start_loc[1:actual_num_reqs + 1] - 1 \
            - num_rejected_tokens_gpu

        return (
            spec_common_attn_metadata,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
        )

    @torch.inference_mode()
    def dummy_run(
        self,
        attn_metadata,
        num_tokens: int,
        use_cudagraphs=True,
        is_graph_capturing=False,
        is_profile: bool = False,
    ) -> None:

        if self.runner.batch_execution_and_padding_state is None:
            raise ValueError(
                "dummy_run of drafter should be executed after "
                "runner._determine_batch_execution_and_padding"
            )
        cudagraph_mode, batch_descriptor, num_tokens_across_dp = self.runner.batch_execution_and_padding_state
        self.runner.batch_execution_and_padding_state = None
        num_input_tokens = batch_descriptor.num_tokens
        if attn_metadata is not None:
            per_layer_attn_metadata = {}
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata[layer_name]
        else:
            per_layer_attn_metadata = None
        # NOTE: when using tree-based specdec, adjust number of forward-passes
        # according to the depth of the tree.
        for fwd_idx in range(
            self.num_speculative_tokens
            if not is_graph_capturing else min(self.num_speculative_tokens, self.n_predict)
        ):
            if self.n_predict == 1 and fwd_idx == 1 and cudagraph_mode == CUDAGraphMode.NONE:
                num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    num_tokens_padded=num_tokens,
                )
                num_input_tokens = num_tokens_dp_padded
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[self.dp_rank] = num_input_tokens

            # Adapt: pass attn_metadata and batch_descriptor to set_forward_context, change cudagraph_runtime_mode
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_descriptor
            ):
                forward_context = get_forward_context()
                forward_context.capturing = False
                if self.supports_mm_inputs:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                elif self.n_predict > 1:
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                else:
                    input_ids = self.input_ids[:num_input_tokens]
                    inputs_embeds = None

                model_kwargs = dict(
                    input_ids=input_ids,
                    positions=self._get_positions(num_input_tokens),
                    hidden_states=self.hidden_states[:num_input_tokens],
                    inputs_embeds=inputs_embeds,
                )
                if self.method == "mtp" and self.n_predict > 1:
                    model_kwargs["spec_step_idx"] = fwd_idx
                ret_hidden_states = self.model(**model_kwargs)

                # Mirror active rank's per-iter (forward, compute_logits)
                # order so idle DP ranks stay lock-step on the DP collectives
                # inside compute_logits. Active's propose/propose_multi_mtp
                # interleaves these two calls per spec step; batching all
                # compute_logits after the loop deadlocks with num_spec > 1.
                if (
                    getattr(self.runner, "dp_parallel_lmhead", False)
                    or getattr(self.runner, "local_parallel_lmhead", False)
                ) and not is_profile:
                    last_hs = (
                        ret_hidden_states if self.method == "mtp"
                        else ret_hidden_states[0]
                    )
                    sample_hs = last_hs[:1]
                    if self.method == "mtp" and self.n_predict > 1:
                        self.model.compute_logits(sample_hs, spec_step_idx=fwd_idx)
                    else:
                        self.model.compute_logits(sample_hs)


    def load_model(self, target_model: nn.Module) -> None:
        draft_model_config = self.vllm_config.speculative_config.draft_model_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
        )
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("eagle_head"):
            self.model = get_model(
                vllm_config=self.vllm_config, model_config=draft_model_config
            )

        # All draft AttentionLayerBase layers — no type-specific separation.
        # The per-group DraftAttnGroup structure handles heterogeneous attention types.
        self.attn_layer_names = sorted(
            list(
                get_layers_from_vllm_config(
                    self.vllm_config, AttentionLayerBase
                ).keys()
                - target_attn_layer_names
            )
        )

        # Per-group structures — populated in validate_same_kv_cache_group()
        # when the kv_cache_config and runner.attn_groups are available.
        self.draft_attn_groups: list[DraftAttnGroup] = []
        self.base_kv_cache_group_id: int | None = None

        if self.supports_mm_inputs:
            # Even if the target model is multimodal, we can also use
            # text-only draft models
            try:
                dummy_input_ids = torch.tensor([[1]], device=self.input_ids.device)
                self.model.embed_input_ids(dummy_input_ids, multimodal_embeddings=None)
            except (NotImplementedError, AttributeError, TypeError):
                logger.warning(
                    "Draft model does not support multimodal inputs, "
                    "falling back to text-only mode"
                )
                self.supports_mm_inputs = False

        if supports_multimodal(target_model):
            # handle multimodality
            if self.get_model_name(target_model) in [
                "Qwen2_5_VLForConditionalGeneration",
                "Qwen3VLForConditionalGeneration",
                #####patch start: for pangu72B-VL/OMNI
                "OpenPanguVLForConditionalGeneration",
                "OpenPanguV2VLForConditionalGeneration",
                "OpenPanguUltraOmniForConditionalGeneration",
                #####patch end
                "Qwen3_5ForConditionalGeneration",
                "Qwen3_5MoeForConditionalGeneration",
            ]:
                self.model.config.image_token_index = target_model.config.image_token_id
            elif self.get_model_name(target_model) == "PixtralForConditionalGeneration":
                self.model.config.image_token_index = (
                    target_model.config.vision_config.image_token_id
                )
            else:
                self.model.config.image_token_index = (
                    target_model.config.image_token_index
                )
            target_language_model = target_model.get_language_model()
        else:
            target_language_model = target_model

        # share embed_tokens with the target model if needed
        if get_pp_group().world_size == 1:
            if hasattr(target_language_model.model, "embed_tokens"):
                target_embed_tokens = target_language_model.model.embed_tokens
            elif hasattr(target_language_model.model, "embedding"):
                target_embed_tokens = target_language_model.model.embedding
            else:
                raise AttributeError(
                    "Target model does not have 'embed_tokens' or 'embedding' attribute"
                )

            share_embeddings = False
            if hasattr(self.model, "has_own_embed_tokens"):
                # EAGLE model
                if not self.model.has_own_embed_tokens:
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model without its own embed_tokens in the"
                        " checkpoint. Sharing target model embedding weights with the"
                        " draft model."
                    )
                elif (
                    isinstance(target_embed_tokens.weight, torch.Tensor)
                    and isinstance(self.model.model.embed_tokens.weight, torch.Tensor)
                    # NOTE: Offload to CPU for comparison to avoid extra GPU memory
                    # usage in CI testing environments with limited GPU memory
                    and torch.equal(
                        target_embed_tokens.weight.cpu(),
                        self.model.model.embed_tokens.weight.cpu(),
                    )
                ):
                    share_embeddings = True
                    logger.info(
                        "Detected EAGLE model with embed_tokens identical to the target"
                        " model. Sharing target model embedding weights with the draft"
                        " model."
                    )
                else:
                    logger.info(
                        "Detected EAGLE model with distinct embed_tokens weights. "
                        "Keeping separate embedding weights from the target model."
                    )
            else:
                # MTP model
                share_embeddings = True
                logger.info(
                    "Detected MTP model. "
                    "Sharing target model embedding weights with the draft model."
                )

            if share_embeddings:
                if hasattr(self.model.model, "embed_tokens"):
                    del self.model.model.embed_tokens
                self.model.model.embed_tokens = target_embed_tokens
        else:
            logger.info(
                "The draft model's vocab embedding will be loaded separately"
                " from the target model."
            )

        # share lm_head with the target model if needed
        share_lm_head = False
        if hasattr(self.model, "has_own_lm_head"):
            # EAGLE model
            if not self.model.has_own_lm_head:
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model without its own lm_head in the checkpoint. "
                    "Sharing target model lm_head weights with the draft model."
                )
            elif (
                hasattr(target_language_model, "lm_head")
                and isinstance(target_language_model.lm_head.weight, torch.Tensor)
                and isinstance(self.model.lm_head.weight, torch.Tensor)
                # TODO: Offload to CPU for comparison to avoid extra GPU memory
                # usage in CI testing environments with limited GPU memory
                and torch.equal(
                    target_language_model.lm_head.weight.cpu(),
                    self.model.lm_head.weight.cpu(),
                )
            ):
                share_lm_head = True
                logger.info(
                    "Detected EAGLE model with lm_head identical to the target model. "
                    "Sharing target model lm_head weights with the draft model."
                )
            else:
                logger.info(
                    "Detected EAGLE model with distinct lm_head weights. "
                    "Keeping separate lm_head weights from the target model."
                )
        else:
            # MTP model
            share_lm_head = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model lm_head weights with the draft model."
            )

        if share_lm_head and hasattr(target_language_model, "lm_head"):
            if hasattr(self.model, "lm_head"):
                del self.model.lm_head
            self.model.lm_head = target_language_model.lm_head

        if hasattr(self.model, "set_shared_weight"):
            self.model.set_shared_weight(target_language_model)

    def validate_same_kv_cache_group(self, kv_cache_config) -> None:
        """Build per-group draft layer structure from runner's attn_groups.

        Instead of requiring all draft layers to be in one KV cache group,
        this discovers each draft layer's (kv_cache_gid, attn_group) and
        groups them accordingly.  Each DraftAttnGroup shares one block table
        and one metadata builder.
        """
        base_layer = (
            self.attn_layer_names[0] if self.attn_layer_names else None
        )

        self.draft_attn_groups = []
        for kv_cache_group_id, kv_attn_groups in enumerate(self.runner.attn_groups):
            for attn_group in kv_attn_groups:
                draft_in_group = set(attn_group.layer_names) & set(self.attn_layer_names)
                if draft_in_group:
                    is_base = (
                        base_layer is not None
                        and base_layer in draft_in_group
                    )
                    self.draft_attn_groups.append(
                        DraftAttnGroup(
                            kv_cache_group_id=kv_cache_group_id,
                            layer_names=sorted(list(draft_in_group)),
                            builder=attn_group.get_metadata_builder(),
                            is_base=is_base,
                        )
                    )

        # Sort so the base group comes first.  This ensures the base CM is
        # updated in-place before other groups shallow-copy from it.
        self.draft_attn_groups.sort(
            key=lambda g: (not g.is_base, g.kv_cache_group_id)
        )
        self.base_kv_cache_group_id = next(
            (g.kv_cache_group_id for g in self.draft_attn_groups if g.is_base), None
        )

    def _build_common_attn_metadata_for_group(
        self,
        kv_cache_group_id: int,
        base_cm: CommonAttentionMetadata,
    ) -> CommonAttentionMetadata:
        """Return a CommonAttentionMetadata with the group's block table.

        For the base group the original *base_cm* is returned as-is (no
        copy).  For other groups a shallow copy is made and the block table
        / slot_mapping are swapped.
        """
        if kv_cache_group_id == self.base_kv_cache_group_id:
            return base_cm
        cm = copy(base_cm)
        blk_table = self.runner.input_batch.block_table[kv_cache_group_id]
        cm.block_table_tensor = blk_table.get_device_tensor(base_cm.num_reqs)
        cm.slot_mapping = blk_table.slot_mapping.gpu[:base_cm.num_actual_tokens]

        if hasattr(base_cm, "num_reqs_unpadded"):
            cm.block_table_tensor[base_cm.num_reqs_unpadded:].fill_(PADDING_SLOT_ID)
        if hasattr(base_cm, "num_tokens_unpadded"):
            cm.slot_mapping[base_cm.num_tokens_unpadded:].fill_(PADDING_SLOT_ID)
        return cm

    def _build_per_group_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> dict:
        """Build per-layer attention metadata for all draft groups."""

        per_layer_attn_metadata: dict = {}
        for group in self.draft_attn_groups:
            extra_attn_metadata_args = {}
            if isinstance(group.builder, NPUMomeAttentionMetadataBuilder):
                num_reqs = common_attn_metadata.num_reqs
                extra_attn_metadata_args['num_accepted_tokens'] = \
                    self.runner.num_accepted_tokens.gpu[:num_reqs]
                extra_attn_metadata_args['num_prompt_tokens'] = \
                    self.runner.num_prompt_tokens.gpu[:num_reqs]

            cm = self._build_common_attn_metadata_for_group(
                group.kv_cache_group_id,
                common_attn_metadata,
            )
            attn_metadata = group.builder.build_for_drafting(
                common_attn_metadata=cm,
                draft_index=draft_index,
                **extra_attn_metadata_args,
            )
            for name in group.layer_names:
                per_layer_attn_metadata[name] = attn_metadata
        return per_layer_attn_metadata

    def _rebuild_per_group_metadata_for_step(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        clamped_positions: torch.Tensor,
        exceeds_max_model_len: torch.Tensor,
        token_index: int,
        per_layer_attn_metadata: dict,
    ) -> None:
        """Rebuild metadata for all draft groups after a seq_lens increment.

        Computes per-group slot_mapping using each group's own block table
        and block size, then rebuilds backend-specific metadata.

        ``self.draft_attn_groups`` is sorted base-first so that
        ``common_attn_metadata`` is updated in-place before other groups
        shallow-copy from it.
        """
        for group in self.draft_attn_groups:
            builder = group.builder
            block_size = builder.kv_cache_spec.block_size
            blk_table = self.runner.input_batch.block_table[group.kv_cache_group_id]
            blk_table_tensor = blk_table.get_device_tensor(
                common_attn_metadata.num_reqs
            )

            # Compute slot_mapping for this group
            if self.uses_mrope:
                block_numbers = clamped_positions[0] // block_size
            else:
                block_numbers = clamped_positions // block_size
            block_ids = blk_table_tensor.gather(
                dim=1, index=block_numbers.view(-1, 1)
            ).view(-1)
            if self.uses_mrope:
                slot_mapping = (
                    block_ids * block_size
                    + clamped_positions[0] % block_size
                )
            else:
                slot_mapping = (
                    block_ids * block_size
                    + clamped_positions % block_size
                )
            slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)

            if group.kv_cache_group_id == self.base_kv_cache_group_id:
                # Update base CM in-place (preserves current behaviour)
                common_attn_metadata.block_table_tensor = blk_table_tensor
                common_attn_metadata.slot_mapping = slot_mapping
                group_cm = common_attn_metadata
            else:
                group_cm = copy(common_attn_metadata)
                group_cm.block_table_tensor = blk_table_tensor
                group_cm.slot_mapping = slot_mapping

            extra_attn_metadata_args = {}
            if isinstance(builder, NPUMomeAttentionMetadataBuilder):
                num_reqs = common_attn_metadata.num_reqs
                extra_attn_metadata_args['num_accepted_tokens'] = \
                    self.runner.num_accepted_tokens.gpu[:num_reqs]
                extra_attn_metadata_args['num_prompt_tokens'] = \
                    self.runner.num_prompt_tokens.gpu[:num_reqs]

            attn_metadata = builder.build_for_drafting(
                common_attn_metadata=group_cm,
                draft_index=token_index + 1,
                **extra_attn_metadata_args,
            )
            for name in group.layer_names:
                per_layer_attn_metadata[name] = attn_metadata

    def propose(
        self,
        target_token_ids: torch.Tensor,  # shape: [num_tokens]
        target_positions: torch.Tensor,  # shape: [num_tokens] or [3, num_tokens] when M-RoPE
        target_hidden_states: torch.Tensor,  # shape: [num_tokens, hidden_size]
        next_token_ids: torch.Tensor,  # shape: [batch_size]
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if (self.method == 'mtp' and self.n_predict > 1
            and self.num_speculative_tokens <= self.n_predict
        ):
            return self.propose_multi_mtp(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                last_token_indices=last_token_indices,
                common_attn_metadata=common_attn_metadata,
                sampling_metadata=sampling_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )

        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]

        # Adapt: handle the padding of query_start_loc
        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:batch_size + 1] - 1
        # End Adapt

        if self.method == "eagle3":
            draft_model = self.model.unwrap() if hasattr(self.model, "unwrap") else self.model
            if not isinstance(draft_model, Eagle3LlamaForCausalLM):
                raise TypeError("draft_model must be Eagle3LlamaForCausalLM")
            target_hidden_states = draft_model.combine_hidden_states(
                target_hidden_states
            )
            if target_hidden_states.shape[-1] != self.hidden_size:
                raise ValueError(
                    f"target_hidden_states shape mismatch: "
                    f"{target_hidden_states.shape[-1]} != {self.hidden_size}"
                )
        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[: num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids

        if self.runner is None:
            raise ValueError("runner is not initialized")

        per_layer_attn_metadata = self._build_per_group_metadata(
            common_attn_metadata,
            draft_index=0,
        )

        # Adapt start: get token info from target model batch descriptor
        if self.runner.batch_execution_and_padding_state is None:
            raise ValueError(
                "propose of drafter should be executed after "
                "runner._determine_batch_execution_and_padding"
            )
        cudagraph_runtime_mode, batch_descriptor, num_tokens_across_dp = self.runner.batch_execution_and_padding_state
        self.runner.batch_execution_and_padding_state = None
        num_input_tokens = batch_descriptor.num_tokens
        # Adapt end

        # copy inputs to buffer for cudagraph
        self._set_positions(num_tokens, target_positions)
        self.hidden_states[:num_tokens] = target_hidden_states

        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )
            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        elif self.n_predict > 1:
            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
            )
            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        # Adapt: pass batch_descriptor to set_forward_context
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode, # Adapt : get cudagraph mode from model runner
            batch_descriptor=batch_descriptor, # Adapt : get batch_descriptor from model runner
        ):
            forward_context = get_forward_context()
            forward_context.capturing = False
        # End Adapt
            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=self._get_positions(num_input_tokens),
                hidden_states=self.hidden_states[:num_input_tokens],
                inputs_embeds=inputs_embeds,
            )
            if self.method == "mtp":
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            draft_token_ids = logits.argmax(dim=-1)
            return draft_token_ids.view(-1, 1)

        if self.uses_mrope:
            positions = target_positions[:, last_token_indices]
        else:
            positions = target_positions[last_token_indices]
        # Adapt: remove deepseek_mtp
        if self.method in (
            "ernie_mtp",
            "longcat_flash_mtp",
            "openpangu_mtp",
        ):
            hidden_states = self.hidden_states[last_token_indices]
        else:
            hidden_states = hidden_states[last_token_indices]

        if any(isinstance(metadata, TreeAttentionMetadata)
               for metadata in per_layer_attn_metadata.values()):
            # Draft using tree attention.
            draft_token_ids_list = self.propose_tree(
                batch_size=batch_size,
                logits=logits,
                positions=positions,
                hidden_states=hidden_states,
                common_attn_metadata=common_attn_metadata,
            )
            # Return shape: [batch_size, num_tree_tokens]
            return torch.cat(draft_token_ids_list, dim=1)

        draft_token_ids = logits.argmax(dim=-1)

        if self.allowed_attn_types is not None:
            for layer_name, metadata in per_layer_attn_metadata.items():
                if not isinstance(metadata, self.allowed_attn_types):
                    raise ValueError(
                        f"Unsupported attention metadata type for speculative "
                        f"decoding with num_speculative_tokens > 1: "
                        f"{type(metadata)} (layer {layer_name}). Supported types "
                        f"are: {self.allowed_attn_types}"
                    )

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        if cudagraph_runtime_mode == CUDAGraphMode.NONE:
            batch_descriptor = None
            batch_size_dp_padded, batch_size_across_dp = self._pad_batch_across_dp(
                num_tokens_unpadded=batch_size, num_tokens_padded=batch_size
            )
            input_batch_size = batch_size_dp_padded
            if batch_size_across_dp is not None:
                batch_size_across_dp[self.dp_rank] = input_batch_size
        else:
            input_batch_size = num_input_tokens
            batch_size_across_dp = num_tokens_across_dp

        common_attn_metadata.num_actual_tokens = input_batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc[: batch_size + 1] = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc[batch_size:] = self.arange[batch_size]
        common_attn_metadata.query_start_loc_cpu[: batch_size + 1] = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()
        common_attn_metadata.query_start_loc_cpu[batch_size:] = common_attn_metadata.query_start_loc_cpu[batch_size]

        # In padded drafter batch, we need to adjust the sequence lengths
        # to remove the "padding" (i.e. rejected tokens).
        # Only apply this adjustment when we have rejected tokens
        # (i.e., not the first proposal).
        if self.num_speculative_tokens > 1 and num_rejected_tokens_gpu is not None:
            actual_batch_size = num_rejected_tokens_gpu.shape[0]
            common_attn_metadata.seq_lens[:actual_batch_size] -= num_rejected_tokens_gpu
            # Invalidate the CPU-side shadows to avoid H<>D sync.
            common_attn_metadata._seq_lens_cpu = None
            common_attn_metadata._num_computed_tokens_cpu = None

        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()
            if self.uses_mrope:
                positions += 1
                # NOTE(woosuk): We should handle the case where the draft model
                # generates tokens beyond the max model length.
                # Since it is complex to remove such requests from the batch,
                # we keep them in the batch but adjust the position ids
                # and slot mappings to avoid the
                # out-of-range access during the model execution.
                # The draft tokens generated with this adjustment
                # should be ignored.
                exceeds_max_model_len = positions[0] >= self.max_model_len
                # Mask out the position ids that exceed the max model length.
                # Otherwise, we may get out-of-range error in RoPE.
                clamped_positions = torch.where(
                    exceeds_max_model_len.unsqueeze(0),
                    torch.zeros_like(positions),
                    positions,
                )
            else:
                positions += 1
                exceeds_max_model_len = positions >= self.max_model_len
                clamped_positions = torch.where(exceeds_max_model_len, 0, positions)
            # For data integrity when async scheduling, we shouldn't use in place
            # operations in case they are modified in next step's `prepare_input`
            # of main model.
            # Increment the sequence lengths.
            common_attn_metadata.seq_lens[:batch_size] += 1
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            common_attn_metadata.seq_lens[:batch_size].masked_fill_(exceeds_max_model_len, 1)

            # Also update the CPU-side shadow; NOTE: this is hacky and should be
            # removed in when common_attn_metadata.seq_lens_cpu is deprecated.
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu[:batch_size] += 1
            if common_attn_metadata._num_computed_tokens_cpu is not None:
                common_attn_metadata._num_computed_tokens_cpu[:batch_size] += 1

            # Compute slot mapping and rebuild attention metadata for all
            # draft groups (each with its own block table and block size).
            self._rebuild_per_group_metadata_for_step(
                common_attn_metadata=common_attn_metadata,
                clamped_positions=clamped_positions,
                exceeds_max_model_len=exceeds_max_model_len,
                token_index=token_index,
                per_layer_attn_metadata=per_layer_attn_metadata,
            )

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self._set_positions(batch_size, clamped_positions)
            self.hidden_states[:batch_size] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:input_batch_size]
            else:
                input_ids = self.input_ids[:input_batch_size]
                inputs_embeds = None

            # Run the model.
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor, # Adapt : get batch_descriptor from model runner
            ):
                forward_context = get_forward_context()
                forward_context.capturing = False
                ret_hidden_states = self.model(
                    input_ids=input_ids,
                    positions=self._get_positions(input_batch_size),
                    hidden_states=self.hidden_states[:input_batch_size],
                    inputs_embeds=inputs_embeds,
                )
                if self.method == "mtp":
                    last_hidden_states = ret_hidden_states
                    hidden_states = ret_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states
            hidden_states = hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size])
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # Output shape: [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def propose_multi_mtp(
        self,
        target_token_ids: torch.Tensor,  # shape: [num_tokens]
        target_positions: torch.Tensor,  # shape: [num_tokens] or [3, num_tokens] when M-RoPE
        target_hidden_states: torch.Tensor,  # shape: [num_tokens, hidden_size]
        next_token_ids: torch.Tensor,  # shape: [batch_size]
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:batch_size + 1] - 1

        self._save_and_change_target_input(
            target_token_ids,
            target_positions,
            target_hidden_states,
            next_token_ids,
            last_token_indices,
            common_attn_metadata,
            num_rejected_tokens_gpu,
        )

        if self.runner is None:
            raise ValueError("runner is not initialized")

        per_layer_attn_metadata = self._build_per_group_metadata(
            common_attn_metadata,
            draft_index=0,
        )

        # Adapt start: get token info from target model batch descriptor
        if self.runner.batch_execution_and_padding_state is None:
            raise ValueError(
                "propose of drafter should be executed after "
                "runner._determine_batch_execution_and_padding"
            )
        cudagraph_runtime_mode, batch_descriptor, num_tokens_across_dp = self.runner.batch_execution_and_padding_state
        self.runner.batch_execution_and_padding_state = None
        num_input_tokens = batch_descriptor.num_tokens
        # Adapt end

        # copy inputs to buffer for cudagraph
        self._set_positions(num_tokens, target_positions)

        previous_input_ids = target_token_ids
        previous_next_token_ids = next_token_ids
        previous_hidden_states = target_hidden_states
        draft_token_ids_list = []
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode, # Adapt : get cudagraph mode from model runner
            batch_descriptor=batch_descriptor, # Adapt : get batch_descriptor from model runner
        ):
            forward_context = get_forward_context()
            forward_context.capturing = False
            for token_index in range(self.num_speculative_tokens):

                # Shift the input ids by one token.
                # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
                self.input_ids[: num_tokens] = torch.roll(previous_input_ids, -1)
                # Replace the last token with the next token.
                # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
                self.input_ids[last_token_indices] = previous_next_token_ids

                self.hidden_states[:num_tokens] = previous_hidden_states

                if self.supports_mm_inputs:
                    mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

                    self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                        self.input_ids[:num_tokens],
                        multimodal_embeddings=mm_embeds,
                        is_multimodal=is_mm_embed,
                    )

                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]
                else:
                    self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                        self.input_ids[:num_tokens],
                    )
                    input_ids = None
                    inputs_embeds = self.inputs_embeds[:num_input_tokens]

                ret_hidden_states = self.model(
                    input_ids=input_ids,
                    positions=self._get_positions(num_input_tokens),
                    hidden_states=self.hidden_states[:num_input_tokens],
                    inputs_embeds=inputs_embeds,
                    spec_step_idx=token_index,
                )
                if self.method == "mtp":
                    last_hidden_states = ret_hidden_states
                    hidden_states = last_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states
                sample_hidden_states = last_hidden_states[last_token_indices]
                logits = self.model.compute_logits(
                    hidden_states=sample_hidden_states,
                    spec_step_idx=token_index,
                )
                new_next_token_ids = logits.argmax(dim=-1).int()
                draft_token_ids_list.append(new_next_token_ids)
                
                # update previous
                previous_input_ids = self.input_ids[: num_tokens]
                previous_hidden_states = last_hidden_states[: num_tokens]
                previous_next_token_ids = new_next_token_ids

        return torch.stack(draft_token_ids_list, dim=1)

    def _save_and_change_target_input(
        self,
        target_token_ids,
        target_positions,
        target_hidden_states,
        next_token_ids,
        last_token_indices,
        common_attn_metadata,
        num_rejected_tokens_gpu,
    ):
        if self.fix_multi_mtp_kvcache:
            device = target_token_ids.device
            batch_size = next_token_ids.numel()
            cu_num_draft_tokens = self.runner.cu_num_draft_tokens.gpu[:batch_size]
            num_draft_tokens = cu_num_draft_tokens.clone()
            num_draft_tokens[1:] = cu_num_draft_tokens[1:] - cu_num_draft_tokens[:-1]
            has_draft_tokens = num_draft_tokens > 0

            basic_range = torch.arange(
                1 + self.num_speculative_tokens,
                device=device, dtype=common_attn_metadata.query_start_loc.dtype
            )
            # FIXME: this assumes every request contributes >= (num_speculative_tokens + 1)
            # query tokens (always true for a decode step: 1 token + num_speculative_tokens
            # drafts). In PD-mixed batches a prefilling request whose prompt is shorter than
            # (num_speculative_tokens + 1) makes final_n_token_indices underflow past its own
            # slice start into the previous request (or go negative), so the gather/write-back
            # touches the neighbour's tokens/KV. The very-short-prompt case is not handled yet;
            # currently safe only because fix_multi_mtp_kvcache is default-off.
            final_n_token_indices = (
                common_attn_metadata.query_start_loc[1:batch_size + 1, None]
                - self.num_speculative_tokens - 1 + basic_range)

            final_token_ids = target_token_ids[final_n_token_indices]
            final_hidden_states = target_hidden_states[final_n_token_indices]

            previous_token_ids = self.input_batch.target_token_ids_cache[:batch_size]
            previous_hidden_states = self.input_batch.target_model_hidden_states_cache[:batch_size]

            token_ids = torch.cat([previous_token_ids, final_token_ids], dim=1)
            hidden_states = torch.cat([previous_hidden_states, final_hidden_states], dim=1)

            selected_indices = basic_range[None, :] + 1 + self.num_speculative_tokens + torch.arange(
                batch_size, dtype=basic_range.dtype, device=device,
            )[:, None] * (1 + self.num_speculative_tokens) * 2
            if num_rejected_tokens_gpu is not None:
                selected_indices = torch.where(
                    has_draft_tokens[:, None],
                    selected_indices - num_rejected_tokens_gpu[:, None],
                    selected_indices,
                )

            selected_indices = selected_indices.view(-1)
            seleted_token_ids = token_ids.view(-1)[selected_indices]
            seleted_hidden_states = hidden_states.view(-1, hidden_states.shape[-1])[selected_indices]

            target_token_ids[final_n_token_indices.view(-1)] = seleted_token_ids
            target_hidden_states[final_n_token_indices.view(-1)] = seleted_hidden_states
            self.input_batch.target_token_ids_cache[:batch_size] = seleted_token_ids.view(batch_size, -1)
            self.input_batch.target_model_hidden_states_cache[:batch_size] = seleted_hidden_states.view(
                batch_size, -1, seleted_hidden_states.shape[-1])

            last_token_indices[:] = common_attn_metadata.query_start_loc[1:batch_size + 1] - 1
            if num_rejected_tokens_gpu is not None:
                target_positions[final_n_token_indices.view(-1)] = torch.where(
                    has_draft_tokens[:, None],
                    target_positions[final_n_token_indices] - num_rejected_tokens_gpu[:, None],
                    target_positions[final_n_token_indices],
                ).view(-1)

            self.runner.num_accepted_tokens.gpu[:batch_size].fill_(1 + self.num_speculative_tokens)
