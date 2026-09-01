# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU V2 model runner.

The upstream runner is imported at module scope, which is safe because this
module itself is imported lazily from npu_worker's V2 branch -- the MRv1 path
never reaches it (pinned by test_mrv1_path_does_not_import_upstream_gpu_package).
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from omni_npu.worker.npu.utils import install_torch_cuda_aliases

logger = init_logger(__name__)


def _dp_lmhead_enabled() -> bool:
    """Whether the LM head runs its DP/local collectives."""
    from omni_npu.model_config.config_loader.loader import model_extra_config

    parall = model_extra_config.parall_config
    return bool(getattr(parall, "ena_dp_lmhead_parallel", False)
                or getattr(parall, "ena_local_lmhead_parallel", False))


class NPUModelRunnerV2(GPUModelRunner):
    """NPU runner using the upstream V2 implementation."""

    def __init__(self, vllm_config, device):
        """Alias torch.cuda onto torch.npu, then build the upstream runner.

        The aliases have to precede super().__init__, which creates a
        torch.cuda.Stream on its third line. They stay here rather than in a
        patch because rewriting torch.cuda is global and MRv1 must not get it;
        the module patches themselves are applied by the plugin (see
        omni/vllm_patches/usefull_patch/patch_mrv2_*.py).
        """
        install_torch_cuda_aliases()
        logger.info("[omni-npu/mrv2] building NPUModelRunnerV2")
        super().__init__(vllm_config, device)

    def prepare_inputs(self, scheduler_output, batch_desc):
        """Stash the step's cudagraph mode for prepare_attn.

        Upstream hands prepare_attn only the input batch. The two run
        back to back on one thread with no other caller, so the instance
        is a safe place to carry it.
        """
        self._omni_cg_mode = batch_desc.cg_mode
        return super().prepare_inputs(scheduler_output, batch_desc)

    def prepare_attn(self, input_batch):
        """Pad the slot mappings only for FULL graphs, as MRv1 does.

        MRv1 drives pad_attn = cudagraph_mode == FULL through both the
        slot mapping and the attention metadata, so
        len(slot_mapping) == num_actual_tokens holds and DP padding
        never reaches attention or KV. V2 pads slot mappings always,
        which only works while eager never pads -- the early return
        overridden in dp_utils. has_separate_kv_update is MRv1's second
        term (False today, kept so a change cannot break silently).
        """
        from vllm.config.compilation import CUDAGraphMode

        block_tables, slot_mappings = super().prepare_attn(input_batch)

        cg_mode = getattr(self, "_omni_cg_mode", None)
        if cg_mode is None:
            raise RuntimeError(
                "[omni-npu/mrv2] prepare_attn ran without a stashed "
                "cudagraph mode; prepare_inputs no longer precedes it")

        pad_attn = cg_mode == CUDAGraphMode.FULL
        if pad_attn or self._omni_has_separate_kv_update():
            return block_tables, slot_mappings

        num_tokens = input_batch.num_tokens
        if slot_mappings.shape[-1] > num_tokens:
            slot_mappings = slot_mappings[..., :num_tokens]
        return block_tables, slot_mappings

    def _omni_has_separate_kv_update(self) -> bool:
        """Whether any backend updates the KV cache outside forward()."""
        cached = getattr(self, "_omni_separate_kv_update", None)
        if cached is not None:
            return cached

        from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec

        cached = False
        for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):
            if isinstance(group.kv_cache_spec, EncoderOnlyAttentionSpec):
                continue

            for attn_group in self.attn_groups[gid]:
                if not attn_group.backend.forward_includes_kv_cache_update:
                    cached = True
                    break

            if cached:
                break
        self._omni_separate_kv_update = cached
        return cached

    def _sync_dummy_main_compute_logits(self, is_profile: bool) -> None:
        """Mirror MRv1's dummy target lm_head sync point.

        Active MRv2 ranks run target ``compute_logits`` after target forward
        and before speculative drafter/propose. Upstream dummy ranks run the
        same target forward through ``execute_model(dummy_run=True)``, then
        continue into the drafter without sampling. When lm_head is sharded
        over DP/local ranks, ``compute_logits`` contains collectives, so idle
        ranks must join them at this exact boundary.
        """
        if is_profile or not _dp_lmhead_enabled():
            return

        state = getattr(self, "execute_model_state", None)
        if state is None:
            return

        hidden_states = getattr(state, "hidden_states", None)
        if hidden_states is None:
            return

        input_batch = state.input_batch
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        self.model.compute_logits(sample_hidden_states)

    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
    ):
        output = super().execute_model(
            scheduler_output,
            intermediate_tensors=intermediate_tensors,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            is_profile=is_profile,
        )
        if dummy_run:
            self._sync_dummy_main_compute_logits(is_profile)
        return output
