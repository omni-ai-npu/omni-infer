# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
import os
import types
from copy import deepcopy
from pathlib import Path
from vllm.logger import init_logger

import torch
import numpy as np

logger = init_logger("vllm.v1.omni")

# Import mock_schedule at module level — this runs as soon as plugin.py
# is loaded by omni-npu, which is the EARLIEST hook point that works
# in all processes (main, EngineCore-fork, Worker-fork).
try:
    from tools.diagnostics import mock_schedule  # noqa: F401

    mock_schedule.install_in_worker()
except Exception:
    pass

try:
    from tools.diagnostics import input_swap  # noqa: F401

    input_swap.install()
except Exception:
    pass

_DEBUG = int(os.getenv("OMNI_CACHE_DEBUG", "0"))

from omni_cache.cache.core.constants import ENABLE_HOST_MAPPING
from omni_cache.cache.utils.debug import (
    apc_debug_enabled,
    should_log_rank,
    summarize_array,
    summarize_map,
)


def staged_bts_np(bt_item):
    """Return the numpy-backed block_table for a per-group staged entry.

    `bt_item` is one element of `input_batch.block_table.block_tables`.
    Depending on build it may be a `BlockTable` (with `.block_table.np`)
    or a `StagedWriteTensor` (with `.gpu`). Returns None when neither
    attribute exists so callers can skip gracefully.
    """
    if hasattr(bt_item, "block_table") and hasattr(bt_item.block_table, "np"):
        return bt_item.block_table.np
    if hasattr(bt_item, "gpu"):
        return bt_item.gpu.cpu().numpy()
    return None


class LoadModelPlugin:
    """
    Plugin for handling omni_cache initialization in load_model.
    """

    def pre_load(self, *args, **kwargs):
        """
        Pre-load hook called before load_model execution.

        This hook is responsible for initializing decode-side omni cache
        when ENABLE_OMNI_CACHE is enabled and kv_role is kv_consumer.

        Args:
            *args: Positional arguments passed to the decorated method
                   (first arg is self, i.e., NPUWorker instance)
            **kwargs: Keyword arguments passed to the decorated method
        """
        if not int(os.getenv("ENABLE_OMNI_CACHE", "0")):
            return

        if not args:
            return

        # Extract self (NPUWorker instance)
        worker = args[0]

        # Check environment and configuration
        if int(os.getenv("ENABLE_OMNI_CACHE", "0")) and worker.vllm_config.kv_transfer_config.kv_role == "kv_consumer":
            from omni_cache.cache.decode import DecodeOmniCache

            # load_model after omni cache is created to register a larger host tensor in decode side
            DecodeOmniCache.initialize_decode_omni_cache(worker.vllm_config, worker.model_runner)

    def post_load(self, *args, **kwargs):
        # At this point NPUModelRunner is imported in the Worker.
        # Finish input_swap install if the module-level daemon
        # thread hasn't patched yet.
        try:
            from tools.diagnostics.input_swap import install as _swap_install

            _swap_install()
        except Exception:
            pass


class InitConfigPlugin:
    """
    Plugin for handling omni_cache initialization in initialize_from_config.

    This plugin is responsible for calling initialize_omni_kv_cache after
    the default initialize_kv_cache is called, when ENABLE_OMNI_CACHE is enabled.
    """

    def _initialize_omni_kv_cache(self, model_runner, kv_cache_config):
        """
        Initialize omni cache for the model runner.

        This is the original implementation from omni-npu's npu_model_runner.py,
        moved here to allow omni_cache to be installed as a plugin without
        requiring code changes in omni-npu.

        Args:
            model_runner: The NPUModelRunner instance
            kv_cache_config: KV cache configuration
        """
        kv_cache_config = deepcopy(kv_cache_config)
        model_runner.kv_cache_config = kv_cache_config
        model_runner.may_add_encoder_only_layers_to_kv_cache_config()
        model_runner.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        model_runner.initialize_attn_backend(kv_cache_config)

        # The kernel block size for all KV cache groups
        kernel_block_sizes = model_runner._prepare_kernel_block_sizes(kv_cache_config)

        # create metadata builders
        model_runner.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

        # Reinitialize need to after initialize_attn_backend
        model_runner.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)

        if model_runner.speculative_config and model_runner.speculative_config.use_eagle():
            from vllm.v1.spec_decode.eagle import EagleProposer

            assert isinstance(model_runner.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache group
            model_runner.drafter.validate_same_kv_cache_group(kv_cache_config)

        logger.warning(f"<<< {model_runner.kv_cache_config=}")

        from omni_cache.cache import omni_cache

        is_decode = (
            omni_cache is not None and hasattr(omni_cache, "hbm_buffer_pool") and omni_cache.hbm_buffer_pool is not None
        )

        if not is_decode and model_runner.vllm_config.kv_transfer_config.kv_role == "kv_producer":
            from omni_cache.cache.core.base import create_omni_cache

            create_omni_cache(
                kv_cache_config=model_runner.kv_cache_config,
                vllm_config=model_runner.vllm_config,
                runner=model_runner,
            )
            from omni_cache.cache import omni_cache as omni_cache

        model_runner.omni_cache = omni_cache

        omni_cache.update_kv_cache_spec(kv_cache_config, model_runner.vllm_config)
        omni_cache.update_model_runner(model_runner)
        omni_cache.ensure_device_cache_initialized()

        if model_runner.vllm_config.kv_transfer_config.kv_role == "kv_consumer" or is_decode:
            from vllm.v1.worker.utils import bind_kv_cache

            assert omni_cache.device_cache is not None
            # replace kv_a and k_pe in device.kv_caches by host swap caches
            if model_runner.omni_cache.enable_dsa and not model_runner.omni_cache.is_pangu_v2:
                for i, layer_name in enumerate(list(omni_cache.device_cache.keys())):
                    rest = omni_cache.device_cache[layer_name]
                    t0 = omni_cache.host_swap_tensor[i][0]
                    t1 = omni_cache.host_swap_tensor[i][1]
                    logger.warning(f"<<< before bind_kv_cache: {t0.shape=}, {t1.shape=}, {rest[0].shape=}")
                    omni_cache.device_cache[layer_name] = (t0, t1, *rest)

            if model_runner.omni_cache.is_pangu_v2:
                # Create a patched version of bind_kv_cache that removes the layer name check
                from collections import defaultdict
                from vllm.attention.layer import Attention
                from vllm.model_executor.models.utils import extract_layer_index
                import vllm.v1.worker.gpu_model_runner as gpu_model_runner

                def bind_kv_cache_patched(
                    kv_caches: dict[str, torch.Tensor],
                    forward_context: dict[str, Attention],
                    runner_kv_caches: list[torch.Tensor],
                    num_attn_module: int = 1,
                ) -> None:
                    # Bind kv_caches to ModelRunner
                    assert len(runner_kv_caches) == 0

                    # Convert kv_caches dict to a list of tensors in the order of layer_index.
                    index2name = defaultdict(list)
                    for layer_name in kv_caches:
                        index2name[extract_layer_index(layer_name, num_attn_module)].append(layer_name)

                    for layer_index in sorted(index2name.keys()):
                        layer_names = index2name[layer_index]
                        layer_name = layer_names[0]
                        runner_kv_caches.append(kv_caches[layer_name])

                    # Bind kv_caches to forward context
                    for layer_name, kv_cache in kv_caches.items():
                        # NOTE: Use list because of v0 PP virtual engine.
                        forward_context[layer_name].kv_cache = [kv_cache]

                bind_kv_cache = bind_kv_cache_patched

            bind_kv_cache(
                omni_cache.device_cache,
                model_runner.vllm_config.compilation_config.static_forward_context,
                model_runner.kv_caches,
            )

        from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group

        if has_kv_transfer_group:
            get_kv_transfer_group().register_kv_caches(
                omni_cache.MEMMAP_PATH,
                omni_cache.dtype,
                block_len_dtype=omni_cache.block_len_dtype,
                omni_cache=omni_cache,
            )

    def pre_init_config(self, *args, **kwargs):
        """
        Pre-init hook called before initialize_kv_cache execution.

        This hook checks if ENABLE_OMNI_CACHE is enabled. If enabled,
        it returns True to skip the original super().initialize_kv_cache()
        function. The actual omni cache initialization is done in post_init_config.

        If ENABLE_OMNI_CACHE is not enabled, returns False to allow
        the original function to execute.

        Args:
            *args: Positional arguments:
                - args[0]: self (NPUModelRunner instance)
                - args[1]: kv_cache_config (KVCacheConfig)
            **kwargs: Keyword arguments

        Returns:
            bool: True to skip original function, False to execute it
        """
        if not int(os.getenv("ENABLE_OMNI_CACHE", "0")):
            return False  # Execute original function

        return True  # Skip original initialize_kv_cache

    def post_init_config(self, *args, result=None, **kwargs):
        """
        Post-init hook called after initialize_kv_cache execution.

        When ENABLE_OMNI_CACHE=1: This hook executes the omni cache initialization
        after the original super().initialize_kv_cache() was skipped.
        When ENABLE_OMNI_CACHE=0: This hook does nothing (original already executed).

        Args:
            *args: Positional arguments:
                - args[0]: self (NPUModelRunner instance)
                - args[1]: kv_cache_config (KVCacheConfig)
            result: Return value of the decorated method
            **kwargs: Keyword arguments
        """
        if not int(os.getenv("ENABLE_OMNI_CACHE", "0")):
            return

        if len(args) < 2:
            return

        model_runner = args[0]
        kv_cache_config = args[1]

        # Execute omni cache initialization
        self._initialize_omni_kv_cache(model_runner, kv_cache_config)


class InputBatchPlugin:
    """Plugin hook for installing OmniCache's decode InputBatch scaffold."""

    def pre_reinitialize_input_batch(self, *args, **kwargs):
        return None

    def post_reinitialize_input_batch(self, *args, **kwargs):
        if not int(os.getenv("ENABLE_OMNI_CACHE", "0")):
            return
        if not int(os.getenv("USE_OMNI_INPUT_BATCH", "0")):
            return
        if len(args) < 3:
            return

        model_runner = args[0]
        kv_cache_config = args[1]
        kernel_block_sizes = args[2]

        kv_role = getattr(model_runner.vllm_config.kv_transfer_config, "kv_role", None)
        if kv_role != "kv_consumer":
            return

        from omni_cache.cache.input_batch import OmniCacheInputBatch

        current_input_batch = getattr(model_runner, "input_batch", None)
        if isinstance(current_input_batch, OmniCacheInputBatch):
            logger.warning(
                "[OMNI-INPUT-BATCH] input batch already installed; kv_role=%s",
                kv_role,
            )
            return
        if current_input_batch is None:
            return

        kv_cache_specs = self._collect_kv_cache_specs(kv_cache_config)
        block_sizes = [spec.block_size for spec in kv_cache_specs]

        from omni_cache.cache import omni_cache as omni_cache_obj

        if omni_cache_obj is None:
            raise RuntimeError(f"Error! OmniCache object is not found.")
        logger.warning(
            "[OMNI-INPUT-BATCH] installing OmniCacheInputBatch kv_role=%s specs=%s kernel_block_sizes=%s",
            kv_role,
            [type(spec).__name__ for spec in kv_cache_specs],
            list(kernel_block_sizes),
        )

        model_runner.input_batch = OmniCacheInputBatch(
            max_num_reqs=model_runner.max_num_reqs,
            max_model_len=max(
                model_runner.max_model_len,
                getattr(model_runner, "max_encoder_len", 0),
            ),
            max_num_batched_tokens=model_runner.max_num_tokens,
            device=model_runner.device,
            pin_memory=model_runner.pin_memory,
            vocab_size=model_runner.model_config.get_vocab_size(),
            block_sizes=block_sizes,
            kernel_block_sizes=list(kernel_block_sizes),
            kv_cache_specs=kv_cache_specs,
            kv_cache_config=kv_cache_config,
            omni_cache=omni_cache_obj,
            logitsprocs=current_input_batch.logitsprocs,
            logitsprocs_need_output_token_ids=(current_input_batch.logitsprocs_need_output_token_ids),
            is_spec_decode=bool(model_runner.vllm_config.speculative_config),
            is_pooling_model=model_runner.is_pooling_model,
            num_speculative_tokens=getattr(model_runner, "num_spec_tokens", 0),
            cp_kv_cache_interleave_size=self._cp_kv_cache_interleave_size(model_runner, current_input_batch),
        )

        # make OmniCacheInputBatch available to omni_cache itself
        setattr(omni_cache_obj, "input_batch", model_runner.input_batch)

    @staticmethod
    def _collect_kv_cache_specs(kv_cache_config) -> list:
        try:
            from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec
        except Exception:
            EncoderOnlyAttentionSpec = ()  # type: ignore[assignment]

        specs = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            if EncoderOnlyAttentionSpec and isinstance(spec, EncoderOnlyAttentionSpec):
                continue
            specs.append(spec)
        return specs

    @staticmethod
    def _cp_kv_cache_interleave_size(model_runner, input_batch) -> int:
        block_table = getattr(input_batch, "block_table", None)
        block_tables = getattr(block_table, "block_tables", None)
        if isinstance(block_tables, (list, tuple)) and block_tables:
            size = getattr(block_tables[0], "cp_kv_cache_interleave_size", None)
            if size is not None:
                return int(size)
        parallel_config = getattr(model_runner, "parallel_config", None)
        return int(getattr(parallel_config, "cp_kv_cache_interleave_size", 1))


class PrepareInputsPlugin:
    """Handle omni_cache operations in prepare_inputs.

    This plugin is responsible for updating selection_kv_block workspace
    when using gather selection in DSA mode.
    """

    def pre_prepare_inputs(self, *args, **kwargs):
        """Pre-prepare hook. For prefill with OMNI_CACHE_ENABLE_VOLATILE=1,
        rewrite all groups' block_table.np with flat fake IDs (1..N) and
        stash the real IDs on input_batch for D2H. vLLM's own pipeline
        then sees only fake IDs for commit_block_table / slot_mapping /
        metadata build."""
        if _DEBUG:
            logger.warning(
                "[PRE-PREP] fired nargs=%d ENABLE_OC=%s ENABLE_VOL=%s",
                len(args),
                os.getenv("ENABLE_OMNI_CACHE", "0"),
                os.getenv("OMNI_CACHE_ENABLE_VOLATILE", "0"),
            )
        if not int(os.getenv("ENABLE_OMNI_CACHE", "0")):
            return
        # PACKED_HBM shrinks the physical buffer to ~100 blocks.  Without
        # the volatile swap remapping real block IDs into the packed range,
        # kernels can access OOB and trigger AICore crashes (507015).
        if not int(os.getenv("OMNI_CACHE_ENABLE_VOLATILE", "0")) and not int(os.getenv("OMNI_CACHE_PACKED_HBM", "0")):
            return
        if len(args) < 1:
            return
        model_runner = args[0]
        has_oc = hasattr(model_runner, "omni_cache")
        oc_none = has_oc and model_runner.omni_cache is None
        if not has_oc or oc_none:
            return
        kv_role = getattr(model_runner.vllm_config.kv_transfer_config, "kv_role", None)
        if kv_role != "kv_producer":
            return
        input_batch = getattr(model_runner, "input_batch", None)
        if input_batch is None:
            return
        # pre_prepare_inputs fires AFTER GPUModelRunner._update_states
        # has called `append_row` (populating block_table.np with the
        # current batch's real block_ids) and BEFORE super()._prepare_inputs
        # runs commit_block_table / compute_slot_mapping / metadata build.
        # Rewrite np to flat fake IDs in place; vLLM's own pipeline then
        # pushes fake values through the whole derivation chain.
        num_reqs = int(getattr(input_batch, "num_reqs", 0))
        if num_reqs <= 0:
            return
        try:
            if _DEBUG:
                logger.warning("[VOLATILE-SWAP] firing for %d reqs", num_reqs)
            self._apply_volatile_swap_to_np(input_batch, num_reqs)
        except Exception as e:
            logger.warning("[VOLATILE-SWAP] swap failed: %s", e)
            if _DEBUG:
                logger.warning("[pre_prepare_inputs] swap failed: %s", e)

    def post_prepare_inputs(self, *args, **kwargs):
        """
        Post-prepare hook called after prepare_inputs execution.

        This hook checks if DSA and omni_cache are enabled, and updates
        the selection_kv_block workspace when using gather selection.

        Args:
            *args: Positional arguments
                - args[0]: self (ModelRunner instance)
                - args[1]: scheduler_output (SchedulerOutput)
                - args[2]: num_scheduled_tokens (List)
            result: Return value of prepare_inputs (InputBatch)
            **kwargs: Keyword arguments
        """
        if _DEBUG:
            logger.warning("[POST-PREP] fired nargs=%d", len(args))
        if not int(os.getenv("ENABLE_OMNI_CACHE", "0")):
            return

        model_runner = args[0]
        input_batch = getattr(model_runner, "input_batch", None)

        if not (hasattr(model_runner, "omni_cache") and model_runner.omni_cache is not None):
            return

        kv_role = getattr(model_runner.vllm_config.kv_transfer_config, "kv_role", None)

        # Prefill (kv_producer): swap is restored at the first D2H entry,
        # after attention metadata/prefix_meta have been built from fake ids.
        if kv_role == "kv_producer":
            return

        if kv_role != "kv_consumer":
            return

        if model_runner.omni_cache.use_input_batch_lane_mapping:
            return

        if not ENABLE_HOST_MAPPING:
            return

        # Record the current batch order for downstream metadata consumers.
        # With USE_OMNI_INPUT_BATCH=1, H2D owns lane reservation and this hook
        # only syncs row order / device lookup state.
        from omni_cache.cache.decode.static_utils import record_current_batch_order

        record_current_batch_order(input_batch, model_runner.omni_cache)

        if model_runner.omni_cache.enable_gs:
            from omni_cache.cache import omni_cache
            from omni_cache.cache.decode import DecodeOmniCache

            DecodeOmniCache.maybe_update_selection_kv_block_status(input_batch, omni_cache, args[2])

    def _apply_volatile_swap_to_np(self, input_batch, num_reqs):
        """Inline np-only swap. Called from the monkey-patched
        `commit_block_table(num_reqs)` BEFORE vLLM's own cpu→gpu copy.
        Does NOT call copy_to_gpu — vLLM's commit handles that.

        Writes fake ids in-place to each group's `block_table.np`.
        Stashes the pre-swap real ids on
        `input_batch._real_block_tables_per_group` for D2H.
        """
        import omni_cache.cache as _cm

        _oc = _cm.omni_cache
        if _oc is None:
            return
        from omni_cache.cache.prefill import PrefillOmniCache

        if not isinstance(_oc, PrefillOmniCache):
            return
        bt_mgr = input_batch.block_table
        staged_bts = getattr(bt_mgr, "block_tables", None)
        if staged_bts is None:
            return
        # 1. Collect all non-zero real block_ids across groups in traversal
        # order and assign flat fake ids 1..N. Stash the pre-swap np
        # arrays for D2H.
        # Iterate only the VALID cells (columns written by append_row for
        # this request) using num_blocks_per_row. Cells beyond that may
        # contain stale fake-IDs from previous requests — pulling them
        # into real_to_fake would pollute the mapping and reorder fake
        # counter assignments across requests.
        # Per-group real→fake maps with a GLOBAL fake counter — each
        # group's real_id=X gets its own fake, distinct from any other
        # group's real_id=X. They refer to different physical blocks in
        # each group's independent allocator; using a single shared
        # dict would collide them in the shared (packed) HBM.
        fake_counter = 1
        per_group_real = []
        per_group_restore_real = []
        per_group_maps = []
        n_reqs = int(num_reqs) if num_reqs is not None else 0
        prev_maps = getattr(input_batch, "_volatile_real_to_fake_per_group", None)
        if prev_maps is None:
            prev_maps = getattr(_oc, "_volatile_real_to_fake_per_group", None)
        apc_dbg = apc_debug_enabled() and should_log_rank(_oc)
        if apc_dbg:
            logger.warning(
                "[APCDBG/SWAP] tp_rank=%s dp_rank=%s stage=%s group_idx=%s layer_name=%s num_reqs=%d",
                getattr(_oc, "tp_rank", None),
                getattr(_oc, "dp_local_rank", None),
                getattr(_oc, "stage_record", None),
                None,
                None,
                n_reqs,
            )

        # Import MambaSpec for type checking
        try:
            from vllm.v1.kv_cache_interface import MambaSpec
        except Exception:
            MambaSpec = None  # type: ignore[assignment]

        for grp_idx, bt_item in enumerate(staged_bts):
            if not (hasattr(bt_item, "block_table") and hasattr(bt_item.block_table, "np")):
                per_group_real.append(None)
                per_group_restore_real.append(None)
                per_group_maps.append({})
                continue

            kv_cache_spec = _oc.kv_cache_config.kv_cache_groups[grp_idx].kv_cache_spec
            sliding_window = getattr(kv_cache_spec, "sliding_window", None)

            bt_np = bt_item.block_table.np
            num_computed_tokens_cpu = getattr(input_batch, "num_computed_tokens_cpu", None)
            num_blocks_per_row = getattr(bt_item, "num_blocks_per_row", None)

            def valid_cols(row, bt_np, num_blocks_per_row=None):
                if num_blocks_per_row is None:
                    return bt_np.shape[1]
                return max(0, min(int(num_blocks_per_row[row]) + 1, bt_np.shape[1]))

            if apc_dbg:
                logger.warning(
                    "[APCDBG/SWAP] tp_rank=%s dp_rank=%s stage=%s "
                    "group_idx=%d layer_name=%s spec=%s block_size=%s "
                    "%s %s %s",
                    getattr(_oc, "tp_rank", None),
                    getattr(_oc, "dp_local_rank", None),
                    getattr(_oc, "stage_record", None),
                    grp_idx,
                    None,
                    type(kv_cache_spec).__name__,
                    getattr(bt_item, "block_size", None),
                    summarize_array("num_blocks_per_row", num_blocks_per_row),
                    summarize_array("num_computed_tokens_cpu", num_computed_tokens_cpu),
                    summarize_array("bt_np_pre", bt_np[:n_reqs]),
                )

            prev_group_map = prev_maps[grp_idx] if prev_maps is not None and grp_idx < len(prev_maps) else None
            if prev_group_map and num_computed_tokens_cpu is not None:
                fake_to_real = {int(v): int(k) for k, v in prev_group_map.items()}
                block_size = int(getattr(bt_item, "block_size", 1))
                for i in range(n_reqs):
                    computed_tokens = int(num_computed_tokens_cpu[i])
                    computed_blocks = (computed_tokens + block_size - 1) // block_size if computed_tokens > 0 else 0
                    prefix_cols = min(computed_blocks, valid_cols(i, bt_np, num_blocks_per_row))

            if apc_dbg:
                if prev_group_map and num_computed_tokens_cpu is not None:
                    logger.warning(
                        "[APCDBG/SWAP] tp_rank=%s dp_rank=%s stage=%s group_idx=%d layer_name=%s after_realize %s",
                        getattr(_oc, "tp_rank", None),
                        getattr(_oc, "dp_local_rank", None),
                        getattr(_oc, "stage_record", None),
                        grp_idx,
                        None,
                        summarize_array("bt_np_realized", bt_np[:n_reqs]),
                    )

            per_group_restore_real.append(bt_np[:n_reqs].copy() if n_reqs else bt_np[:0].copy())

            # MambaSpec: zero out blocks for already-computed tokens so they
            # are excluded from the volatile swap mapping and D2H.
            if MambaSpec is not None and isinstance(kv_cache_spec, MambaSpec) and num_computed_tokens_cpu is not None:
                for i in range(n_reqs):
                    computed_blocks = int(num_computed_tokens_cpu[i]) // bt_item.block_size
                    out_of_window = max(0, computed_blocks - 2)
                    if out_of_window > 0:
                        bt_np[i, :out_of_window] = 0

            # Sliding-window: zero out blocks that have fallen outside the
            # window so they are excluded from the volatile swap and D2H.
            # A block is out-of-window if its position < computed_blocks - window_blocks.
            elif sliding_window is not None and num_computed_tokens_cpu is not None:
                window_blocks = int(sliding_window) // bt_item.block_size + 2
                for i in range(n_reqs):
                    computed_blocks = int(num_computed_tokens_cpu[i]) // bt_item.block_size
                    out_of_window = max(0, computed_blocks - window_blocks)
                    if out_of_window > 0:
                        bt_np[i, :out_of_window] = 0
            if apc_dbg:
                logger.warning(
                    "[APCDBG/SWAP] tp_rank=%s dp_rank=%s stage=%s group_idx=%d layer_name=%s after_zero %s",
                    getattr(_oc, "tp_rank", None),
                    getattr(_oc, "dp_local_rank", None),
                    getattr(_oc, "stage_record", None),
                    grp_idx,
                    None,
                    summarize_array("bt_np_zeroed", bt_np[:n_reqs]),
                )
            per_group_real.append(bt_np[:n_reqs].copy() if n_reqs else bt_np[:0].copy())
            group_map = {}
            for row in range(n_reqs):
                n_cols = valid_cols(row, bt_np, num_blocks_per_row)
                for col in range(n_cols):
                    r = int(bt_np[row, col])
                    if r != 0 and r not in group_map:
                        group_map[r] = fake_counter
                        fake_counter += 1
            per_group_maps.append(group_map)
            if apc_dbg:
                logger.warning(
                    "[APCDBG/SWAP] tp_rank=%s dp_rank=%s stage=%s group_idx=%d layer_name=%s fake_counter=%d %s",
                    getattr(_oc, "tp_rank", None),
                    getattr(_oc, "dp_local_rank", None),
                    getattr(_oc, "stage_record", None),
                    grp_idx,
                    None,
                    fake_counter,
                    summarize_map("real_to_fake", group_map),
                )
        if fake_counter == 1:
            return

        # Build per-group CPU LUT tensors from the dict maps.
        per_group_luts = []
        for group_map in per_group_maps:
            if not group_map:
                per_group_luts.append((None, None))
                continue
            _keys = torch.tensor(list(group_map.keys()), dtype=torch.int64)
            _vals = torch.tensor(list(group_map.values()), dtype=torch.int64)
            r2f = torch.arange(int(_keys.max()) + 1, dtype=torch.int64)
            r2f[_keys] = _vals
            f2r = torch.arange(int(_vals.max()) + 1, dtype=torch.int64)
            f2r[_vals] = _keys
            per_group_luts.append((r2f, f2r))

        # Rewrite only the VALID cells in place using numpy vectorized ops.
        for grp_idx, (bt_item, group_map) in enumerate(zip(staged_bts, per_group_maps)):
            if not group_map:
                continue
            if not (hasattr(bt_item, "block_table") and hasattr(bt_item.block_table, "np")):
                continue
            bt_np = bt_item.block_table.np
            num_blocks_per_row = getattr(bt_item, 'num_blocks_per_row', None)
            r2f_lut, _ = per_group_luts[grp_idx]
            if r2f_lut is not None:
                r2f_np = r2f_lut.numpy()
                lut_max = r2f_np.shape[0] - 1
                view = bt_np[:n_reqs]
                clipped = view.clip(0, lut_max)
                remapped = r2f_np[clipped]
                if num_blocks_per_row is not None:
                    col_idx = np.arange(bt_np.shape[1])[np.newaxis, :]
                    limit = np.array([
                        max(0, min(int(num_blocks_per_row[r]) + 1, bt_np.shape[1]))
                        for r in range(n_reqs)
                    ], dtype=np.int64)[:, np.newaxis]
                    mask = (col_idx < limit) & (view != 0)
                else:
                    mask = view != 0
                view[mask] = remapped[mask]
            if apc_dbg:
                logger.warning(
                    "[APCDBG/SWAP] tp_rank=%s dp_rank=%s stage=%s group_idx=%d layer_name=%s after_rewrite %s",
                    getattr(_oc, "tp_rank", None),
                    getattr(_oc, "dp_local_rank", None),
                    getattr(_oc, "stage_record", None),
                    grp_idx,
                    None,
                    summarize_array("bt_np_fake", bt_np[:n_reqs]),
                )
        # Stash per-group pre-swap real IDs for D2H.
        # A new volatile swap means this step has not been restored yet;
        # the first D2H entry will flip it back to real block ids.
        input_batch._volatile_swap_np_restored = False
        input_batch._real_block_tables_per_group = per_group_real
        input_batch._restore_real_block_tables_per_group = per_group_restore_real
        input_batch._volatile_real_to_fake_per_group = per_group_maps
        input_batch._volatile_per_group_luts = per_group_luts

        # Also stash on omni_cache for next chunk restoration. Keep the
        # zeroed table for D2H scatter, and the full realized table for
        # restoring vLLM's scheduler block table after packed-HBM use.
        _oc._real_block_tables_per_group = per_group_real
        _oc._restore_real_block_tables_per_group = per_group_restore_real
        _oc._volatile_real_to_fake_per_group = per_group_maps
        _oc._volatile_per_group_luts = per_group_luts


def _register_kv_connectors() -> None:
    """Call connector package to register KV connectors into vLLM."""
    try:
        from omni_cache.connector import register_connectors
    except Exception as e:
        logger.warning(
            "omni_cache: failed to import connector.register_connectors, skip KV connector registration: %s",
            e,
        )
        return

    register_connectors()


def _init_cache() -> None:
    """Initialization hook for cache module (currently no-op)."""
    pass


def _init_ox() -> None:
    """Initialization hook for ox binary (path checks, logging, etc.)."""
    pass


def register() -> None:
    """Unified registration entry for the omni-cache plugin."""
    logger.info("omni_cache: starting unified registration (connector/cache/ox)")
    _register_kv_connectors()
    _init_cache()
    _init_ox()
    _register_attn_plugins()
    _init_diagnostics()
    logger.info("omni_cache: unified registration finished")


def _init_diagnostics() -> None:
    """Initialize KV diagnostics (gear resolver + step dumper)."""
    try:
        from tools.diagnostics.dump_controller import install as _diag_install

        _diag_install()
    except Exception:
        pass

    # Import and install mock_schedule monkey-patch
    try:
        from tools.diagnostics import mock_schedule as mock_schedule_module

        mock_schedule_module.install_in_worker()
    except Exception:
        pass

    # Import and install input_swap monkey-patch (OMNI_MOCK_SCHEDULE=2)
    try:
        from tools.diagnostics import input_swap as input_swap_module

        input_swap_module.install()
    except Exception:
        pass


def _register_attn_plugins() -> None:
    """Register Omni-Cache attention plugins.

    This function should be called when omni_cache plugin is loaded.
    It registers attention plugins based on the OMNI_CACHE_ATTN_PLUGINS
    environment variable.

    Usage (in omni-npu import_kernels):
        if os.environ.get("ENABLE_OMNI_CACHE") == "1":
            from omni_cache.plugin import register_attn_plugins
            register_attn_plugins()
    """
    try:
        from omni_cache.attn_plugins import register_omni_cache_plugins

        register_omni_cache_plugins()
    except Exception as e:
        logger.warning(
            "omni_cache: failed to register attention plugins: %s",
            e,
        )


def get_decorators():
    """Return all available attention decorators and register plugins.

    This function is called via entry_points to dynamically
    load decorators into the namespace. It also automatically
    registers the plugins so that decorators can work properly.

    Returns:
        dict: Mapping of decorator names to decorator functions

    """
    try:
        # First, register plugins to ensure decorators can find them
        from .attn_plugins import register_omni_cache_plugins

        register_omni_cache_plugins()
        logger.info("omni_cache: plugins registered via get_decorators()")

        # Then, return the decorators
        from .attn_plugins import (
            compressed_mqa_attn_decorator,
            mqa_attn_decorator,
            mla_attn_decorator,
            dsa_attn_decorator,
        )

        return {
            "compressed_mqa_attn_decorator": compressed_mqa_attn_decorator,
            "mqa_attn_decorator": mqa_attn_decorator,
            "mla_attn_decorator": mla_attn_decorator,
            "dsa_attn_decorator": dsa_attn_decorator,
        }
    except Exception as e:
        logger.warning(
            "omni_cache: failed to get decorators: %s",
            e,
        )
        return {}
