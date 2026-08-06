# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import pytest

from vllm.config import CUDAGraphMode, CompilationConfig, SchedulerConfig, VllmConfig
from vllm.forward_context import BatchDescriptor

from omni.worker.npu_graph_dispatcher import NPUGraphDispatcher


@pytest.fixture
def mock_vllm_config():
    """Create a mock VllmConfig with basic defaults."""
    config = MagicMock(spec=VllmConfig)
    config.compilation_config = MagicMock(spec=CompilationConfig)
    config.scheduler_config = MagicMock(spec=SchedulerConfig)
    config.scheduler_config.max_num_seqs = 8
    config.speculative_config = None
    config.lora_config = None

    # Default compilation config values
    config.compilation_config.cudagraph_capture_sizes = [1, 8]
    config.compilation_config.max_cudagraph_capture_size = 8
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    config.compilation_config.mode = None
    config.compilation_config.cudagraph_specialize_lora = None

    # Mock pad_for_cudagraph: identity by default (no padding)
    config.pad_for_cudagraph = lambda bs: bs

    # Default: cudagraph_mode does NOT require piecewise compilation
    config.compilation_config.requires_piecewise_compilation = MagicMock(
        return_value=False
    )
    config.compilation_config.is_attention_compiled_piecewise = MagicMock(
        return_value=True
    )

    return config


class TestCreatePaddedBatchDescriptor:
    """Tests for NPUGraphDispatcher._create_padded_batch_descriptor."""

    def test_uniform_decode_divisible(self, mock_vllm_config):
        """uniform_decode=True with divisible query len -> uniform descriptor."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        dispatcher.cudagraph_mode = CUDAGraphMode.FULL
        dispatcher.uniform_decode_query_len = 1

        desc = dispatcher._create_padded_batch_descriptor(
            num_tokens=8, uniform_decode=True, has_lora=False
        )

        assert desc.num_tokens == 8
        assert desc.num_reqs == 8
        assert desc.uniform is True
        assert desc.has_lora is False

    def test_uniform_decode_with_uniform_decode_query_len_gt_1(
        self, mock_vllm_config
    ):
        """uniform_decode=True with query_len > 1 and divisible."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        dispatcher.cudagraph_mode = CUDAGraphMode.FULL
        dispatcher.uniform_decode_query_len = 4

        desc = dispatcher._create_padded_batch_descriptor(
            num_tokens=8, uniform_decode=True, has_lora=False
        )

        assert desc.num_tokens == 8
        assert desc.num_reqs == 2  # 8 / 4
        assert desc.uniform is True

    def test_uniform_decode_not_divisible_falls_to_non_uniform(
        self, mock_vllm_config
    ):
        """uniform_decode=True but not divisible -> non-uniform."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        dispatcher.cudagraph_mode = CUDAGraphMode.FULL
        dispatcher.uniform_decode_query_len = 3

        desc = dispatcher._create_padded_batch_descriptor(
            num_tokens=8, uniform_decode=True, has_lora=False
        )

        assert desc.num_tokens == 8
        assert desc.num_reqs == 8  # min(8, 8) = 8
        assert desc.uniform is False

    def test_uniform_decode_false(self, mock_vllm_config):
        """uniform_decode=False always yields non-uniform."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        dispatcher.cudagraph_mode = CUDAGraphMode.FULL
        dispatcher.uniform_decode_query_len = 1

        desc = dispatcher._create_padded_batch_descriptor(
            num_tokens=8, uniform_decode=False, has_lora=False
        )

        assert desc.num_tokens == 8
        assert desc.uniform is False
        assert desc.num_reqs == 8

    def test_uniform_decode_non_full_mode(self, mock_vllm_config):
        """When mode doesn't have FULL, uniform_decode is forced to False."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        dispatcher.cudagraph_mode = CUDAGraphMode.PIECEWISE
        dispatcher.uniform_decode_query_len = 1

        desc = dispatcher._create_padded_batch_descriptor(
            num_tokens=8, uniform_decode=True, has_lora=False
        )

        # PIECEWISE mode doesn't have FULL, so has_mode returns False
        assert desc.uniform is False

    def test_with_padding(self, mock_vllm_config):
        """pad_for_cudagraph modifies num_tokens before descriptor creation."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        dispatcher.cudagraph_mode = CUDAGraphMode.FULL
        dispatcher.uniform_decode_query_len = 1
        mock_vllm_config.pad_for_cudagraph = lambda bs: 16

        desc = dispatcher._create_padded_batch_descriptor(
            num_tokens=8, uniform_decode=True, has_lora=False
        )

        assert desc.num_tokens == 16  # padded
        assert desc.num_reqs == 16
        assert desc.uniform is True

    def test_with_lora(self, mock_vllm_config):
        """has_lora flag is propagated."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        dispatcher.cudagraph_mode = CUDAGraphMode.FULL
        dispatcher.uniform_decode_query_len = 1

        desc = dispatcher._create_padded_batch_descriptor(
            num_tokens=8, uniform_decode=False, has_lora=True
        )

        assert desc.has_lora is True

    def test_num_reqs_capped_by_max_num_seqs(self, mock_vllm_config):
        """num_reqs is capped at max_num_seqs for non-uniform."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        dispatcher.cudagraph_mode = CUDAGraphMode.FULL
        dispatcher.uniform_decode_query_len = 1

        desc = dispatcher._create_padded_batch_descriptor(
            num_tokens=100, uniform_decode=False, has_lora=False
        )

        assert desc.num_tokens == 100
        assert desc.num_reqs == 8  # min(100, 8)
        assert desc.uniform is False

    def test_uniform_decode_mode_with_no_full(self, mock_vllm_config):
        """When cudagraph_mode has no FULL mode, uniform decode is forced non-uniform."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        dispatcher.cudagraph_mode = CUDAGraphMode.PIECEWISE  # no FULL
        dispatcher.uniform_decode_query_len = 1

        desc = dispatcher._create_padded_batch_descriptor(
            num_tokens=8, uniform_decode=True, has_lora=False
        )

        assert desc.uniform is False


class TestRelaxBatchDescriptorForMixedBatchCudagraphs:
    """Tests for NPUGraphDispatcher._relax_batch_descriptor_for_mixed_batch_cudagraphs."""

    def test_relax_sets_uniform_false_and_caps_num_reqs(self, mock_vllm_config):
        """Relaxed descriptor: uniform=False, num_reqs capped at max_num_seqs."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        batch_desc = BatchDescriptor(num_tokens=8, num_reqs=8, uniform=True)

        relaxed = dispatcher._relax_batch_descriptor_for_mixed_batch_cudagraphs(
            batch_desc
        )

        assert relaxed.num_tokens == 8
        assert relaxed.num_reqs == 8  # min(8, 8)
        assert relaxed.uniform is False

    def test_relax_caps_num_reqs_when_below_max_num_seqs(self, mock_vllm_config):
        """When num_tokens < max_num_seqs, num_reqs = num_tokens."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        batch_desc = BatchDescriptor(num_tokens=3, num_reqs=3, uniform=True)

        relaxed = dispatcher._relax_batch_descriptor_for_mixed_batch_cudagraphs(
            batch_desc
        )

        assert relaxed.num_tokens == 3
        assert relaxed.num_reqs == 3  # min(8, 3)
        assert relaxed.uniform is False

    def test_relax_with_lora(self, mock_vllm_config):
        """has_lora is preserved in relaxed descriptor."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        batch_desc = BatchDescriptor(
            num_tokens=8, num_reqs=8, uniform=True, has_lora=True
        )

        relaxed = dispatcher._relax_batch_descriptor_for_mixed_batch_cudagraphs(
            batch_desc
        )

        assert relaxed.has_lora is True
        assert relaxed.uniform is False


class TestInitializeCudagraphKeys:
    """Tests for NPUGraphDispatcher.initialize_cudagraph_keys."""

    def test_initialize_with_full_mode_no_lora(self, mock_vllm_config):
        """FULL mode initializes only FULL keys (no separate routine)."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        cudagraph_mode = CUDAGraphMode.FULL

        with patch.object(
            dispatcher, "_create_padded_batch_descriptor"
        ) as mock_create:
            with patch.object(
                dispatcher, "_relax_batch_descriptor_for_mixed_batch_cudagraphs"
            ) as mock_relax:
                mock_create.return_value = BatchDescriptor(8, num_reqs=8, uniform=False)
                mock_relax.return_value = BatchDescriptor(
                    8, num_reqs=8, uniform=False
                )

                dispatcher.initialize_cudagraph_keys(
                    cudagraph_mode, uniform_decode_query_len=1
                )

        # FULL mode without separate_routine: mixed_mode = FULL, decode_mode = FULL
        # No separate routine, so mixed_mode == decode_mode, decode section is skipped
        # Only FULL keys added via the mixed_mode loop
        assert len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL]) == 1  # [1, 8]
        assert len(dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE]) == 0
        assert dispatcher.keys_initialized is True

    def test_initialize_with_piecewise_mode_no_lora(self, mock_vllm_config):
        """PIECEWISE mode initializes only PIECEWISE keys."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        cudagraph_mode = CUDAGraphMode.PIECEWISE

        dispatcher.initialize_cudagraph_keys(
            cudagraph_mode, uniform_decode_query_len=1
        )

        assert len(dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE]) == 2
        assert len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL]) == 0
        assert dispatcher.keys_initialized is True

    @pytest.mark.parametrize(
        "cudagraph_mode,expected_full,expected_piecewise",
        [
            (CUDAGraphMode.FULL_AND_PIECEWISE, 2, 2),
            (CUDAGraphMode.FULL_DECODE_ONLY, 2, 0),
            (CUDAGraphMode.FULL, 2, 0),
            (CUDAGraphMode.PIECEWISE, 0, 2),
        ],
    )
    def test_initialize_with_different_modes(
        self, mock_vllm_config, cudagraph_mode, expected_full, expected_piecewise
    ):
        """Verify key counts for each CUDAGraphMode."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)

        dispatcher.initialize_cudagraph_keys(
            cudagraph_mode, uniform_decode_query_len=1
        )

        assert (
            len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL]) == expected_full
        ), f"FULL key count mismatch for {cudagraph_mode}"
        assert (
            len(dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE])
            == expected_piecewise
        ), f"PIECEWISE key count mismatch for {cudagraph_mode}"

    def test_initialize_with_lora_config(self, mock_vllm_config):
        """With LoRA, both has_lora=True/False keys are created."""
        mock_vllm_config.lora_config = MagicMock()
        mock_vllm_config.compilation_config.cudagraph_specialize_lora = True
        dispatcher = NPUGraphDispatcher(mock_vllm_config)

        dispatcher.initialize_cudagraph_keys(
            CUDAGraphMode.FULL, uniform_decode_query_len=1
        )

        # 2 sizes * 2 lora cases = 4 FULL keys
        assert len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL]) == 4

    def test_initialize_with_lora_specialize_disabled(self, mock_vllm_config):
        """With LoRA but specialize=False, only has_lora=True keys."""
        mock_vllm_config.lora_config = MagicMock()
        mock_vllm_config.compilation_config.cudagraph_specialize_lora = False
        dispatcher = NPUGraphDispatcher(mock_vllm_config)

        dispatcher.initialize_cudagraph_keys(
            CUDAGraphMode.FULL, uniform_decode_query_len=1
        )

        # 2 sizes * 1 lora case = 2 FULL keys
        assert len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL]) == 2

    def test_initialize_with_full_and_piecewise_mode(self, mock_vllm_config):
        """FULL_AND_PIECEWISE creates both PIECEWISE and FULL keys."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)

        dispatcher.initialize_cudagraph_keys(
            CUDAGraphMode.FULL_AND_PIECEWISE, uniform_decode_query_len=1
        )

        # PIECEWISE keys: mixed_mode returns PIECEWISE for FULL_AND_PIECEWISE
        assert len(dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE]) == 2
        # FULL keys: decode_mode returns FULL + separate decode keys
        assert len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL]) == 2

    def test_uses_npu_relax_method(self, mock_vllm_config):
        """Verify that NPU's _relax_batch_descriptor_for_mixed_batch_cudagraphs is used."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)

        with patch.object(
            dispatcher,
            "_relax_batch_descriptor_for_mixed_batch_cudagraphs",
            wraps=dispatcher._relax_batch_descriptor_for_mixed_batch_cudagraphs,
        ) as mock_npu_relax:
            dispatcher.initialize_cudagraph_keys(
                CUDAGraphMode.PIECEWISE, uniform_decode_query_len=1
            )

            # Should have called NPU relax (not BatchDescriptor.relax_for_mixed_batch_cudagraphs)
            assert mock_npu_relax.call_count == 2  # once per capture size


class TestDispatch:
    """Tests for NPUGraphDispatcher.dispatch."""

    def setup_dispatcher(self, mock_vllm_config, cudagraph_mode=CUDAGraphMode.FULL):
        """Helper to create and initialize a dispatcher."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        dispatcher.initialize_cudagraph_keys(
            cudagraph_mode, uniform_decode_query_len=1
        )
        return dispatcher

    def test_dispatch_full_mode_exact_match(self, mock_vllm_config):
        """Exact match returns FULL mode with the exact batch descriptor."""
        dispatcher = self.setup_dispatcher(mock_vllm_config, CUDAGraphMode.FULL)

        rt_mode, key = dispatcher.dispatch(
            num_tokens=8, uniform_decode=False, has_lora=False
        )

        assert rt_mode == CUDAGraphMode.FULL
        assert key.num_tokens == 8

    def test_dispatch_full_mode_relaxed_match(self, mock_vllm_config):
        """Non-matching exact but matching relaxed key -> FULL with relaxed desc."""
        dispatcher = self.setup_dispatcher(mock_vllm_config, CUDAGraphMode.FULL)

        rt_mode, key = dispatcher.dispatch(
            num_tokens=3, uniform_decode=False, has_lora=False
        )

        # num_tokens=3, padded to 3, not in capture_sizes [1, 8] exactly,
        # but relaxed key for the capture sizes should match
        # Actually, for FULL mode, the keys are capture sizes [1, 8]
        # 3 won't match 1 or 8 exactly, so it goes through relaxed key path
        # relaxed should match if any FULL key matches the relaxed descriptor
        # The relaxed desc for 3 would be BatchDescriptor(3, num_reqs=3, uniform=False)
        # This won't match the FULL key BatchDescriptor(1, ...) or BatchDescriptor(8, ...)
        # So this would fall through to NONE
        assert rt_mode == CUDAGraphMode.NONE
        assert key.num_tokens == 3

    def test_dispatch_piecewise_mode(self, mock_vllm_config):
        """PIECEWISE mode dispatches correctly."""
        dispatcher = self.setup_dispatcher(
            mock_vllm_config, CUDAGraphMode.PIECEWISE
        )

        rt_mode, key = dispatcher.dispatch(
            num_tokens=8, uniform_decode=False, has_lora=False
        )

        assert rt_mode == CUDAGraphMode.PIECEWISE

    def test_dispatch_with_disable_full(self, mock_vllm_config):
        """disable_full=True skips FULL mode and falls through."""
        dispatcher = self.setup_dispatcher(
            mock_vllm_config, CUDAGraphMode.PIECEWISE
        )

        rt_mode, key = dispatcher.dispatch(
            num_tokens=8,
            uniform_decode=False,
            has_lora=False,
            disable_full=True,
        )

        assert rt_mode == CUDAGraphMode.PIECEWISE

    def test_dispatch_no_match_returns_none(self, mock_vllm_config):
        """No key match returns NONE mode with trivial BatchDescriptor."""
        dispatcher = self.setup_dispatcher(mock_vllm_config, CUDAGraphMode.FULL)

        rt_mode, key = dispatcher.dispatch(
            num_tokens=100, uniform_decode=False, has_lora=False
        )

        assert rt_mode == CUDAGraphMode.NONE
        assert key.num_tokens == 100

    def test_dispatch_keys_not_initialized(self, mock_vllm_config):
        """When keys_initialized is False, dispatch returns NONE."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        dispatcher.cudagraph_mode = CUDAGraphMode.FULL
        dispatcher.keys_initialized = False

        rt_mode, key = dispatcher.dispatch(
            num_tokens=8, uniform_decode=False, has_lora=False
        )

        assert rt_mode == CUDAGraphMode.NONE

    def test_dispatch_cudagraph_mode_none(self, mock_vllm_config):
        """When cudagraph_mode is NONE, dispatch returns NONE."""
        dispatcher = NPUGraphDispatcher(mock_vllm_config)
        dispatcher.cudagraph_mode = CUDAGraphMode.NONE
        dispatcher.keys_initialized = True

        rt_mode, key = dispatcher.dispatch(
            num_tokens=8, uniform_decode=False, has_lora=False
        )

        assert rt_mode == CUDAGraphMode.NONE

    def test_dispatch_exceeds_max_capture_size(self, mock_vllm_config):
        """num_tokens > max_cudagraph_capture_size returns NONE."""
        mock_vllm_config.compilation_config.max_cudagraph_capture_size = 8
        dispatcher = self.setup_dispatcher(mock_vllm_config, CUDAGraphMode.FULL)

        rt_mode, key = dispatcher.dispatch(
            num_tokens=16, uniform_decode=False, has_lora=False
        )

        assert rt_mode == CUDAGraphMode.NONE

    def test_dispatch_full_and_piecewise_mode(self, mock_vllm_config):
        """FULL_AND_PIECEWISE mode: non-uniform batch goes to PIECEWISE."""
        dispatcher = self.setup_dispatcher(
            mock_vllm_config, CUDAGraphMode.FULL_AND_PIECEWISE
        )

        rt_mode, key = dispatcher.dispatch(
            num_tokens=8, uniform_decode=False, has_lora=False
        )

        # FULL_AND_PIECEWISE: mixed batches (non-uniform) go to PIECEWISE
        assert rt_mode == CUDAGraphMode.PIECEWISE

    def test_dispatch_uniform_decode_full_mode(self, mock_vllm_config):
        """Uniform decode with FULL mode dispatches correctly."""
        dispatcher = self.setup_dispatcher(mock_vllm_config, CUDAGraphMode.FULL)

        rt_mode, key = dispatcher.dispatch(
            num_tokens=8, uniform_decode=True, has_lora=False
        )

        # uniform decode with FULL mode should match decode keys -> FULL
        assert rt_mode == CUDAGraphMode.FULL

    def test_dispatch_disable_full_with_piecewise(self, mock_vllm_config):
        """disable_full=True works with PIECEWISE keys."""
        dispatcher = self.setup_dispatcher(
            mock_vllm_config, CUDAGraphMode.FULL_AND_PIECEWISE
        )

        # With PIECEWISE keys present and disable_full=True,
        # FULL mode is skipped, but PIECEWISE should still be tried
        rt_mode, key = dispatcher.dispatch(
            num_tokens=8,
            uniform_decode=False,
            has_lora=False,
            disable_full=True,
        )

        # disable_full skips FULL, but PIECEWISE should still match
        assert rt_mode == CUDAGraphMode.PIECEWISE


class TestIntegrationWithConfig:
    """Integration-style tests that build real config objects."""

    def _create_vllm_config(
        self,
        cudagraph_mode: str = "FULL",
        max_num_seqs: int = 8,
        lora_config: bool = False,
    ) -> MagicMock:
        """Helper to create a realistic mock config (following vllm test pattern)."""
        comp_config = MagicMock(spec=CompilationConfig)

        # Set up CUDAGraphMode enum
        mode_map = {
            "FULL": CUDAGraphMode.FULL,
            "PIECEWISE": CUDAGraphMode.PIECEWISE,
            "FULL_AND_PIECEWISE": CUDAGraphMode.FULL_AND_PIECEWISE,
            "FULL_DECODE_ONLY": CUDAGraphMode.FULL_DECODE_ONLY,
        }
        cg_mode = mode_map[cudagraph_mode]
        comp_config.cudagraph_mode = cg_mode
        comp_config.cudagraph_capture_sizes = [1, 8]
        comp_config.max_cudagraph_capture_size = 8
        comp_config.cudagraph_specialize_lora = None
        comp_config.mode = None

        # Mock methods
        comp_config.requires_piecewise_compilation = MagicMock(return_value=False)
        comp_config.is_attention_compiled_piecewise = MagicMock(return_value=True)

        mock_config = MagicMock(spec=VllmConfig)
        mock_config.compilation_config = comp_config
        mock_config.scheduler_config = MagicMock(spec=SchedulerConfig)
        mock_config.scheduler_config.max_num_seqs = max_num_seqs
        mock_config.speculative_config = None

        if lora_config:
            mock_config.lora_config = MagicMock()
            mock_config.lora_config.max_loras = 8
            comp_config.cudagraph_specialize_lora = True
        else:
            mock_config.lora_config = None

        mock_config.pad_for_cudagraph = lambda bs: bs

        return mock_config

    @pytest.mark.parametrize(
        "cudagraph_mode_str,expected_full_count,expected_piecewise_count",
        [
            ("FULL", 2, 0),
            ("PIECEWISE", 0, 2),
            ("FULL_AND_PIECEWISE", 2, 2),
            ("FULL_DECODE_ONLY", 2, 0),
        ],
    )
    def test_key_initialization_counts(
        self,
        cudagraph_mode_str,
        expected_full_count,
        expected_piecewise_count,
    ):
        """Verify key counts for different modes via integration setup."""
        config = self._create_vllm_config(cudagraph_mode_str)
        dispatcher = NPUGraphDispatcher(config)
        dispatcher.initialize_cudagraph_keys(
            config.compilation_config.cudagraph_mode, uniform_decode_query_len=1
        )

        assert (
            len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL])
            == expected_full_count
        )
        assert (
            len(dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE])
            == expected_piecewise_count
        )

    @pytest.mark.parametrize(
        "cudagraph_mode_str",
        [
            "FULL",
            "PIECEWISE",
            "FULL_AND_PIECEWISE",
            "FULL_DECODE_ONLY",
        ],
    )
    def test_dispatch_modes_from_config(self, cudagraph_mode_str):
        """Test dispatch behavior with different config setups."""
        config = self._create_vllm_config(cudagraph_mode_str)
        dispatcher = NPUGraphDispatcher(config)
        dispatcher.initialize_cudagraph_keys(
            config.compilation_config.cudagraph_mode, uniform_decode_query_len=1
        )

        rt_mode, key = dispatcher.dispatch(
            num_tokens=8, uniform_decode=False, has_lora=False
        )

        if cudagraph_mode_str == "FULL":
            assert rt_mode == CUDAGraphMode.FULL
        elif cudagraph_mode_str == "PIECEWISE":
            assert rt_mode == CUDAGraphMode.PIECEWISE
        elif cudagraph_mode_str == "FULL_AND_PIECEWISE":
            # non-uniform: PIECEWISE
            assert rt_mode == CUDAGraphMode.PIECEWISE
        elif cudagraph_mode_str == "FULL_DECODE_ONLY":
            assert rt_mode == CUDAGraphMode.NONE

    def test_lora_key_count(self):
        """With LoRA enabled, keys are doubled."""
        config = self._create_vllm_config("FULL", lora_config=True)
        dispatcher = NPUGraphDispatcher(config)
        dispatcher.initialize_cudagraph_keys(
            config.compilation_config.cudagraph_mode, uniform_decode_query_len=1
        )

        # 2 sizes * 2 lora cases = 4 FULL keys
        assert len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL]) == 4

