# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import hashlib
import os
import pytest
from unittest.mock import patch, MagicMock

import torch
import torch_npu
import torchair

from vllm.config import VllmConfig, SchedulerConfig, SpeculativeConfig
try:
    from vllm.config import KvTransferConfig
except ImportError:
    # native VLLM without KvTransferConfig, simulate a compatible class
    class KvTransferConfig:
        def __init__(self, kv_role: str = None):
            self.kv_role = kv_role
import vllm.config
vllm.config.KvTransferConfig = KvTransferConfig

from omni_npu.compilation.ge_compile_config import(
    MAX_GEAR_NUM,
    BLOCK_NUM_FLOATING_RANGE,
    NPUCompilationConfig
)


@pytest.fixture
def default_vllm_config() -> VllmConfig:
    return VllmConfig(
        model="test-model",
        scheduler_config=SchedulerConfig(
            max_num_seqs=64, 
            max_num_batched_tokens=128,
            max_model_len=2048, 
            is_encoder_decoder=False),
        speculative_config=None,
        kv_transfer_config=None,
        additional_config={}
    )


@pytest.fixture
def default_compile_config() -> NPUCompilationConfig:
    return NPUCompilationConfig()


class TestNPUCompilationConfig:
    def test_build_from_cli_no_env(self, default_compile_config, default_vllm_config):
        """Test the default configuration without the environment variable TORCH_COMPILE_GE."""
        raw_graph_config = {"backend": "ge", "use_ge_graph_cached": True}

        with patch.dict(os.environ, {}, clear=True):
            default_compile_config.build_from_cli(raw_graph_config, default_vllm_config)
        
        assert default_compile_config.use_gegraph is False
        assert default_compile_config.backend == "ge"
        assert default_compile_config.use_ge_graph_cached is True
        assert default_compile_config.decode_gear_list == [128]
        assert default_compile_config.block_num_floating_range == BLOCK_NUM_FLOATING_RANGE
    

    def test_build_from_cli_with_env(self, default_compile_config, default_vllm_config):
        """Test the default configuration with the environment variable TORCH_COMPILE_GE."""
        raw_graph_config = {"backend": None, "use_ge_graph_cached": False}

        with patch.dict(os.environ, {"TORCH_COMPILE_GE": "True"}):
            default_compile_config.build_from_cli(raw_graph_config, default_vllm_config)

        assert default_compile_config.use_gegraph is True
        assert default_compile_config.backend is None
        assert default_compile_config.use_ge_graph_cached is False


    def test_build_from_cli_gear_list_type_error(self, default_compile_config, default_vllm_config):
        """Test that decode_gear_list throws an TypeError exception when it is not a list."""
        raw_graph_config = {"decode_gear_list": 32}
        with pytest.raises(TypeError, match="decode_gear_list must be a list"):
            default_compile_config.build_from_cli(raw_graph_config, default_vllm_config)


    def test_update_gear_options_over_max_num(self, default_compile_config, default_vllm_config):
        """
        Test that decode_gear_list throws an ValueError exception 
        when the number of gears in the decode_gear_list is greater than MAX_GEAR_NUM.
        """
        default_compile_config.decode_gear_list = [5, 10, 15, 20, 25, 30, 40]

        with pytest.raises(ValueError, match=f"Max gear num supported is {MAX_GEAR_NUM} now."):
            default_compile_config.update_gear_options(default_vllm_config)


    def test_update_gear_options_over_batched_tokens(self, default_compile_config, default_vllm_config):
        """Test the values of gears in the decode_gear_list exceed max_num_batched_tokens will be automatically truncated."""
        default_compile_config.decode_gear_list = [64, 128, 256]
        
        with patch("omni_npu.compilation.ge_compile_config.logger.warning") as mock_warn:
            default_compile_config.update_gear_options(default_vllm_config)

        assert default_compile_config.decode_gear_list == [64, 128]
        mock_warn.assert_called_once()

    
    def test_update_gear_options_no_set_gear_list(self, default_compile_config, default_vllm_config):
        """Test the decode_gear_list will be automatically set to [max_num_batched_tokens] when decode_gear_list is empty."""
        default_compile_config.decode_gear_list = None
        default_compile_config.update_gear_options(default_vllm_config)
        assert default_compile_config.decode_gear_list == [128]


    def test_update_gear_options_use_spec_decode(self, default_compile_config, default_vllm_config):
        """
        Test enable speculative decoding mode with enable_adaptive set to true, 
        the decode_gear_list will not add the max_num_batched_tokens option.
        """
        with patch("vllm.config.SpeculativeConfig") as mock_speculative_config:
            mock_speculative_config_instance = MagicMock()
            mock_speculative_config_instance.enable_adaptive = True
            mock_speculative_config_instance.num_speculative_tokens = 3
            mock_speculative_config.return_value = mock_speculative_config_instance

            default_vllm_config.speculative_config = mock_speculative_config_instance
            default_compile_config.decode_gear_list = [64]
            default_compile_config.update_gear_options(default_vllm_config)

            assert default_compile_config.decode_gear_list == [64]


    def test_update_gear_options_kv_consumer(self, default_compile_config, default_vllm_config):
        """Test the kv_consumer mode and change the upper limit of the gear to max_num_seqs."""
        default_vllm_config.kv_transfer_config = KvTransferConfig(kv_role="kv_consumer")
        default_compile_config.decode_gear_list = [10, 20]
        default_compile_config.update_gear_options(default_vllm_config)
        assert default_compile_config.decode_gear_list == [10, 20, 64]


    def test_init_backend_no_set_gegraph(self, default_compile_config, default_vllm_config):
        """Test set 'use_gegraph=False' will raise an ValueError exception."""
        default_compile_config.use_gegraph = False
        with pytest.raises(ValueError, match="use_gegraph is not set."):
            default_compile_config.init_backend(default_vllm_config)


    def test_init_backend_default(self, default_compile_config, default_vllm_config):
        """Test use the default backend for initialization."""
        default_compile_config.use_gegraph = True
        default_compile_config.backend = None

        with patch("omni_npu.compilation.ge_compile_config.torchair.get_npu_backend") as mock_get_npu_backend:
            mock_get_npu_backend.return_value = "torchair_npu_backend"
            result = default_compile_config.init_backend(default_vllm_config)
        
        assert result == "torchair_npu_backend"
        mock_get_npu_backend.assert_called_once()


    def test_init_backend_custom(self, default_compile_config, default_vllm_config):
        """Test use the custom backend for initialization."""
        default_compile_config.use_gegraph = True
        default_compile_config.backend = "custom_backend"

        result = default_compile_config.init_backend(default_vllm_config)
        assert result == "custom_backend"


    def test_compute_hash_consistent(self, default_compile_config):
        """
        Test with the same configuration, the hash values are the same; 
        the configuration changes, the hash values also change.
        """
        default_compile_config.level = 0
        default_compile_config.backend = "ge"
        default_compile_config.block_num_floating_range = BLOCK_NUM_FLOATING_RANGE

        hash1 = default_compile_config.compute_hash()
        hash2 = default_compile_config.compute_hash()
        assert hash1 == hash2

        default_compile_config.backend = "torchair"
        hash3 = default_compile_config.compute_hash()
        assert hash1 != hash3

        # Verify whether the hash algorithm is correct
        factors = [0, "ge", BLOCK_NUM_FLOATING_RANGE]
        manual_hash = hashlib.sha256(str(factors).encode()).hexdigest()
        assert hash1 == manual_hash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])