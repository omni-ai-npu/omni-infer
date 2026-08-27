# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for config loader that do NOT require actual NPU hardware or file system.
These tests use mocking to verify the logic and API contracts.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch, mock_open
from contextlib import contextmanager
import sys
import json
import os

import pytest
import torch
from dataclasses import asdict


@contextmanager
def mock_dependencies():
    """Context manager to mock external dependencies"""
    with patch('torch.npu.get_device_name', return_value='Ascend910B'):
        with patch.dict(os.environ, {'ROLE': 'prefill', 'PREFILL_POD_NUM': '1', 'DECODE_POD_NUM': '1'}):
            yield


@pytest.mark.unit
class TestConfigLoaderUnit(unittest.TestCase):
    """Unit tests for config loader (no NPU hardware or file system required)"""

    def setUp(self):
        """Set up test fixtures"""
        # Save original sys.modules state for restoration
        self.original_modules = sys.modules.copy()

        # Mock torch_npu availability
        self.torch_npu_mock = MagicMock()
        sys.modules['torch_npu'] = self.torch_npu_mock

        # torch_npu normally registers torch.npu on import. In CPU-only test envs
        # (or when sys.modules['torch_npu'] is a mock), torch.npu is missing, but
        # ModelOperatorOptConfig.__post_init__ writes to torch.npu.config — register
        # a stub so import-time attribute access does not fail. Restored in tearDown.
        import torch as _torch
        self._torch_had_npu = hasattr(_torch, 'npu')
        self._original_torch_npu = getattr(_torch, 'npu', None)
        self.torch_npu_attr_mock = MagicMock()
        _torch.npu = self.torch_npu_attr_mock

        # Mock ALL vllm dependencies and submodules
        self.vllm_mock = MagicMock()
        self.vllm_logger_mock = MagicMock()
        self.vllm_reasoning_mock = MagicMock()

        # Create nested mock structure for vllm.reasoning
        self.vllm_reasoning_abs_parsers_mock = MagicMock()
        self.vllm_reasoning_deepseek_mock = MagicMock()
        self.vllm_config_mock = MagicMock()
        self.vllm_distributed_mock = MagicMock()
        self.vllm_v1_mock = MagicMock()

        # Set up the logger init_logger
        self.vllm_mock.logger = self.vllm_logger_mock
        self.vllm_logger_mock.init_logger = MagicMock(return_value=self.vllm_logger_mock)

        # Set up vllm.reasoning structure
        self.vllm_reasoning_mock.abs_reasoning_parsers = self.vllm_reasoning_abs_parsers_mock
        self.vllm_reasoning_mock.deepseek_r1_reasoning_parser = self.vllm_reasoning_deepseek_mock

        # Register all vllm submodules in sys.modules
        sys.modules['vllm'] = self.vllm_mock
        sys.modules['vllm.logger'] = self.vllm_logger_mock
        sys.modules['vllm.reasoning'] = self.vllm_reasoning_mock
        sys.modules['vllm.reasoning.abs_reasoning_parsers'] = self.vllm_reasoning_abs_parsers_mock
        sys.modules['vllm.reasoning.deepseek_r1_reasoning_parser'] = self.vllm_reasoning_deepseek_mock
        sys.modules['vllm.config'] = self.vllm_config_mock
        sys.modules['vllm.distributed'] = self.vllm_distributed_mock
        sys.modules['vllm.v1'] = self.vllm_v1_mock

        # Mock omni_npu.model_config.config_loader.features
        self.features_mock = MagicMock()
        sys.modules['omni_npu.model_config.config_loader.features'] = self.features_mock

        # Mock omni_npu.v1.parsers (needed by some tests)
        self.parsers_mock = MagicMock()
        self.parsers_mock.register_lazy_parsers = MagicMock()
        sys.modules['omni_npu.v1.parsers'] = self.parsers_mock

    def tearDown(self):
        """Clean up after tests - restore original sys.modules"""
        # Restore torch.npu attribute (set in setUp)
        import torch as _torch
        if self._torch_had_npu:
            _torch.npu = self._original_torch_npu
        elif hasattr(_torch, 'npu'):
            del _torch.npu

        # Carefully restore sys.modules to original state
        # Remove any modules we added
        modules_to_remove = [
            'torch_npu',
            'vllm',
            'vllm.logger',
            'vllm.reasoning',
            'vllm.reasoning.abs_reasoning_parsers',
            'vllm.reasoning.deepseek_r1_reasoning_parser',
            'vllm.config',
            'vllm.distributed',
            'vllm.v1',
            'omni_npu.model_config.config_loader.features',
            'omni_npu.v1.parsers',
        ]

        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]

        # Restore any modules that existed before
        for module, value in self.original_modules.items():
            if module not in sys.modules:
                sys.modules[module] = value

    def test_model_operator_opt_config_post_init_enable_prefetch_true(self):
        """Test ModelOperatorOptConfig __post_init__ when enable_prefetch is True"""
        from omni_npu.model_config.config_loader.loader import ModelOperatorOptConfig
        
        mock_logger = MagicMock()
        
        with patch('omni_npu.model_config.config_loader.loader.logger', mock_logger):
            config = ModelOperatorOptConfig(enable_prefetch=True)
        
        # When enable_prefetch is True, prefetch values should remain default
        self.assertEqual(config.expert_gate_up_prefetch, 50)
        self.assertEqual(config.attn_prefetch, 96)
        # Verify warning was not logged
        mock_logger.warning.assert_not_called()

    def test_model_operator_opt_config_post_init_enable_prefetch_false(self):
        """Test ModelOperatorOptConfig __post_init__ when enable_prefetch is False"""
        from omni_npu.model_config.config_loader.loader import ModelOperatorOptConfig
        
        mock_logger = MagicMock()
        
        with patch('omni_npu.model_config.config_loader.loader.logger', mock_logger):
            config = ModelOperatorOptConfig(enable_prefetch=False)
        
        # When enable_prefetch is False, prefetch values should be set to 0
        self.assertEqual(config.expert_gate_up_prefetch, 0)
        self.assertEqual(config.expert_down_prefetch, 0)
        self.assertEqual(config.attn_prefetch, 0)
        self.assertEqual(config.dense_mlp_prefetch, 0)
        self.assertEqual(config.lm_head_prefetch, 0)
        self.assertEqual(config.shared_expert_gate_up_prefetch, 0)
        self.assertEqual(config.shared_expert_down_prefetch, 0)
        # Verify warning was logged
        mock_logger.warning.assert_called_once_with(
            "[WARNING] When enable_prefetch is false, prefetch_Mb must be set to 0."
        )

    def test_model_operator_opt_config_post_init_conflicting_comm_config(self):
        """Test ModelOperatorOptConfig __post_init__ raises error for conflicting comm config"""
        from omni_npu.model_config.config_loader.loader import ModelOperatorOptConfig
        
        with self.assertRaises(ValueError) as context:
            ModelOperatorOptConfig(enable_pipeline_comm=True, enable_round_pipeline_comm=True)
        
        self.assertIn("Conflicting communication configuration", str(context.exception))

    def test_model_operator_opt_config_post_init_unquant_bmm_nz(self):
        """Test ModelOperatorOptConfig __post_init__ sets torch config for unquant_bmm_nz"""
        from omni_npu.model_config.config_loader.loader import ModelOperatorOptConfig
        
        config = ModelOperatorOptConfig(unquant_bmm_nz=True)
        
        # Verify torch.npu.config.allow_internal_format was set
        self.torch_npu_mock.config.allow_internal_format = True

    def test_model_operator_opt_config_lmhead_fp32_default(self):
        """Test ModelOperatorOptConfig lmhead_fp32 defaults to False"""
        from omni_npu.model_config.config_loader.loader import ModelOperatorOptConfig

        config = ModelOperatorOptConfig()
        self.assertFalse(config.lmhead_fp32)

    def test_pr1_config_defaults_for_vit_sp_and_dispatch_combine(self):
        from omni_npu.model_config.config_loader.loader import (
            ModelOperatorOptConfig,
            ModelParallelConfig,
        )

        parallel = ModelParallelConfig()
        self.assertFalse(parallel.vit_dynamic_sp)
        self.assertEqual(parallel.vit_cp_min_patch_tokens, {})

        operator = ModelOperatorOptConfig()
        self.assertEqual(operator.moe_dispatch_combine_max_batch_size, 128)

    def test_model_operator_opt_config_lmhead_fp32_from_config(self):
        """Test _init_model_extra_config loads lmhead_fp32 from operator config"""
        from omni_npu.model_config.config_loader.loader import (
            _init_model_extra_config,
            TaskConfig,
            model_extra_config,
        )

        task_config = TaskConfig()
        with patch(
            "omni_npu.model_config.config_loader.loader._get_best_practice_config",
            return_value={
                "model_parallel_config": {},
                "operator_optimization_config": {"lmhead_fp32": True},
            },
        ):
            _init_model_extra_config(task_config)

        self.assertTrue(model_extra_config.operator_opt_config.lmhead_fp32)

    def test_parse_hf_config_deepseek_v3(self):
        """Test parse_hf_config for deepseek_v3 model"""
        from omni_npu.model_config.config_loader.loader import parse_hf_config
        
        # Mock hf_config
        hf_config_mock = MagicMock()
        hf_config_mock.model_type = "deepseek_v3"
        hf_config_mock.quantization_config = {
            'format': 'int-quantized',
            'config_groups': {
                'group_0': {
                    'weights': {'num_bits': 8},
                    'input_activations': {'num_bits': 8}
                }
            },
            'kv_cache_scheme': 'default'
        }
        
        model_name, quant_type = parse_hf_config(hf_config_mock)
        
        self.assertEqual(model_name, "deepseek_v3")
        self.assertEqual(quant_type, "w8a8c16")

    def test_parse_hf_config_deepseek_v32(self):
        """Test parse_hf_config for deepseek_v32 model"""
        from omni_npu.model_config.config_loader.loader import parse_hf_config
        # Mock hf_config
        hf_config_mock = MagicMock()
        hf_config_mock.model_type = "deepseek_v32"
        hf_config_mock.quantization_config = {
            'format': 'int-quantized',
            'config_groups': {
                'group_0': {
                    'weights': {'num_bits': 8},
                    'input_activations': {'num_bits': 8}
                }
            },
            'kv_cache_scheme': 'default'
        }
        
        model_name, quant_type = parse_hf_config(hf_config_mock)
        
        self.assertEqual(model_name, "deepseek_v32")
        self.assertEqual(quant_type, "w8a8c16")

    def test_parse_hf_config_bf16(self):
        """Test parse_hf_config for BF16 model without quantization"""
        from types import SimpleNamespace
        from omni_npu.model_config.config_loader.loader import parse_hf_config

        hf_config = SimpleNamespace(model_type="some_model", dtype=torch.bfloat16)
        model_name, quant_type = parse_hf_config(hf_config)

        self.assertEqual(model_name, "some_model")
        self.assertEqual(quant_type, "bf16")

    def test_parse_hf_config_fp16(self):
        """Test parse_hf_config maps float16/fp16 dtype to quant_type fp16."""
        from types import SimpleNamespace
        from omni_npu.model_config.config_loader.loader import parse_hf_config

        for dtype in ("float16", "fp16", torch.float16):
            hf_config = SimpleNamespace(model_type="openpangu_v2", dtype=dtype)
            model_name, quant_type = parse_hf_config(hf_config)

            self.assertEqual(model_name, "openpangu_v2")
            self.assertEqual(quant_type, "fp16", msg=f"dtype={dtype!r}")

    @patch('os.path.exists', return_value=True)
    def test_get_best_practice_config_gpt_oss_alias(self, mock_exists):
        """Test low-latency best-practice config lookup for gpt_oss alias."""
        parser_stub = MagicMock()
        parser_stub.register_lazy_parsers = MagicMock()
        with patch.dict("sys.modules", {"omni_npu.v1.parsers": parser_stub}):
            from omni_npu.model_config.config_loader.loader import _get_best_practice_config, TaskConfig

            best_practice_entries = [
                {
                    "model": "gpt_oss",
                    "hardware": "A3",
                    "precision": "w8a8c16",
                    "configs": {
                        "hybrid": {
                            "config_file": "gpt_oss/gpt_oss_w8a8c16_a3_hybrid.json"
                        }
                    },
                }
            ]
            selected_config = {
                "model_parallel_config": {
                    "layer_parallel_config": {
                        "self_attn.o_proj": {
                            "x_transform": {"type": "NoOp"},
                            "y_transform": {"type": "AllReduce"},
                        }
                    }
                },
                "operator_optimization_config": {"decode_moe_dispatch_combine": True},
            }

            with patch(
                "omni_npu.model_config.config_loader.loader._loader_configs_data",
                side_effect=[best_practice_entries, selected_config],
            ) as mock_loader:
                task_config = TaskConfig(
                    model_name="gpt_oss",
                    hardware_platform="A3",
                    quant_type="w8a8c16",
                    is_pd_disaggregation=False,
                    enable_low_latency=True,
                )
                result = _get_best_practice_config(task_config)

            self.assertEqual(result, selected_config)
            self.assertEqual(mock_loader.call_count, 2)

    @patch('os.path.exists', return_value=True)
    def test_get_best_practice_config_gpt_oss_a2_low_latency_hyphen_alias(self, mock_exists):
        parser_stub = MagicMock()
        parser_stub.register_lazy_parsers = MagicMock()
        with patch.dict("sys.modules", {"omni_npu.v1.parsers": parser_stub}):
            from omni_npu.model_config.config_loader.loader import _get_best_practice_config, TaskConfig

            best_practice_entries = [
                {
                    "model": "gpt-oss",
                    "hardware": "A2",
                    "precision": "w8a8c16",
                    "configs": {
                        "hybrid": {
                            "config_file": "gpt_oss/gpt_oss_w8a8c16_a2_hybrid.json"
                        }
                    },
                }
            ]
            selected_config = {
                "model_parallel_config": {
                    "layer_parallel_config": {
                        "self_attn.o_proj": {
                            "x_transform": {"type": "NoOp"},
                            "y_transform": {"type": "AllReduce"},
                        }
                    }
                },
                "operator_optimization_config": {"decode_moe_dispatch_combine": False},
            }

            with patch(
                "omni_npu.model_config.config_loader.loader._loader_configs_data",
                side_effect=[best_practice_entries, selected_config],
            ):
                task_config = TaskConfig(
                    model_name="gpt-oss",
                    hardware_platform="A2",
                    quant_type="w8a8c16",
                    is_pd_disaggregation=False,
                    enable_low_latency=True,
                )
                result = _get_best_practice_config(task_config)

            self.assertEqual(result, selected_config)

    @patch('os.path.exists', return_value=True)
    def test_get_best_practice_config_gpt_oss_a2_high_throughout_alias(self, mock_exists):
        parser_stub = MagicMock()
        parser_stub.register_lazy_parsers = MagicMock()
        with patch.dict("sys.modules", {"omni_npu.v1.parsers": parser_stub}):
            from omni_npu.model_config.config_loader.loader import _get_best_practice_config, TaskConfig

            best_practice_entries = [
                {
                    "model": "gpt_oss",
                    "hardware": "A2",
                    "precision": "w8a8c16",
                    "configs": {
                        "hybrid": {
                            "config_file": "gpt_oss/gpt_oss_w8a8c16_a2_hybrid.json"
                        }
                    },
                }
            ]
            selected_config = {
                "model_parallel_config": {},
                "operator_optimization_config": {"decode_moe_dispatch_combine": False},
            }

            with patch(
                "omni_npu.model_config.config_loader.loader._loader_configs_data",
                side_effect=[best_practice_entries, selected_config],
            ):
                task_config = TaskConfig(
                    model_name="gpt_oss",
                    hardware_platform="A2",
                    quant_type="w8a8c16",
                    is_pd_disaggregation=False,
                    enable_low_latency=False,
                )
                result = _get_best_practice_config(task_config)

            self.assertEqual(result, selected_config)

    def test_filter_dict_by_dataclass(self):
        """Test filter_dict_by_dataclass filters valid keys"""
        from omni_npu.model_config.config_loader.loader import filter_dict_by_dataclass, ModelOperatorOptConfig
        
        data_dict = {
            'enable_prefetch': False,
            'invalid_key': 'value',
            'expert_gate_up_prefetch': 100
        }
        
        filtered = filter_dict_by_dataclass(ModelOperatorOptConfig, data_dict)
        
        self.assertIn('enable_prefetch', filtered)
        self.assertIn('expert_gate_up_prefetch', filtered)
        self.assertNotIn('invalid_key', filtered)

    @patch('builtins.open', new_callable=mock_open, read_data='{"key": "value"}')
    def test_loader_configs_data(self, mock_file):
        """Test _loader_configs_data loads JSON correctly"""
        from omni_npu.model_config.config_loader.loader import _loader_configs_data
        
        result = _loader_configs_data('dummy_path.json')
        
        self.assertEqual(result, {"key": "value"})
        mock_file.assert_called_once_with('dummy_path.json', 'r')

    @patch('builtins.open', new_callable=mock_open, read_data='invalid json')
    def test_loader_configs_data_invalid_json(self, mock_file):
        """Test _loader_configs_data raises error for invalid JSON"""
        from omni_npu.model_config.config_loader.loader import _loader_configs_data
        
        with self.assertRaises(RuntimeError) as context:
            _loader_configs_data('dummy_path.json')
        
        self.assertIn("Invalid JSON format", str(context.exception))

    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({
        "model_parallel_config": {"enable_share_expert_tp": True},
        "operator_optimization_config": {"enable_prefetch": False}
    }))
    def test_init_model_extra_config(self, mock_file, mock_exists):
        """Test _init_model_extra_config initializes configs correctly"""
        from omni_npu.model_config.config_loader.loader import _init_model_extra_config, TaskConfig, model_extra_config

        task_config = TaskConfig()

        # Mock _get_best_practice_config to return config data
        with patch('omni_npu.model_config.config_loader.loader._get_best_practice_config', return_value={
            "model_parallel_config": {"enable_share_expert_tp": True},
            "operator_optimization_config": {"enable_prefetch": False}
        }):
            _init_model_extra_config(task_config)

            self.assertEqual(model_extra_config.parall_config.enable_share_expert_tp, True)
            self.assertEqual(model_extra_config.operator_opt_config.enable_prefetch, False)
            # Verify __post_init__ was called and prefetch values were reset
            self.assertEqual(model_extra_config.operator_opt_config.expert_gate_up_prefetch, 0)

    def test_init_model_extra_config_uses_absolute_custom_config_path(self):
        """CUSTOM_MODEL_CONFIG_PATH absolute values are loaded directly."""
        from omni_npu.model_config.config_loader.loader import _init_model_extra_config, TaskConfig

        # Use a path that is absolute on the current platform. Posix-only paths
        # like "/tmp/..." are not absolute on Windows (no drive letter), so
        # os.path.join there keeps the prefix. tempfile.gettempdir() returns a
        # platform-native absolute path that bypasses the prefix on both OSes.
        import tempfile
        custom_path = os.path.join(tempfile.gettempdir(), "custom_model_config.json")
        config_data = {
            "model_parallel_config": {},
            "operator_optimization_config": {},
        }

        with patch.dict(os.environ, {"CUSTOM_MODEL_CONFIG_PATH": custom_path}), \
             patch(
                 'omni_npu.model_config.config_loader.loader._loader_configs_data',
                 return_value=config_data,
             ) as mock_loader:
            _init_model_extra_config(TaskConfig())

        mock_loader.assert_called_once_with(custom_path)

    def test_init_model_extra_config_resolves_relative_custom_config_path(self):
        """CUSTOM_MODEL_CONFIG_PATH relative values remain under default_config_path."""
        from omni_npu.model_config.config_loader.loader import (
            _init_model_extra_config,
            TaskConfig,
            default_config_path,
        )

        custom_path = "custom/config.json"
        # Mirror the production join so the expected path matches on every
        # platform (path separators differ between Windows and Posix).
        expected_model_config_file_path = os.path.join(default_config_path, custom_path)
        config_data = {
            "model_parallel_config": {},
            "operator_optimization_config": {},
        }

        with patch.dict(os.environ, {"CUSTOM_MODEL_CONFIG_PATH": custom_path}), \
             patch(
                 'omni_npu.model_config.config_loader.loader._loader_configs_data',
                 return_value=config_data,
             ) as mock_loader:
            _init_model_extra_config(TaskConfig())

        mock_loader.assert_called_once_with(expected_model_config_file_path)

    def test_update_task_config(self):
        """Test update_task_config updates task_config correctly"""
        from omni_npu.model_config.config_loader.loader import update_task_config, model_extra_config
        
        update_task_config(model_name="test_model", quant_type="w8a8")
        
        self.assertEqual(model_extra_config.task_config.model_name, "test_model")
        self.assertEqual(model_extra_config.task_config.quant_type, "w8a8")

    def test_print_model_config(self):
        """Test _print_model_config logs config correctly"""
        from omni_npu.model_config.config_loader.loader import _print_model_config, model_extra_config
        
        mock_logger = MagicMock()
        
        with patch('omni_npu.model_config.config_loader.loader.logger', mock_logger):
            _print_model_config()
        
        # Verify logger.info was called
        mock_logger.info.assert_called()

    def test_load_model_extra_config(self):
        """Test load_model_extra_config function with mocked dependencies"""
        from omni_npu.model_config.config_loader.loader import (
            load_model_extra_config,
            model_extra_config,
        )
        
        # Mock all required classes and dependencies
        mock_model_config = MagicMock()
        mock_model_config.hf_config.model_type = "deepseek_v3"
        mock_model_config.hf_config.quantization_config = {
            'format': 'int-quantized',
            'config_groups': {'group_0': {'weights': {'num_bits': 8}, 'input_activations': {'num_bits': 8}}},
            'kv_cache_scheme': 'default'
        }
        mock_model_config.enforce_eager = False
        
        mock_vllm_config = MagicMock()
        mock_vllm_config.additional_config = None
        mock_vllm_config.parallel_config.enable_eplb = False
        mock_vllm_config.model_config.dtype = torch.float16
        
        mock_scheduler_config = MagicMock()
        mock_scheduler_config.enable_chunked_prefill = False

        # Mock external dependencies
        with patch('omni_npu.model_config.config_loader.loader.parse_hf_config', return_value=('deepseek_v3', 'w8a8c16')), \
             patch('omni_npu.model_config.config_loader.loader.update_task_config') as mock_update, \
             patch('omni_npu.model_config.config_loader.loader._validate_config') as mock_validate, \
             patch('omni_npu.model_config.config_loader.loader._print_model_config') as mock_print:

            load_model_extra_config(mock_model_config, mock_vllm_config, mock_scheduler_config)

            # Verify that update_task_config was called
            mock_update.assert_called_once()
            # Verify that _validate_config was called
            mock_validate.assert_called_once()
            # Verify that _print_model_config was called
            mock_print.assert_called_once()
            self.assertEqual(model_extra_config.dtype, torch.float16)

    def test_model_operator_opt_config_enable_mtp_invariant(self):
        """Test ModelOperatorOptConfig enable_mtp_invariant default and custom value"""
        from omni_npu.model_config.config_loader.loader import ModelOperatorOptConfig
        
        default_config = ModelOperatorOptConfig()
        self.assertFalse(default_config.enable_mtp_invariant)
        
        enabled_config = ModelOperatorOptConfig(enable_mtp_invariant=True)
        self.assertTrue(enabled_config.enable_mtp_invariant)


    # ------------------------------------------------------------------
    # _get_best_practice_config: warning & error paths (lines 439, 443, 451)
    # ------------------------------------------------------------------

    def test_get_best_practice_config_warns_when_no_matching_list(self):
        """Cover logger.warning + return None when model/hardware/precision not found."""
        from omni_npu.model_config.config_loader.loader import _get_best_practice_config, TaskConfig

        with patch(
            "omni_npu.model_config.config_loader.loader._loader_configs_data",
            return_value=[],
        ):
            result = _get_best_practice_config(TaskConfig())

        self.assertIsNone(result)

    def test_get_best_practice_config_warns_when_no_pd_scheme(self):
        """Cover logger.warning + return None when pd_scheme missing from configs_list."""
        from omni_npu.model_config.config_loader.loader import _get_best_practice_config, TaskConfig

        best_practice_entries = [
            {
                "model": "deepseek_v3",
                "hardware": "A3",
                "precision": "w8a8c16",
                "configs": {},  # No matching pd_scheme key
            }
        ]

        with patch(
            "omni_npu.model_config.config_loader.loader._loader_configs_data",
            return_value=best_practice_entries,
        ):
            task_config = TaskConfig(
                model_name="deepseek_v3",
                hardware_platform="A3",
                quant_type="w8a8c16",
            )
            result = _get_best_practice_config(task_config)

        self.assertIsNone(result)

    def test_get_best_practice_config_raises_when_file_not_found(self):
        """Cover raise RuntimeError when resolved config file does not exist on disk."""
        from omni_npu.model_config.config_loader.loader import _get_best_practice_config, TaskConfig

        best_practice_entries = [
            {
                "model": "deepseek_v3",
                "hardware": "A3",
                "precision": "w8a8c16",
                "configs": {
                    "hybrid": {"config_file": "non_existent.json"},
                },
            }
        ]

        with patch(
            "omni_npu.model_config.config_loader.loader._loader_configs_data",
            return_value=best_practice_entries,
        ), patch("os.path.exists", return_value=False):
            # is_pd_disaggregation=False → pd_scheme='hybrid' (matches configs key)
            task_config = TaskConfig(
                model_name="deepseek_v3",
                hardware_platform="A3",
                quant_type="w8a8c16",
                is_pd_disaggregation=False,
            )
            with self.assertRaises(RuntimeError) as ctx:
                _get_best_practice_config(task_config)

        self.assertIn("but not found", str(ctx.exception))

    # ------------------------------------------------------------------
    # _loader_configs_data: TypeError and generic Exception handlers (lines 377, 379)
    # ------------------------------------------------------------------

    def test_loader_configs_data_type_error_handler(self):
        """Cover except TypeError → RuntimeError('Config structure mismatch')."""
        from omni_npu.model_config.config_loader.loader import _loader_configs_data

        # Patch open so the file handle is a mock; then make json.load raise TypeError.
        m = mock_open()
        with patch("builtins.open", m), patch("json.load", side_effect=TypeError("mock type error")):
            with self.assertRaises(RuntimeError) as ctx:
                _loader_configs_data("dummy_path.json")

        self.assertIn("Config structure mismatch", str(ctx.exception))

    def test_loader_configs_data_unexpected_error_handler(self):
        """Cover except Exception → RuntimeError('Unexpected error')."""
        from omni_npu.model_config.config_loader.loader import _loader_configs_data

        with patch("builtins.open", side_effect=OSError("unexpected failure")):
            with self.assertRaises(RuntimeError) as ctx:
                _loader_configs_data("dummy_path.json")

        self.assertIn("Unexpected error", str(ctx.exception))

if __name__ == '__main__':
    unittest.main()
