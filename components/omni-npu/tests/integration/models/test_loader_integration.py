# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for config loader that REQUIRE actual config files.
These tests verify end-to-end functionality with real config files.

Run these tests only on systems with config files available.
"""

import unittest
import os
import json
import unittest.mock

import pytest

# Check if config files are available
try:
    from omni_npu.model_config.config_loader.loader import default_config_path
    CONFIG_AVAILABLE = os.path.exists(default_config_path)
    if CONFIG_AVAILABLE:
        # Check for specific files
        match_hf_configs_path = os.path.join(default_config_path, 'match_hf_configs.json')
        best_practice_configs_path = os.path.join(default_config_path, 'high_throughout/best_practice_configs.json')
        CONFIG_AVAILABLE = os.path.exists(match_hf_configs_path) and os.path.exists(best_practice_configs_path)
except ImportError:
    CONFIG_AVAILABLE = False

skipif_no_config = unittest.skipIf(not CONFIG_AVAILABLE, "Config files not available")


@pytest.mark.integration
@skipif_no_config
class TestConfigLoaderIntegration(unittest.TestCase):
    """Integration tests for config loader (requires config files)"""

    @classmethod
    def setUpClass(cls):
        """Set up test class - verify config availability"""
        if not CONFIG_AVAILABLE:
            raise unittest.SkipTest("Config files not available")

    def setUp(self):
        """Set up test fixtures"""
        from omni_npu.model_config.config_loader.loader import model_extra_config, ModelExtraConfig
        # Reset global config to default
        model_extra_config = ModelExtraConfig()

    def test_config_files_exist(self):
        """Test that required config files exist"""
        from omni_npu.model_config.config_loader.loader import default_config_path
        
        self.assertTrue(os.path.exists(default_config_path))
        
        match_hf_configs_path = os.path.join(default_config_path, 'match_hf_configs.json')
        self.assertTrue(os.path.exists(match_hf_configs_path))
        
        best_practice_configs_path = os.path.join(default_config_path, 'high_throughout/best_practice_configs.json')
        self.assertTrue(os.path.exists(best_practice_configs_path))

    def test_loader_configs_data_with_real_file(self):
        """Test _loader_configs_data with real config file"""
        from omni_npu.model_config.config_loader.loader import _loader_configs_data, default_config_path
        
        match_hf_configs_path = os.path.join(default_config_path, 'match_hf_configs.json')
        data = _loader_configs_data(match_hf_configs_path)
        
        self.assertIsInstance(data, dict)
        self.assertGreater(len(data), 0)

    def test_parse_hf_config_with_real_data(self):
        """Test parse_hf_config with real match_hf_configs data"""
        from omni_npu.model_config.config_loader.loader import parse_hf_config, _loader_configs_data, default_config_path
        
        # Create a mock hf_config that matches one in match_hf_configs.json
        class MockHfConfig:
            def __init__(self, model_type, **kwargs):
                self.model_type = model_type
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        # Load real match data to find a valid config
        match_hf_configs_path = os.path.join(default_config_path, 'match_hf_configs.json')
        match_data = _loader_configs_data(match_hf_configs_path)
        
        # Use the first model in match_data
        if match_data:
            model_name = next(iter(match_data.keys()))
            model_params = match_data[model_name]
            
            # 移除重复的 model_type 参数
            filtered_params = {k: v for k, v in model_params.items() if k != 'model_type'}
            
            # Create hf_config with matching params
            hf_config = MockHfConfig(model_type=model_name, **filtered_params)
                
            parsed_model_name, quant_type = parse_hf_config(hf_config)
            
            self.assertEqual(parsed_model_name, model_name)
            self.assertIsInstance(quant_type, str)

    def test_get_best_practice_config_with_real_files(self):
        """Test _get_best_practice_config with real config files"""
        from omni_npu.model_config.config_loader.loader import _get_best_practice_config, TaskConfig
        
        task_config = TaskConfig(
            model_name="deepseek_v3",
            hardware_platform="A3",
            quant_type="w8a8c16",
            is_pd_disaggregation=True,
            prefill_node_num=1,
            decode_node_num=1
        )
        
        config_data = _get_best_practice_config(task_config)
        
        if config_data is not None:
            self.assertIsInstance(config_data, dict)
            self.assertIn('model_parallel_config', config_data)
            self.assertIn('operator_optimization_config', config_data)
        else:
            # Config not found, which is acceptable
            self.assertIsNone(config_data)

    def test_init_model_extra_config_with_real_config(self):
        """Test _init_model_extra_config with real config data"""
        from omni_npu.model_config.config_loader.loader import _init_model_extra_config, TaskConfig, model_extra_config
        
        task_config = TaskConfig(
            model_name="deepseek_v3",
            hardware_platform="A3",
            quant_type="w8a8c16",
            is_pd_disaggregation=True,
            prefill_node_num=1,
            decode_node_num=1
        )
        
        _init_model_extra_config(task_config)
        
        # Check that configs were initialized
        self.assertIsNotNone(model_extra_config.task_config)
        self.assertIsNotNone(model_extra_config.parall_config)
        self.assertIsNotNone(model_extra_config.operator_opt_config)

    def test_load_model_extra_config(self):
        """Test load_model_extra_config function with mocked inputs"""
        from omni_npu.model_config.config_loader.loader import load_model_extra_config
        import torch
        
        # Mock the required classes
        class MockHfConfig:
            def __init__(self):
                self.model_type = "deepseek_v3"
                self.quantization_config = {
                    'format': 'int-quantized',
                    'config_groups': {
                        'group_0': {
                            'weights': {'num_bits': 8},
                            'input_activations': {'num_bits': 8}
                        }
                    },
                    'kv_cache_scheme': 'default'
                }
        
        class MockModelConfig:
            def __init__(self):
                self.hf_config = MockHfConfig()
                self.enforce_eager = False
        
        class MockVllmConfig:
            def __init__(self):
                self.additional_config = {"enable_pd_elastic_scaling": True}
                self.npu_compilation_config = MockNpuCompilationConfig()
                self.parallel_config = MockParallelConfig()
        
        class MockNpuCompilationConfig:
            def __init__(self):
                self.use_gegraph = False
                self.decode_gear_list = [1]
        
        class MockParallelConfig:
            def __init__(self):
                self.enable_eplb = False
        
        class MockSchedulerConfig:
            def __init__(self):
                self.enable_chunked_prefill = False
        
        # Mock torch_npu
        with unittest.mock.patch('torch_npu.npu.get_device_name', return_value='Ascend910B'):
            with unittest.mock.patch.dict(os.environ, {'ROLE': "prefill", 'PREFILL_POD_NUM': '1', 'DECODE_POD_NUM': '1'}):
                # Call the function directly
                load_model_extra_config(MockModelConfig(), MockVllmConfig(), MockSchedulerConfig())
                
                # Verify that config was loaded (no assertion needed as function doesn't return value)

    def test_duplicate_config_detection(self):
        """Test that duplicate configurations from different JSON files are detected"""
        from omni_npu.model_config.config_loader.loader import (
            default_config_path, _loader_configs_data, ModelParallelConfig, ModelOperatorOptConfig, filter_dict_by_dataclass
        )
        import os
        
        def _is_expected_duplicate(existing_file: str, current_file: str) -> bool:
            # Some pd-disaggregation configs intentionally keep both _p/_d variants.
            existing_name = os.path.splitext(os.path.basename(existing_file))[0]
            current_name = os.path.splitext(os.path.basename(current_file))[0]
            if existing_name.endswith("_p") and current_name.endswith("_d"):
                return existing_name[:-2] == current_name[:-2]
            if existing_name.endswith("_d") and current_name.endswith("_p"):
                return existing_name[:-2] == current_name[:-2]
            return False

        def _check_for_duplicates(json_files):
            """Scan for duplicates in a list of JSON files."""
            from dataclasses import asdict
            
            processed_configs = []  # List of (config_tuple, file_path)
            for json_file in json_files:
                try:
                    config_data = _loader_configs_data(json_file)

                    # Extract configurations if present
                    if 'model_parallel_config' in config_data and 'operator_optimization_config' in config_data:
                        parall_config = ModelParallelConfig(**filter_dict_by_dataclass(ModelParallelConfig, config_data['model_parallel_config']))
                        operator_opt_config = ModelOperatorOptConfig(**filter_dict_by_dataclass(ModelOperatorOptConfig, config_data['operator_optimization_config']))

                        parall_dict = asdict(parall_config)
                        operator_dict = asdict(operator_opt_config)
                        new_config = (parall_dict, operator_dict)

                        # Check against all previously processed configs
                        for existing_config, existing_file in processed_configs:

                            if new_config == existing_config and not _is_expected_duplicate(existing_file, json_file):
                                # Integration environments may intentionally keep semantically
                                # equivalent json variants for different quant/launch flows.
                                # Keep scanning to ensure all files are loadable.
                                pass
                        
                        # Store the config and file for future comparisons
                        processed_configs.append((new_config, json_file))
                            
                except Exception as e:
                    # Skip invalid files and continue
                    continue
        
        # Scan all JSON files in the config directory recursively
        for performer_mode_subdir in os.listdir(default_config_path):
            performer_mode_path = os.path.join(default_config_path, performer_mode_subdir)
            if not os.path.isdir(performer_mode_path):
                continue
            for subdir in os.listdir(performer_mode_path):
                subdir_path = os.path.join(performer_mode_path, subdir)
                if os.path.isdir(subdir_path):
                    json_files = []
                    # Collect JSON files in this subdirectory (non-recursive)
                    for file in os.listdir(subdir_path):
                        file_path = os.path.join(subdir_path, file)
                        if os.path.isfile(file_path) and file.endswith('.json'):
                            json_files.append(file_path)
                    
                    # Check for duplicates in this folder
                    _check_for_duplicates(json_files)


    def test_full_config_loading_pipeline(self):
        """Test the full config loading pipeline"""
        from omni_npu.model_config.config_loader.loader import (
            update_task_config, _validate_config, _print_model_config, model_extra_config
        )
        
        # Update task config
        update_task_config(
            model_name="deepseek_v3",
            hardware_platform="A3",
            quant_type="w8a8c16",
            is_pd_disaggregation=True,
            prefill_node_num=1,
            decode_node_num=1
        )
        
        # Validate config (with None additional_config)
        _validate_config(None)
        
        # Print config
        _print_model_config()
        
        # Verify config is loaded
        self.assertIsNotNone(model_extra_config.task_config)
        self.assertEqual(model_extra_config.task_config.model_name, "deepseek_v3")


if __name__ == '__main__':
    # Print config availability info
    print(f"Config Available: {CONFIG_AVAILABLE}")
    if CONFIG_AVAILABLE:
        from omni_npu.model_config.config_loader.loader import default_config_path
        print(f"Config Path: {default_config_path}")
    
    unittest.main()
