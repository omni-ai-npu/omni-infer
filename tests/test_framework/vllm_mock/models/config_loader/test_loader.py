# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import copy
import unittest
from dataclasses import is_dataclass

class ConfigObj(dict):
    """
    一个同时支持属性访问和字典访问的配置对象
    """
    def __init__(self, data=None):
        super().__init__()
        if data is None:
            data = {}
        self.update(data)
        
        # 递归转换嵌套字典
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = ConfigObj(value)
            elif isinstance(value, list):
                self[key] = [ConfigObj(item) if isinstance(item, dict) else item for item in value]
    
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"No such attribute: {name}")
    
    def __setattr__(self, name, value):
        self[name] = value
    
    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError(f"No such attribute: {name}")

def dict_to_object(d):
    """
    将嵌套字典转换为ConfigObj对象
    """
    return ConfigObj(d)


class TestModelAdditionalConfig(unittest.TestCase):

    hf_config_dict = {
        "model_type":"deepseek_v3",
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "max_position_embeddings": 163840,
        "vocab_size": 129280,
        "intermediate_size": 18432,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2048,
        "quantization_config": {
            "config_groups": {
                "group_0":{
                    "input_activations":{
                        "num_bits": 8
                    },
                    "weights":{
                        "num_bits": 8
                    }
                }
            },
            "format": "int-quantized"
        }
    }
    hf_config = dict_to_object(hf_config_dict)
    
    def test_basic_load_json_config(self):
        """测试基本配置加载功能"""
        from omni.models.config_loader.loader import model_extra_config, call_config_updater
        config = model_extra_config
        
        # 验证配置结构
        self.assertTrue(is_dataclass(config))
        self.assertTrue(is_dataclass(config.task_config))
        self.assertTrue(is_dataclass(config.parall_config))
        self.assertTrue(is_dataclass(config.operator_opt_config))
        
        call_config_updater(
            config_updater_name = 'update_task_config',
            hf_config = self.hf_config,
            hardware_platform = 'A3',
            is_pd_disaggregation = True,
            is_prefill_node = True,
            enable_chunked_prefill = False,
            enable_omni_placement = False,
            decode_gear_list = [32],
            enable_graph_mode = False
        )
        # 验证配置值
        self.assertEqual(config.task_config.enable_omni_placement, False)
        self.assertEqual(config.task_config.is_prefill_node, True)

        self.assertEqual(config.parall_config.dense_mlp_tp_size, 4)
        
        self.assertEqual(config.operator_opt_config.enable_kv_rmsnorm_rope_cache, True)
        self.assertEqual(config.operator_opt_config.prefill_moe_all_to_all, True)
        self.assertEqual(config.operator_opt_config.moe_multi_stream_tune, False)
        self.assertEqual(config.operator_opt_config.merge_qkv, False)
        self.assertEqual(config.operator_opt_config.two_stage_comm, False)
        self.assertEqual(config.operator_opt_config.gmm_nz, True)
        self.assertEqual(config.operator_opt_config.decode_moe_dispatch_combine, True)
        self.assertEqual(config.operator_opt_config.control_accept_rate, -1)
        
    def test_default_config_when_no_json(self):
         # 准备测试数据
        modified_config = copy.deepcopy(self.hf_config)
        modified_config["model_type"] = "qwen2.5"
        from omni.models.config_loader.loader import model_extra_config, call_config_updater, ModelExtraConfig
        model_extra_config = ModelExtraConfig()
        config = model_extra_config
        
        # 验证配置结构
        self.assertTrue(is_dataclass(config))
        self.assertTrue(is_dataclass(config.task_config))
        self.assertTrue(is_dataclass(config.parall_config))
        self.assertTrue(is_dataclass(config.operator_opt_config))
        
        call_config_updater(
            config_updater_name = 'update_task_config',
            hf_config = modified_config,
            hardware_platform = 'A3',
            is_pd_disaggregation = True,
            is_prefill_node = True,
            enable_chunked_prefill = False,
            enable_omni_placement = False,
            decode_gear_list = [32],
            enable_graph_mode = False
        )
        # 验证配置值
        self.assertEqual(config.task_config.enable_omni_placement, False)
        self.assertEqual(config.task_config.is_prefill_node, True)

        self.assertEqual(config.parall_config.dense_mlp_tp_size, 1)
        
        self.assertEqual(config.operator_opt_config.enable_kv_rmsnorm_rope_cache, True)
        self.assertEqual(config.operator_opt_config.prefill_moe_all_to_all, True)
        self.assertEqual(config.operator_opt_config.moe_multi_stream_tune, False)
        self.assertEqual(config.operator_opt_config.merge_qkv, False)
        self.assertEqual(config.operator_opt_config.two_stage_comm, False)
        self.assertEqual(config.operator_opt_config.gmm_nz, False)
        self.assertEqual(config.operator_opt_config.decode_moe_dispatch_combine, True)
        self.assertEqual(config.operator_opt_config.control_accept_rate, -1)

    def test_basic_config_check(self):
        """测试动态配置冲突修改功能"""
        from omni.models.config_loader.loader import model_extra_config, call_config_updater
        config = model_extra_config
        
        # 验证配置结构
        self.assertTrue(is_dataclass(config))
        self.assertTrue(is_dataclass(config.task_config))
        self.assertTrue(is_dataclass(config.parall_config))
        self.assertTrue(is_dataclass(config.operator_opt_config))

        
        call_config_updater(
            config_updater_name = 'update_task_config',
            hf_config = self.hf_config,
            hardware_platform = 'A2',
            is_pd_disaggregation = True,
            is_prefill_node = True,
            enable_chunked_prefill = False,
            enable_omni_placement = False,
            decode_gear_list = [32],
            enable_graph_mode = False
        )
        # 验证配置值
        
        self.assertEqual(config.operator_opt_config.moe_multi_stream_tune, False)
        self.assertEqual(config.operator_opt_config.use_super_kernel, False)
        self.assertEqual(config.operator_opt_config.use_prefetch, False)
        self.assertEqual(config.operator_opt_config.use_mlaprolog, False)
        self.assertEqual(config.operator_opt_config.fa_quant, False)
        self.assertEqual(config.operator_opt_config.expert_gate_up_prefetch, 0)
        self.assertEqual(config.operator_opt_config.expert_down_prefetch, 0)
        self.assertEqual(config.operator_opt_config.attn_prefetch, 0)


if __name__ == '__main__':
    unittest.main()
