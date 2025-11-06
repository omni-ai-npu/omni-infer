# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

def is_mlp_weight_prefetch_on():
    from .loader import model_extra_config
    return (model_extra_config.operator_opt_config.use_prefetch and 
            model_extra_config.operator_opt_config.expert_gate_up_prefetch and
            model_extra_config.operator_opt_config.expert_down_prefetch)


def apply_eager_mode_config(operator_opt_config):
    operator_opt_config.moe_multi_stream_tune = False
    operator_opt_config.use_super_kernel = False
    operator_opt_config.use_prefetch = False
    operator_opt_config.expert_gate_up_prefetch = 0
    operator_opt_config.expert_down_prefetch = 0
    operator_opt_config.attn_prefetch = 0


def apply_fusion_pass(model_extra_config):
    if model_extra_config.task_config.enable_graph_mode:
        if model_extra_config.task_config.model_name == "pangu_ultra_moe" and model_extra_config.task_config.quant_type.startswith("w4a8"):
            model_extra_config.operator_opt_config.inplace_add_rms_norm_fustion_pass = True