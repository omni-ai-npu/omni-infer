# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from vllm.logger import logger


def is_mlp_weight_prefetch_on():
    from .loader import model_extra_config
    return (model_extra_config.operator_opt_config.enable_prefetch and 
            model_extra_config.operator_opt_config.expert_gate_up_prefetch and
            model_extra_config.operator_opt_config.expert_down_prefetch)


def apply_eager_mode_config(model_extra_config):
    """Apply eager-mode modifications to the given ModelExtraConfig.

    """
    
    if model_extra_config.task_config.graph_mode == "eager_mode":
        model_extra_config.operator_opt_config.enable_scmoe_multi_stream = False
        model_extra_config.operator_opt_config.enable_super_kernel = False
        model_extra_config.operator_opt_config.enable_prefetch = False
        model_extra_config.operator_opt_config.expert_gate_up_prefetch = 0
        model_extra_config.operator_opt_config.expert_down_prefetch = 0
        model_extra_config.operator_opt_config.attn_prefetch = 0
        model_extra_config.operator_opt_config.dense_mlp_prefetch = 0
        model_extra_config.operator_opt_config.lm_head_prefetch = 0
        model_extra_config.operator_opt_config.shared_expert_gate_up_prefetch = 0
        model_extra_config.operator_opt_config.shared_expert_down_prefetch = 0
        logger.warning(
            f"[WARNING] Eager mode disables all these optimization configurations by default."
        )


def apply_seq_parallel(model_extra_config):
    # validate seq_parallel
    cfg = model_extra_config.parall_config
    ena_sp = cfg.ena_seq_parallel
    ena_cp = cfg.ena_context_parallel

    plugins = os.environ.get("VLLM_PLUGINS", "").split(',')
    if "omni_custom_models" not in plugins and "omni_pangu_models" not in plugins:
        cfg.ena_seq_parallel = False
        cfg.ena_context_parallel = False
        logger.warning(f"[WARNING] omni_custom_models and omni_pangu_models both not in VLLM_PLUGINS, disabled ena_seq_parallel and ena_context_parallel")

    if ena_cp:
        assert ena_sp, "ena_context_parallel requires ena_seq_parallel"


def apply_omni_cache(additional_config):
    pass
