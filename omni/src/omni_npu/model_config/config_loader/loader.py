# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass, field, fields, asdict
from typing import Any, Literal
import json
import os

import torch
import torch_npu

# import logging
from vllm.logger import init_logger

from omni_npu import envs
from omni_npu.configs import OmniAdditionalConfig

from .features import (
    apply_eager_mode_config,
    apply_omni_cache,
    apply_seq_parallel,
)


logger = init_logger(__name__)
default_config_path = os.path.normpath(os.path.join(os.path.abspath(__file__), '../../configs'))

MoECommStrategyType = Literal["allreduce", "allgather_reducescatter", "dispatch_combine", "all2allv"]

def load_model_extra_config(model_config, vllm_config, scheduler_config):
    model_name, quant_type= parse_hf_config(model_config.hf_config)
    is_pd_disaggregation = False
    is_prefill_node = None
    if envs.OMNI_PD_ROLE:
        is_pd_disaggregation = True
        is_prefill_node = True if envs.OMNI_PD_ROLE == 'prefill' else False
    if vllm_config.additional_config is not None:
        omni_add = OmniAdditionalConfig.from_vllm_config(vllm_config)
        enable_pd_elastic_scaling = omni_add.enable_pd_elastic_scaling
        enable_low_latency = omni_add.enable_low_latency
        enable_omni_cache = omni_add.enable_omni_cache
    else:
        enable_pd_elastic_scaling = False
        enable_low_latency = False
        enable_omni_cache = False

    if model_config.enforce_eager:
        graph_mode = 'eager_mode'
    else:
        graph_mode = 'acl_graph'

    enable_chunked_prefill = scheduler_config.enable_chunked_prefill
    enable_eplb=vllm_config.parallel_config.enable_eplb
    
    device_name = torch_npu.npu.get_device_name(0)

    if device_name.startswith("Ascend910B"):
        hardware_platform = "A2"
    elif device_name.startswith("Ascend910"):
        hardware_platform = "A3"
    elif device_name.startswith("Ascend950"):
        hardware_platform = "A5"
    else:
        raise ValueError(f"Unsupported device: {device_name}. Only Ascend910/Ascend910B/Ascend950 are supported.")

    global model_extra_config
    model_extra_config.dtype = vllm_config.model_config.dtype

    update_task_config(
        model_name = model_name,
        hardware_platform = hardware_platform,
        is_pd_disaggregation = is_pd_disaggregation,
        is_prefill_node = is_prefill_node,
        quant_type = quant_type,
        prefill_node_num = envs.OMNI_PD_PREFILL_POD_NUM,
        decode_node_num = envs.OMNI_PD_DECODE_POD_NUM,
        enable_eplb = enable_eplb,
        enable_chunked_prefill = enable_chunked_prefill,
        enable_low_latency = enable_low_latency,
        graph_mode = graph_mode,
        enable_pd_elastic_scaling = enable_pd_elastic_scaling,
        enable_omni_cache = enable_omni_cache
    )
    _validate_config(vllm_config.additional_config)
    _print_model_config()


@dataclass
class TaskConfig:
    model_name: str = "deepseek_v3"
    hardware_platform: str = "A3"
    is_pd_disaggregation: bool = True
    is_prefill_node: bool = True # decode_node when it's False
    quant_type: str = "w8a8c16"
    prefill_node_num: int = 0
    decode_node_num: int = 0
    enable_eplb: bool = False
    enable_pd_elastic_scaling: bool = False # 是否支持动态扩缩容，开启时使用默认一套配置
    enable_chunked_prefill: bool = False
    graph_mode: str = "eager_mode"
    enable_low_latency: bool = False
    enable_omni_cache: bool = False


@dataclass
class ModelParallelConfig:
    enable_share_expert_tp: bool = False
    layer_parallel_config: dict[str, Any] = field(default_factory=dict)
    input_split: bool = False
    ena_seq_parallel: bool = False # 模型内，全局序列并行（tp切分token，对齐到tp_size，从vocab_embbeding到最后一个layernorm之间）
    ena_context_parallel: bool = False # DSA的序列并行，依赖ena_seq_parallel开启
    enable_flashcomm2: bool = False
    sharded_o_proj: bool = False # prefill侧o_proj采用sharded linear
    ena_dp_lmhead_parallel: bool = False  # lm_head 按 DP 维度切 vocab，显存降到 1/dp_size
    ena_local_lmhead_parallel: bool = False  # lm_head 按单机 local_world_group 切 vocab
    enable_aicpu_dp_sync: bool = False # dp coordination 是否使用aicpu, 需要HCCL_OP_EXPANSION_MODE=AI_CPU


@dataclass
class ModelOperatorOptConfig:
    enable_kv_rmsnorm_rope_cache: bool = False
    enable_prefill_mla_absorb_pa: bool = False
    kv_nz: bool = False
    gmm_nz: bool = True
    lmhead_fp32: bool = False
    moe_multi_stream_tune: bool = False
    # Shared-experts multi-stream:
    #   shared_expert_multi_stream is the on/off switch. When True, shared experts
    #   run on a dedicated side stream so they overlap with the routed-experts
    #   path. When False, shared experts run synchronously on the main stream and
    #   shared_expert_parallel_schedule is ignored.
    #
    #   shared_expert_parallel_schedule chooses *how* to overlap (only consulted
    #   when shared_expert_multi_stream=True):
    #     "with_finalize"           — (default) shared experts overlap the
    #                                 unpermute/finalize step that runs after
    #                                 apply_experts. Coarser-grained, simpler.
    #     "with_routed_experts_cv"  — finer-grained: shared experts' act_fn and
    #                                 down_proj are interleaved with routed
    #                                 experts' grouped-matmul / vector ops
    #                                 inside apply_experts.
    shared_expert_multi_stream: bool = True
    shared_expert_parallel_schedule: str = "with_finalize"
    enable_mhc_multistream: bool = False  # overlap mhc_sinkhorn with o_proj / w2_gmm on a side stream
    split_q_up_in_multistream: bool = False  # in MLA multi-stream prologs, replace q_b_proj with q_b_nope_proj / q_b_pe_proj children so q_b_pe runs on side_stream parallel to W_UK_T absorb on main
    enable_prefill_shared_exp_multi_stream: bool = False
    enable_agrs_finalize_metadata_overlap: bool = False
    unquant_bmm_nz: bool = True
    prefill_grouped_matmul_finalize_routing: bool = False
    decode_moe_dispatch_combine: bool = False
    enable_super_kernel: bool = False
    sk_code: bool = False # 是否调用 torch.npu.super_kernel_scope_begin/end；老驱动一旦加载到 sk_scope 代码就报错，因此需要开关守卫
    enable_mlaprolog: bool = False
    mtp_remove_redundant_kv: bool = False # MTP场景下，去除FIA算子对同一请求的冗余KV cache搬运，当前不支持与Omni Attention同时使用
    enable_prefetch: bool = True # 是否开启预取
    expert_gate_up_prefetch: int = 50 # 默认预取大小为 50Mb；如果是权重是BF16型，设置为 30Mb
    expert_down_prefetch: int = 28 # 当权重是w8a8且ep_size > 64 时，默认预取大小为 28Mb，否则为0
    dense_mlp_prefetch: int = 56 # 默认预取大小为 56Mb
    lm_head_prefetch: int = 135 # 默认预取大小为 135Mb
    attn_prefetch: int = 96 # 默认预取大小为 96Mb
    shared_expert_gate_up_prefetch: int = 28
    shared_expert_down_prefetch: int = 14

    enable_moe_agrs: bool = False
    enable_moe_allreduce: bool = False
    enable_round_pipeline_comm: bool = False
    enable_pipeline_comm: bool = False
    use_omni_cache: bool = False

    omni_disable_npu_add_rms_norm: bool = False
    use_noncontiguous_kv: bool = False
    use_aicpu_fa_tiling: bool = False
    merge_q_kv_conv: bool = False # 是否合并qa_conv和compresskv_conv
    disable_npu_top_k_top_p_sample: bool = False
    use_mome_inplace_update: bool = False
    enable_multi_stream: bool = False
    use_batch_invariant_op: bool = False
    enable_precision_strong_consistency: bool = False # 是否开启精度强一致性，默认关闭
    router_gating_in_fp32: bool = False # moe router gate 是否使用fp32格式
    enable_mtp_invariant: bool = False # 是否使能MTP一致性

    gmm_fr_token_threshold: int = 0 # 开启gmm_fr的token数阈值，小于等于阈值时开启，默认不开启
    if envs.OMNI_ENABLE_OMNI_CACHE:
        moe_seq_split_length: int = 128 * 10 # omni-cache 开启时使用该配置
    else:
        moe_seq_split_length: int = 10**9 # 开启chunk moe的token数阈值，大于阈值会触发chunk moe, 默认不开启
    use_rope_fusion_op: bool = False # 是否使用npu_apply_rotary_pos_emb融合算子，默认不开启，CANN>=9.0.0可开启
    use_mhc_fusion_op: bool = False # 是否使用mhc大融合算子，默认不开启
    use_mome_inplace_update: bool = False
    moe_comm_strategy: MoECommStrategyType = "dispatch_combine" # MoE通信策略，默认为dispatch_combine
    fix_multi_mtp_kvcache: bool = False # Whether or not to fix kvcache in multi-mtp
    use_topk_topp_stream: bool = False # 是否开启采样topk_topp多流
    li_prolog_multi_stream: bool = False # 是否在 li_prolog 中使用 ki_stream 和 wi_stream 多流并行，会影响确定性
    num_extra_reserved_blocks: int = 0 # 保留额外block数用于APC
    optimize_first_chunk: bool = False # Whether turn off PA (non-absorb mode) and use MoME SP for the first chunk

    def __post_init__(self):

        # Check the dependencies of use_prefetch and prefetch_Mb
        if not self.enable_prefetch:
            self.expert_gate_up_prefetch = 0
            self.expert_down_prefetch = 0
            self.attn_prefetch = 0
            self.dense_mlp_prefetch = 0
            self.lm_head_prefetch = 0
            self.shared_expert_gate_up_prefetch = 0
            self.shared_expert_down_prefetch = 0
            logger.warning(f"[WARNING] When enable_prefetch is false, prefetch_Mb must be set to 0.")

        if envs.OMNI_ENABLE_OMNI_CACHE:
            self.use_omni_cache = True

        # Check for mutually exclusive configuration options
        if self.enable_pipeline_comm and \
                self.enable_round_pipeline_comm:
            raise ValueError(
                "Conflicting communication configuration: "
                "'enable_pipeline_comm' and 'enable_round_pipeline_comm' cannot both be True. "
                "Please disable one of these communication modes."
            )
        
        if self.unquant_bmm_nz:
            # if use weight nz, this config must be True
            torch.npu.config.allow_internal_format = True


@dataclass
class ModelExtraConfig:
    dtype: torch.dtype = torch.bfloat16
    parall_config: ModelParallelConfig = field(default_factory = ModelParallelConfig)
    operator_opt_config: ModelOperatorOptConfig = field(default_factory = ModelOperatorOptConfig)
    task_config: TaskConfig = field(default_factory = TaskConfig)


model_extra_config = ModelExtraConfig()


def filter_dict_by_dataclass(dataclass_type, data_dict):
    valid_keys = {f.name for f in fields(dataclass_type)}
    return {k: v for k, v in data_dict.items() if k in valid_keys}


def update_task_config(**kwargs):
    global model_extra_config
    task_config = model_extra_config.task_config
    if task_config is None:
        task_config = TaskConfig()
    if kwargs:
        for key, value in kwargs.items():
            if hasattr(task_config, key):
                setattr(task_config, key, value)

    _init_model_extra_config(task_config)


def parse_hf_config(hf_config):

    vars_hf_config = vars(hf_config)

    matches = []
    match_hf_configs_path = os.path.join(default_config_path,'match_hf_configs.json')

    match_hf_configs_data = _loader_configs_data(match_hf_configs_path)

    for model_name, model_params in match_hf_configs_data.items():
        # Check if all extracted_params match model parameters
        is_match = True
        for key, value in model_params.items():
            # If model doesn't have this parameter or parameter values don't match
            if key not in vars_hf_config or vars_hf_config[key] != value:
                is_match = False
                break

        if is_match:
            matches.append(model_name)

    # Check matching results
    if len(matches) == 0:
        model_name = hf_config.model_type
    elif len(matches) > 1:
        if hf_config.model_type == "deepseek_v3":
            model_name = "deepseek_v3" 
        elif hf_config.model_type == "deepseek_v32": 
            model_name = "deepseek_v32"
        else:
            raise RuntimeError(
                f"[ERROR] Multiple matching model names found: {matches}. Unable to determine the correct model name."
            )
    else:
        model_name = matches[0]

    if hasattr(hf_config, "quantization_config"):
        quantization_config = hf_config.quantization_config
    elif hasattr(hf_config, "text_config"):
        quantization_config = getattr(hf_config.text_config, "quantization_config", None)
    else:
        quantization_config = None

    if quantization_config is not None and quantization_config.get('format', '').strip() == 'int-quantized':
        weights_type = quantization_config["config_groups"]["group_0"]["weights"]["num_bits"]
        if isinstance(weights_type, dict):
            num_bits_values = weights_type.values()
            weights_type = f"{min(num_bits_values)}"

        input_activations_type = quantization_config["config_groups"]["group_0"]["input_activations"]["num_bits"]
        if isinstance(input_activations_type, dict):
            num_bits_values = input_activations_type.values()
            input_activations_type = f"{min(num_bits_values)}"

        kv_cache_scheme_type = quantization_config["kv_cache_scheme"]
        quant_type = f"w{weights_type}a{input_activations_type}"
        if kv_cache_scheme_type == "Opti-C8":
            quant_type = quant_type+"_fa_c8"
        elif isinstance(kv_cache_scheme_type, dict):
            num_bits_values = kv_cache_scheme_type["num_bits"]
            quant_type = f"{quant_type}c{num_bits_values}"
        else:
            quant_type = f"{quant_type}c16"
    elif quantization_config is not None and quantization_config.get('quant_method', '').strip() == 'hifloat8':
        quant_type = "hif8"
    elif quantization_config is not None and quantization_config.get('quant_method', '').strip() == 'mxfp8':
        quant_type = "mxfp8"
    else:
        if hasattr(hf_config, "dtype") and hf_config.dtype in ["float16", "fp16", torch.float16]:
            quant_type = "fp16"
        else:
            quant_type = "bf16"

    return model_name, quant_type


def _init_model_extra_config(task_config):

    custom_model_config_path = envs.OMNI_CUSTOM_MODEL_CONFIG_PATH
    if custom_model_config_path:
        logger.info(
            f"Get custom_model_config_path from environ: {custom_model_config_path}"
        )
        # load best_pratice_model_config_path from os.environ
        best_practice_model_config_path = os.path.join(default_config_path, custom_model_config_path)
        config_data = _loader_configs_data(best_practice_model_config_path)
    else:
        config_data = _get_best_practice_config(task_config)

    setattr(model_extra_config, 'task_config', task_config)

    if config_data:

        parall_config = ModelParallelConfig(
            **filter_dict_by_dataclass(
                ModelParallelConfig, config_data["model_parallel_config"]
            )
        )
        operator_opt_config = ModelOperatorOptConfig(
            **filter_dict_by_dataclass(
                ModelOperatorOptConfig, config_data["operator_optimization_config"]
            )
        )

        setattr(model_extra_config, 'parall_config', parall_config)
        setattr(model_extra_config, 'operator_opt_config', operator_opt_config)

    else:
        # Set default configs if no config data found
        setattr(model_extra_config, 'parall_config', ModelParallelConfig())
        setattr(model_extra_config, 'operator_opt_config', ModelOperatorOptConfig())


def _loader_configs_data(file_path):
    try:
        with open(file_path, 'r') as f:
            configs_data = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[ERROR] Invalid JSON format in config file: {e}")
    except KeyError as e:
        raise RuntimeError(f"[ERROR] Missing required key in config data: {e}")
    except TypeError as e:
        raise RuntimeError(f"[ERROR] Config structure mismatch or incorrect field types: {e}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Unexpected error while loading model extra config: {e}")

    return configs_data


def _get_best_practice_config(task_config):

    performance_mode = "low_latency" if task_config.enable_low_latency else "high_throughout"

    configs_data = _loader_configs_data(
        os.path.join(
            default_config_path, f"{performance_mode}/best_practice_configs.json"
        )
    )

    configs_list = None
    for data in configs_data:
        if data["model"] == task_config.model_name and \
            data["hardware"] == task_config.hardware_platform and \
                data["precision"] == task_config.quant_type:
            configs_list = data["configs"]
            break

    if task_config.is_pd_disaggregation and not task_config.enable_pd_elastic_scaling:
        pd_scheme = f'{task_config.prefill_node_num}P{task_config.decode_node_num}D'
    elif task_config.is_pd_disaggregation and task_config.enable_pd_elastic_scaling:
        pd_scheme = "pd_elastic_scaling"
    else:
        pd_scheme = 'hybrid'

    task_info = f'{task_config.model_name}_{task_config.quant_type}_{task_config.hardware_platform}_{pd_scheme}'

    if not configs_list:
        logger.warning(
            f"The configuration for {task_config.model_name}_{task_config.quant_type} "
            f"on {task_config.hardware_platform} with performance mode '{performance_mode}' "
            f"was not found in best_practice_configs.json. Loading default configuration."
        )
        return None
    else:
        files_data = configs_list.get(pd_scheme, None)
        if not files_data:
            logger.warning(
                f"The configuration for {task_info} with performance mode '{performance_mode}' "
                f"was not found in best_practice_configs.json. Loading default configuration."
            )
            return None

        if task_config.is_pd_disaggregation and task_config.is_prefill_node:
            model_config_file_path = files_data.get("prefill_config_file")
        elif task_config.is_pd_disaggregation and not task_config.is_prefill_node:
            model_config_file_path = files_data.get("decode_config_file")
        else:
            model_config_file_path = files_data.get("config_file")

        best_practice_model_config_path = os.path.join(
            default_config_path, f"{performance_mode}/{model_config_file_path}"
        )

        if not os.path.exists(best_practice_model_config_path):
            raise RuntimeError(
                f"[ERROR] Task {task_info} requires configuration file {best_practice_model_config_path}, "
                f"but not found."
            )
        else:
            logger.info(
                f"The task about {task_info} load configuration file from {best_practice_model_config_path}")
            config_data = _loader_configs_data(best_practice_model_config_path)

        return config_data


def _validate_config(additional_config):
    global model_extra_config
    apply_eager_mode_config(model_extra_config)
    apply_omni_cache(additional_config)
    apply_seq_parallel(model_extra_config)


def _print_model_config():
    try:
        model_info = json.dumps(asdict(model_extra_config), indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        model_info = repr(model_extra_config)
        logger.warning(f"Failed to JSON-serialize model_extra_config: {e}")
    logger.info(f"ModelExtraConfig: {model_info}")
