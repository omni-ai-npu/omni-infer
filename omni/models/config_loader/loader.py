# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass, field, fields, asdict
import json
import os
import torch

import logging
from omni.models.config_loader.features import apply_eager_mode_config, apply_fusion_pass

def init_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(console_handler)
    return logger

logger = init_logger(__name__)

default_config_path = os.path.normpath(os.path.join(os.path.abspath(__file__), '../../configs'))

_MODEL_EXTRA_CONFIG_UPDATERS = {}

def register_config_updaters(config_updater_name: str):
    def warpper(func):
        if _MODEL_EXTRA_CONFIG_UPDATERS.get(config_updater_name, False):
            raise ValueError(f"Duplicate registration: {config_updater_name}")
        _MODEL_EXTRA_CONFIG_UPDATERS[config_updater_name] = func
        return func
    return warpper


def call_config_updater(config_updater_name: str, **kwargs):
    if config_updater_name not in _MODEL_EXTRA_CONFIG_UPDATERS:
        raise KeyError(f"Unknown task config updater: {config_updater_name}. Avaliable: {list(_MODEL_EXTRA_CONFIG_UPDATERS.keys())}")
    func = _MODEL_EXTRA_CONFIG_UPDATERS[config_updater_name]
    func(**kwargs)
    logger.info(f"Task config updated via '{config_updater_name}'")


@dataclass
class TaskConfig:
    model_name: str = "deepseek_v3"
    hardware_platform: str = "A3"
    is_pd_disaggregation: bool = True
    is_prefill_node: bool = True # decode_node when it's False
    quant_type: str = "w8a8"
    prefill_node_num: int = 0
    decode_node_num: int = 0
    enable_omni_placement: bool = False
    enable_pd_elastic_scaling: bool = False # 是否支持动态扩缩容，开启时使用默认一套配置
    decode_gear_list: list[int] = field(default_factory = lambda: [1])
    enable_chunked_prefill: bool = False
    enable_graph_mode: bool = True
    enable_attn_ffn_disaggregation: bool = False
    low_latency: bool = False


@dataclass
class ModelParallelConfig:
    dense_mlp_tp_size: int = 1
    o_proj_tp_size: int = 1
    attn_sp_size: int = 1
    redundancy_shared_expert_num: int = 0
    attn_dies: int = 0
    enable_share_expert_tp: bool = False
    eh_proj_tp_size: int = 1

 
@dataclass
class ModelOperatorOptConfig:
    enable_kv_rmsnorm_rope_cache: bool = True
    prefill_moe_all_to_all: bool = True
    enable_moe_prefill_multi_stream: bool = False
    moe_multi_stream_tune: bool = False
    best_ep: bool = False
    merge_qkv: bool = False
    two_stage_comm: bool = False
    gmm_nz: bool = False
    unquant_bmm_nz: bool = False
    decode_moe_dispatch_combine: bool = True
    decode_flash_comm_1: bool = True # decode节点开启FlashComm1优化
    use_super_kernel: bool = False
    enable_prefill_micro_batch: bool = False
    use_mlaprolog: bool = False
    cast_w2_scale_f32: bool = False
    control_accept_rate: float = -1 # <0 or >1 不控制, >=0 and <=1 控制MTP开启时接受率为该值，几乎必然导致输出结果异常，仅保证只投机1个token时满足这一数值
    mla_multistream_limit_core: str = '' # 空字符串代表不开启多流分核，形如'20|36'代表主流分配的AIC和AIV核数分别为20和36
    shared_experts_to_gmm: bool = False # 当redundancy_shared_expert_num > 0时，共享专家使用GMM代替BMM进行计算（限定收益场景：EP288 + 单die bs >= 48，仅针对Decode阶段）
    enable_gmm_swiglu_quant: bool = False # 当redundancy_shared_expert_num > 0时，使用npu_grouped_matmul_swiglu_quant_v2融合算子
    mtp_remove_redundant_kv: bool = False # MTP场景下，去除FIA算子对同一请求的冗余KV cache搬运，当前不支持与Omni Attention同时使用
    use_prefetch: bool = True # 是否开启预取
    expert_gate_up_prefetch: int = 50 # 默认预取大小为 50Mb；如果是权重是BF16型，设置为 30Mb
    expert_down_prefetch: int = 28 # 当权重是w8a8且ep_size > 64 时，默认预取大小为 28Mb，否则为0
    dense_mlp_prefetch: int = 56 # 默认预取大小为 56Mb
    lm_head_prefetch: int = 135 # 默认预取大小为 135Mb
    attn_prefetch: int = 96 # 默认预取大小为 96Mb
    shared_expert_gate_up_prefetch: int = 28
    shared_expert_down_prefetch: int = 14

    enable_round_pipeline_comm: bool = False
    enable_pipeline_comm: bool = False
    prefill_enable_long_seq: bool = False
    prefill_enable_mla_alltoall: bool = False
    prefill_enable_mla_alltoall_local: bool = False
    fa_quant: bool = False
    use_omni_cache: bool = False
    tp_nnodes: int = 1
    c8_calib_path: str = None # 计算faquant的scale采集的kv_cache的calib地址，在test_config_prefill.json赋值
    experts_pruning: bool = False
    use_tnd_pa: bool = True  # 稠密模型使用新CANN包FIA算子，以TND+PA格式计算attention

    enable_dsa: bool = False # 使能mla = Indexer + select FA
    enable_indexer_quant: bool = False # 使能indexer量化
    max_split_token_ratio_threshold: float = 0.8 # Split hidden_states in prefill if token duplication ratio exceeds threshold, to avoid GMM OOM.
    max_split_token_count_threshold: int = 32768 # Split hidden_states in prefill if token duplication count exceeds threshold, to avoid GMM OOM.
   
    enable_topktoppsample_op: bool = False # 使用topktoppsample算子

    enable_scale_parallel: bool = False #用于qwen235b的scale_parallel优化启用开关，默认关闭
    ascend_operator_fusion_pass_set: str = '' #用于控制关闭算子融合，为空代表不关闭任何算子融合

    enable_mlp_seq_split: bool = False # 模型大 + 权重大 + 长序列场景下会OOM，需要切分长度时打开以避免OOM，默认切分大小为4096
    enable_mla_prefill_multistream: bool = False # mla prefill阶段qkv计算启用多流
    decode_experts_pruning: bool = False
    new_w4_op: bool = False # w4a8新算子
    enable_c8: bool = False # GQA使能C8
    
    enable_scmoe_multi_stream: bool = False # 龙猫ScMoe架构多流开启
    load_rms_bias: bool = False # RMSNorm是否启用bias
    enable_e_score_correction_bias: bool = False # moe部分是否启用e_score_correction_bias
    use_dcp: bool = False # 开启KV切片和decode context parallel
    enable_omni_attn_optimization: bool = True
    enable_attn_update: bool = False # use_dcp场景下是否使用attention update融合算子; 1230商发CANN不支持入图

    def __post_init__(self):

        # Check the dependencies of use_prefetch and prefetch_Mb
        if not self.use_prefetch:
            self.expert_gate_up_prefetch = 0
            self.expert_down_prefetch = 0
            self.attn_prefetch = 0
            self.dense_mlp_prefetch = 0
            self.lm_head_prefetch = 0
            self.shared_expert_gate_up_prefetch = 0
            self.shared_expert_down_prefetch = 0
            logger.warning(f"[WARNING] When enable_prefetch is false, prefetch_Mb must be set to 0.")

            
        if os.getenv("ENABLE_OMNI_CACHE", "0") == "1":
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
    parall_config: ModelParallelConfig = field(default_factory = ModelParallelConfig)
    operator_opt_config: ModelOperatorOptConfig = field(default_factory = ModelOperatorOptConfig)
    task_config: TaskConfig = field(default_factory = TaskConfig)



def filter_dict_by_dataclass(dataclass_type, data_dict):
    valid_keys = {f.name for f in fields(dataclass_type)}
    return {k: v for k, v in data_dict.items() if k in valid_keys}


def parse_hf_config(hf_config):
    
    # Fixed parameter key list (parameters to check)
    FIXED_KEYS = [
    "hidden_size",
    "num_attention_heads",
    "max_position_embeddings",
    "vocab_size",
    "intermediate_size",
    "n_routed_experts",
    "n_shared_experts",
    "moe_intermediate_size"
    ]
    
    extracted_params = {}
    
    vars_hf_config = vars(hf_config)
    for key in FIXED_KEYS:
        if key in vars_hf_config:
            extracted_params[key] = vars_hf_config[key]
        else:
            extracted_params[key] = None

    matches = []
    match_hf_configs_path = os.path.join(default_config_path,'match_hf_configs.json')

    match_hf_configs_data = _loader_configs_data(match_hf_configs_path)

    for model_name, model_params in match_hf_configs_data.items():
        # Check if all extracted_params match model parameters
        is_match = True
        for key, value in extracted_params.items():
            # If model doesn't have this parameter or parameter values don't match
            if key not in model_params or model_params[key] != value:
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
            raise RuntimeError(f"[ERROR] Multiple matching model names found: {matches}. Unable to determine the correct model name.")
    else:
        model_name = matches[0]

    if hasattr(hf_config, "quantization_config") and hf_config.quantization_config['format'].strip() == 'int-quantized':
        weights_type = hf_config.quantization_config["config_groups"]["group_0"]["weights"]["num_bits"]
        if isinstance(weights_type, dict):
            num_bits_values = weights_type.values()
            weights_type = min(num_bits_values)

        input_activations_type = hf_config.quantization_config["config_groups"]["group_0"]["input_activations"]["num_bits"]
        if isinstance(input_activations_type, dict):
            num_bits_values = input_activations_type.values()
            input_activations_type = min(num_bits_values)
        
        kv_cache_scheme_type = hf_config.quantization_config["kv_cache_scheme"]
        quant_type = f"w{weights_type}a{input_activations_type}"
        if kv_cache_scheme_type == "Opti-C8":
            quant_type = quant_type+"_fa_c8"
        elif isinstance(kv_cache_scheme_type, dict):
            num_bits_values = kv_cache_scheme_type["num_bits"]
            quant_type = f"{quant_type}c{num_bits_values}"
    else:
        quant_type = "bf16"

    return model_name, quant_type


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


def _load_best_practice_config():
    best_practice_configs_path = os.path.join(default_config_path, 'best_practice_configs.json')
    
    if not os.path.exists(best_practice_configs_path):
        raise RuntimeError(f"[ERROR] Best practice configs file not found: {best_practice_configs_path}")
    
    configs_data = _loader_configs_data(best_practice_configs_path)
    
    config_map = {
        (c["model"], c["hardware"], c["precision"], c["pd_disaggregation"],c["prefill_node_num"],c["decode_node_num"]): \
        (c["prefill_config_file"], c["decode_config_file"])
        for c in configs_data if c.get("pd_disaggregation") is not None and c.get("attn_ffn_disaggregation") is None \
            and c.get("low_latency") is None
    }

    node_elasticly_config_map = {
        (c["model"], c["hardware"], c["precision"],c["enable_pd_elastic_scaling"]): \
        (c["prefill_config_file"], c["decode_config_file"])
        for c in configs_data if c.get("enable_pd_elastic_scaling") is not None
    }

    afd_config_map = {
        (c["model"], c["hardware"], c["precision"], c["pd_disaggregation"],c["prefill_node_num"],c["decode_node_num"]): \
        (c["prefill_config_file"], c["decode_config_file"])
        for c in configs_data if c.get("pd_disaggregation") is not None and c.get("attn_ffn_disaggregation") is not None
    }

    low_latency_map = {
        (c["model"], c["hardware"], c["precision"], c["pd_disaggregation"],c["prefill_node_num"],c["decode_node_num"]): \
        (c["prefill_config_file"], c["decode_config_file"])
        for c in configs_data if c.get("pd_disaggregation") is not None and c.get("attn_ffn_disaggregation") is None \
            and c.get("low_latency") is not None
    }

    return config_map, node_elasticly_config_map, afd_config_map, low_latency_map



def _get_best_practice_config(task_config):
    best_practice_model_config_path = os.environ.get("BEST_PRATICE_MODEL_CONFIG_PATH", None)
    if best_practice_model_config_path:
        logger.info(f"Get best_practice_model_config_path from environ")
        # load best_pratice_model_config_path from os.environ
        best_practice_model_config_path = json.loads(best_practice_model_config_path)
        best_practice_model_config_path = [best_practice_model_config_path["prefill_config_file"],
                                           best_practice_model_config_path["decode_config_file"]]
    else:
        config_map, node_elasticly_config_map, afd_config_map, low_latency_map = _load_best_practice_config()

        if task_config.enable_attn_ffn_disaggregation:
            best_practice_model_config_path = afd_config_map.get((task_config.model_name,
                task_config.hardware_platform, task_config.quant_type, task_config.is_pd_disaggregation,
                task_config.prefill_node_num,task_config.decode_node_num), None)
        elif task_config.low_latency:
            best_practice_model_config_path = low_latency_map.get((task_config.model_name,
                task_config.hardware_platform, task_config.quant_type, task_config.is_pd_disaggregation,
                task_config.prefill_node_num,task_config.decode_node_num), None)
        elif not task_config.enable_pd_elastic_scaling:
            best_practice_model_config_path = config_map.get((task_config.model_name,
                task_config.hardware_platform, task_config.quant_type, task_config.is_pd_disaggregation,
                task_config.prefill_node_num,task_config.decode_node_num), None)
        else:
            best_practice_model_config_path = node_elasticly_config_map.get((task_config.model_name,
                task_config.hardware_platform, task_config.quant_type, task_config.enable_pd_elastic_scaling), None)
            
        logger.info(f"Get best_practice_model_config_path from best_practice_configs.json")
    
    task_info = f'{task_config.model_name}_{task_config.quant_type}_{task_config.hardware_platform}'
    if best_practice_model_config_path:
        if task_config.is_prefill_node:
            best_practice_model_config_path = best_practice_model_config_path[0]
        else:  
            best_practice_model_config_path = best_practice_model_config_path[1]

        best_practice_model_config_path = os.path.join(default_config_path, best_practice_model_config_path)

        if not os.path.exists(best_practice_model_config_path):
            raise RuntimeError(f"[ERROR] Task {task_info} requires configuration file {best_practice_model_config_path}, but not found.")
        else:
            logger.info(
                f"The task about {task_info} load configuration file from {best_practice_model_config_path}")
            config_data = _loader_configs_data(best_practice_model_config_path)

    else:
        config_data = None
        logger.info(
            f"The task about {task_info} does not require configuration file, using default configuration.")
    
    return config_data


def _init_model_extra_config(task_config):

    config_data = _get_best_practice_config(task_config)

    if config_data:

        parall_config = ModelParallelConfig(**filter_dict_by_dataclass(ModelParallelConfig,config_data['model_parallel_config']))
        try:
            operator_opt_config = ModelOperatorOptConfig(**filter_dict_by_dataclass(ModelOperatorOptConfig, config_data['operator_optimization_config']))
        except KeyError:
            operator_opt_config = ModelOperatorOptConfig(**filter_dict_by_dataclass(ModelOperatorOptConfig, config_data['operator_optimizition_config']))

        setattr(operator_opt_config, 'use_omni_cache', getattr(operator_opt_config, 'use_omni_cache', False))
        setattr(operator_opt_config, 'use_omni_cache', os.getenv("ENABLE_OMNI_CACHE") == "1")
        setattr(operator_opt_config, 'tp_nnodes', int(os.environ.get("OMNI_CACHE_TP_NNODES", "1")))

        setattr(model_extra_config, 'task_config', task_config)
        setattr(model_extra_config, 'parall_config', parall_config)
        setattr(model_extra_config, 'operator_opt_config', operator_opt_config)

model_extra_config = ModelExtraConfig()

def _validate_config():
    global model_extra_config
    apply_eager_mode_config(model_extra_config)
    apply_fusion_pass(model_extra_config)


@register_config_updaters('update_task_config')
def update_task_config(**kwargs):
    global model_extra_config
    task_config = model_extra_config.task_config
    if task_config is not None and kwargs:
        for key, value in kwargs.items():
            if hasattr(task_config, key):
                setattr(task_config, key, value)
                logger.info(f"{key} loads parameters from framework : {value}")

    hf_config = kwargs.get('hf_config')
    if hf_config is None:
        raise KeyError("hf_config is required for update_task_config")

    task_config.model_name, task_config.quant_type = parse_hf_config(hf_config)
    _init_model_extra_config(task_config)
    _validate_config()

    try:
        model_info = json.dumps(asdict(model_extra_config), indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        model_info = repr(model_extra_config)
        logger.warning(f"Failed to JSON-serialize model_extra_config: {e}")
    logger.info(f"ModelExtraConfig: {model_info}")
    





