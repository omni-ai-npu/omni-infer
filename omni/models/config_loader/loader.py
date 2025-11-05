# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass, field, fields, asdict
import json
import os
import torch.distributed

from vllm.logger import logger
import omni.adaptors.vllm.envs as envs
from omni.models.config_loader.features import apply_eager_mode_config

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


@dataclass
class ModelParallelConfig:
    dense_mlp_tp_size: int = 1
    o_proj_tp_size: int = 1
    attn_sp_size: int = 1
    redundancy_shared_expert_num: int = 0
    enable_attn_ffn_disaggregation: bool = False
    attn_dies: int = 0

 
@dataclass
class ModelOperatorOptConfig:
    enable_kv_rmsnorm_rope_cache: bool = True
    prefill_moe_all_to_all: bool = True
    moe_multi_stream_tune: bool = False
    best_ep: bool = False
    merge_qkv: bool = False
    two_stage_comm: bool = False
    gmm_nz: bool = False
    unquant_bmm_nz: bool = False
    decode_moe_dispatch_combine: bool = True
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
    attn_prefetch: int = 96 # 默认预取大小为 96Mb

    enable_round_pipeline_comm: bool = False
    enable_pipeline_comm: bool = False
    prefill_enable_long_seq: bool = False
    prefill_enable_mla_alltoall: bool = False
    prefill_enable_mla_alltoall_local: bool = False
    fa_quant: bool = False
    use_omni_cache: bool = False
    c8_calib_path: str = None # 计算faquant的scale采集的kv_cache的calib地址，在test_config_prefill.json赋值
    experts_pruning: bool = False
    use_tnd_pa: bool = False  # 稠密模型使用新CANN包FIA算子，以TND+PA格式计算attention

    enable_dsa: bool = False # 使能mla = Indexer + select FA
    max_split_token_ratio_threshold: float = 0.8 # Split hidden_states in prefill if token duplication ratio exceeds threshold, to avoid GMM OOM.
    max_split_token_count_threshold: int = 32768 # Split hidden_states in prefill if token duplication count exceeds threshold, to avoid GMM OOM.
   
    enable_topktoppsample_op: bool = False # 使用topktoppsample算子

    enable_scale_parallel: bool = False #用于qwen235b的scale_parallel优化启用开关，默认关闭

    enable_mlp_seq_split: bool = False # 模型大 + 权重大 + 长序列场景下会OOM，需要切分长度时打开以避免OOM，默认切分大小为4096

@dataclass 
class ModelExtraConfig:
    parall_config: ModelParallelConfig = field(default_factory = ModelParallelConfig)
    operator_opt_config: ModelOperatorOptConfig = field(default_factory = ModelOperatorOptConfig)
    task_config: TaskConfig = field(default_factory = TaskConfig)

    def __post_init__(self):

        # Check the dependencies of use_prefetch and prefetch_Mb
        if not self.operator_opt_config.use_prefetch:
            self.operator_opt_config.expert_gate_up_prefetch = 0
            self.operator_opt_config.expert_down_prefetch = 0
            self.operator_opt_config.attn_prefetch = 0
            logger.warning(f"[WARNING] When enable_prefetch is false, prefetch_Mb must be set to 0.")
        

        if self.operator_opt_config.enable_round_pipeline_comm:
            # Determine NPU count based on hardware platform
            npu_device_count = 8 if self.task_config.hardware_platform.startswith("A2") else 16
            num_nodes = torch.distributed.get_world_size() // npu_device_count
            
            # Validate that exactly 4 nodes are used for round pipeline communication
            if num_nodes != 4:
                raise ValueError(
                    "Invalid node configuration: "
                    f"Expected exactly 4 nodes when using round pipeline communication, "
                    f"but found {num_nodes} nodes (calculated from {torch.distributed.get_world_size()} "
                    f"world size and {npu_device_count} NPUs per node)."
                )
            
        if os.getenv("ENABLE_OMNI_CACHE", "0") == "1":
            self.operator_opt_config.use_omni_cache = True

        # Check for mutually exclusive configuration options
        if self.operator_opt_config.enable_pipeline_comm and \
                self.operator_opt_config.enable_round_pipeline_comm:
            raise ValueError(
                "Conflicting communication configuration: "
                "'enable_pipeline_comm' and 'enable_round_pipeline_comm' cannot both be True. "
                "Please disable one of these communication modes."
            )


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
        input_activations_type = hf_config.quantization_config["config_groups"]["group_0"]["input_activations"]["num_bits"]
        kv_cache_scheme_type = hf_config.quantization_config["kv_cache_scheme"]
        quant_type = f"w{weights_type}a{input_activations_type}"
        if isinstance(weights_type, dict) and weights_type.get('mlp.experts', None) == 4:
            quant_type = "w4a8"
        if kv_cache_scheme_type == "Opti-C8":
            quant_type = quant_type+"_fa_c8"
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
        for c in configs_data if c.get("pd_disaggregation") is not None
    }

    node_elasticly_config_map = {
        (c["model"], c["hardware"], c["precision"],c["enable_pd_elastic_scaling"]): \
        (c["prefill_config_file"], c["decode_config_file"])
        for c in configs_data if c.get("enable_pd_elastic_scaling") is not None
    }

    return config_map, node_elasticly_config_map



def _get_best_practice_config(task_config):
    config_map, node_elasticly_config_map = _load_best_practice_config()

    if not task_config.enable_pd_elastic_scaling:
        best_practice_model_config_path = config_map.get((task_config.model_name,
            task_config.hardware_platform, task_config.quant_type, task_config.is_pd_disaggregation,
            task_config.prefill_node_num,task_config.decode_node_num), None)
    else:
        best_practice_model_config_path = node_elasticly_config_map.get((task_config.model_name,
            task_config.hardware_platform, task_config.quant_type, task_config.enable_pd_elastic_scaling), None)
    
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


def _init_model_extra_config(task_config) -> ModelExtraConfig:

    config_data = _get_best_practice_config(task_config)

    if config_data:

        parall_config = ModelParallelConfig(**filter_dict_by_dataclass(ModelParallelConfig,config_data['model_parallel_config']))
        try:
            operator_opt_config = ModelOperatorOptConfig(**filter_dict_by_dataclass(ModelOperatorOptConfig, config_data['operator_optimization_config']))
        except KeyError:
            operator_opt_config = ModelOperatorOptConfig(**filter_dict_by_dataclass(ModelOperatorOptConfig, config_data['operator_optimizition_config']))

        setattr(model_extra_config, 'task_config', task_config)
        setattr(model_extra_config, 'parall_config', parall_config)
        setattr(model_extra_config, 'operator_opt_config', operator_opt_config)

model_extra_config = ModelExtraConfig()

def _validate_config():
    global model_extra_config
    if not model_extra_config.task_config.enable_graph_mode:
        apply_eager_mode_config(model_extra_config.operator_opt_config)
        logger.warning(
            f"[WARNING] Eager mode disables all these optimization configurations by default."
        )


@register_config_updaters('update_task_config')
def update_task_config(**kwargs):
    global model_extra_config
    task_config = model_extra_config.task_config
    if task_config is not None and kwargs:
        for key, value in kwargs.items():
            if hasattr(task_config, key):
                setattr(task_config, key, value)
                logger.info(f"{key} loads from vllm config: {value}")
    task_config.model_name, task_config.quant_type = parse_hf_config(kwargs['hf_config'])
    _init_model_extra_config(task_config)
    _validate_config()

    try:
        model_info = json.dumps(asdict(model_extra_config), indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        model_info = repr(model_extra_config)
        logger.warning(f"Failed to JSON-serialize model_extra_config: {e}")
    logger.info(f"ModelExtraConfig: {model_info}")
    





