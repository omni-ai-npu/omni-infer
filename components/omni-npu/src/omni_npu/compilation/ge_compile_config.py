# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import hashlib
import os
from typing import Any, Callable, Optional, Union

import torch
import torch_npu
import torchair
from torchair import patch_for_hcom

from vllm.config import VllmConfig
from vllm.logger import init_logger

#from omni_npu.models.config_loader.loader import model_extra_config

logger = init_logger(__name__)

MAX_GEAR_NUM = 6
BLOCK_NUM_FLOATING_RANGE = 30

def get_torchair_config(vllm_config: VllmConfig):
    patch_for_hcom()
    config = torchair.CompilerConfig()
    if os.environ.get("FROZEN_PARAMETER_DISABLED", "0") == "0":
        config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    torch.npu.set_compile_mode(jit_compile=False)
    return config


class NPUCompilationConfig:
    use_gegraph: bool = False
    """Whether to use ge-graph inside compilation.
    - False: ge-graph inside compilation is not used.
    - True: ge-graph inside compilation is used."""

    backend: Optional[str] = None
    """The backend for compilation."""

    use_ge_graph_cached: bool = False
    """Whether to use ge backend graph caching."""

    decode_gear_list: Optional[list[int]] = None
    """The gear size of the different static plots"""

    block_num_floating_range: int = BLOCK_NUM_FLOATING_RANGE
    """The compilation cache allows for the range of fluctuations"""

    def build_from_cli(self, raw_graph_config: dict[str,Any], vllm_config: VllmConfig):
        """Parse the CLI value for the compilation config.
        """
        import os
        self.use_gegraph = os.getenv("TORCH_COMPILE_GE", "False").lower() == "true"
        self.backend = raw_graph_config.get("backend", None)
        self.use_ge_graph_cached = raw_graph_config.get("use_ge_graph_cached", False)
        self.decode_gear_list = raw_graph_config.get("decode_gear_list", None)
        if self.decode_gear_list and not isinstance(self.decode_gear_list, list):
            raise TypeError("decode_gear_list must be a list")

        self.update_gear_options(vllm_config)

        logger.info(f"the NPUCompilationConfig value is: {self}")

    def update_gear_options(self, vllm_config: VllmConfig):
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        use_spec_decode = vllm_config.speculative_config is not None
        enable_adaptive = use_spec_decode and vllm_config.speculative_config.enable_adaptive
        max_gear_size = vllm_config.scheduler_config.max_num_batched_tokens
        if vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.kv_role == "kv_consumer" or vllm_config.additional_config.get("enable_hybrid_graph_mode", False):
            max_gear_size = max_num_reqs if not use_spec_decode else max_num_reqs * (1 + vllm_config.speculative_config.num_speculative_tokens)
    
        if self.decode_gear_list is not None and len(self.decode_gear_list) > MAX_GEAR_NUM:
            raise ValueError(f"Max gear num supported is {MAX_GEAR_NUM} now.")

        if self.decode_gear_list and max(self.decode_gear_list) > max_gear_size:
            decode_gear_list = [gear for gear in self.decode_gear_list if gear <= max_gear_size]
            logger.warning(
                f"PTA_TORCHAIR_DECODE_GEAR_LIST({self.decode_gear_list}) becomes ({decode_gear_list}) due to max_batch_size({max_gear_size})")
            self.decode_gear_list = decode_gear_list

        if not self.decode_gear_list:
            self.decode_gear_list = [max_gear_size]

        if (not enable_adaptive and len(self.decode_gear_list) < MAX_GEAR_NUM and max(self.decode_gear_list) < max_gear_size):
            self.decode_gear_list.append(max_gear_size)

    def init_backend(self, vllm_config: VllmConfig) -> Union[str, Callable]:
        if not self.use_gegraph:
            raise ValueError("use_gegraph is not set.")

        if not self.backend or self.backend == "":
            config = get_torchair_config(vllm_config)
            npu_backend = torchair.get_npu_backend(compiler_config=config)
            logger.debug(f"Using torchair backend!")
            return npu_backend

        logger.debug(f"Using user-defined backend!")
        return self.backend

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.
        """
        factors: list[Any] = [self.level, self.backend, self.block_num_floating_range]
        return hashlib.sha256(str(factors).encode()).hexdigest()
