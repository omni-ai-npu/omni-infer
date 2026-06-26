# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# This patch is used for enable_eplb fix in ParallelConfig and FusedMoE
# Please use this patch by adding VLLM_PLUGINS="omni-npu,omni_npu_patches" OMNI_NPU_VLLM_PATCHES="EPLBEngineConfig,EPLBSharedFusedMoE" before vllm serve

import torch
from typing import Optional, Sequence
from vllm.logger import init_logger
from vllm.config import ModelConfig, ParallelConfig
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.distributed.eplb.eplb_state import EplbState
logger = init_logger(__name__)
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
try:
    from omni_placement.omni_planner import OmniPlanner
except ImportError:
    logger.warning("OmniPlanner is not installed, please install it when use vllm and set --enable_eplb True")
    OmniPlanner = None

@register_patch("EPLBState", EplbState)
class EPLBStatePatch(VLLMPatch):
    """
    EPLBState Patch for EPLB on NPU devices (when eplb enabled).
    """
    _attr_names_to_apply = ['__init__', 'build_initial_global_physical_to_logical_map', 'add_model', 'step', 'rearrange', 'start_async_loop', 'recv_state', 'get_eep_state']

    planner: Optional[OmniPlanner] = None
    start_step: bool = False

    def __init__(self, parallel_config: ParallelConfig, device: torch.device):
        self.planner = None
        self.start_step = None
        self.is_async = False

    @staticmethod
    def build_initial_global_physical_to_logical_map(
        num_routed_experts: int,
        num_redundant_experts: int,
    ) -> Sequence[int]:
        global_physical_to_logical_map = list(range(num_routed_experts))
        global_physical_to_logical_map += [
            i % num_routed_experts for i in range(num_redundant_experts)
        ]
        return global_physical_to_logical_map

    def add_model(
        self,
        model: MixtureOfExperts,
        model_config: ModelConfig,
        global_expert_load: torch.Tensor | None = None,
        old_global_expert_indices: torch.Tensor | None = None,
        rank_mapping: dict[int, int] | None = None,
    ):
        if OmniPlanner is not None and model_config.runner != "draft":
            param_dict = dict(model.named_parameters())
            planner = OmniPlanner()
            planner.init_dram_weights(param_dict, first_k_dense_replace=model_config.hf_config.first_k_dense_replace)
            self.planner = planner
        else:
            self.planner = None

    def step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False,
    ) -> None:
        if OmniPlanner is not None:
            if not self.start_step:
                self.planner.start_dynamic_optimize_expert_load_balance()
                self.start_step = True
            self.planner.place_experts()
        else:
            pass

    def rearrange(
        self,
        is_profile: bool = False,
        execute_shuffle: bool = True,
        global_expert_loads: list[torch.Tensor] | None = None,
        rank_mapping: dict[int, int] | None = None,
    ) -> torch.Tensor | None:
        pass

    def start_async_loop(
        self,
        rank_mapping: dict[int, int] | None = None,
        is_profile: bool = False,
    ):
        pass

    @staticmethod
    def recv_state() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        pass

    @classmethod
    def get_eep_state(
        cls, parallel_config: ParallelConfig
    ) -> tuple[
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
        dict[int, int] | None,
    ]:
        global_expert_loads = None
        old_global_expert_indices_per_model = None
        rank_mapping = None
        return (
            global_expert_loads, 
            old_global_expert_indices_per_model, 
            rank_mapping,
        )