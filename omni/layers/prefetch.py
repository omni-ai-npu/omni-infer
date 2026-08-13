# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from typing import Any, Dict, Optional, List

import torch
import torch_npu

from omni_npu.model_config.config_loader.loader import model_extra_config

_ATTN_LINEAR_NAMES = (
    "fused_qkv_a_proj",
    "q_a_proj",
    "q_b_proj",
    "q_proj",
    "kv_a_proj_with_mqa",
)

_PREFETCH_GROUP_REGISTRY: Dict[str, str] = {
    "next_attn": "_prefetch_next_attn",
    "moe": "_prefetch_moe",
}

PREFETCH_UNIT_SIZE = 1 << 20
# Names of linear layers that support attention prefetching
_attn_prefetch_linear: List[str] = []

# Cache for next decoder layer's prefetched data
_prefetch_next_decoder_layer: Dict[Any, Any] = {}


def setup_prefetch_for_model(model: Any) -> None:
    if not model_extra_config.operator_opt_config.enable_prefetch:
        return
    _prefetch_next_decoder_layer.clear()
    _attn_prefetch_linear.clear()
    layers = getattr(model, "layers", None)
    if not layers or not hasattr(layers, "__getitem__"):
        return

    if model_extra_config.operator_opt_config.attn_prefetch <= 0:
        return
        
    self_attn = getattr(layers[0], "self_attn", None)
    for name in _ATTN_LINEAR_NAMES:
        linear = getattr(self_attn, name, None)
        if linear is not None:
            _attn_prefetch_linear.append(name)

    for idx in range(len(layers) - 1):
        mlp = getattr(layers[idx], "mlp", None)
        if mlp is None:
            continue
        moe = getattr(mlp, "experts", None)
        if moe is None:
            continue
        _prefetch_next_decoder_layer[moe] = layers[idx + 1]


class PrefetchManager:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def prefetch_weight(
        self,
        weight: Optional[torch.Tensor],
        trigger: Optional[torch.Tensor],
        prefetch_size: Optional[int],
    ) -> None:
        missing_required_input = weight is None or trigger is None or prefetch_size is None
        invalid_prefetch_size = prefetch_size is not None and prefetch_size <= 0
        if missing_required_input or invalid_prefetch_size:
            return
        torch_npu.npu_prefetch(weight, trigger, prefetch_size * PREFETCH_UNIT_SIZE)

    def prefetch(
        self,
        group: str,
        trigger: torch.Tensor,
        layer: Optional[torch.nn.Module] = None,
    ) -> None:
        if layer is None:
            return
        prefetch_method_name = _PREFETCH_GROUP_REGISTRY.get(group)
        if prefetch_method_name is None:
            return
        prefetch_for_group = getattr(self, prefetch_method_name, None)
        if prefetch_for_group is None or not callable(prefetch_for_group):
            return
        prefetch_for_group(layer, trigger)

    def _prefetch_next_attn(self, layer: Any, trigger: torch.Tensor) -> None:
        operator_opt = model_extra_config.operator_opt_config
        next_layer = _prefetch_next_decoder_layer.get(layer)
        if next_layer is None:
            return
        next_attn = getattr(next_layer, "self_attn", None)
        if next_attn is None:
            return
        for name in _attn_prefetch_linear:
            proj = getattr(next_attn, name, None)
            if proj is None:
                continue
            weight = getattr(proj, "weight", None)
            self.prefetch_weight(weight, trigger, operator_opt.attn_prefetch)

    def _prefetch_moe(self, layer: Any, trigger: torch.Tensor) -> None:
        operator_opt = model_extra_config.operator_opt_config
        if hasattr(layer, "w13_weight") and layer.w13_weight.numel() > 0:
            self.prefetch_weight(layer.w13_weight, trigger, operator_opt.expert_gate_up_prefetch)
        if hasattr(layer, "w2_weight") and layer.w2_weight.numel() > 0:
            self.prefetch_weight(layer.w2_weight, trigger, operator_opt.expert_down_prefetch)
        shared_experts = getattr(layer, "shared_experts", None)
        if shared_experts is not None:
            if getattr(shared_experts, "gate_up_proj", None):
                self.prefetch_weight(
                    shared_experts.gate_up_proj.weight,
                    trigger,
                    operator_opt.shared_expert_gate_up_prefetch,
                )
            if getattr(shared_experts, "down_proj", None):
                self.prefetch_weight(
                    shared_experts.down_proj.weight,
                    trigger,
                    operator_opt.shared_expert_down_prefetch,
                )
