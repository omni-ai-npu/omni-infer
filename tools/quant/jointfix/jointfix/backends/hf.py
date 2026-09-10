# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
HuggingFace backend — generic LLaMA / Qwen / Mistral / Qwen2-MoE.

Proves the model axis on a public model: `--backend hf --method jointfix`
should quantize an off-the-shelf LLaMA. Uses transformers for config + module
construction and model.safetensors.index.json for the shard map.

STATUS: experimental scaffold — not yet implemented. The tricky part is
downstream_consumers() for MoE variants (router stays BF16).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from jointfix.backends.base import LayerSpec, ModelBackend
from jointfix.core.primitives import UNIVERSAL_SKIP_PATTERNS
from jointfix.registry import register_backend


@register_backend("hf")
class HFBackend(ModelBackend):
    def skip_patterns(self) -> List[str]:
        return list(UNIVERSAL_SKIP_PATTERNS)

    def config(self) -> dict:
        raise NotImplementedError("hf backend:load config.json")

    def layer_specs(self) -> List[LayerSpec]:
        raise NotImplementedError("hf backend:parse HF config.json -> LayerSpec list")

    def weight_map(self) -> Dict[str, str]:
        raise NotImplementedError("hf backend:model.safetensors.index.json")

    def load_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("hf backend:read via model.safetensors.index.json")

    def embed(self, input_ids, device):
        raise NotImplementedError("hf backend:embed_tokens row-gather")

    def save_quantized(self, out_dir, layer_idx, tensors, quant_meta) -> None:
        raise NotImplementedError("hf backend")

    def build_layer(self, spec, weights, device) -> nn.Module:
        raise NotImplementedError("hf backend:transformers decoder layer (meta build)")

    def install_stat_hooks(self, layer, layer_idx, collectors, stats_config) -> list:
        raise NotImplementedError("hf backend:nn.Module forward hooks")
