# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
Pangu backend — built on _pangu_torch_calib.

Wraps PanguTorchDecoderLayer / build_pangu_torch_layer_specs / the
PanguTorchMoE.forward monkey-patch behind the ModelBackend interface. The heavy
vendored module is imported LAZILY (inside methods) so `import jointfix` stays
light.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn

from jointfix.backends.base import LayerSpec, ModelBackend
from jointfix.core.primitives import UNIVERSAL_SKIP_PATTERNS
from jointfix.registry import register_backend

# Pangu-specific non-quantized linears (DSA LightningIndexer + MHC).
_PANGU_SKIP_PATTERNS = [
    "indexer.wk",
    "indexer.weights_proj",
    "indexer.wq_b",
    "mhc_module.phi",
]

# MHC and gate params keep their initialized dtype (not the safetensors dtype).
_PRESERVE_PATTERNS = (".attn_mhc_module.", ".mlp_mhc_module.", "mlp.gate.weight")


def _has_npu() -> bool:
    try:
        return hasattr(torch, "npu") and torch.npu.is_available()
    except (AttributeError, RuntimeError):
        return False


def _meta_build_layer(layer_tensors, model_config, pangu_spec, device):
    """
    Meta-device build of a PanguTorchDecoderLayer.

    meta construct (0 mem, 0 init) -> to_empty(device) -> per-param H2D load.
    H2D is non_blocking=False + an NPU sync barrier: on 505B (~22 GB/layer) async
    copies raced the forward and were read as NaN. Do NOT reintroduce
    non_blocking=True without an explicit synchronize.
    """
    from jointfix.backends._pangu_torch_calib import PanguTorchDecoderLayer

    pfx = f"model.layers.{pangu_spec.layer_idx}."
    with torch.device("meta"):
        layer = PanguTorchDecoderLayer(model_config, pangu_spec, full_layer=True)
    layer = layer.to_empty(device=device)

    for name, param in layer.named_parameters():
        full = pfx + name
        if full not in layer_tensors:
            # safe-fill unloaded so to_empty garbage can't propagate NaN
            param.data.fill_(1.0) if ("norm" in name.lower()) else param.data.zero_()
            continue
        t = layer_tensors[full]
        new_dtype = param.dtype if any(p in f".{name}." or name == p
                                       for p in _PRESERVE_PATTERNS) else t.dtype
        param.data = t.to(device=device, dtype=new_dtype, non_blocking=False)

    for name, buf in layer.named_buffers():
        full = pfx + name
        if full not in layer_tensors:
            buf.data.zero_()
            continue
        buf.data = layer_tensors[full].to(device=device, dtype=buf.dtype, non_blocking=False)

    if _has_npu() and "npu" in str(device):
        try:
            torch.npu.synchronize()
        except RuntimeError as e:
            print(f"  [WARN] npu.synchronize after build failed: {str(e)[:80]}")

    layer.train(False)   # .eval() without the literal string
    return layer


@register_backend("pangu")
class PanguBackend(ModelBackend):
    def __init__(self, model_dir: str):
        super().__init__(model_dir)
        self._cfg = None
        self._wmap = None

    # ── topology ──────────────────────────────────────────────────────────────
    def skip_patterns(self) -> List[str]:
        return UNIVERSAL_SKIP_PATTERNS + _PANGU_SKIP_PATTERNS

    def mtp_skip_patterns(self) -> List[str]:
        """
        MTP prefixes for methods that explicitly preserve MTP as BF16.

        This is intentionally separate from ``skip_patterns`` so the original
        language-only JointFix path keeps its historical finalize behaviour.
        """
        depth = int(self._config().get("num_hidden_layers", 0))
        n_mtp = int(self._config().get("num_nextn_predict_layers", 0))
        return [f"model.layers.{idx}." for idx in range(depth, depth + n_mtp)]

    def layer_specs(self) -> List[LayerSpec]:
        from jointfix.backends._pangu_torch_calib import build_pangu_torch_layer_specs

        specs = []
        for ps in build_pangu_torch_layer_specs(self.model_dir):
            specs.append(LayerSpec(
                layer_idx=ps.layer_idx,
                is_moe=ps.is_moe,
                hidden_size=ps.hidden_size,
                extra={
                    "attention_type": ps.attention_type,
                    "is_dsa": ps.is_dsa,
                    "use_mhc": ps.use_mhc,
                    "use_mome": ps.use_mome,
                    "mhc_num_stream": ps.mhc_num_stream,
                    "has_block_post_layernorm": ps.has_block_post_layernorm,
                    "_pangu_spec": ps,
                },
            ))
        return specs

    # ── config + shard index (cached) ─────────────────────────────────────────
    def config(self) -> dict:
        return self._config()

    def _config(self) -> dict:
        if self._cfg is None:
            import json
            import os
            with open(os.path.join(self.model_dir, "config.json")) as f:
                self._cfg = json.load(f)
        return self._cfg

    def weight_map(self) -> Dict[str, str]:
        return self._weight_index()

    def _weight_index(self) -> Dict[str, str]:
        """
        tensor-name -> shard-filename, from model.safetensors.index.json
        (or the single-file model.safetensors).
        """
        if self._wmap is not None:
            return self._wmap
        import json
        import os
        idx_path = os.path.join(self.model_dir, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                self._wmap = json.load(f)["weight_map"]
        else:
            from safetensors import safe_open
            single = "model.safetensors"
            wmap = {}
            with safe_open(os.path.join(self.model_dir, single), framework="pt") as f:
                for k in f.keys():
                    wmap[k] = single
            self._wmap = wmap
        return self._wmap

    # ── weights I/O ─────────────────────────────────────────────────────────
    def load_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        import os
        from safetensors import safe_open

        idx = self._weight_index()
        pfx = f"model.layers.{layer_idx}."
        by_shard = defaultdict(list)
        for name, shard in idx.items():
            if name.startswith(pfx):
                by_shard[shard].append(name)
        out: Dict[str, torch.Tensor] = {}
        for shard, keys in by_shard.items():
            with safe_open(os.path.join(self.model_dir, shard), framework="pt") as f:
                for k in keys:
                    out[k] = f.get_tensor(k)
        return out

    def save_quantized(self, out_dir, layer_idx, tensors, quant_meta) -> None:
        """
        v1: one safetensors per layer (sufficient for the weight-level
        equivalence test). Deployment-format reassembly (compressed-tensors index +
        config.json) is a later finalization step.
        """
        import os
        from pathlib import Path

        from jointfix.core.checkpoint import atomic_save

        os.makedirs(out_dir, exist_ok=True)
        atomic_save(tensors, Path(out_dir) / f"layer_{layer_idx:04d}.safetensors")

    # ── forward ───────────────────────────────────────────────────────────────
    def build_layer(self, spec: LayerSpec, weights, device) -> nn.Module:
        return _meta_build_layer(weights, self._config(), spec.extra["_pangu_spec"], device)

    def embed(self, input_ids: torch.Tensor, device) -> torch.Tensor:
        """
        Embed token ids -> initial hidden states.

        NPU: float32 row-gather `embed_w.float()[ids]` (the model only runs on
        NPU, which is lenient about float32 input @ bf16 weights; every norm/MHC
        op upcasts to float32 internally).

        CPU: cast to bf16 — CPU's F.linear is strict (input must match the bf16
        params). CPU is for quick pipeline checks only.
        """
        import os
        from safetensors import safe_open

        idx = self._weight_index()
        shard = idx["model.embed_tokens.weight"]
        with safe_open(os.path.join(self.model_dir, shard), framework="pt") as f:
            embed_w = f.get_tensor("model.embed_tokens.weight").float()
        hidden = embed_w[input_ids]
        dev = device if isinstance(device, torch.device) else torch.device(device)
        if dev.type == "cpu":
            hidden = hidden.to(torch.bfloat16)
        return hidden.to(device)

    def install_stat_hooks(self, layer: nn.Module, layer_idx: int,
                           collectors: dict, stats_config) -> list:
        """
        Register per-site activation-stat pre-hooks (minus the NaN-trace
        diagnostic hooks).

        Collection-point key convention (the contract with process_layer):
            {pfx}.q_b_in      {pfx}.o_in       {pfx}.mlp_in
            {pfx}.exp{eid}_down_in   {pfx}.shared_down_in   {pfx}.dense_down_in
        where pfx = "model.layers.{layer_idx}". The per-expert hook fires only for
        experts that actually receive routed tokens, so empty experts never create
        a collector (the search/quantize side handles that via RTN fallback).
        """
        from jointfix.core.stats import AccumActStats

        pfx = f"model.layers.{layer_idx}"
        handles = []

        def _update(key, x, token_is_text=None, priority=None):
            if token_is_text is not None:
                rows = x.reshape(-1, x.shape[-1]).shape[0]
                token_is_text = token_is_text.reshape(-1)
                if rows % token_is_text.numel() != 0:
                    raise ValueError(
                        f"collector {key}: activation rows {rows} are not divisible "
                        f"by modality rows {token_is_text.numel()}"
                    )
                repeat = rows // token_is_text.numel()
                token_is_text = token_is_text.repeat_interleave(repeat)
                if priority is not None:
                    priority = priority.reshape(-1).repeat_interleave(repeat)
            if key not in collectors:
                collectors[key] = AccumActStats(x.shape[-1], stats_config)
            if token_is_text is None and priority is None:
                # Preserve the original language-only collector call exactly.
                collectors[key].update(x)
            else:
                collectors[key].update(
                    x, token_is_text=token_is_text, priority=priority)

        def _pre(key):
            def hook(_mod, inputs):
                if stats_config.modality_aware:
                    _update(
                        key, inputs[0],
                        token_is_text=getattr(_mod, "_jointfix_token_is_text", None),
                        priority=getattr(_mod, "_jointfix_priority", None),
                    )
                else:
                    _update(key, inputs[0])
            return hook

        handles.append(layer.self_attn.q_b_proj.register_forward_pre_hook(_pre(f"{pfx}.q_b_in")))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(_pre(f"{pfx}.o_in")))

        if hasattr(layer.mlp, "experts"):
            if stats_config.modality_aware:
                from jointfix.core.modality import MoERouteStats
                route_key = f"{pfx}.moe_routes"
                route_stats = MoERouteStats(
                    len(layer.mlp.experts), layer.mlp.routed_scaling_factor)
                collectors[route_key] = route_stats
                layer.mlp._jointfix_route_collector = route_stats
            # hook the MoE module itself (router uses F.linear, bypassing gate hooks)
            handles.append(layer.mlp.register_forward_pre_hook(_pre(f"{pfx}.mlp_in")))
            for eid, expert in enumerate(layer.mlp.experts):
                handles.append(expert.down_proj.register_forward_pre_hook(
                    _pre(f"{pfx}.exp{eid}_down_in")))
            if hasattr(layer.mlp, "shared_experts"):
                handles.append(layer.mlp.shared_experts.down_proj.register_forward_pre_hook(
                    _pre(f"{pfx}.shared_down_in")))
        else:
            handles.append(layer.mlp.register_forward_pre_hook(_pre(f"{pfx}.mlp_in")))
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(_pre(f"{pfx}.dense_down_in")))

        return handles

    def set_calibration_context(self, layer: nn.Module,
                                token_is_text: torch.Tensor | None) -> None:
        if token_is_text is None or not hasattr(layer.mlp, "experts"):
            return
        layer.mlp._jointfix_token_is_text = token_is_text
        # MLP input statistics are collected at the MoE module boundary.
        layer.mlp._jointfix_priority = torch.ones_like(
            token_is_text, dtype=torch.float32)
