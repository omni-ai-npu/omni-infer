# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
ModelBackend — the MODEL axis.

A backend encapsulates everything model-specific: how to enumerate layers, read a
layer's weights, build a forward-capable module, install stat hooks, declare the
smooth-absorption consumer map, and write quantized output.

The runner (core/runner.py) and methods (methods/*) talk ONLY to this interface —
they never import a concrete model implementation. Adding a new model = adding one
subclass here; nothing in core/ or methods/ changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn


@dataclass
class LayerSpec:
    """
    Per-layer metadata, model-agnostic core + opaque model extras.

    Generalises Pangu's `PanguTorchLayerSpec`. Model-specific fields
    (Pangu's is_dsa / use_mhc / use_mome, sliding_window, …) ride in `extra`
    so the universal interface stays small.
    """
    layer_idx: int
    is_moe: bool
    hidden_size: int
    extra: dict = field(default_factory=dict)


class ModelBackend(ABC):
    """Model-specific operations the quantization pipeline needs."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir

    # ── topology ────────────────────────────────────────────────────────────
    @abstractmethod
    def layer_specs(self) -> List[LayerSpec]:
        """Enumerate decoder layers (generalises build_*_layer_specs)."""

    @abstractmethod
    def skip_patterns(self) -> List[str]:
        """
        Weight-name substrings to leave in BF16, EXTENDING
        core.primitives.UNIVERSAL_SKIP_PATTERNS with model-specific entries
        (e.g. Pangu's indexer.* / mhc_module.phi).
        """

    @abstractmethod
    def config(self) -> dict:
        """
        The model's config.json as a dict (dims, head counts, expert counts).
        Consumed by model-coupled methods for absorption topology.
        """

    # ── weights I/O ─────────────────────────────────────────────────────────
    @abstractmethod
    def weight_map(self) -> Dict[str, str]:
        """tensor-name -> shard-filename (the safetensors index), for resume."""

    @abstractmethod
    def load_layer_weights(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Read all tensors for one layer from sharded safetensors."""

    @abstractmethod
    def save_quantized(self, out_dir: str, layer_idx: int,
                       tensors: Dict[str, torch.Tensor], quant_meta: dict) -> None:
        """Persist a quantized layer (int8 weights + scales + meta)."""

    # ── forward ─────────────────────────────────────────────────────────────
    @abstractmethod
    def embed(self, input_ids: torch.Tensor, device) -> torch.Tensor:
        """Token ids [n, seq] -> initial hidden states for the calibration forward."""

    @abstractmethod
    def build_layer(self, spec: LayerSpec, weights: Dict[str, torch.Tensor],
                    device) -> nn.Module:
        """
        Build a forward-capable decoder layer (generalises _build_layer_module).

        This is the decoupling boundary: the runner calls this instead of hardcoding a
        concrete decoder class. Implementations should prefer a meta-device build
        (do NOT wrap this in a ThreadPool — that was NET NEGATIVE on MoE).
        """

    @abstractmethod
    def install_stat_hooks(self, layer: nn.Module, layer_idx: int,
                           collectors: dict, stats_config) -> list:
        """
        Register forward pre-hooks that populate `collectors` with this layer's
        per-site activation stats (lazily creating AccumActStats(in_features,
        stats_config)), keyed by a model-specific convention. Returns the hook
        handles for removal after the calibration forward.

        The key convention is the contract with the method's process_layer — both
        are model-coupled in v1.
        """

    def set_calibration_context(self, layer: nn.Module,
                                token_is_text: torch.Tensor | None) -> None:
        """
        Attach per-sample calibration metadata before a layer forward.

        Backends without modality-aware routing can keep this no-op default.
        """
        del layer, token_is_text

    # NOTE: there is deliberately NO downstream_consumers() / absorb() here.
    # Smooth-scale absorption lives in the METHOD (JointFixMethod.process_layer)
    # in v1 — it is Pangu-coupled and marked as a seam to extract into the
    # backend when a second model needs jointfix.
