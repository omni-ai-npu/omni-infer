# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""
QuantMethod — the METHOD axis.

A method encapsulates the transform + quantizer: given a built layer (+ optional
activation stats + the backend), produce the quantized tensors for that layer.

Adding a method = adding one subclass here + registering it; nothing in core/ or
backends/ changes. Methods contribute their own CLI args, so the core CLI never
grows a method-specific flag.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from jointfix.backends.base import LayerSpec, ModelBackend


class QuantMethod(ABC):
    """A quantization method (jointfix, quarot, spinquant, awq, …)."""

    #: registry name, e.g. "jointfix"
    name: str = "base"

    #: Activation-aware methods (smooth/awq/gptq) need a calibration forward;
    #: weight-only rotation methods (quarot/spinquant) can set this False and the
    #: runner skips stat collection for them.
    needs_activations: bool = True

    @abstractmethod
    def add_cli_args(self, parser) -> None:
        """
        Register this method's argument group (e.g. smooth's
        --objective / --write-quant / --num-iterations). Keeps core CLI clean.
        """

    def stats_config(self):
        """
        StatsConfig governing the calibration-forward stat collection. Default
        is fine for most methods; jointfix overrides from its JointSearchConfig.
        """
        from jointfix.core.stats import StatsConfig
        return StatsConfig()

    def configure(self, args) -> None:
        """
        Apply the parsed CLI args (from add_cli_args) onto the method's config.
        Default no-op; methods with a config override this.
        """

    def dump_traces(self, out_dir) -> None:
        """
        Persist any per-layer search diagnostics the method accumulated (e.g.
        jointfix's (a,b) traces -> joint_search_traces.json). The runner calls this
        after the layer loop. Default no-op.
        """

    @abstractmethod
    def process_layer(
        self,
        layer_tensors: Dict[str, "object"],
        collectors: dict,
        spec: LayerSpec,
        backend: ModelBackend,
        device,
        devices=None,
    ) -> Dict[str, "object"]:
        """
        Transform + quantize one layer.

        Args:
            layer_tensors: raw {name: weight} for this layer (from
                           backend.load_layer_weights) — the method operates on
                           these, not on the built nn.Module (that was only needed
                           for the calibration forward).
            collectors:    per-site activation stats keyed by the backend's
                           convention (empty for weight-only methods).
            spec:          the LayerSpec.
            backend:       for model config / skip patterns / topology.
            device:        search / compute device.

        Returns:
            {tensor_name: tensor} ready for backend.save_quantized
            (int8 weights, bf16 scales, and any pass-through BF16 tensors).
        """
