# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace


def as_moe_runner(layer, *, use_ep=None):
    """Give a test layer the attributes vLLM 0.25's MoERunner exposes.

    NPU quant methods receive the runner, not the experts: weights live on
    ``layer.routed_experts``, the shared-expert MLP on
    ``layer._shared_experts._layer`` and the EP switch on
    ``layer.moe_config.moe_parallel_config``. Test layers that own the weights
    themselves act as their own ``routed_experts``.
    """
    # object.__setattr__ keeps torch.nn.Module from registering a self-cycle.
    if getattr(layer, "routed_experts", None) is None:
        object.__setattr__(layer, "routed_experts", layer)
    if getattr(layer, "_shared_experts", None) is None:
        shared = getattr(layer, "shared_experts", None)
        object.__setattr__(
            layer,
            "_shared_experts",
            None if shared is None else SimpleNamespace(_layer=shared),
        )
    moe_config = getattr(layer, "moe_config", None)
    if moe_config is None:
        moe_config = SimpleNamespace(is_sequence_parallel=False, num_experts=4)
        object.__setattr__(layer, "moe_config", moe_config)
    if use_ep is not None or getattr(moe_config, "moe_parallel_config", None) is None:
        moe_config.moe_parallel_config = SimpleNamespace(
            use_ep=True if use_ep is None else use_ep
        )
    return layer


def moe_layer(**kwargs):
    return as_moe_runner(SimpleNamespace(**kwargs))
