# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace


def moe_layer(**kwargs):
    layer = SimpleNamespace(**kwargs)
    if getattr(layer, "routed_experts", None) is None:
        layer.routed_experts = layer
    if getattr(layer, "_shared_experts", None) is None:
        shared = getattr(layer, "shared_experts", None)
        layer._shared_experts = (
            None if shared is None else SimpleNamespace(_layer=shared)
        )
    moe_config = getattr(layer, "moe_config", None)
    if moe_config is None:
        layer.moe_config = SimpleNamespace(
            moe_parallel_config=SimpleNamespace(use_ep=True),
            is_sequence_parallel=False,
            num_experts=4,
        )
    elif getattr(moe_config, "moe_parallel_config", None) is None:
        moe_config.moe_parallel_config = SimpleNamespace(use_ep=True)
    return layer
