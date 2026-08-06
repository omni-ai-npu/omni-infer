# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
from types import SimpleNamespace

from omni_npu.model_config.config_loader.features import apply_eager_mode_config


pytestmark = pytest.mark.unit


def _make_config(graph_mode="eager_mode", moe_multi_stream_tune=False):
    return SimpleNamespace(
        task_config=SimpleNamespace(
            graph_mode=graph_mode,
        ),
        operator_opt_config=SimpleNamespace(
            moe_multi_stream_tune=moe_multi_stream_tune,
            enable_scmoe_multi_stream=True,
            enable_super_kernel=True,
            enable_prefetch=True,
            expert_gate_up_prefetch=50,
            expert_down_prefetch=28,
            attn_prefetch=96,
            dense_mlp_prefetch=56,
            lm_head_prefetch=135,
            shared_expert_gate_up_prefetch=28,
            shared_expert_down_prefetch=14,
        ),
    )


def test_eager_mode_does_not_disable_moe_multi_stream_tune():
    """Eager mode should NOT touch moe_multi_stream_tune."""
    cfg = _make_config(graph_mode="eager_mode", moe_multi_stream_tune=True)
    apply_eager_mode_config(cfg)
    assert cfg.operator_opt_config.moe_multi_stream_tune is True


def test_eager_mode_disables_other_optimizations():
    """Eager mode should disable prefetch, super_kernel, etc."""
    cfg = _make_config(graph_mode="eager_mode")
    apply_eager_mode_config(cfg)
    assert cfg.operator_opt_config.enable_prefetch is False
    assert cfg.operator_opt_config.enable_super_kernel is False
    assert cfg.operator_opt_config.enable_scmoe_multi_stream is False
    assert cfg.operator_opt_config.expert_gate_up_prefetch == 0


def test_non_eager_mode_does_not_touch_config():
    """Non-eager mode should not modify any config."""
    cfg = _make_config(graph_mode="acl_graph", moe_multi_stream_tune=True)
    apply_eager_mode_config(cfg)
    assert cfg.operator_opt_config.moe_multi_stream_tune is True
    assert cfg.operator_opt_config.enable_prefetch is True
