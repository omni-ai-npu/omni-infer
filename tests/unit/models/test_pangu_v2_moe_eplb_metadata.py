# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""EPLB metadata update contract of OpenPanguV2ForCausalLM.

A rebalance may move experts between devices but must never change how many
physical experts a single device holds: the expert weight buffers were sized
for that count at build time. The check has to survive `python -O`, so it is a
raise rather than an assert.
"""

from types import SimpleNamespace

import pytest

from omni_npu.v1.models.pangu import pangu_v2_moe as model_mod


pytestmark = pytest.mark.unit


def _bare_model(num_local_physical_experts, num_logical_experts=8):
    model = model_mod.OpenPanguV2ForCausalLM.__new__(
        model_mod.OpenPanguV2ForCausalLM
    )
    model.num_local_physical_experts = num_local_physical_experts
    model.num_logical_experts = num_logical_experts
    model.model = SimpleNamespace(layers=[])
    return model


def test_update_metadata_rejects_a_changed_per_device_count():
    """A different per-device count would desync the expert weight buffers."""
    model = _bare_model(num_local_physical_experts=4)

    with pytest.raises(ValueError, match="per-device expert count"):
        model.update_physical_experts_metadata(16, 8)


def test_update_metadata_recomputes_redundant_experts():
    """Matching counts update the physical/redundant totals in place."""
    model = _bare_model(num_local_physical_experts=4, num_logical_experts=8)

    model.update_physical_experts_metadata(12, 4)

    assert model.num_physical_experts == 12
    assert model.num_local_physical_experts == 4
    assert model.num_redundant_experts == 4
