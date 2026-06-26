# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture(autouse=True)
def _mock_vllm_deps(monkeypatch):
    """Mock all vllm dependencies needed by the import chain."""
    distributed_module = types.ModuleType("vllm.distributed")
    distributed_module.get_ep_group = lambda: SimpleNamespace(
        world_size=2, rank=0, rank_in_group=0,
        all_gather=lambda tensor, dim=0: torch.cat([tensor, tensor], dim=dim),
        reduce_scatter=lambda tensor, dim=0: tensor,
    )
    distributed_module.get_dp_group = lambda: SimpleNamespace(
        world_size=1,
        all_gather=lambda tensor, dim=0: tensor,
        reduce_scatter=lambda tensor, dim=0: tensor,
    )
    distributed_module.get_tp_group = lambda: SimpleNamespace(all_reduce=lambda x: x)
    distributed_module.get_tensor_model_parallel_world_size = lambda: 1
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed_module)

    forward_context_module = types.ModuleType("vllm.forward_context")
    forward_context_module.get_forward_context = lambda: SimpleNamespace(attn_metadata=None)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward_context_module)

    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = lambda name: MagicMock()
    logger_module.logger = MagicMock()
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)

    platforms_module = types.ModuleType("vllm.platforms")
    platforms_module.current_platform = SimpleNamespace(device_type="cpu")
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_module)

    math_utils_module = types.ModuleType("vllm.utils.math_utils")
    math_utils_module.cdiv = lambda a, b: (a + b - 1) // b
    monkeypatch.setitem(sys.modules, "vllm.utils.math_utils", math_utils_module)

    torch_npu = MagicMock()
    torch_npu.npu_moe_init_routing_v2 = MagicMock()
    torch_npu.npu_dynamic_quant = MagicMock()
    torch_npu.npu_moe_re_routing = MagicMock()
    torch_npu.npu_moe_finalize_routing = MagicMock()
    torch_npu.npu_grouped_matmul_finalize_routing = MagicMock()
    torch_npu.npu_moe_distribute_dispatch_v2 = MagicMock()
    torch_npu.npu_moe_distribute_combine_v2 = MagicMock()
    torch_npu.npu = SimpleNamespace(get_device_name=lambda _: "Ascend910C")
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)


def _make_dummy_method_cls():
    from omni_npu.layers.fused_moe.fused_moe_method_base import NPUFusedMoEMethodBase

    class _DummyMethod(NPUFusedMoEMethodBase):
        def apply_experts(self, layer, prepare_permute_result, activation="silu"):
            return prepare_permute_result.hidden_states_sorted_by_experts

    return _DummyMethod


class _DummyStrategy:
    def __init__(self):
        self.called = False

    def prepare_permute(self, layer, x, topk_ids):
        self.called = True
        return SimpleNamespace(
            hidden_states_sorted_by_experts=x,
            expert_tokens=torch.tensor([x.shape[0]], dtype=torch.int64),
            dynamic_scale=None,
        )

    def prepare_finalize_metadata(self, layer, topk_weights, result):
        self.called = True
        return {"weights": topk_weights}

    def unpermute_finalize(
        self, layer, hidden_states, topk_ids, topk_weights, result,
        finalize_params=None, finalize_metadata=None,
    ):
        self.called = True
        self.finalize_params = finalize_params
        self.finalize_metadata = finalize_metadata
        return hidden_states + 1


@pytest.mark.unit
def test_method_base_delegates_prepare_and_finalize():
    method = _make_dummy_method_cls()()
    strategy = _DummyStrategy()
    x = torch.ones(2, 3)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    topk_weights = torch.ones(2, 1, dtype=torch.float32)

    prepare_result = method.apply_prepare_permute(strategy, SimpleNamespace(), x, topk_ids)
    metadata = method.prepare_finalize_metadata(
        strategy, SimpleNamespace(), topk_weights, prepare_result
    )
    out = method.apply_unpermute_finalize(
        strategy,
        SimpleNamespace(),
        prepare_result.hidden_states_sorted_by_experts,
        topk_ids,
        topk_weights,
        prepare_result,
        finalize_metadata=metadata,
    )

    assert strategy.called is True
    assert torch.equal(out, torch.full((2, 3), 2.0))
    assert strategy.finalize_metadata is metadata


@pytest.mark.unit
def test_make_communication_strategy_selector_sets_selector(monkeypatch):
    from omni_npu.layers.fused_moe import fused_moe_method_base as base_module

    method = _make_dummy_method_cls()()
    fake_selector = SimpleNamespace(select_communication_strategy=lambda n: ("agrs", object()))
    monkeypatch.setattr(base_module, "CommunicationStrategySelector", lambda _moe: fake_selector)

    method.make_communication_strategy_selector(SimpleNamespace())
    strategy, _ = method.select_communication_strategy(4)

    assert method.communication_strategy_selector is fake_selector
    assert strategy == "agrs"
