# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib
import sys
import types
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CUDAGraphMode

if "vllm.logger" not in sys.modules or not hasattr(sys.modules["vllm.logger"], "init_logger"):
    _logger_module = types.ModuleType("vllm.logger")
    _logger_module.init_logger = lambda _name: MagicMock()
    sys.modules["vllm.logger"] = _logger_module


pytestmark = pytest.mark.unit


@pytest.fixture
def prepare_module(monkeypatch):
    torch_npu = MagicMock()
    torch_npu.npu_moe_init_routing_v2 = MagicMock()
    torch_npu.npu_dynamic_quant = MagicMock()
    torch_npu.npu_moe_re_routing = MagicMock()
    torch_npu.npu_moe_finalize_routing = MagicMock()
    torch_npu.npu_grouped_matmul_finalize_routing = MagicMock()
    torch_npu.npu_moe_distribute_dispatch_v2 = MagicMock()
    torch_npu.npu_moe_distribute_combine_v2 = MagicMock()
    torch_npu.npu = SimpleNamespace(get_device_name=lambda _: "Ascend910C")

    class DummyBackend:
        def get_hccl_comm_name(self, rank_in_group):
            return f"hccl_{rank_in_group}"

    class DummyDeviceGroup:
        def _get_backend(self, device):
            return DummyBackend()

    ep_group = SimpleNamespace(
        world_size=2,
        rank=1,
        rank_in_group=1,
        device_group=DummyDeviceGroup(),
        all_gather=lambda tensor, dim=0: torch.cat([tensor, tensor], dim=dim),
        reduce_scatter=lambda tensor, dim=0: tensor,
    )
    context_holder = SimpleNamespace(attn_metadata=None)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    distributed_module = types.ModuleType("vllm.distributed")
    distributed_module.get_ep_group = lambda: ep_group
    distributed_module.get_dp_group = lambda: SimpleNamespace(
        world_size=1,
        all_gather=lambda tensor, dim=0: tensor,
        reduce_scatter=lambda tensor, dim=0: tensor,
    )
    distributed_module.get_tp_group = lambda: SimpleNamespace(all_reduce=lambda x: x)
    distributed_module.get_tensor_model_parallel_world_size = lambda: 1
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed_module)

    forward_context_module = types.ModuleType("vllm.forward_context")
    forward_context_module.get_forward_context = lambda: context_holder
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
    logger_module = types.ModuleType("vllm.logger")
    logger_module.logger = MagicMock()
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)
    vllm_utils = sys.modules.get("vllm.utils")
    if vllm_utils is not None and not hasattr(vllm_utils, "random_uuid"):
        # Compat: older/newer vllm variants may not expose random_uuid.
        vllm_utils.random_uuid = lambda: str(uuid.uuid4())
    parsers_module = types.ModuleType("omni_npu.v1.parsers")
    parsers_module.register_lazy_parsers = lambda: None
    monkeypatch.setitem(sys.modules, "omni_npu.v1.parsers", parsers_module)

    monkeypatch.setattr(
        torch.distributed,
        "all_to_all_single",
        lambda output, input, *args, **kwargs: output.copy_(input),
        raising=False,
    )

    module_name = "omni_npu.layers.fused_moe.prepare_permute_unpermute_finalize"
    sys.modules.pop(module_name, None)
    logger_mod = importlib.import_module("vllm.logger")
    monkeypatch.setattr(logger_mod, "init_logger", lambda _name: MagicMock(), raising=False)
    module = importlib.import_module(module_name)
    importlib.reload(module)

    stubs = SimpleNamespace(
        torch_npu=torch_npu,
        ep_group=ep_group,
        context_holder=context_holder,
    )
    return module, stubs


def test_all2all_prepare_permute_no_quant(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        ep_size=stubs.ep_group.world_size,
        quant_method=SimpleNamespace(
            moe_quant_config=None,
            num_of_redundant_experts=0,
        ),
        quant_config=None,
        w13_weight=torch.zeros(2, 1, 1),
    )
    expanded_x = torch.arange(6, dtype=torch.float32).view(2, 3)
    expanded_row_idx = torch.tensor([0, 1], dtype=torch.int32)
    tokens_per_expert = torch.tensor([1, 0, 1, 0], dtype=torch.int32)
    stubs.torch_npu.npu_moe_init_routing_v2.return_value = (
        expanded_x,
        expanded_row_idx,
        tokens_per_expert,
        None,
    )
    sorted_states = torch.full((2, 3), 2.0)
    gathered_idxs_unsort = torch.tensor([1, 0], dtype=torch.int32)
    tokens_per_local_expert = torch.tensor([1, 1], dtype=torch.int32)
    stubs.torch_npu.npu_moe_re_routing.return_value = (
        sorted_states,
        None,
        gathered_idxs_unsort,
        tokens_per_local_expert,
    )

    handler = module.All2AllPrepPmtAndUnpmtFinal(layer)
    result = handler.prepare_permute(
        layer=layer,
        x=torch.ones(2, 3),
        topk_ids=torch.zeros(2, 2, dtype=torch.int32),
    )

    assert result.hidden_states_sorted_by_experts is sorted_states
    assert result.dynamic_scale is None
    assert result.input_splits == [1, 1]
    assert result.output_splits == [1, 1]
    assert result.expert_tokens.equal(tokens_per_local_expert)
    assert stubs.torch_npu.npu_moe_init_routing_v2.call_args.kwargs["quant_mode"] == -1


def test_agrs_prepare_permute_no_quant_all_gathers_before_routing(prepare_module):
    """AGRS non-quant path all_gather(x) and all_gather(topk_ids) before npu_moe_init_routing_v2."""
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        ep_size=stubs.ep_group.world_size,
        quant_config=None,
        quant_method=SimpleNamespace(
            moe_quant_config=None,
            num_of_redundant_experts=0,
        ),
        w13_weight=torch.zeros(2, 1, 1),
    )
    x_local = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    topk_local = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    expanded_x = torch.ones(4, 3)
    expanded_row_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    expert_tokens = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
    stubs.torch_npu.npu_moe_init_routing_v2.return_value = (
        expanded_x,
        expanded_row_idx,
        expert_tokens,
        None,
    )

    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    result = handler.prepare_permute(layer=layer, x=x_local, topk_ids=topk_local)

    expected_gathered_x = stubs.ep_group.all_gather(x_local, dim=0)
    expected_gathered_topk = stubs.ep_group.all_gather(topk_local, dim=0)
    args, kwargs = stubs.torch_npu.npu_moe_init_routing_v2.call_args
    assert torch.equal(args[0], expected_gathered_x)
    assert torch.equal(args[1], expected_gathered_topk)
    assert kwargs["active_expert_range"] == [2, 4]
    assert kwargs["row_idx_type"] == 0
    assert kwargs["quant_mode"] == -1
    assert result.dynamic_scale is None
    assert torch.equal(result.hidden_states_sorted_by_experts, expanded_x)
    assert result.dtype == x_local.dtype


def test_agrs_prepare_permute_with_quant(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        ep_size=stubs.ep_group.world_size,
        quant_config=object(),
        quant_method=SimpleNamespace(moe_quant_config=None),
        w13_weight=torch.zeros(2, 1, 1),  # shape: (local_num_experts, ...)
    )
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    x_int8 = x.to(torch.int8)
    x_scale = torch.tensor([0.1, 0.1], dtype=torch.float32)

    stubs.torch_npu.npu_dynamic_quant.return_value = (x_int8, x_scale)

    expanded_x = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    expanded_row_idx = torch.tensor([0, 1], dtype=torch.int32)
    expert_tokens = torch.tensor([1, 1, 0, 0], dtype=torch.int32)
    expanded_scale = torch.tensor([0.1, 0.1], dtype=torch.float32)

    stubs.torch_npu.npu_moe_init_routing_v2.return_value = (
        expanded_x,
        expanded_row_idx,
        expert_tokens,
        expanded_scale,
    )
    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    result = handler.prepare_permute(
        layer=layer,
        x=torch.ones(2, 3),
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
    )

    stubs.torch_npu.npu_moe_init_routing_v2.assert_called_once()
    assert torch.equal(result.hidden_states_sorted_by_experts, expanded_x)
    assert torch.equal(result.expert_tokens, expert_tokens)
    assert result.dtype == x.dtype


def test_agrs_prepare_permute_hifloat8_uses_cast_without_scale(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        ep_size=stubs.ep_group.world_size,
        quant_config=object(),
        quant_method=SimpleNamespace(
            moe_quant_config=SimpleNamespace(
                use_hifloat8_w8a8=True,
                use_mxfp8_w8a8=False,
            ),
        ),
        w13_weight=torch.zeros(2, 1, 1),
    )
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    x_hif8 = x.to(torch.int8)
    expanded_x = torch.ones(2, 2, dtype=torch.float32)
    expanded_row_idx = torch.tensor([0, 1], dtype=torch.int32)
    expert_tokens = torch.tensor([1, 1], dtype=torch.int32)
    dirty_dynamic_scale = torch.ones(2, dtype=torch.float32)

    stubs.torch_npu.hifloat8 = object()
    stubs.torch_npu.npu_dtype_cast.return_value = x_hif8
    stubs.torch_npu.npu_moe_init_routing_v2.return_value = (
        expanded_x,
        expanded_row_idx,
        expert_tokens,
        dirty_dynamic_scale,
    )

    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    result = handler.prepare_permute(layer=layer, x=x, topk_ids=topk_ids)

    stubs.torch_npu.npu_dtype_cast.assert_called_once()
    cast_args = stubs.torch_npu.npu_dtype_cast.call_args.args
    assert torch.equal(cast_args[0], x)
    assert cast_args[1] is stubs.torch_npu.hifloat8
    _, kwargs = stubs.torch_npu.npu_moe_init_routing_v2.call_args
    assert kwargs["scale"] is None
    assert result.dynamic_scale is None
    assert torch.equal(result.hidden_states_sorted_by_experts, expanded_x)


def test_agrs_prepare_permute_mxfp8_routes_bf16_then_quants_expanded(prepare_module):
    """mxfp8 keeps x in bf16 through npu_moe_init_routing_v2 and runs
    npu_dynamic_mx_quant on `expanded_x` after routing (workaround for the
    buggy mxfp8 scale path through routing)."""
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        ep_size=stubs.ep_group.world_size,
        quant_config=object(),
        quant_method=SimpleNamespace(
            moe_quant_config=SimpleNamespace(
                use_hifloat8_w8a8=False,
                use_mxfp8_w8a8=True,
            ),
        ),
        w13_weight=torch.zeros(2, 1, 1),
    )
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    expanded_x_bf16 = torch.full((2, 2), 7.0, dtype=torch.float32)
    expanded_row_idx = torch.tensor([0, 1], dtype=torch.int32)
    expert_tokens = torch.tensor([1, 1], dtype=torch.int32)
    dirty_dynamic_scale = torch.ones(2, dtype=torch.float32)

    expanded_x_fp8 = torch.zeros(2, 2, dtype=torch.int8)
    expanded_scale = torch.full((2, 2 // 32 + 1,), 3, dtype=torch.uint8)

    stubs.torch_npu.npu_moe_init_routing_v2.return_value = (
        expanded_x_bf16,
        expanded_row_idx,
        expert_tokens,
        dirty_dynamic_scale,
    )
    stubs.torch_npu.npu_dynamic_mx_quant.return_value = (expanded_x_fp8, expanded_scale)

    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    result = handler.prepare_permute(layer=layer, x=x, topk_ids=topk_ids)

    # Routing got the bf16 x (gathered via the EP all_gather stub) with scale=None
    _, kwargs = stubs.torch_npu.npu_moe_init_routing_v2.call_args
    assert kwargs["scale"] is None
    # Post-routing quant fired on expanded_x_bf16, returned expanded_x_fp8 + scale.
    stubs.torch_npu.npu_dynamic_mx_quant.assert_called_once()
    quant_args, _ = stubs.torch_npu.npu_dynamic_mx_quant.call_args
    assert torch.equal(quant_args[0], expanded_x_bf16)
    assert torch.equal(result.hidden_states_sorted_by_experts, expanded_x_fp8)
    assert torch.equal(result.dynamic_scale, expanded_scale)


def test_agrs_prepare_permute_cv_mxfp8_quants_on_side_stream(prepare_module, monkeypatch):
    """with_routed_experts_cv + mxfp8: gate_up_proj is launched on the side
    stream with a freshly mx-quantised payload, then expanded_x is quantised
    post-routing."""
    module, stubs = prepare_module
    monkeypatch.setattr(
        module.model_extra_config.operator_opt_config,
        "shared_expert_multi_stream", True, raising=False,
    )
    monkeypatch.setattr(
        module.model_extra_config.operator_opt_config,
        "shared_expert_parallel_schedule", "with_routed_experts_cv", raising=False,
    )

    class _DummyEvent:
        def __init__(self):
            self.recorded = False
            self.waits = []
        def record(self):
            self.recorded = True
        def wait(self, stream):
            self.waits.append(stream)

    class _NoopCtx:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False

    npu_ns = SimpleNamespace(
        Event=_DummyEvent,
        stream=lambda _s: _NoopCtx(),
        get_device_name=lambda _: "Ascend910C",
    )
    monkeypatch.setattr(torch, "npu", npu_ns, raising=False)
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda self, _s: None, raising=False,
    )

    shared_gate_up = torch.full((4, 16), 11.0, dtype=torch.float32)
    gate_up_proj = MagicMock(return_value=shared_gate_up)
    shared_experts = SimpleNamespace(gate_up_proj=gate_up_proj)
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        ep_size=stubs.ep_group.world_size,
        quant_config=object(),
        quant_method=SimpleNamespace(
            moe_quant_config=SimpleNamespace(
                use_hifloat8_w8a8=False,
                use_mxfp8_w8a8=True,
            ),
        ),
        w13_weight=torch.zeros(2, 1, 1),
        shared_experts=shared_experts,
    )

    x = torch.zeros(2, 8, dtype=torch.float32)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    expanded_x_bf16 = torch.zeros(2, 8, dtype=torch.float32)
    expanded_row_idx = torch.tensor([0, 1], dtype=torch.int32)
    expert_tokens = torch.tensor([1, 1], dtype=torch.int32)
    side_fp8 = torch.zeros(4, 8, dtype=torch.int8)
    side_scale = torch.zeros(4, 8 // 32 + 1, dtype=torch.uint8)
    post_fp8 = torch.ones(2, 8, dtype=torch.int8)
    post_scale = torch.ones(2, 8 // 32 + 1, dtype=torch.uint8)
    stubs.torch_npu.npu_dynamic_mx_quant.side_effect = [
        (side_fp8, side_scale),
        (post_fp8, post_scale),
    ]
    stubs.torch_npu.npu_moe_init_routing_v2.return_value = (
        expanded_x_bf16,
        expanded_row_idx,
        expert_tokens,
        torch.zeros(2),
    )

    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    result = handler.prepare_permute(layer=layer, x=x, topk_ids=topk_ids)

    # Side-stream quant + gate_up_proj launched with the mxfp8 dict payload.
    gate_up_proj.assert_called_once()
    dict_arg = gate_up_proj.call_args.args[0]
    assert isinstance(dict_arg, dict)
    assert torch.equal(dict_arg["x_mxfp8"], side_fp8)
    assert torch.equal(dict_arg["pertoken_scale"], side_scale)
    assert result.shared_expert_gate_up is shared_gate_up
    # Both mx_quants happened (side stream + post-routing).
    assert stubs.torch_npu.npu_dynamic_mx_quant.call_count == 2
    assert torch.equal(result.hidden_states_sorted_by_experts, post_fp8)


def test_agrs_prepare_permute_cv_unsupported_quant_raises(prepare_module, monkeypatch):
    """with_routed_experts_cv only supports hif8 and mxfp8; any other quant
    method must raise NotImplementedError."""
    module, stubs = prepare_module
    monkeypatch.setattr(
        module.model_extra_config.operator_opt_config,
        "shared_expert_multi_stream", True, raising=False,
    )
    monkeypatch.setattr(
        module.model_extra_config.operator_opt_config,
        "shared_expert_parallel_schedule", "with_routed_experts_cv", raising=False,
    )

    class _DummyEvent:
        def record(self):
            pass
        def wait(self, _stream):
            pass

    class _NoopCtx:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False

    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        Event=_DummyEvent,
        stream=lambda _s: _NoopCtx(),
        get_device_name=lambda _: "Ascend910C",
    ), raising=False)
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda self, _s: None, raising=False,
    )

    # moe_quant_config exists but flags both False → falls into the int8
    # branch upstream, then the cv block raises because it's neither hif8
    # nor mxfp8.
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        ep_size=stubs.ep_group.world_size,
        quant_config=object(),
        quant_method=SimpleNamespace(
            moe_quant_config=SimpleNamespace(
                use_hifloat8_w8a8=False,
                use_mxfp8_w8a8=False,
            ),
        ),
        w13_weight=torch.zeros(2, 1, 1),
        shared_experts=SimpleNamespace(gate_up_proj=MagicMock()),
    )
    x = torch.zeros(2, 4, dtype=torch.float32)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    stubs.torch_npu.npu_dynamic_quant.return_value = (
        x.to(torch.int8),
        torch.ones(2, dtype=torch.float32),
    )

    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    with pytest.raises(NotImplementedError, match="hifloat8 or"):
        handler.prepare_permute(layer=layer, x=x, topk_ids=topk_ids)


def test_agrs_prepare_permute_quant_decode_on_a2_sets_row_idx_type(prepare_module):
    module, stubs = prepare_module
    stubs.torch_npu.npu.get_device_name = lambda _: "Ascend910B"
    stubs.context_holder.attn_metadata = {0: SimpleNamespace(num_prefills=0)}
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        ep_size=stubs.ep_group.world_size,
        quant_config=object(),
        quant_method=SimpleNamespace(moe_quant_config=None),
        w13_weight=torch.zeros(2, 1, 1),
    )
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    stubs.torch_npu.npu_dynamic_quant.return_value = (
        x.to(torch.int8),
        torch.tensor([0.1, 0.2], dtype=torch.float32),
    )
    stubs.torch_npu.npu_moe_init_routing_v2.return_value = (
        torch.ones(2, 2, dtype=torch.float32),
        torch.tensor([-3, 5], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
        torch.tensor([0.1, 0.2], dtype=torch.float32),
    )

    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    result = handler.prepare_permute(
        layer=layer,
        x=x,
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
    )

    assert result.row_idx_type == 1
    assert torch.equal(result.expanded_row_idx, torch.tensor([-3, 5], dtype=torch.int32))
    assert stubs.torch_npu.npu_moe_init_routing_v2.call_args.kwargs["row_idx_type"] == 1


def test_all2all_unpermute_finalize_reorders_and_finalizes(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        quant_method=SimpleNamespace(
            moe_quant_config=None,
            num_of_redundant_experts=0,
        ),
    )
    handler = module.All2AllPrepPmtAndUnpmtFinal(layer)
    stubs.torch_npu.npu_moe_finalize_routing.return_value = torch.full((2, 2), 5.0)

    prepare_result = module.All2AllPreparePermuteResult(
        hidden_states_sorted_by_experts=torch.zeros(2, 2),
        expert_tokens=torch.tensor([1, 1], dtype=torch.int32),
        dynamic_scale=None,
        gathered_idxs_unsort=torch.tensor([1, 0], dtype=torch.int32),
        expanded_x=torch.zeros(2, 2),
        expanded_row_idx=torch.tensor([0, 1], dtype=torch.int32),
        input_splits=[1, 1],
        output_splits=[1, 1],
    )
    hidden_states = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
    topk_weights = torch.ones(2, 1, dtype=torch.float32)

    output = handler.unpermute_finalize(
        layer=layer,
        hidden_states=hidden_states,
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
        topk_weights=topk_weights,
        all2all_prepare_permute_result=prepare_result,
    )

    assert torch.equal(output, torch.full((2, 2), 5.0))
    stubs.torch_npu.npu_moe_finalize_routing.assert_called_once()
    assert (
        stubs.torch_npu.npu_moe_finalize_routing.call_args.kwargs["scales"].dtype
        == hidden_states.dtype
    )


def test_dispatch_prepare_permute_passes_quant_mode_and_mask(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        quant_config=object(),
    )
    stubs.context_holder.attn_metadata = {
        0: SimpleNamespace(decode=SimpleNamespace(mc2_mask=torch.tensor([1, 0, 1], dtype=torch.bool)))
    }
    dispatch_out = (
        torch.ones(2, 3),
        torch.ones(2),
        torch.zeros(2, dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
    )
    stubs.torch_npu.npu_moe_distribute_dispatch_v2.return_value = dispatch_out

    handler = module.DispatchCombinePrepPmtAndUnpmtFinal(layer)
    result = handler.prepare_permute(
        layer=layer,
        x=torch.ones(2, 3),
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
    )

    assert result.expert_tokens.dtype == torch.int64
    kwargs = stubs.torch_npu.npu_moe_distribute_dispatch_v2.call_args.kwargs
    assert kwargs["quant_mode"] == 2
    assert kwargs["group_ep"] == "hccl_1"
    assert torch.equal(kwargs["x_active_mask"], torch.tensor([1, 0], dtype=torch.bool))


def test_dispatch_unpermute_finalize_passes_counts_and_mask(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        quant_config=None,
    )
    stubs.context_holder.attn_metadata = SimpleNamespace(
        decode=SimpleNamespace(mc2_mask=torch.tensor([1, 1], dtype=torch.bool))
    )
    stubs.torch_npu.npu_moe_distribute_combine_v2.return_value = torch.full((2, 2), 7.0)
    handler = module.DispatchCombinePrepPmtAndUnpmtFinal(layer)
    prepare_result = module.DispatchCombinePreparePermuteResult(
        hidden_states_sorted_by_experts=torch.zeros(2, 2),
        expert_tokens=torch.tensor([1, 1], dtype=torch.int32),
        dynamic_scale=torch.ones(2),
        tp_recv_counts=torch.tensor([1, 1], dtype=torch.int32),
        ep_recv_counts=torch.tensor([1, 1], dtype=torch.int32),
        expand_idx=torch.tensor([0, 1], dtype=torch.int32),
    )

    output = handler.unpermute_finalize(
        layer=layer,
        hidden_states=torch.ones(2, 2),
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        dispatch_combine_prepare_permute_result=prepare_result,
    )

    assert torch.equal(output, torch.full((2, 2), 7.0))
    kwargs = stubs.torch_npu.npu_moe_distribute_combine_v2.call_args.kwargs
    assert kwargs["ep_send_counts"] is prepare_result.ep_recv_counts
    assert kwargs["tp_send_counts"] is prepare_result.tp_recv_counts
    assert torch.equal(kwargs["x_active_mask"], torch.tensor([1, 1], dtype=torch.bool))


def test_agrs_unpermute_finalize_quant_uses_gathered_weights_drop_pad_mode_3(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        quant_config=object(),
    )
    stubs.torch_npu.npu_moe_finalize_routing.return_value = torch.ones(2, 2, dtype=torch.float32)
    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    # all_gather concatenates local topk along dim=0; ids must match gathered weights shape.
    prepare_result = module.AGRSPreparePermuteResult(
        hidden_states_sorted_by_experts=torch.zeros(2, 2),
        expert_tokens=torch.tensor([1, 1], dtype=torch.int32),
        dynamic_scale=torch.ones(2),
        expert_range=[2, 4],
        expanded_row_idx=torch.tensor([0, 1], dtype=torch.int32),
        gathered_topk_ids=torch.tensor([[0], [3], [0], [3]], dtype=torch.int32),
        dtype=torch.float16,
    )
    topk_weights = torch.ones(2, 1, dtype=torch.float32)
    hidden_states = torch.ones(2, 2, dtype=torch.float32)

    y = handler.unpermute_finalize(
        layer=layer,
        hidden_states=hidden_states,
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
        topk_weights=topk_weights,
        agrs_prepare_permute_result=prepare_result,
    )

    assert y.dtype == torch.float16
    args, kwargs = stubs.torch_npu.npu_moe_finalize_routing.call_args
    assert torch.equal(args[0], hidden_states.unsqueeze(0))
    assert torch.equal(kwargs["scales"], stubs.ep_group.all_gather(topk_weights, dim=0).float())
    assert kwargs["expanded_src_to_dst_row"] is prepare_result.expanded_row_idx
    assert kwargs["export_for_source_row"] is prepare_result.gathered_topk_ids
    assert kwargs["drop_pad_mode"] == 3


def test_agrs_prepare_finalize_metadata_gathers_topk_weights(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        quant_config=object(),
    )
    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    prepare_result = module.AGRSPreparePermuteResult(
        hidden_states_sorted_by_experts=torch.zeros(2, 2),
        expert_tokens=torch.tensor([1, 1], dtype=torch.int32),
        dynamic_scale=torch.ones(2),
        expert_range=[2, 4],
        expanded_row_idx=torch.tensor([0, 1], dtype=torch.int32),
        gathered_topk_ids=torch.tensor([[0], [3], [0], [3]], dtype=torch.int32),
        dtype=torch.float16,
    )
    topk_weights = torch.ones(2, 1, dtype=torch.bfloat16)

    metadata = handler.prepare_finalize_metadata(
        layer=layer,
        topk_weights=topk_weights,
        agrs_prepare_permute_result=prepare_result,
    )

    assert torch.equal(
        metadata.gathered_topk_weights,
        stubs.ep_group.all_gather(topk_weights, dim=0),
    )


def test_agrs_unpermute_finalize_uses_prepared_metadata_without_regather(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        quant_config=object(),
    )
    stubs.torch_npu.npu_moe_finalize_routing.return_value = torch.ones(2, 2, dtype=torch.float32)
    all_gather_mock = MagicMock(side_effect=stubs.ep_group.all_gather)
    stubs.ep_group.all_gather = all_gather_mock
    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    prepare_result = module.AGRSPreparePermuteResult(
        hidden_states_sorted_by_experts=torch.zeros(2, 2),
        expert_tokens=torch.tensor([1, 1], dtype=torch.int32),
        dynamic_scale=torch.ones(2),
        expert_range=[2, 4],
        expanded_row_idx=torch.tensor([0, 1], dtype=torch.int32),
        gathered_topk_ids=torch.tensor([[0], [3], [0], [3]], dtype=torch.int32),
        dtype=torch.float16,
    )
    metadata = module.AGRSFinalizeMetadata(
        gathered_topk_weights=torch.ones(4, 1, dtype=torch.bfloat16),
    )

    y = handler.unpermute_finalize(
        layer=layer,
        hidden_states=torch.ones(2, 2, dtype=torch.float32),
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
        topk_weights=torch.ones(2, 1, dtype=torch.float32),
        agrs_prepare_permute_result=prepare_result,
        finalize_metadata=metadata,
    )

    assert y.dtype == torch.float16
    all_gather_mock.assert_not_called()
    scales = stubs.torch_npu.npu_moe_finalize_routing.call_args.kwargs["scales"]
    assert torch.equal(scales, metadata.gathered_topk_weights.float())


def test_agrs_unpermute_finalize_quant_row_idx_type_1_uses_grouped_finalize(
    prepare_module, monkeypatch
):
    module, stubs = prepare_module
    original_arange = torch.arange

    def _cpu_arange(*args, **kwargs):
        kwargs.pop("device", None)
        return original_arange(*args, **kwargs)

    monkeypatch.setattr(module.torch, "arange", _cpu_arange)
    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        quant_config=object(),
        dp_size=1,
        w2_weight=torch.ones(2, 3, 2, dtype=torch.int8),
        w2_weight_scale=torch.ones(2, 3, dtype=torch.bfloat16),
        w2_bias=torch.zeros(2, 3, dtype=torch.bfloat16),
    )
    stubs.torch_npu.npu_grouped_matmul_finalize_routing.return_value = torch.ones(
        2, 3, dtype=torch.bfloat16
    )
    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    handler.batch_size = 2
    prepare_result = module.AGRSPreparePermuteResult(
        hidden_states_sorted_by_experts=torch.zeros(4, 3),
        expert_tokens=torch.tensor([1, 1], dtype=torch.int32),
        dynamic_scale=torch.ones(4),
        expert_range=[2, 4],
        expanded_row_idx=torch.tensor([0, 1, 4, 5], dtype=torch.int32),
        gathered_topk_ids=torch.tensor([[2, 3], [2, 3]], dtype=torch.int32),
        dtype=torch.float16,
        row_idx_type=1,
    )
    topk_ids = torch.tensor([[2, 3], [2, 3]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.2, 0.8], [0.3, 0.7]], dtype=torch.float32)
    hidden_states = (
        torch.ones(4, 3, dtype=torch.int8),
        torch.tensor([1.0, 1.1, 1.2, 1.3], dtype=torch.float32),
    )

    y = handler.unpermute_finalize(
        layer=layer,
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        agrs_prepare_permute_result=prepare_result,
    )

    assert y.dtype == torch.float16
    kwargs = stubs.torch_npu.npu_grouped_matmul_finalize_routing.call_args.kwargs
    assert torch.allclose(kwargs["logit"], torch.tensor([0.2, 0.8, 0.3, 0.3]))
    assert torch.equal(kwargs["row_index"], torch.tensor([0, 0, 1, 1], dtype=torch.int64))
    assert kwargs["output_bs"] == prepare_result.gathered_topk_ids.shape[0]
    assert kwargs["group_list_type"] == 1
    assert kwargs["bias"] is layer.w2_bias
    assert kwargs["shared_input"] is None

def test_strategy_selector_enable_moe_agrs(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(global_num_experts=4)
    stubs.torch_npu.npu.get_device_name = lambda _: "Ascend910"
    module.get_tensor_model_parallel_world_size = lambda: 1
    module.get_dp_group = lambda: SimpleNamespace(world_size=2)
    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={0: SimpleNamespace(decode_threshold=0)}
    )
    module.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(decode_moe_dispatch_combine=False, enable_moe_agrs=True),
        parall_config=SimpleNamespace(ena_seq_parallel=False),
    )

    selector = module.CommunicationStrategySelector(layer)
    strategy, _ = selector.select_communication_strategy(num_tokens=999)

    assert strategy == "agrs"

def test_strategy_selector_lazily_constructs_selected_strategy(
    prepare_module, monkeypatch
):
    module, stubs = prepare_module
    layer = SimpleNamespace(global_num_experts=4)
    stubs.torch_npu.npu.get_device_name = lambda _: "Ascend910B"
    module.get_tensor_model_parallel_world_size = lambda: 1
    module.get_dp_group = lambda: SimpleNamespace(world_size=2)
    module.get_forward_context = lambda: SimpleNamespace(attn_metadata=None)
    module.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(decode_moe_dispatch_combine=False, enable_moe_agrs=False),
        parall_config=SimpleNamespace(ena_seq_parallel=False),
    )

    class DummyAGRS:
        def __init__(self, moe):
            self.moe = moe

    def fail_dispatch(_moe):
        raise AssertionError("dispatch_combine should be constructed lazily")

    monkeypatch.setattr(module, "AGRSPrepPmtAndUnpmtFinal", DummyAGRS)
    monkeypatch.setattr(module, "DispatchCombinePrepPmtAndUnpmtFinal", fail_dispatch)

    selector = module.CommunicationStrategySelector(layer)
    assert selector.prepare_permute_and_unpermute_finalize_dict == {}

    strategy, strategy_impl = selector.select_communication_strategy(num_tokens=999)
    assert strategy == "agrs"
    assert isinstance(strategy_impl, DummyAGRS)

    _, cached_strategy_impl = selector.select_communication_strategy(num_tokens=999)
    assert cached_strategy_impl is strategy_impl


def test_strategy_selector_a2_tp_dp_branches(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(global_num_experts=4)
    stubs.torch_npu.npu.get_device_name = lambda _: "Ascend910B"
    module.get_tensor_model_parallel_world_size = lambda: 2
    module.get_dp_group = lambda: SimpleNamespace(world_size=2)
    module.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(decode_moe_dispatch_combine=False, enable_moe_agrs=False),
        parall_config=SimpleNamespace(ena_seq_parallel=False),
    )

    selector = module.CommunicationStrategySelector(layer)

    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={0: SimpleNamespace(decode_threshold=4)},
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
    )
    strategy_small, _ = selector.select_communication_strategy(num_tokens=4)

    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={0: SimpleNamespace(decode_threshold=4)},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    strategy_large, _ = selector.select_communication_strategy(num_tokens=8)

    assert strategy_small == "agrs"
    assert strategy_large == "all2all"


def test_strategy_selector_a2_tp_only_always_agrs(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(global_num_experts=4)
    stubs.torch_npu.npu.get_device_name = lambda _: "Ascend910B"
    module.get_tensor_model_parallel_world_size = lambda: 1
    module.get_dp_group = lambda: SimpleNamespace(world_size=2)
    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={0: SimpleNamespace(decode_threshold=0)}
    )
    module.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(decode_moe_dispatch_combine=False, enable_moe_agrs=False),
        parall_config=SimpleNamespace(ena_seq_parallel=False),
    )

    selector = module.CommunicationStrategySelector(layer)
    strategy, _ = selector.select_communication_strategy(num_tokens=999)

    assert strategy == "agrs"


def test_strategy_selector_a2_dp_only_always_agrs(prepare_module):
    """A2 device, dp=1, tp=4, dispatch_combine=False → always agrs."""
    module, stubs = prepare_module
    layer = SimpleNamespace(global_num_experts=4)
    stubs.torch_npu.npu.get_device_name = lambda _: "Ascend910B"
    module.get_tensor_model_parallel_world_size = lambda: 4
    module.get_dp_group = lambda: SimpleNamespace(world_size=1)
    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={0: SimpleNamespace(decode_threshold=0)}
    )
    module.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(decode_moe_dispatch_combine=False, enable_moe_agrs=False),
        parall_config=SimpleNamespace(ena_seq_parallel=False),
    )

    selector = module.CommunicationStrategySelector(layer)
    strategy, _ = selector.select_communication_strategy(num_tokens=999)

    assert strategy == "agrs"


def test_strategy_selector_a2_dispatch_combine_enabled_small_tokens(prepare_module):
    """A2 device, dispatch_combine=True, small local tokens → dispatch_combine."""
    module, stubs = prepare_module
    layer = SimpleNamespace(global_num_experts=4)
    stubs.torch_npu.npu.get_device_name = lambda _: "Ascend910B"
    module.get_tensor_model_parallel_world_size = lambda: 4
    module.get_dp_group = lambda: SimpleNamespace(world_size=1)
    module.get_forward_context = lambda: SimpleNamespace(attn_metadata=None)
    module.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(decode_moe_dispatch_combine=True, enable_moe_agrs=False),
        parall_config=SimpleNamespace(ena_seq_parallel=False)
    )

    selector = module.CommunicationStrategySelector(layer)
    # local_num_tokens = cdiv(16, 4) = 4, threshold default 64 → dispatch_combine
    strategy, _ = selector.select_communication_strategy(num_tokens=16)

    assert strategy == "dispatch_combine"


def test_strategy_selector_a2_dispatch_combine_enabled_large_tokens(prepare_module):
    """A2 device, dispatch_combine=True, large local tokens → all2all."""
    module, stubs = prepare_module
    layer = SimpleNamespace(global_num_experts=4)
    stubs.torch_npu.npu.get_device_name = lambda _: "Ascend910B"
    module.get_tensor_model_parallel_world_size = lambda: 2
    module.get_dp_group = lambda: SimpleNamespace(world_size=1)
    module.get_forward_context = lambda: SimpleNamespace(attn_metadata=None)
    module.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(decode_moe_dispatch_combine=True, enable_moe_agrs=False),
        parall_config=SimpleNamespace(ena_seq_parallel=False)
    )

    selector = module.CommunicationStrategySelector(layer)
    # local_num_tokens = cdiv(200, 2) = 100, threshold 64 → all2all
    strategy, _ = selector.select_communication_strategy(num_tokens=200)

    assert strategy == "all2all"


def test_strategy_selector_non_a2_dp1_tp_gt1_small_tokens(prepare_module):
    """Non-A2, dp=1, tp=4, small local tokens → dispatch_combine."""
    module, _ = prepare_module
    layer = SimpleNamespace(global_num_experts=4)
    module.get_tensor_model_parallel_world_size = lambda: 4
    module.get_dp_group = lambda: SimpleNamespace(world_size=1)
    module.get_forward_context = lambda: SimpleNamespace(attn_metadata=None)

    selector = module.CommunicationStrategySelector(layer)
    # local_num_tokens = cdiv(32, 4) = 8, threshold 64 → dispatch_combine
    strategy, _ = selector.select_communication_strategy(num_tokens=32)

    assert strategy == "dispatch_combine"


def test_strategy_selector_non_a2_dp1_tp_gt1_large_tokens(prepare_module):
    """Non-A2, dp=1, tp=2, large local tokens → all2all."""
    module, _ = prepare_module
    layer = SimpleNamespace(global_num_experts=4)
    module.get_tensor_model_parallel_world_size = lambda: 2
    module.get_dp_group = lambda: SimpleNamespace(world_size=1)
    module.get_forward_context = lambda: SimpleNamespace(attn_metadata=None)

    selector = module.CommunicationStrategySelector(layer)
    # local_num_tokens = cdiv(200, 2) = 100, threshold 64 → all2all
    strategy, _ = selector.select_communication_strategy(num_tokens=200)

    assert strategy == "all2all"


def test_strategy_selector_non_a2_dp_gt1_tp1_small_tokens(prepare_module):
    """Non-A2, dp>1, tp=1 → dispatch_combine for small tokens."""
    module, _ = prepare_module
    layer = SimpleNamespace(global_num_experts=4)
    module.get_tensor_model_parallel_world_size = lambda: 1
    module.get_dp_group = lambda: SimpleNamespace(world_size=4)
    module.get_forward_context = lambda: SimpleNamespace(attn_metadata=None)

    selector = module.CommunicationStrategySelector(layer)
    # tp=1, dp>1 → TP/DP only path
    # local_num_tokens = cdiv(8, 1) = 8, threshold 64 → dispatch_combine
    strategy, _ = selector.select_communication_strategy(num_tokens=8)

    assert strategy == "dispatch_combine"


def test_strategy_selector_non_a2_dp_gt1_tp1_large_tokens(prepare_module):
    """Non-A2, dp>1, tp=1, large tokens → all2all."""
    module, _ = prepare_module
    layer = SimpleNamespace(global_num_experts=4)
    module.get_tensor_model_parallel_world_size = lambda: 1
    module.get_dp_group = lambda: SimpleNamespace(world_size=4)
    module.get_forward_context = lambda: SimpleNamespace(attn_metadata=None)

    selector = module.CommunicationStrategySelector(layer)
    # local_num_tokens = cdiv(200, 1) = 200, threshold 64 → all2all
    strategy, _ = selector.select_communication_strategy(num_tokens=200)

    assert strategy == "all2all"


def test_strategy_selector_returns_dispatch_for_small_token_on_non_a2(prepare_module):
    module, _ = prepare_module
    layer = SimpleNamespace(global_num_experts=4)

    selector = module.CommunicationStrategySelector(layer)
    strategy, _ = selector.select_communication_strategy(num_tokens=4)
    assert strategy == "dispatch_combine"


def test_strategy_selector_non_a2_tp_dp_branches(prepare_module):
    module, _ = prepare_module
    layer = SimpleNamespace(global_num_experts=4)
    module.get_tensor_model_parallel_world_size = lambda: 2
    module.get_dp_group = lambda: SimpleNamespace(world_size=2)

    selector = module.CommunicationStrategySelector(layer)

    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={0: SimpleNamespace(decode_threshold=4)},
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
    )

    strategy_small, _ = selector.select_communication_strategy(num_tokens=8)
    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata={0: SimpleNamespace(decode_threshold=4)},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    strategy_large, _ = selector.select_communication_strategy(num_tokens=200)

    assert strategy_small == "agrs"
    assert strategy_large == "all2all"


def _cpu_arange(original_arange):
    def wrapper(*args, **kwargs):
        kwargs.pop("device", None)
        return original_arange(*args, **kwargs)
    return wrapper


def test_prepare_finalize_params_returns_none_no_quant(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4, quant_config=None, w13_weight=torch.zeros(2, 1, 1),
    )
    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    result = handler.prepare_finalize_params(
        layer=layer,
        topk_ids=torch.zeros(2, 2, dtype=torch.int32),
        agrs_prepare_permute_result=module.AGRSPreparePermuteResult(
            hidden_states_sorted_by_experts=torch.zeros(4, 3),
            expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
            dynamic_scale=None, row_idx_type=1,
        ),
    )
    assert result is None


def test_prepare_finalize_params_returns_none_for_row_idx_0(prepare_module):
    module, stubs = prepare_module
    layer = SimpleNamespace(
        global_num_experts=4, quant_config=object(), w13_weight=torch.zeros(2, 1, 1),
    )
    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    result = handler.prepare_finalize_params(
        layer=layer,
        topk_ids=torch.zeros(2, 2, dtype=torch.int32),
        agrs_prepare_permute_result=module.AGRSPreparePermuteResult(
            hidden_states_sorted_by_experts=torch.zeros(4, 3),
            expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
            dynamic_scale=None, row_idx_type=0,
        ),
    )
    assert result is None


def test_prepare_finalize_params_w8a8(prepare_module, monkeypatch):
    module, stubs = prepare_module
    monkeypatch.setattr(module.torch, "arange", _cpu_arange(torch.arange))
    layer = SimpleNamespace(
        global_num_experts=4, quant_config=object(), dp_size=1,
        w2_weight_scale=torch.ones(2, 3, dtype=torch.bfloat16),
        w2_bias=torch.zeros(2, 3, dtype=torch.bfloat16),
        w13_weight=torch.zeros(2, 1, 1),
    )
    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    handler.batch_size = 2
    prepare_result = module.AGRSPreparePermuteResult(
        hidden_states_sorted_by_experts=torch.zeros(4, 3),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        dynamic_scale=torch.ones(4),
        expanded_row_idx=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        gathered_topk_ids=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
        dtype=torch.float16, row_idx_type=1,
    )
    result = handler.prepare_finalize_params(
        layer=layer,
        topk_ids=torch.zeros(2, 2, dtype=torch.int32),
        agrs_prepare_permute_result=prepare_result,
    )
    assert isinstance(result, module.AGRSFinalizeParams)
    assert result.expanded_row_idx is not None
    assert result.row_index.dtype == torch.int64
    assert result.batch_size == prepare_result.gathered_topk_ids.shape[0]
    assert result.w2_bias is layer.w2_bias


def test_prepare_finalize_params_w4a8(prepare_module, monkeypatch):
    module, stubs = prepare_module
    monkeypatch.setattr(module.torch, "arange", _cpu_arange(torch.arange))
    w2_int4_scale = torch.ones(2, 3, dtype=torch.float32)
    w2_bias = torch.zeros(2, 3)
    layer = SimpleNamespace(
        global_num_experts=4, quant_config=object(),
        w2_weight_int4_scale=w2_int4_scale, w2_weight_bias=w2_bias,
        w13_weight=torch.zeros(2, 1, 1), dp_size=1,
    )
    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    handler.batch_size = 2
    prepare_result = module.AGRSPreparePermuteResult(
        hidden_states_sorted_by_experts=torch.zeros(4, 3),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        dynamic_scale=torch.ones(4), row_idx_type=1,
        expanded_row_idx=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        gathered_topk_ids=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
    )
    result = handler.prepare_finalize_params(
        layer=layer,
        topk_ids=torch.zeros(2, 2, dtype=torch.int32),
        agrs_prepare_permute_result=prepare_result,
    )
    assert result.w2_scale is w2_int4_scale
    assert result.w2_bias is w2_bias


def test_unpermute_finalize_with_finalize_params(prepare_module, monkeypatch):
    module, stubs = prepare_module
    monkeypatch.setattr(module.torch, "arange", _cpu_arange(torch.arange))
    stubs.torch_npu.npu_grouped_matmul_finalize_routing.return_value = torch.ones(2, 3, dtype=torch.bfloat16)
    layer = SimpleNamespace(
        quant_config=object(), dp_size=1,
        w2_weight=torch.ones(2, 3, 2, dtype=torch.int8),
    )
    handler = module.AGRSPrepPmtAndUnpmtFinal(
        SimpleNamespace(global_num_experts=4, w13_weight=torch.zeros(2, 1, 1))
    )
    handler.batch_size = 2
    finalize_params = module.AGRSFinalizeParams(
        expanded_row_idx=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        row_index=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        batch_size=4,
        w2_scale=torch.ones(2, 3, dtype=torch.float32),
        w2_bias=None,
    )
    prepare_result = module.AGRSPreparePermuteResult(
        hidden_states_sorted_by_experts=torch.zeros(4, 3),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        dynamic_scale=torch.ones(4),
        expanded_row_idx=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        gathered_topk_ids=torch.zeros(2, 2, dtype=torch.int32),
        dtype=torch.float16, row_idx_type=1,
    )
    y = handler.unpermute_finalize(
        layer=layer,
        hidden_states=(torch.ones(4, 3, dtype=torch.int8), torch.ones(4)),
        topk_ids=torch.zeros(2, 2, dtype=torch.int32),
        topk_weights=torch.ones(2, 2, dtype=torch.float32),
        agrs_prepare_permute_result=prepare_result,
        finalize_params=finalize_params,
    )
    assert y.dtype == torch.float16
    kwargs = stubs.torch_npu.npu_grouped_matmul_finalize_routing.call_args.kwargs
    assert kwargs["logit"] is not None
    assert kwargs["row_index"] is finalize_params.row_index
    assert kwargs["output_bs"] == 4
    assert kwargs["scale"] is finalize_params.w2_scale


def test_unpermute_finalize_serial_prefill_gmmfr_float_cast(prepare_module, monkeypatch):
    """Serial fallback path with prefill_grouped_matmul_finalize_routing casts sorted_topk_weight to float."""
    module, stubs = prepare_module
    monkeypatch.setattr(module.torch, "arange", _cpu_arange(torch.arange))

    # Enable prefill GMMFR config
    module.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(
            decode_moe_dispatch_combine=False,
            prefill_grouped_matmul_finalize_routing=True,
        )
    )

    layer = SimpleNamespace(
        global_num_experts=4,
        quant_config=object(),
        dp_size=1,
        w2_weight=torch.ones(2, 3, 2, dtype=torch.int8),
        w2_weight_scale=torch.ones(2, 3, dtype=torch.bfloat16),
        w2_bias=torch.zeros(2, 3, dtype=torch.bfloat16),
    )
    stubs.torch_npu.npu_grouped_matmul_finalize_routing.return_value = torch.ones(2, 3, dtype=torch.bfloat16)

    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    handler.batch_size = 2

    prepare_result = module.AGRSPreparePermuteResult(
        hidden_states_sorted_by_experts=torch.zeros(4, 3),
        expert_tokens=torch.tensor([2, 2], dtype=torch.int32),
        dynamic_scale=torch.ones(4),
        expert_range=[0, 4],
        expanded_row_idx=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        gathered_topk_ids=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
        dtype=torch.float16,
        row_idx_type=1,
    )
    hidden_states = (
        torch.ones(4, 3, dtype=torch.int8),
        torch.tensor([1.0, 1.1, 1.0, 1.1], dtype=torch.float32),
    )
    # Call without finalize_params -> serial fallback path
    y = handler.unpermute_finalize(
        layer=SimpleNamespace(
            quant_config=object(),
            w2_weight=torch.ones(2, 3, 2, dtype=torch.int8),
            w2_weight_scale=torch.ones(2, 3, dtype=torch.bfloat16),
            w2_bias=torch.zeros(2, 3, dtype=torch.bfloat16),
            dp_size=1,
        ),
        hidden_states=hidden_states,
        topk_ids=torch.zeros(2, 2, dtype=torch.int32),
        topk_weights=torch.ones(2, 2, dtype=torch.float32),
        agrs_prepare_permute_result=prepare_result,
        finalize_params=None,
    )

    assert y.dtype == torch.float16
    kwargs = stubs.torch_npu.npu_grouped_matmul_finalize_routing.call_args.kwargs
    # sorted_topk_weight should be float32 due to .float() cast
    assert kwargs["logit"].dtype == torch.float32


def test_agrs_prepare_permute_hifloat8_with_routed_experts_cv_runs_shared_gate_up(
    prepare_module,
):
    """When shared_expert_parallel_schedule == 'with_routed_experts_cv', prepare_permute
    schedules layer.shared_experts.gate_up_proj on the side stream and records an event
    on AGRSPreparePermuteResult.shared_expert_gate_up_proj_finished_event."""
    module, stubs = prepare_module
    module.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(
            shared_expert_multi_stream=True,
            shared_expert_parallel_schedule="with_routed_experts_cv",
            decode_moe_dispatch_combine=False,
            enable_moe_agrs=False,
            prefill_grouped_matmul_finalize_routing=False,
        ),
        parall_config=SimpleNamespace(ena_seq_parallel=False),
    )

    recorded_events = []

    class FakeEvent:
        def __init__(self):
            self.recorded = False
            self.waits = []
            recorded_events.append(self)

        def record(self):
            self.recorded = True

        def wait(self, stream):
            self.waits.append(stream)

    class FakeStreamCtx:
        def __init__(self, _stream):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    import torch as _torch
    _torch.npu.Event = FakeEvent
    _torch.npu.stream = FakeStreamCtx

    shared_gate_up_proj = MagicMock(return_value=torch.full((2, 4), 7.0))
    shared_experts = SimpleNamespace(gate_up_proj=shared_gate_up_proj)

    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        ep_size=stubs.ep_group.world_size,
        quant_config=object(),
        quant_method=SimpleNamespace(
            moe_quant_config=SimpleNamespace(
                use_hifloat8_w8a8=True,
                use_mxfp8_w8a8=False,
            ),
        ),
        w13_weight=torch.zeros(2, 1, 1),
        shared_experts=shared_experts,
    )

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    x_hif8 = torch.zeros_like(x, dtype=torch.int8)
    stubs.torch_npu.hifloat8 = object()
    stubs.torch_npu.npu_dtype_cast.return_value = x_hif8
    stubs.torch_npu.npu_moe_init_routing_v2.return_value = (
        torch.ones(2, 2, dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
        None,
    )

    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    # The handler stores its own side_stream attribute via named_stream.
    fake_side_stream = SimpleNamespace(wait_stream=lambda _other: None)
    handler.side_stream = fake_side_stream

    # x_quant.record_stream(side_stream) needs a real Stream; patch to a no-op.
    import torch as _torch
    orig_record_stream = _torch.Tensor.record_stream
    _torch.Tensor.record_stream = lambda self, _stream: None
    try:
        result = handler.prepare_permute(
            layer=layer, x=x, topk_ids=torch.zeros(2, 1, dtype=torch.int32)
        )
    finally:
        _torch.Tensor.record_stream = orig_record_stream

    # gate_up_proj was called on the dict containing the hifloat8-quantised tensor.
    shared_gate_up_proj.assert_called_once()
    arg = shared_gate_up_proj.call_args.args[0]
    assert "x_hif8" in arg
    assert torch.equal(result.shared_expert_gate_up, torch.full((2, 4), 7.0))
    # Two events: one for ready_to_dispatch (which records) and one for the finished event.
    assert len(recorded_events) == 2
    assert all(ev.recorded for ev in recorded_events)
    assert result.shared_expert_gate_up_proj_finished_event is recorded_events[1]


def test_agrs_prepare_permute_default_finalize_no_shared_gate_up(prepare_module):
    """Default 'with_finalize' path leaves shared_expert_gate_up* fields as None."""
    module, stubs = prepare_module
    module.model_extra_config = SimpleNamespace(
        operator_opt_config=SimpleNamespace(
            shared_expert_multi_stream=True,
            shared_expert_parallel_schedule="with_finalize",
            decode_moe_dispatch_combine=False,
            enable_moe_agrs=False,
            prefill_grouped_matmul_finalize_routing=False,
        ),
        parall_config=SimpleNamespace(ena_seq_parallel=False),
    )

    layer = SimpleNamespace(
        global_num_experts=4,
        local_num_experts=2,
        ep_size=stubs.ep_group.world_size,
        quant_config=object(),
        quant_method=SimpleNamespace(
            moe_quant_config=SimpleNamespace(
                use_hifloat8_w8a8=True,
                use_mxfp8_w8a8=False,
            ),
        ),
        w13_weight=torch.zeros(2, 1, 1),
        shared_experts=SimpleNamespace(gate_up_proj=MagicMock()),
    )
    x = torch.ones(2, 2, dtype=torch.float32)
    stubs.torch_npu.hifloat8 = object()
    stubs.torch_npu.npu_dtype_cast.return_value = x.to(torch.int8)
    stubs.torch_npu.npu_moe_init_routing_v2.return_value = (
        torch.ones(2, 2, dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
        None,
    )

    handler = module.AGRSPrepPmtAndUnpmtFinal(layer)
    result = handler.prepare_permute(
        layer=layer, x=x, topk_ids=torch.zeros(2, 1, dtype=torch.int32)
    )

    assert result.shared_expert_gate_up is None
    assert result.shared_expert_gate_up_proj_finished_event is None
    layer.shared_experts.gate_up_proj.assert_not_called()
