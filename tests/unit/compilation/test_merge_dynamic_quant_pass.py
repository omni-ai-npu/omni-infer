# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass

from omni_npu.compilation.passes.merge_dynamic_quant_pass import (
    MergeDynamicQuantPass,
    MergeDynamicQuantPattern,
)


def test_pattern_get_inputs_builds_expected_npu_tensors():
    pattern = MergeDynamicQuantPattern(MagicMock())
    tensors = [object() for _ in range(5)]

    with patch(
        "omni_npu.compilation.passes.merge_dynamic_quant_pass.torch.empty",
        side_effect=tensors,
    ) as empty:
        result = pattern.get_inputs()

    assert result == tensors
    assert empty.call_count == 5
    assert empty.call_args_list[0].args == ((8, 4),)
    assert empty.call_args_list[0].kwargs == {
        "dtype": torch.bfloat16,
        "device": "npu",
        "requires_grad": False,
    }
    assert empty.call_args_list[1].args == ((4, 2),)
    assert empty.call_args_list[2].args == ((2,),)


def test_pattern_and_replacement_quantize_independently_then_share_quantized_input():
    pattern = MergeDynamicQuantPattern(MagicMock())
    example_inputs = ["hidden", "w0", "s0", "w1", "s1"]

    with patch.object(pattern, "get_inputs", return_value=example_inputs), patch(
        "omni_npu.compilation.passes.merge_dynamic_quant_pass.torchair.register_replacement"
    ) as register:
        pattern.register()

    pattern_fn, replacement_fn, registered_inputs = register.call_args.args
    assert registered_inputs == example_inputs

    with patch(
        "omni_npu.compilation.passes.merge_dynamic_quant_pass.torch_npu.npu_dynamic_quant",
        side_effect=[("q0", "qs0"), ("q1", "qs1")],
    ) as dynamic_quant, patch(
        "omni_npu.compilation.passes.merge_dynamic_quant_pass.torch_npu.npu_quant_matmul",
        side_effect=["y0", "y1"],
    ) as quant_matmul:
        assert pattern_fn(*example_inputs) == ("y0", "y1")

    assert dynamic_quant.call_count == 2
    assert quant_matmul.call_args_list[0].kwargs["x1"] == "q0"
    assert quant_matmul.call_args_list[1].kwargs["x1"] == "q1"

    with patch(
        "omni_npu.compilation.passes.merge_dynamic_quant_pass.torch_npu.npu_dynamic_quant",
        return_value=("shared_q", "shared_scale"),
    ) as dynamic_quant, patch(
        "omni_npu.compilation.passes.merge_dynamic_quant_pass.torch_npu.npu_quant_matmul",
        side_effect=["merged_y0", "merged_y1"],
    ) as quant_matmul:
        assert replacement_fn(*example_inputs) == ("merged_y0", "merged_y1")

    dynamic_quant.assert_called_once_with("hidden", smooth_scales=None)
    assert quant_matmul.call_count == 2
    assert all(
        call.kwargs["x1"] == "shared_q"
        and call.kwargs["pertoken_scale"] == "shared_scale"
        for call in quant_matmul.call_args_list
    )


def test_pass_registers_pattern_only_for_bfloat16():
    with patch.object(VllmInductorPass, "__init__", return_value=None), patch.object(
        MergeDynamicQuantPattern, "register"
    ) as register:
        MergeDynamicQuantPass(
            SimpleNamespace(model_config=SimpleNamespace(dtype=torch.float16))
        )
        register.assert_not_called()

        config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
        MergeDynamicQuantPass(config)

    register.assert_called_once()


def test_pass_call_is_noop_because_torchair_owns_rewrite():
    merge_pass = object.__new__(MergeDynamicQuantPass)
    assert merge_pass(MagicMock()) is None
