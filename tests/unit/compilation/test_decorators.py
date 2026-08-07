# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
import os
import pytest

import vllm.compilation.decorators as _dec_mododule
from vllm.config import VllmConfig, CUDAGraphMode

from omni_npu.compilation.decorators import (
    _bypass_prefill,
    _wrap_call,
    patch_compile_decorators,
)
import omni_npu.compilation.decorators as decorators_mod


@pytest.fixture(autouse=True)
def setup_teardown():
    """Before the test, back up all the global states.
    After the test, restore them to avoid environmental contamination between test cases.
    """
    original_support_compile = _dec_mododule._support_torch_compile
    import vllm.compilation.piecewise_backend as _piecewise_module
    original_piecewise_call = _piecewise_module.PiecewiseBackend.__call__
    original_piecewise_patched = getattr(
        _piecewise_module.PiecewiseBackend, "_omni_npu_patched", False
    )
    yield
    _dec_mododule._support_torch_compile = original_support_compile
    _piecewise_module.PiecewiseBackend.__call__ = original_piecewise_call
    _piecewise_module.PiecewiseBackend._omni_npu_patched = original_piecewise_patched


class TestModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = '', **kwargs):
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class TestModelWithTuple(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = '', **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x * 2, )


class TestModelWithList(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = '', **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [x * 2]


def _support_torch_compile(x):
    torch._dynamo.mark_dynamic(x, 0)
    return x

@pytest.mark.parametrize(
    "attn_metadata, cudagraph_runtime_mode, expected_hit",
    [
        (None, CUDAGraphMode.FULL, True),
        ({"0": MagicMock(num_prefills=1)}, CUDAGraphMode.FULL, True),
        ({"0": MagicMock(num_prefills=0)}, CUDAGraphMode.FULL, False),
        ({"0": MagicMock(num_prefills=0)}, CUDAGraphMode.NONE, True),
    ]
)
def test_bypass_prefill(attn_metadata, cudagraph_runtime_mode, expected_hit):
    """Test whether bypass compilation is required and proceed with the native forward method."""
    test_model = TestModel(vllm_config=VllmConfig(), prefix="test")
    test_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    with patch("omni_npu.compilation.decorators.get_forward_context") as mock_get_forward_context:
        mock_get_forward_context.return_value.attn_metadata = attn_metadata
        mock_get_forward_context.return_value.cudagraph_runtime_mode = cudagraph_runtime_mode

        hit, retval = _bypass_prefill(test_model, test_tensor)

        assert hit == expected_hit
        if expected_hit:
            assert torch.allclose(retval, test_tensor * 2)
        else:
            assert retval is None


@pytest.mark.parametrize(
    "bypass_result, model_cls, expected",
    [
        ((True, "bypassed"), TestModel, "bypassed"),
        ((False, None), TestModelWithTuple, torch.tensor([2.0])),
        ((False, None), TestModelWithList, torch.tensor([2.0])),
    ]
)
def test_wrap_call(bypass_result, model_cls, expected):
    """Test wrap_call bypasses and unwraps single-element outputs correctly."""
    model = model_cls(vllm_config=VllmConfig(), prefix="test")
    wrapped_call = _wrap_call(type(model).__call__)
    test_tensor = torch.tensor([1.0])

    with patch("omni_npu.compilation.decorators._bypass_prefill", return_value=bypass_result):
        retval = wrapped_call(model, test_tensor)

    if isinstance(expected, torch.Tensor):
        assert torch.equal(retval, expected.to(retval.device))
    else:
        assert retval == expected


def test_wrap_call_returns_original_output_for_non_singleton_sequence():
    """Test wrap_call keeps the original output when it should not unwrap."""
    original_call = MagicMock(return_value=(torch.tensor([1.0]), torch.tensor([2.0])))
    wrapped_call = _wrap_call(original_call)
    model = TestModel(vllm_config=VllmConfig(), prefix="test")
    test_tensor = torch.tensor([1.0])

    with patch("omni_npu.compilation.decorators._bypass_prefill", return_value=(False, None)):
        retval = wrapped_call(model, test_tensor)

    assert retval == original_call.return_value


def test_wrap_call_unwraps_nested_single_tensor_list():
    """Test wrap_call unwraps tuple -> list -> tensor outputs."""
    original_call = MagicMock(return_value=([torch.tensor([3.0])],))
    wrapped_call = _wrap_call(original_call)
    model = TestModel(vllm_config=VllmConfig(), prefix="test")

    with patch("omni_npu.compilation.decorators._bypass_prefill", return_value=(False, None)):
        retval = wrapped_call(model, torch.tensor([1.0]))

    assert torch.equal(retval, torch.tensor([3.0]).to(retval.device))


def test_patched_mark_dynamic_replaces_mark_dynamic():
    """Test _patched_mark_dynamic rewrites mark_dynamic to maybe_mark_dynamic."""
    original_support_compile = _dec_mododule._support_torch_compile
    _dec_mododule._support_torch_compile = _support_torch_compile

    with patch("torch._dynamo.mark_dynamic") as mock_mark_dynamic, patch(
        "torch._dynamo.maybe_mark_dynamic"
    ) as mock_maybe_mark_dynamic:
        decorators_mod._patched_mark_dynamic()
        tensor = torch.tensor([1.0])
        result = _dec_mododule._support_torch_compile(tensor)

    mock_mark_dynamic.assert_not_called()
    mock_maybe_mark_dynamic.assert_called_once_with(tensor, 0)
    assert result is tensor
    _dec_mododule._support_torch_compile = original_support_compile


def test_patch_compile_decorators_no_ge_compile():
    mock_original_decorator = MagicMock(return_value=TestModel)
    _dec_mododule._support_torch_compile = mock_original_decorator

    with patch("omni_npu.compilation.decorators._patched_mark_dynamic") as mock_patched_mark_dynamic:
        with patch("omni_npu.compilation.decorators._wrap_call", MagicMock(side_effect=lambda x: x)) as mock_wrap:
            patch_compile_decorators()

            patched_cls = _dec_mododule._support_torch_compile(TestModel)

            mock_original_decorator.assert_called_once()
            mock_patched_mark_dynamic.assert_called_once()
            mock_wrap.assert_called_once_with(TestModel.__call__)
            assert issubclass(patched_cls, nn.Module)


def test_patch_compile_decorators_no_env():
    with patch("omni_npu.compilation.decorators._patched_mark_dynamic"):
        patch_compile_decorators()
        assert _dec_mododule._support_torch_compile.__name__ == "_patched_support_torch_compile"


def test_patch_compile_decorators_skips_repatching_piecewise_backend():
    """Test patching is a no-op when PiecewiseBackend is already patched."""
    import vllm.compilation.piecewise_backend as _piecewise_module

    sentinel_call = _piecewise_module.PiecewiseBackend.__call__
    _piecewise_module.PiecewiseBackend._omni_npu_patched = True

    with patch("omni_npu.compilation.decorators._patched_mark_dynamic"):
        patch_compile_decorators()

    assert _piecewise_module.PiecewiseBackend.__call__ is sentinel_call


@pytest.mark.parametrize(
    "compile_sizes, compile_ranges, expected_range, expected_ret",
    [
        ([64], [], (64, 64), "compile-size"),
        ([], [(128, 256)], (128, 256), "compile-range"),
    ]
)
def test_patched_call_runtime_shape_none_fallback_compile_sizes(
    compile_sizes, compile_ranges, expected_range, expected_ret
):
    """Test patched call falls back to compile_sizes or compile_ranges when runtime shape is unavailable."""
    import vllm.compilation.piecewise_backend as _piecewise_module
    from vllm.config.utils import Range

    mock_backend = MagicMock()
    compile_range = Range(start=expected_range[0], end=expected_range[1])
    range_entry = MagicMock()
    range_entry.runnable.return_value = expected_ret
    mock_backend.range_entries = {compile_range: range_entry}
    mock_backend.compile_sizes = compile_sizes
    mock_backend.compile_ranges = [
        Range(start=start, end=end) for start, end in compile_ranges
    ]
    mock_backend.sym_shape_indices = [0]

    with patch("omni_npu.compilation.decorators._patched_mark_dynamic"):
        _piecewise_module.PiecewiseBackend._omni_npu_patched = False
        patch_compile_decorators()

        ret = _piecewise_module.PiecewiseBackend.__call__(mock_backend, None)

    mock_backend._maybe_compile_for_range_entry.assert_called_once_with(range_entry, (None,))
    range_entry.runnable.assert_called_once_with(None)
    assert ret == expected_ret


def test_patched_call_runtime_shape_none_without_fallback_raises():
    """Test patched call raises when no runtime shape or fallback range can be determined."""
    import vllm.compilation.piecewise_backend as _piecewise_module

    mock_backend = SimpleNamespace(
        range_entries={},
        to_be_compiled_ranges=set(),
        compile_sizes=[],
        compile_ranges=[],
        sym_shape_indices=[],
        _find_range_for_shape=MagicMock(),
        _maybe_compile_for_range_entry=MagicMock(),
    )

    with patch("omni_npu.compilation.decorators._patched_mark_dynamic"):
        _piecewise_module.PiecewiseBackend._omni_npu_patched = False
        patch_compile_decorators()

        with pytest.raises(RuntimeError, match="Cannot determine fallback compile range"):
            _piecewise_module.PiecewiseBackend.__call__(mock_backend, None, "x")


def test_patched_call_runtime_shape_hits_existing_range():
    """Test patched call forwards the runtime-shape input to range lookup."""
    import vllm.compilation.piecewise_backend as _piecewise_module

    range_entry = MagicMock()
    range_entry.runnable.return_value = "existing-range"
    mock_backend = SimpleNamespace(
        range_entries={},
        to_be_compiled_ranges=set(),
        compile_sizes=[],
        compile_ranges=[],
        sym_shape_indices=[0],
        _find_range_for_shape=MagicMock(return_value=range_entry),
        _maybe_compile_for_range_entry=MagicMock(),
    )

    with patch("omni_npu.compilation.decorators._patched_mark_dynamic"):
        _piecewise_module.PiecewiseBackend._omni_npu_patched = False
        patch_compile_decorators()

        test_tensor = torch.randn(16, 32)
        ret = _piecewise_module.PiecewiseBackend.__call__(mock_backend, test_tensor, 64)

    mock_backend._find_range_for_shape.assert_called_once()
    runtime_shape = mock_backend._find_range_for_shape.call_args.args[0]
    if isinstance(runtime_shape, torch.Tensor):
        runtime_shape = runtime_shape.shape[0] if runtime_shape.dim() > 0 else runtime_shape.item()
    assert runtime_shape in (16, 64)
    mock_backend._maybe_compile_for_range_entry.assert_called_once_with(range_entry, (test_tensor, 64))
    range_entry.runnable.assert_called_once_with(test_tensor, 64)
    assert ret == "existing-range"


def test_patched_call_runtime_shape_uses_sym_shape_index():
    """Test patched call uses sym_shape_indices when provided."""
    import vllm.compilation.piecewise_backend as _piecewise_module

    range_entry = MagicMock()
    range_entry.runnable.return_value = "indexed-range"
    mock_backend = SimpleNamespace(
        range_entries={},
        to_be_compiled_ranges=set(),
        compile_sizes=[],
        compile_ranges=[],
        sym_shape_indices=[1],
        _find_range_for_shape=MagicMock(return_value=range_entry),
        _maybe_compile_for_range_entry=MagicMock(),
    )

    with patch("omni_npu.compilation.decorators._patched_mark_dynamic"):
        _piecewise_module.PiecewiseBackend._omni_npu_patched = False
        patch_compile_decorators()

        test_tensor = torch.randn(16, 32)
        ret = _piecewise_module.PiecewiseBackend.__call__(mock_backend, test_tensor, 64)

    mock_backend._find_range_for_shape.assert_called_once_with(64)
    mock_backend._maybe_compile_for_range_entry.assert_called_once_with(
        range_entry, (test_tensor, 64)
    )
    range_entry.runnable.assert_called_once_with(test_tensor, 64)
    assert ret == "indexed-range"


def test_patched_mark_dynamic_raises_when_exec_lacks_support_torch_compile():
    """Cover KeyError path when exec'd source no longer defines _support_torch_compile.

    Patches ``inspect.getsource`` to return a trivial source (``pass``) so the
    ``exec`` in ``_patched_mark_dynamic`` populates ``new_torch_compile`` without
    the expected key, triggering the ``raise KeyError`` guard.
    """
    with patch("inspect.getsource", return_value="pass\n"):
        with pytest.raises(KeyError, match="_support_torch_compile"):
            decorators_mod._patched_mark_dynamic()


def test_patched_call_creates_new_range_entry_when_no_existing_match():
    """Cover middle branch of _get_fallback_range_entry.

    When runtime_shape is not None AND _find_range_for_shape returns None AND
    the resulting compile_range is not yet in range_entries, the patched code
    must create a fresh RangeEntry, register it, and mark it for compilation.
    """
    import vllm.compilation.piecewise_backend as _piecewise_module
    from vllm.config.utils import Range

    mock_range_entry = MagicMock()
    mock_range_entry.runnable.return_value = "new-range"

    mock_backend = SimpleNamespace(
        range_entries={},
        to_be_compiled_ranges=set(),
        compile_sizes=[],
        compile_ranges=[],
        sym_shape_indices=[0],
        _find_range_for_shape=MagicMock(return_value=None),
        _maybe_compile_for_range_entry=MagicMock(),
    )

    with patch("omni_npu.compilation.decorators._patched_mark_dynamic"):
        _piecewise_module.PiecewiseBackend._omni_npu_patched = False
        patch_compile_decorators()

        with patch.object(_piecewise_module, "RangeEntry", return_value=mock_range_entry):
            ret = _piecewise_module.PiecewiseBackend.__call__(mock_backend, 128)

    expected_range = Range(start=128, end=128)
    assert expected_range in mock_backend.range_entries
    assert expected_range in mock_backend.to_be_compiled_ranges
    mock_backend._maybe_compile_for_range_entry.assert_called_once()
    assert ret == "new-range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
