# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
import pytest

import vllm.compilation.decorators as _dec_mododule
from vllm.config import VllmConfig, CUDAGraphMode
from vllm.config.utils import Range

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
    import torch._dynamo as dynamo

    original_piecewise_call = _piecewise_module.PiecewiseBackend.__call__
    had_piecewise_flag = hasattr(
        _piecewise_module.PiecewiseBackend, "_omni_npu_static_range_patched"
    )
    original_piecewise_flag = getattr(
        _piecewise_module.PiecewiseBackend,
        "_omni_npu_static_range_patched",
        None,
    )
    original_compile_flag = decorators_mod._COMPILE_DECORATORS_PATCHED
    original_mark_dynamic = dynamo.mark_dynamic
    had_dynamic_flag = hasattr(dynamo, "_omni_npu_maybe_mark_dynamic")
    original_dynamic_flag = getattr(dynamo, "_omni_npu_maybe_mark_dynamic", None)

    decorators_mod._COMPILE_DECORATORS_PATCHED = False
    if had_piecewise_flag:
        delattr(_piecewise_module.PiecewiseBackend, "_omni_npu_static_range_patched")
    if had_dynamic_flag:
        delattr(dynamo, "_omni_npu_maybe_mark_dynamic")
    yield

    _dec_mododule._support_torch_compile = original_support_compile
    _piecewise_module.PiecewiseBackend.__call__ = original_piecewise_call
    decorators_mod._COMPILE_DECORATORS_PATCHED = original_compile_flag
    dynamo.mark_dynamic = original_mark_dynamic
    if had_piecewise_flag:
        _piecewise_module.PiecewiseBackend._omni_npu_static_range_patched = (
            original_piecewise_flag
        )
    elif hasattr(_piecewise_module.PiecewiseBackend, "_omni_npu_static_range_patched"):
        delattr(_piecewise_module.PiecewiseBackend, "_omni_npu_static_range_patched")
    if had_dynamic_flag:
        dynamo._omni_npu_maybe_mark_dynamic = original_dynamic_flag
    elif hasattr(dynamo, "_omni_npu_maybe_mark_dynamic"):
        delattr(dynamo, "_omni_npu_maybe_mark_dynamic")


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
    _piecewise_module.PiecewiseBackend._omni_npu_static_range_patched = True

    with patch("omni_npu.compilation.decorators._patched_mark_dynamic"):
        patch_compile_decorators()

    assert _piecewise_module.PiecewiseBackend.__call__ is sentinel_call


def test_patched_call_static_shape_dispatches_precompiled_range():
    """Static dispatch infers batch size and runs the matching compiled entry."""
    import vllm.compilation.piecewise_backend as _piecewise_module

    range_entry = MagicMock()
    range_entry.runnable.return_value = "static-range"
    mock_backend = SimpleNamespace(
        sym_shape_indices=[],
        compile_ranges=["configured-range"],
        _find_range_for_shape=MagicMock(return_value=range_entry),
    )

    with patch("omni_npu.compilation.decorators._patched_mark_dynamic"):
        patch_compile_decorators()
        tensor = torch.randn(16, 32)
        ret = _piecewise_module.PiecewiseBackend.__call__(mock_backend, tensor, 64)

    mock_backend._find_range_for_shape.assert_called_once_with(16)
    range_entry.runnable.assert_called_once_with(tensor, 64)
    assert ret == "static-range"


def test_patched_call_static_shape_requires_tensor_input():
    """Static dispatch cannot choose a range without a tensor batch dimension."""
    import vllm.compilation.piecewise_backend as _piecewise_module

    mock_backend = SimpleNamespace(
        compile_ranges=[],
        sym_shape_indices=[],
        _find_range_for_shape=MagicMock(),
    )

    with patch("omni_npu.compilation.decorators._patched_mark_dynamic"):
        patch_compile_decorators()
        with pytest.raises(RuntimeError, match="Cannot determine runtime shape"):
            _piecewise_module.PiecewiseBackend.__call__(mock_backend, None, "x")


def test_patched_call_symbolic_shape_compiles_exact_range_with_runtime_args():
    """An unmatched symbolic shape compiles once with the real MRoPE args."""
    import vllm.compilation.piecewise_backend as _piecewise_module

    runnable = MagicMock(return_value="exact-range")
    compile_graph = MagicMock(return_value=runnable)
    save_to_file = MagicMock()
    graph = object()
    inductor_config = object()
    compilation_config = object()
    mock_backend = SimpleNamespace(
        compile_ranges=[Range(start=1, end=96)],
        sym_shape_indices=[1],
        range_entries={},
        _find_range_for_shape=MagicMock(return_value=None),
        graph=graph,
        vllm_backend=SimpleNamespace(
            compiler_manager=SimpleNamespace(
                compile=compile_graph,
                save_to_file=save_to_file,
            ),
            inductor_config=inductor_config,
            is_encoder=False,
        ),
        compilation_config=compilation_config,
        piecewise_compile_index=2,
        total_piecewise_compiles=4,
        is_last_graph=True,
        _log_compile_start=MagicMock(),
    )
    hidden_states = torch.randn(97, 32)
    mrope_positions = torch.randn(3, 97)

    with patch("omni_npu.compilation.decorators._patched_mark_dynamic"):
        patch_compile_decorators()
        result1 = _piecewise_module.PiecewiseBackend.__call__(
            mock_backend,
            hidden_states,
            97,
            mrope_positions,
        )
        result2 = _piecewise_module.PiecewiseBackend.__call__(
            mock_backend,
            hidden_states,
            97,
            mrope_positions,
        )

    exact_range = Range(start=97, end=97)
    assert exact_range in mock_backend.range_entries
    assert result1 == "exact-range"
    assert result2 == "exact-range"
    assert compile_graph.call_count == 1
    compile_args, compile_kwargs = compile_graph.call_args
    assert compile_args[0] is graph
    assert compile_args[1][0] is hidden_states
    assert compile_args[1][1] == 97
    assert compile_args[1][2] is mrope_positions
    assert compile_args[2] is inductor_config
    assert compile_args[3] is compilation_config
    assert compile_kwargs == {
        "compile_range": exact_range,
        "graph_index": 2,
        "num_graphs": 4,
        "is_encoder": False,
    }
    mock_backend._log_compile_start.assert_called_once_with(exact_range)
    save_to_file.assert_called_once_with()
    assert runnable.call_count == 2


def test_patched_call_static_shape_compiles_exact_range():
    """An unmatched static shape also uses the exact-range fallback."""
    import vllm.compilation.piecewise_backend as _piecewise_module

    runnable = MagicMock(return_value="static-exact-range")
    compile_graph = MagicMock(return_value=runnable)
    mock_backend = SimpleNamespace(
        compile_ranges=[Range(start=1, end=64)],
        sym_shape_indices=[],
        range_entries={},
        _find_range_for_shape=MagicMock(return_value=None),
        graph=object(),
        vllm_backend=SimpleNamespace(
            compiler_manager=SimpleNamespace(
                compile=compile_graph,
                save_to_file=MagicMock(),
            ),
            inductor_config=object(),
            is_encoder=False,
        ),
        compilation_config=object(),
        piecewise_compile_index=0,
        total_piecewise_compiles=1,
        is_last_graph=False,
        _log_compile_start=MagicMock(),
    )
    hidden_states = torch.randn(65, 32)

    with patch("omni_npu.compilation.decorators._patched_mark_dynamic"):
        patch_compile_decorators()
        result = _piecewise_module.PiecewiseBackend.__call__(
            mock_backend,
            hidden_states,
        )

    assert Range(start=65, end=65) in mock_backend.range_entries
    assert result == "static-exact-range"
    assert compile_graph.call_count == 1
    mock_backend.vllm_backend.compiler_manager.save_to_file.assert_not_called()


def test_patched_call_exact_range_requires_original_graph():
    """A backend restored only from artifacts cannot compile a new range."""
    import vllm.compilation.piecewise_backend as _piecewise_module

    mock_backend = SimpleNamespace(
        compile_ranges=[Range(start=1, end=96)],
        sym_shape_indices=[1],
        range_entries={},
        _find_range_for_shape=MagicMock(return_value=None),
        graph=None,
    )

    with patch("omni_npu.compilation.decorators._patched_mark_dynamic"):
        patch_compile_decorators()
        with pytest.raises(RuntimeError, match="precompiled artifacts"):
            _piecewise_module.PiecewiseBackend.__call__(
                mock_backend,
                torch.randn(97, 32),
                97,
                torch.randn(3, 97),
            )

    assert Range(start=97, end=97) in mock_backend.range_entries


def test_patched_call_symbolic_shape_delegates_to_upstream():
    """Symbolic-shape dispatch remains owned by vLLM's original backend."""
    import vllm.compilation.piecewise_backend as _piecewise_module

    calls = MagicMock(return_value="upstream")

    def upstream_call(self, *args):
        return calls(self, *args)

    _piecewise_module.PiecewiseBackend.__call__ = upstream_call
    range_entry = object()
    mock_backend = SimpleNamespace(
        sym_shape_indices=[1],
        compile_ranges=[Range(start=1, end=96)],
        _find_range_for_shape=MagicMock(return_value=range_entry),
    )

    with patch("omni_npu.compilation.decorators._patched_mark_dynamic"):
        patch_compile_decorators()
        ret = _piecewise_module.PiecewiseBackend.__call__(mock_backend, "x", 64)

    calls.assert_called_once_with(mock_backend, "x", 64)
    mock_backend._find_range_for_shape.assert_called_once_with(64)
    assert ret == "upstream"


def test_patched_mark_dynamic_skips_when_maybe_mark_dynamic_is_unavailable():
    """Older torch builds without maybe_mark_dynamic keep mark_dynamic intact."""
    import torch._dynamo as dynamo

    original_mark_dynamic = dynamo.mark_dynamic
    with patch.object(dynamo, "maybe_mark_dynamic", None), patch.object(
        decorators_mod.logger, "warning"
    ) as warning:
        decorators_mod._patched_mark_dynamic()

    assert dynamo.mark_dynamic is original_mark_dynamic
    warning.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
