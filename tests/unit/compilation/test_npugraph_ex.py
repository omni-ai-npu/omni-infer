# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from collections.abc import Callable
import sys
import torch
import torch._inductor.compile_fx
import torch.fx as fx

import pytest
from unittest.mock import patch, MagicMock, ANY

from omni_npu.compilation.npugraph_ex import NpuGraphExAdaptor


@pytest.fixture
def default_npugraph_ex_adaptor():
    return NpuGraphExAdaptor()


@pytest.fixture
def mock_inputs():
    graph = MagicMock(spec=fx.GraphModule)
    example_inputs = [torch.tensor([1, 2, 3])]
    compiler_config = {"key": "value"}
    return graph, example_inputs, compiler_config


@pytest.fixture
def mock_npugraph_ex_module():
    """Mock the npugraph_ex module in sys.modules so that local imports work."""
    mock_module = MagicMock()
    mock_config = MagicMock()
    mock_module.CompilerConfig.return_value = mock_config
    mock_backend = MagicMock()
    mock_backend.return_value = MagicMock(spec=Callable)
    mock_module.get_npu_backend.return_value = mock_backend
    with patch.dict(sys.modules, {"npugraph_ex": mock_module}):
        yield mock_module


class TestNpuGraphExAdaptor:
    def test_compile_with_tuple_output(self, default_npugraph_ex_adaptor, mock_inputs, mock_npugraph_ex_module):
        """Test graph_returns_tuple output is tuple"""
        with patch("torch._inductor.compile_fx.graph_returns_tuple") as mock_graph_returns_tuple:
            with patch("omni_npu.compilation.npugraph_ex_config.get_aclgraph_config") as mock_get_aclgraph_config:
                mock_graph_returns_tuple.return_value = True

                graph, example_inputs, compiler_config = mock_inputs

                mock_config = MagicMock()
                mock_config.npugraph_ex_config = {"enable": True, "static_kernel_compile": True}
                mock_get_aclgraph_config.return_value = mock_config

                mock_compile_fn = MagicMock(spec=Callable)
                mock_npugraph_ex_module.get_npu_backend.return_value = mock_compile_fn

                result = default_npugraph_ex_adaptor.compile(graph, example_inputs, compiler_config)

                assert isinstance(result, tuple)
                assert len(result) == 2
                assert callable(result[0])
                assert result[1] is None

                mock_graph_returns_tuple.assert_called_once_with(graph)
                mock_npugraph_ex_module.get_npu_backend.assert_called_once()
                mock_compile_fn.assert_called_once_with(graph, example_inputs)


    def test_compile_with_non_tuple_output(self, default_npugraph_ex_adaptor, mock_inputs, mock_npugraph_ex_module):
        """Test graph_returns_tuple output is not tuple"""
        with patch("torch._inductor.compile_fx.graph_returns_tuple") as mock_graph_returns_tuple:
            with patch("omni_npu.compilation.npugraph_ex_config.get_aclgraph_config") as mock_get_aclgraph_config:
                # Trigger the logic for rewriting the FX Graph
                mock_graph_returns_tuple.return_value = False

                graph, example_inputs, compiler_config = mock_inputs
                # mock fx graph
                mock_fx_graph = MagicMock()
                mock_output_node = MagicMock()
                mock_return_value = MagicMock()
                mock_output_node.args = (mock_return_value, )
                mock_fx_graph.output_node.return_value = mock_output_node
                # mock create_node and recompile function
                mock_fx_graph.create_node = MagicMock()
                mock_fx_graph.inserting_before = MagicMock()
                graph.graph = mock_fx_graph

                mock_config = MagicMock()
                mock_config.npugraph_ex_config = {"enable": True, "static_kernel_compile": True}
                mock_get_aclgraph_config.return_value = mock_config

                mock_compile_fn = MagicMock(spec=Callable)
                mock_npugraph_ex_module.get_npu_backend.return_value = mock_compile_fn

                result = default_npugraph_ex_adaptor.compile(graph, example_inputs, compiler_config)

                assert isinstance(result, tuple)
                assert len(result) == 2
                assert callable(result[0])
                assert result[1] is None

                mock_graph_returns_tuple.assert_called_once_with(graph)
                mock_npugraph_ex_module.get_npu_backend.assert_called_once()
                mock_compile_fn.assert_called_once_with(graph, example_inputs)

                # Verify that the logic for rewriting the FX Graph has been executed
                mock_fx_graph.create_node.assert_called_once_with("call_function", tuple, args=([mock_return_value],))
                graph.recompile.assert_called_once()


    def test_compile_with_enable_false(self, default_npugraph_ex_adaptor, mock_inputs):
        """Test when npugraph_ex_config.enable is False"""
        with patch("omni_npu.compilation.npugraph_ex_config.get_aclgraph_config") as mock_get_aclgraph_config:
            graph, example_inputs, compiler_config = mock_inputs

            mock_config = MagicMock()
            mock_config.npugraph_ex_config = {"enable": False}
            mock_get_aclgraph_config.return_value = mock_config

            result = default_npugraph_ex_adaptor.compile(graph, example_inputs, compiler_config)

            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] is graph
            assert result[1] is None

            mock_get_aclgraph_config.assert_called_once()

    def test_compile_with_multiple_config_options(self, default_npugraph_ex_adaptor, mock_inputs, mock_npugraph_ex_module):
        """Test with multiple configuration options set"""
        with patch("torch._inductor.compile_fx.graph_returns_tuple") as mock_graph_returns_tuple:
            with patch("omni_npu.compilation.npugraph_ex_config.get_aclgraph_config") as mock_get_aclgraph_config:
                mock_graph_returns_tuple.return_value = True

                graph, example_inputs, compiler_config = mock_inputs

                custom_post_pass = MagicMock()
                custom_pre_pass = MagicMock()
                compiler_config["post_grad_custom_post_pass"] = custom_post_pass
                compiler_config["post_grad_custom_pre_pass"] = custom_pre_pass

                mock_config = MagicMock()
                mock_config.npugraph_ex_config = {
                    "enable": True,
                    "static_kernel_compile": True,
                    "super_kernel_optimize": True,
                    "capture_limit": 100,
                    "clone_input": False,
                    "clone_output": True,
                    "remove_noop_ops": False,
                    "inplace_pass": False,
                    "input_inplace_pass": False,
                    "pattern_fusion_pass": False,
                    "frozen_parameter": True,
                 }
                mock_get_aclgraph_config.return_value = mock_config

                mock_config_obj = MagicMock()
                mock_npugraph_ex_module.CompilerConfig.return_value = mock_config_obj

                mock_compile_fn = MagicMock(spec=Callable)
                mock_npugraph_ex_module.get_npu_backend.return_value = mock_compile_fn

                result = default_npugraph_ex_adaptor.compile(graph, example_inputs, compiler_config)

                # Verify mode and force_eager are set
                assert mock_config_obj.mode == "npugraph_ex"
                assert mock_config_obj.force_eager is True
                # Verify all configurations were applied
                assert mock_config_obj.static_kernel_compile is True
                assert mock_config_obj.super_kernel_optimize is True
                assert mock_config_obj.debug.aclgraph.static_capture_size_limit == 100
                assert mock_config_obj.clone_input is False
                assert mock_config_obj.clone_output is True
                assert mock_config_obj.remove_noop_ops is False
                assert mock_config_obj.inplace_pass is False
                assert mock_config_obj.input_inplace_pass is False
                assert mock_config_obj.pattern_fusion_pass is False
                assert mock_config_obj.frozen_parameter is True
                assert mock_config_obj.post_grad_custom_post_pass == custom_post_pass
                assert mock_config_obj.post_grad_custom_pre_pass == custom_pre_pass
                assert result[0] is not None
if __name__ == "__main__":
    pytest.main([__file__, "-v"])