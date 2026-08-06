# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import pytest
from unittest.mock import patch, MagicMock, Mock
from torch import fx

from vllm.config import VllmConfig, ModelConfig
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from omni_npu.compilation.pass_manager import GraphPassManager


class MockVllmInductorPass(VllmInductorPass):
    """Mock pass implementation for testing."""
    def __init__(self, vllm_config, applicable_ranges=None):
        super().__init__(vllm_config)
        self._stored_vllm_config = vllm_config
        self.applicable_ranges = applicable_ranges or ["decode"]
        self.called = False

    def _get_current_compile_range(self):
        """Get the current compile range from vllm_config or default."""
        if hasattr(self._stored_vllm_config, '_test_compile_range'):
            return self._stored_vllm_config._test_compile_range
        return "decode"

    def is_applicable(self, runtime_shape) -> bool:
        compile_range = self._get_current_compile_range()
        return compile_range in self.applicable_ranges

    def is_applicable_for_range(self, compile_range: str) -> bool:
        return compile_range in self.applicable_ranges

    def __call__(self, graph: fx.Graph):
        self.called = True
        # Mark graph as modified for testing
        graph._test_modified_by = self.__class__.__name__


def test_graph_pass_manager_init():
    """Test GraphPassManager initialization."""
    manager = GraphPassManager()
    assert hasattr(manager, 'passes')
    assert isinstance(manager.passes, list)
    assert len(manager.passes) == 0
    assert hasattr(manager, 'add')
    assert hasattr(manager, 'configure')
    assert hasattr(manager, '__call__')


def test_add_pass():
    """Test adding a pass to the manager."""
    manager = GraphPassManager()
    vllm_config = MagicMock(spec=VllmConfig)
    pass1 = MockVllmInductorPass(vllm_config)

    manager.add(pass1)

    assert len(manager.passes) == 1
    assert manager.passes[0] is pass1

    # Add another pass
    pass2 = MockVllmInductorPass(vllm_config, ["prefill"])
    manager.add(pass2)

    assert len(manager.passes) == 2
    assert manager.passes[1] is pass2


class TestGraphPassManagerCall:
    """Test __call__ method of GraphPassManager."""

    @pytest.fixture
    def vllm_config(self):
        """Create a mock VllmConfig."""
        config = MagicMock(spec=VllmConfig)
        config.model_config = MagicMock(spec=ModelConfig)
        config.model_config.dtype = Mock
        return config

    @pytest.fixture
    def mock_graph(self):
        """Create a mock fx.Graph."""
        graph = MagicMock(spec=fx.Graph)
        graph.recompile = MagicMock()
        return graph

    @pytest.fixture
    def pass_context(self, vllm_config, request):
        """Fixture to mock get_pass_context with different compile ranges."""
        compile_range = getattr(request, "param", "decode")
        # Store compile_range on vllm_config for mock passes to access
        vllm_config._test_compile_range = compile_range
        with patch('omni_npu.compilation.pass_manager.get_pass_context') as mock_ctx:
            ctx = MagicMock()
            ctx.compile_range = compile_range
            mock_ctx.return_value = ctx
            yield ctx
            
            # Clean up after test
            if hasattr(vllm_config, '_test_compile_range'):
                delattr(vllm_config, '_test_compile_range')

    @pytest.mark.parametrize("pass_context,applicable_range", [("decode", "decode"), ("prefill", "prefill")], indirect=["pass_context"])
    def test_call_with_applicable_pass(self, vllm_config, mock_graph, pass_context, applicable_range):
        """Test __call__ with a pass applicable for the current compile range."""
        manager = GraphPassManager()
        pass1 = MockVllmInductorPass(vllm_config, [applicable_range])
        manager.add(pass1)

        result = manager(mock_graph, [], vllm_config)

        assert result is mock_graph
        assert pass1.called is True
        assert mock_graph.recompile.called

    def test_call_with_multiple_passes_mixed_applicability(self, vllm_config, mock_graph, pass_context):
        """Test __call__ with multiple passes, some applicable, some not."""
        manager = GraphPassManager()
        pass1 = MockVllmInductorPass(vllm_config, ["decode"])  # applicable
        pass2 = MockVllmInductorPass(vllm_config, ["prefill"])  # not applicable for decode
        pass3 = MockVllmInductorPass(vllm_config, ["decode", "prefill"])  # applicable
        manager.add(pass1)
        manager.add(pass2)
        manager.add(pass3)

        result = manager(mock_graph, [], vllm_config)

        assert result is mock_graph
        assert pass1.called is True
        assert pass2.called is False
        assert pass3.called is True
        assert mock_graph.recompile.called

    def test_call_with_no_passes(self, vllm_config, mock_graph, pass_context):
        """Test __call__ when no passes are added."""
        manager = GraphPassManager()

        result = manager(mock_graph, [], vllm_config)

        assert result is mock_graph
        assert mock_graph.recompile.called


class TestGraphPassManagerConfigure:
    """Test configure method of GraphPassManager."""

    @pytest.fixture
    def vllm_config(self):
        """Create a mock VllmConfig."""
        config = MagicMock(spec=VllmConfig)
        config.model_config = MagicMock(spec=ModelConfig)
        config.model_config.dtype = Mock
        return config

    def test_configure_with_none_additional_config(self, vllm_config):
        """Test configure when additional_config is None."""
        manager = GraphPassManager()
        vllm_config.additional_config = None

        manager.configure(vllm_config)

        assert hasattr(manager, 'npugraph_ex_config')
        assert manager.npugraph_ex_config == {}
        assert len(manager.passes) == 0

    def test_configure_with_empty_additional_config(self, vllm_config):
        """Test configure when additional_config is an empty dict."""
        manager = GraphPassManager()
        vllm_config.additional_config = {}

        manager.configure(vllm_config)

        assert manager.npugraph_ex_config == {}
        assert len(manager.passes) == 0

    def test_configure_rejects_invalid_npugraph_ex_config_type(
        self, vllm_config
    ):
        """Test that pass-manager config goes through the shared accessor."""
        manager = GraphPassManager()
        vllm_config.additional_config = {"npugraph_ex_config": []}

        with pytest.raises(
            TypeError,
            match="additional_config.npugraph_ex_config must be dict",
        ):
            manager.configure(vllm_config)

    def test_configure_with_no_npugraph_ex_config(self, vllm_config):
        """Test configure when additional_config exists but has no npugraph_ex_config."""
        manager = GraphPassManager()
        vllm_config.additional_config = {"some_other_key": "value"}

        manager.configure(vllm_config)

        assert manager.npugraph_ex_config == {}
        assert len(manager.passes) == 0

    def test_configure_with_merge_dynamic_quant_false(self, vllm_config):
        """Test configure with merge_dynamic_quant set to False."""
        manager = GraphPassManager()
        vllm_config.additional_config = {
            "npugraph_ex_config": {"merge_dynamic_quant": False}
        }

        manager.configure(vllm_config)

        assert manager.npugraph_ex_config == {"merge_dynamic_quant": False}
        assert len(manager.passes) == 0

    def test_configure_with_merge_dynamic_quant_true(self, vllm_config):
        """Test configure with merge_dynamic_quant set to True."""
        manager = GraphPassManager()
        vllm_config.additional_config = {
            "npugraph_ex_config": {"merge_dynamic_quant": True}
        }

        with patch('omni_npu.compilation.passes.merge_dynamic_quant_pass.MergeDynamicQuantPass') as MockMergePass:
            manager.configure(vllm_config)

            # Verify config is set
            assert manager.npugraph_ex_config == {"merge_dynamic_quant": True}

            # Verify MergeDynamicQuantPass was instantiated with config
            MockMergePass.assert_called_once_with(vllm_config)

            # Verify the pass was added to passes list
            assert len(manager.passes) == 1
            assert manager.passes[0] is MockMergePass.return_value

    def test_configure_with_enable_moe_multistream_true(self, vllm_config):
        """Test configure with enable_moe_multistream set to True (currently no-op)."""
        manager = GraphPassManager()
        vllm_config.additional_config = {
            "npugraph_ex_config": {"enable_moe_multistream": True}
        }

        manager.configure(vllm_config)

        assert manager.npugraph_ex_config == {"enable_moe_multistream": True}
        # enable_moe_multistream is a no-op, so no pass is added
        assert len(manager.passes) == 0

    def test_configure_with_multiple_options(self, vllm_config):
        """Test configure with multiple npugraph_ex_config options."""
        manager = GraphPassManager()
        vllm_config.additional_config = {
            "npugraph_ex_config": {
                "merge_dynamic_quant": True,
                "enable_moe_multistream": False,
                "other_option": "value"
            }
        }

        with patch('omni_npu.compilation.passes.merge_dynamic_quant_pass.MergeDynamicQuantPass') as MockMergePass:
            manager.configure(vllm_config)

            # Verify all options are preserved
            assert manager.npugraph_ex_config == {
                "merge_dynamic_quant": True,
                "enable_moe_multistream": False,
                "other_option": "value"
            }

            # Verify only merge_dynamic_quant adds a pass
            MockMergePass.assert_called_once_with(vllm_config)
            assert len(manager.passes) == 1

    def test_configure_preserves_other_additional_config_keys(self, vllm_config):
        """Test that configure preserves other keys in additional_config."""
        manager = GraphPassManager()
        vllm_config.additional_config = {
            "other_key_1": "value1",
            "npugraph_ex_config": {
                "merge_dynamic_quant": False
            },
            "other_key_2": "value2"
        }

        manager.configure(vllm_config)

        # Only npugraph_ex_config is stored
        assert manager.npugraph_ex_config == {"merge_dynamic_quant": False}
        # Config object should not be modified
        assert vllm_config.additional_config["other_key_1"] == "value1"
        assert vllm_config.additional_config["other_key_2"] == "value2"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])