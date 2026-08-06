# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import sys
from unittest.mock import MagicMock

import pytest
from omni.vllm_plugin import plugin


class TestVllmPlugin:
    def test_plugin_basic(self):
        """Test basic plugin function functionality (should not raise exceptions).
        
        Since the test environment may have already imported related modules,
        we mainly verify that the function doesn't raise exceptions.
        """
        result = plugin()
        # Return value should be None or platform class name string
        assert result is None or result == "omni.platform.NPUPlatform"

    def test_plugin_with_torch_npu_attribute(self, monkeypatch):
        """Test plugin when torch has npu attribute (fallback case).
        
        Verifies that when torch_npu import fails but torch has npu attribute,
        the plugin returns the NPU platform class name.
        """
        # Save original __import__ function to avoid recursion
        import builtins
        original_import = builtins.__import__
        
        # Mock torch module to have npu attribute
        original_torch = sys.modules.get("torch")
        mock_torch = MagicMock()
        mock_torch.npu = MagicMock()

        # Mock torch_npu import failure
        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch_npu":
                raise ImportError("No module named torch_npu")
            # For other modules, use original import (avoid recursion)
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setattr("builtins.__import__", mock_import)

        result = plugin()
        assert result == "omni.platform.NPUPlatform"

        # Restore
        if original_torch:
            monkeypatch.setitem(sys.modules, "torch", original_torch)

    def test_plugin_no_torch(self, monkeypatch):
        """Test plugin when torch cannot be imported.
        
        Verifies that when torch import fails, the plugin returns None.
        """
        # Save original __import__ function to avoid recursion
        import builtins
        original_import = builtins.__import__
        
        original_torch = sys.modules.pop("torch", None)

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch":
                raise ImportError("No module named torch")
            # For other modules, use original import (avoid recursion)
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr("builtins.__import__", mock_import)

        result = plugin()
        assert result is None

        # Restore
        if original_torch:
            sys.modules["torch"] = original_torch

    def test_plugin_no_npu_support(self, monkeypatch):
        """Test plugin when torch is available but has no npu support (covers line 23).
        
        Verifies that when torch can be imported but torch_npu import fails
        and torch has no npu attribute, the plugin returns None.
        """
        # Save original __import__ function to avoid recursion
        import builtins
        original_import = builtins.__import__
        
        original_torch = sys.modules.get("torch")
        # Create a mock torch without npu attribute
        # Use SimpleNamespace to ensure hasattr(torch, "npu") returns False
        from types import SimpleNamespace
        mock_torch = SimpleNamespace()
        # Ensure no npu attribute
        assert not hasattr(mock_torch, "npu"), "mock_torch should not have npu attribute"

        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch_npu":
                raise ImportError("No module named torch_npu")
            # For other modules, use original import (avoid recursion)
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setitem(sys.modules, "torch", mock_torch)
        monkeypatch.setattr("builtins.__import__", mock_import)

        result = plugin()
        # Should return None because torch has no npu attribute (covers line 23)
        assert result is None

        # Restore
        if original_torch:
            monkeypatch.setitem(sys.modules, "torch", original_torch)

