# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Unit tests for omni_npu.attention.backends.utils module.
Tests for get_attention_backend, load_plugin_backends, and get_available_backends.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from unittest.mock import Mock

# Load utils.py directly so we do not import omni_npu.attention.backends.__init__
# (that pulls attention.py -> torch.npu / NPU runtime, which breaks CPU-only pytest).
_UTILS_MOD_NAME = "omni_npu.attention.backends.utils"


def _load_utils_standalone():
    if _UTILS_MOD_NAME in sys.modules:
        return sys.modules[_UTILS_MOD_NAME]
    repo_root = Path(__file__).resolve().parents[4]
    utils_path = repo_root / "omni" / "attention" / "backends" / "utils.py"
    spec = importlib.util.spec_from_file_location(_UTILS_MOD_NAME, utils_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_UTILS_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


_utils = _load_utils_standalone()
NPU_ATTENTION_BACKEND = _utils.NPU_ATTENTION_BACKEND
get_attention_backend = _utils.get_attention_backend
load_plugin_backends = _utils.load_plugin_backends
get_available_backends = _utils.get_available_backends
_load_plugin_backends_map = _utils._load_plugin_backends_map
_is_plugin_disabled = _utils._is_plugin_disabled
apply_plugin_overrides = _utils.apply_plugin_overrides


class TestGetAttentionBackend:
    """Tests for get_attention_backend function."""

    def test_get_attention_backend_returns_registered_path(self):
        """Test that get_attention_backend returns the path for a registered backend."""
        backend_name = "TestBackend"
        expected_path = "some.module.TestBackendClass"
        NPU_ATTENTION_BACKEND[backend_name] = expected_path

        result = get_attention_backend(backend_name)
        assert result == expected_path

        # Cleanup
        del NPU_ATTENTION_BACKEND[backend_name]

    def test_get_attention_backend_returns_none_for_unregistered(self):
        """Test that get_attention_backend returns None for unregistered backends."""
        result = get_attention_backend("NonExistentBackend")
        assert result is None


class TestLoadPluginBackends:
    """Tests for load_plugin_backends function."""

    def test_load_plugin_backends_successful_load(self, monkeypatch):
        """Test successful loading of a plugin backend (covers lines 40-44, 48)."""
        # Create a mock entry point with a class that has get_name method
        mock_backend_class = Mock()
        mock_backend_class.get_name = Mock(return_value="PluginBackend")
        mock_backend_class.__module__ = "test_module"
        mock_backend_class.__qualname__ = "TestPluginClass"

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=mock_backend_class)
        mock_ep.name = "test_plugin"

        # Mock entry_points to return our mock
        def mock_entry_points(group):
            if group == "omni_npu.attention_backends":
                return [mock_ep]
            return []

        monkeypatch.setattr(_utils, "entry_points", mock_entry_points)

        # Clear any existing registration
        if "PluginBackend" in NPU_ATTENTION_BACKEND:
            del NPU_ATTENTION_BACKEND["PluginBackend"]

        load_plugin_backends()

        assert "PluginBackend" not in NPU_ATTENTION_BACKEND

    def test_load_plugin_backends_failed_load(self, monkeypatch):
        """Test handling of failed plugin loading (covers lines 49-50)."""
        # Create a mock entry point that raises an exception
        mock_ep = Mock()
        mock_ep.load = Mock(side_effect=ImportError("Failed to load"))
        mock_ep.name = "failing_plugin"

        def mock_entry_points(group):
            if group == "omni_npu.attention_backends":
                return [mock_ep]
            return []

        monkeypatch.setattr(_utils, "entry_points", mock_entry_points)

        # Mock logger to verify warning is called
        mock_logger = Mock()
        monkeypatch.setattr(_utils, "logger", mock_logger)

        # Should not raise, just log a warning
        load_plugin_backends()

        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0]
        assert "Failed to load backend plugin" in call_args[0]

    def test_load_plugin_backends_empty_entry_points(self, monkeypatch):
        """Test load_plugin_backends with no entry points."""
        def mock_entry_points(group):
            return []

        monkeypatch.setattr(_utils, "entry_points", mock_entry_points)

        # Should complete without error
        load_plugin_backends()


class TestGetAvailableBackends:
    """Tests for get_available_backends function (covers line 54)."""

    def test_get_available_backends_returns_list(self):
        """Test that get_available_backends returns a list of backend names."""
        # Register some test backends
        NPU_ATTENTION_BACKEND["Backend1"] = "path1"
        NPU_ATTENTION_BACKEND["Backend2"] = "path2"

        result = get_available_backends()

        assert isinstance(result, list)
        assert "Backend1" in result
        assert "Backend2" in result

        # Cleanup
        del NPU_ATTENTION_BACKEND["Backend1"]
        del NPU_ATTENTION_BACKEND["Backend2"]

    def test_get_available_backends_empty(self):
        """Test get_available_backends when no backends are registered."""
        # Save current backends
        saved = NPU_ATTENTION_BACKEND.copy()
        NPU_ATTENTION_BACKEND.clear()

        result = get_available_backends()

        assert isinstance(result, list)
        assert len(result) == 0

        # Restore
        NPU_ATTENTION_BACKEND.update(saved)

    def test_get_available_backends_returns_copy(self):
        """Test that get_available_backends returns a copy, not the original."""
        NPU_ATTENTION_BACKEND["TestBackend"] = "test_path"

        result = get_available_backends()
        result.append("NewBackend")

        # Original should not be modified
        assert "NewBackend" not in NPU_ATTENTION_BACKEND

        # Cleanup
        del NPU_ATTENTION_BACKEND["TestBackend"]


class TestLoadPluginBackendsMap:
    """Tests for _load_plugin_backends_map function."""

    def test_returns_cached_result(self, monkeypatch):
        """Test that cached result is returned when cache is set."""
        utils_module = sys.modules[_UTILS_MOD_NAME]

        # Set cache directly
        cached_map = {"CachedBackend": object()}
        utils_module._PLUGIN_BACKEND_CACHE = cached_map

        result = _load_plugin_backends_map()
        assert result is cached_map

        # Reset cache
        utils_module._PLUGIN_BACKEND_CACHE = None

    def test_empty_plugin_map_on_exception(self, monkeypatch):
        """Test that entry_points exception returns empty dict and caches it."""
        utils_module = sys.modules[_UTILS_MOD_NAME]

        utils_module._PLUGIN_BACKEND_CACHE = None

        def mock_entry_points(group):
            raise RuntimeError("entry_points failed")

        monkeypatch.setattr(_utils, "entry_points", mock_entry_points)

        result = _load_plugin_backends_map()
        assert result == {}
        assert utils_module._PLUGIN_BACKEND_CACHE == {}

        # Reset cache
        utils_module._PLUGIN_BACKEND_CACHE = None

    def test_successful_plugin_load(self, monkeypatch):
        """Test successful loading of plugin backends."""
        utils_module = sys.modules[_UTILS_MOD_NAME]

        utils_module._PLUGIN_BACKEND_CACHE = None

        mock_backend_cls = Mock()
        mock_backend_cls.get_name = Mock(return_value="TestPluginBackend")

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=mock_backend_cls)
        mock_ep.name = "test_plugin"

        def mock_entry_points(group):
            if group == "omni_npu.attention_backends":
                return [mock_ep]
            return []

        monkeypatch.setattr(_utils, "entry_points", mock_entry_points)

        mock_logger = Mock()
        monkeypatch.setattr(_utils, "logger", mock_logger)

        result = _load_plugin_backends_map()

        assert "TestPluginBackend" in result
        assert result["TestPluginBackend"] is mock_backend_cls
        mock_logger.debug.assert_called()

        # Reset cache
        utils_module._PLUGIN_BACKEND_CACHE = None

    def test_plugin_load_failure_logs_warning(self, monkeypatch):
        """Test that plugin load failure logs warning and continues."""
        utils_module = sys.modules[_UTILS_MOD_NAME]

        utils_module._PLUGIN_BACKEND_CACHE = None

        # One failing, one successful entry point
        mock_fail_ep = Mock()
        mock_fail_ep.load = Mock(side_effect=ImportError("load failed"))
        mock_fail_ep.name = "failing_plugin"

        mock_success_cls = Mock()
        mock_success_cls.get_name = Mock(return_value="SuccessBackend")
        mock_success_ep = Mock()
        mock_success_ep.load = Mock(return_value=mock_success_cls)
        mock_success_ep.name = "success_plugin"

        def mock_entry_points(group):
            if group == "omni_npu.attention_backends":
                return [mock_fail_ep, mock_success_ep]
            return []

        monkeypatch.setattr(_utils, "entry_points", mock_entry_points)

        mock_logger = Mock()
        monkeypatch.setattr(_utils, "logger", mock_logger)

        result = _load_plugin_backends_map()

        # Failed one skipped, successful one loaded
        assert "SuccessBackend" in result
        assert "failing_plugin" not in result
        mock_logger.warning.assert_called()

        # Reset cache
        utils_module._PLUGIN_BACKEND_CACHE = None

    def test_get_name_failure_logs_warning(self, monkeypatch):
        """Test that get_name() failure logs warning."""
        utils_module = sys.modules[_UTILS_MOD_NAME]

        utils_module._PLUGIN_BACKEND_CACHE = None

        mock_backend_cls = Mock()
        mock_backend_cls.get_name = Mock(side_effect=AttributeError("no get_name"))

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=mock_backend_cls)
        mock_ep.name = "bad_plugin"

        def mock_entry_points(group):
            if group == "omni_npu.attention_backends":
                return [mock_ep]
            return []

        monkeypatch.setattr(_utils, "entry_points", mock_entry_points)

        mock_logger = Mock()
        monkeypatch.setattr(_utils, "logger", mock_logger)

        result = _load_plugin_backends_map()

        assert result == {}
        mock_logger.warning.assert_called()

        # Reset cache
        utils_module._PLUGIN_BACKEND_CACHE = None


class TestIsPluginDisabled:
    """Tests for _is_plugin_disabled function."""

    def test_returns_false_when_env_not_set(self, monkeypatch):
        """Test returns False when DISABLE_PLUGIN_BACKENDS is not set."""
        monkeypatch.delenv("DISABLE_PLUGIN_BACKENDS", raising=False)
        assert _is_plugin_disabled("NPUDSA") is False

    def test_returns_false_when_env_empty(self, monkeypatch):
        """Test returns False when DISABLE_PLUGIN_BACKENDS is empty."""
        monkeypatch.setenv("DISABLE_PLUGIN_BACKENDS", "")
        assert _is_plugin_disabled("NPUDSA") is False

    def test_returns_true_when_backend_in_list(self, monkeypatch):
        """Test returns True when backend is in the disabled list."""
        monkeypatch.setenv("DISABLE_PLUGIN_BACKENDS", "NPUDSA,NPUMLA")
        assert _is_plugin_disabled("NPUDSA") is True
        assert _is_plugin_disabled("NPUMLA") is True

    def test_returns_false_when_backend_not_in_list(self, monkeypatch):
        """Test returns False when backend is not in the disabled list."""
        monkeypatch.setenv("DISABLE_PLUGIN_BACKENDS", "NPUDSA,NPUMLA")
        assert _is_plugin_disabled("OTHER") is False

    def test_handles_whitespace_in_env(self, monkeypatch):
        """Test that whitespace is stripped from env values."""
        monkeypatch.setenv("DISABLE_PLUGIN_BACKENDS", "  NPUDSA , NPUMLA  , OTHER ")
        assert _is_plugin_disabled("NPUDSA") is True
        assert _is_plugin_disabled("NPUMLA") is True
        assert _is_plugin_disabled("OTHER") is True

    def test_handles_trailing_comma(self, monkeypatch):
        """Test that trailing commas are handled."""
        monkeypatch.setenv("DISABLE_PLUGIN_BACKENDS", "NPUDSA,")
        assert _is_plugin_disabled("NPUDSA") is True


class TestApplyPluginOverrides:
    """Tests for apply_plugin_overrides function."""

    def test_returns_empty_overrides_when_no_plugins(self, monkeypatch):
        """Test returns empty overrides when no plugins available."""
        utils_module = sys.modules[_UTILS_MOD_NAME]

        utils_module._PLUGIN_BACKEND_CACHE = None

        # Register a base backend
        NPU_ATTENTION_BACKEND["TestBaseBackend"] = "base.module.TestBackend"

        def mock_entry_points(group):
            return []

        monkeypatch.setattr(_utils, "entry_points", mock_entry_points)

        overrides, base_paths = apply_plugin_overrides()

        assert overrides == {}
        assert "TestBaseBackend" in base_paths

        # Cleanup
        del NPU_ATTENTION_BACKEND["TestBaseBackend"]
        utils_module._PLUGIN_BACKEND_CACHE = None

    def test_plugin_replaces_base_backend(self, monkeypatch):
        """Test that plugin replaces base backend when available."""
        utils_module = sys.modules[_UTILS_MOD_NAME]

        utils_module._PLUGIN_BACKEND_CACHE = None

        # Register base backend
        NPU_ATTENTION_BACKEND["ReplaceableBackend"] = "base.module.BaseBackend"

        # Create mock plugin
        mock_plugin_cls = Mock()
        mock_plugin_cls.get_name = Mock(return_value="ReplaceableBackend")
        mock_plugin_cls.__module__ = "plugin.module"
        mock_plugin_cls.__qualname__ = "PluginBackend"

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=mock_plugin_cls)
        mock_ep.name = "replace_plugin"

        def mock_entry_points(group):
            if group == "omni_npu.attention_backends":
                return [mock_ep]
            return []

        monkeypatch.setattr(_utils, "entry_points", mock_entry_points)

        mock_logger = Mock()
        monkeypatch.setattr(_utils, "logger", mock_logger)

        overrides, base_paths = apply_plugin_overrides()

        assert "ReplaceableBackend" in overrides
        assert overrides["ReplaceableBackend"] is mock_plugin_cls
        assert base_paths["ReplaceableBackend"] == "base.module.BaseBackend"
        assert NPU_ATTENTION_BACKEND["ReplaceableBackend"] == "plugin.module.PluginBackend"
        mock_logger.info.assert_called()

        # Cleanup
        del NPU_ATTENTION_BACKEND["ReplaceableBackend"]
        utils_module._PLUGIN_BACKEND_CACHE = None

    def test_disabled_backend_not_replaced(self, monkeypatch):
        """Test that disabled backend is not replaced by plugin."""
        utils_module = sys.modules[_UTILS_MOD_NAME]

        utils_module._PLUGIN_BACKEND_CACHE = None

        monkeypatch.setenv("DISABLE_PLUGIN_BACKENDS", "DisabledBackend")

        # Register base backend
        NPU_ATTENTION_BACKEND["DisabledBackend"] = "base.module.DisabledBackend"

        # Create mock plugin
        mock_plugin_cls = Mock()
        mock_plugin_cls.get_name = Mock(return_value="DisabledBackend")
        mock_plugin_cls.__module__ = "plugin.module"
        mock_plugin_cls.__qualname__ = "PluginDisabled"

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=mock_plugin_cls)
        mock_ep.name = "disabled_plugin"

        def mock_entry_points(group):
            if group == "omni_npu.attention_backends":
                return [mock_ep]
            return []

        monkeypatch.setattr(_utils, "entry_points", mock_entry_points)

        mock_logger = Mock()
        monkeypatch.setattr(_utils, "logger", mock_logger)

        overrides, base_paths = apply_plugin_overrides()

        assert "DisabledBackend" not in overrides
        assert NPU_ATTENTION_BACKEND["DisabledBackend"] == "base.module.DisabledBackend"
        mock_logger.debug.assert_called()

        # Cleanup
        monkeypatch.delenv("DISABLE_PLUGIN_BACKENDS")
        del NPU_ATTENTION_BACKEND["DisabledBackend"]
        utils_module._PLUGIN_BACKEND_CACHE = None

    def test_same_class_not_replaced(self, monkeypatch):
        """Test that same class (same module path) is not replaced."""
        utils_module = sys.modules[_UTILS_MOD_NAME]

        utils_module._PLUGIN_BACKEND_CACHE = None

        # Register base backend with same path as plugin
        NPU_ATTENTION_BACKEND["SameClassBackend"] = "same.module.SameBackend"

        mock_plugin_cls = Mock()
        mock_plugin_cls.get_name = Mock(return_value="SameClassBackend")
        mock_plugin_cls.__module__ = "same.module"
        mock_plugin_cls.__qualname__ = "SameBackend"

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=mock_plugin_cls)
        mock_ep.name = "same_plugin"

        def mock_entry_points(group):
            if group == "omni_npu.attention_backends":
                return [mock_ep]
            return []

        monkeypatch.setattr(_utils, "entry_points", mock_entry_points)

        overrides, base_paths = apply_plugin_overrides()

        assert "SameClassBackend" not in overrides

        # Cleanup
        del NPU_ATTENTION_BACKEND["SameClassBackend"]
        utils_module._PLUGIN_BACKEND_CACHE = None

    def test_multiple_backends_mixed_behavior(self, monkeypatch):
        """Test multiple backends with mixed enabled/disabled states."""
        utils_module = sys.modules[_UTILS_MOD_NAME]

        utils_module._PLUGIN_BACKEND_CACHE = None

        monkeypatch.setenv("DISABLE_PLUGIN_BACKENDS", "DisabledOne")

        # Register multiple backends
        NPU_ATTENTION_BACKEND["EnabledBackend"] = "base.module.EnabledBackend"
        NPU_ATTENTION_BACKEND["DisabledOne"] = "base.module.DisabledOne"
        NPU_ATTENTION_BACKEND["NoPluginBackend"] = "base.module.NoPlugin"

        # Create mock plugins
        mock_enabled_plugin = Mock()
        mock_enabled_plugin.get_name = Mock(return_value="EnabledBackend")
        mock_enabled_plugin.__module__ = "plugin.module"
        mock_enabled_plugin.__qualname__ = "EnabledPlugin"

        mock_disabled_plugin = Mock()
        mock_disabled_plugin.get_name = Mock(return_value="DisabledOne")
        mock_disabled_plugin.__module__ = "plugin.module"
        mock_disabled_plugin.__qualname__ = "DisabledPlugin"

        mock_ep1 = Mock()
        mock_ep1.load = Mock(return_value=mock_enabled_plugin)
        mock_ep1.name = "enabled_plugin"

        mock_ep2 = Mock()
        mock_ep2.load = Mock(return_value=mock_disabled_plugin)
        mock_ep2.name = "disabled_plugin"

        def mock_entry_points(group):
            if group == "omni_npu.attention_backends":
                return [mock_ep1, mock_ep2]
            return []

        monkeypatch.setattr(_utils, "entry_points", mock_entry_points)

        overrides, base_paths = apply_plugin_overrides()

        assert "EnabledBackend" in overrides
        assert "DisabledOne" not in overrides
        assert "NoPluginBackend" not in overrides
        assert NPU_ATTENTION_BACKEND["DisabledOne"] == "base.module.DisabledOne"

        # Cleanup
        monkeypatch.delenv("DISABLE_PLUGIN_BACKENDS")
        for key in ["EnabledBackend", "DisabledOne", "NoPluginBackend"]:
            del NPU_ATTENTION_BACKEND[key]
        utils_module._PLUGIN_BACKEND_CACHE = None


def test_init_cp_with_computed_lens_prefill_all_zero(monkeypatch):
    """init_cp 要求显式传入 computed_lens（全 prefill 时可为每请求 0 长度缓存前缀）。"""
    monkeypatch.setattr(
        _utils,
        "current_platform",
        SimpleNamespace(device_type="cpu"),
    )
    mock_g = Mock()
    mock_g.world_size = 2
    mock_g.rank_in_group = 0
    mock_g.device_group = Mock()
    blk = torch.zeros(1, 2, dtype=torch.int32)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    computed = torch.zeros(cu.size(0) - 1, dtype=cu.dtype)  # [1]
    m = _utils.SPManager.init_cp(
        sp_group=mock_g,
        cumlens=cu,
        computed_lens=computed,
        cumlens_np=None,
        page_size=128,
        table_size=128,
        block_table_ref=blk,
    )
    cl1, _, _, _ = m.cp_attn_meta_2()
    assert cl1 is not None


class TestSPManagerSWA:
    def _manager(
        self,
        monkeypatch,
        rank: int,
        query_cumlens: list[int],
        computed_lens: list[int],
        block_table: torch.Tensor,
        world_size: int = 2,
    ) -> _utils.SPManager:
        monkeypatch.setattr(
            _utils,
            "current_platform",
            SimpleNamespace(device_type="cpu"),
        )
        group = Mock()
        group.world_size = world_size
        group.rank_in_group = rank
        group.device_group = Mock()
        manager = _utils.SPManager.init_sp(tok=query_cumlens[-1], sp_group=group)
        manager.init_sp_attn(
            query_cumlens=query_cumlens,
            computed_lens=computed_lens,
            block_table_ref=block_table,
        )
        return manager

    def test_single_request_uses_local_query_and_prefix_kv(self, monkeypatch):
        block_table = torch.tensor([[4, 5]], dtype=torch.int32)

        rank0 = self._manager(monkeypatch, 0, [0, 5], [3], block_table)
        assert rank0.sp_len == 3
        q_cumlens, kv_lens, local_table = rank0.sp_attn_meta()
        assert rank0.valid_token_count == 3
        assert q_cumlens.tolist() == [3]
        assert kv_lens.tolist() == [6]
        assert torch.equal(local_table, block_table)

        rank1 = self._manager(monkeypatch, 1, [0, 5], [3], block_table)
        q_cumlens, kv_lens, local_table = rank1.sp_attn_meta()
        assert rank1.valid_token_count == 2
        assert rank1.sp_len == 3
        assert q_cumlens.tolist() == [2]
        assert kv_lens.tolist() == [8]
        assert torch.equal(local_table, block_table)

    def test_multi_request_shard_builds_fragment_metadata(self, monkeypatch):
        block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)

        rank0 = self._manager(monkeypatch, 0, [0, 3, 7], [10, 20], block_table)
        q_cumlens, kv_lens, local_table = rank0.sp_attn_meta()
        assert q_cumlens.tolist() == [3, 4]
        assert kv_lens.tolist() == [13, 21]
        assert torch.equal(local_table, block_table)

        rank1 = self._manager(monkeypatch, 1, [0, 3, 7], [10, 20], block_table)
        q_cumlens, kv_lens, local_table = rank1.sp_attn_meta()
        assert rank1.valid_token_count == 3
        assert rank1.sp_len == 4
        assert q_cumlens.tolist() == [3]
        assert kv_lens.tolist() == [24]
        assert torch.equal(local_table, block_table[1:2])

    def test_rank_without_tokens_has_empty_metadata(self, monkeypatch):
        block_table = torch.tensor([[4]], dtype=torch.int32)
        manager = self._manager(
            monkeypatch,
            rank=3,
            query_cumlens=[0, 2],
            computed_lens=[0],
            block_table=block_table,
            world_size=4,
        )

        q_cumlens, kv_lens, local_table = manager.sp_attn_meta()
        assert manager.valid_token_count == 0
        assert manager.sp_len == 1
        assert q_cumlens.numel() == 0
        assert kv_lens.numel() == 0
        assert local_table.shape == (0, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
