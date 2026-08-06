# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Unit tests for omni.attention.backends.utils module.
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

# Load utils.py directly so we do not import omni.attention.backends.__init__
# (that pulls attention.py -> torch.npu / NPU runtime, which breaks CPU-only pytest).
_UTILS_MOD_NAME = "omni.attention.backends.utils"


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
SPManager = _utils.SPManager


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
            if group == "omni.attention_backends":
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
            if group == "omni.attention_backends":
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
            if group == "omni.attention_backends":
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
            if group == "omni.attention_backends":
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
            if group == "omni.attention_backends":
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
            if group == "omni.attention_backends":
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
            if group == "omni.attention_backends":
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
            if group == "omni.attention_backends":
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
            if group == "omni.attention_backends":
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


def _sp_manager_for_mome(monkeypatch, sp_size: int, sp_rank: int) -> SPManager:
    monkeypatch.setattr(
        _utils,
        "current_platform",
        SimpleNamespace(device_type="cpu"),
    )
    mock_group = Mock()
    mock_group.world_size = sp_size
    mock_group.rank_in_group = sp_rank
    mock_group.device_group = Mock()
    return SPManager(mock_group)

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


class TestSPManagerMomeSchemeCpMome:
    """Tests for SPManager._scheme_cp_mome."""

    def test_rank0_single_request(self, monkeypatch):
        """Rank 0: one request, verify split sizes and tail metadata."""
        mgr = _sp_manager_for_mome(monkeypatch, sp_size=2, sp_rank=0)
        seq_lens = np.array([8], dtype=np.int32)
        mgr._scheme_cp_mome(seq_lens, mome_kernel_width=3)

        assert mgr.mome_prefix_size == 2
        assert mgr.cp_mome_phase_split_sizes == (2,)
        assert mgr.cp_mome_req_split_sizes == (4,)
        # core = 2 * cdiv(8,4) + 1 * 2 = 6
        # tail = kernel(3) + prefix(2) = 5
        assert mgr.cp_mome_merged_core_split_sizes == (6,)
        assert mgr.cp_mome_req_tail_append_lens == (5,)
        assert mgr.cp_mome_merged_split_sizes == (11,)
        assert mgr.cp_mome_seq_lens == (8,)
        assert mgr.cp_mome_suffix_block_len == 2
        assert mgr.cp_mome_local_suffix_len == 4
        assert mgr.cp_mome_query_start_loc.tolist() == [0, 11]

    def test_rank1_single_request(self, monkeypatch):
        """Non-zero rank uses factor=2 in core length."""
        mgr = _sp_manager_for_mome(monkeypatch, sp_size=2, sp_rank=1)
        seq_lens = np.array([8], dtype=np.int32)
        mgr._scheme_cp_mome(seq_lens, mome_kernel_width=3)

        assert mgr.cp_mome_merged_core_split_sizes == (8,)
        assert mgr.cp_mome_merged_split_sizes == (13,)

    def test_two_requests_start_loc_cumsum(self, monkeypatch):
        """Two requests: cdiv(seq_len, 4) phase splits; cumsum of merged per-req lengths."""
        mgr = _sp_manager_for_mome(monkeypatch, sp_size=2, sp_rank=0)
        seq_lens = np.array([8, 16], dtype=np.int32)
        mgr._scheme_cp_mome(seq_lens, mome_kernel_width=3)

        # frag_num=4, cp_query_split_lens = cdiv([8,16],4) = [2,4]
        assert mgr.cp_mome_phase_split_sizes == (2, 4)
        # rank0 factor=1: core = [2*2+2, 2*4+2] = [6,10], tail = min(seq_len, 5) each -> merged [11,15]
        assert mgr.cp_mome_query_start_loc.tolist() == [0, 11, 26]


class TestSPManagerMomeSuffixExchange:
    """Tests for SPManager.mome_suffix_exchange (all_gather mocked)."""

    def test_rank0_phase1_suffix_from_gather(self, monkeypatch):
        mgr = _sp_manager_for_mome(monkeypatch, sp_size=2, sp_rank=0)
        mgr._scheme_cp_mome(np.array([8], dtype=np.int32), mome_kernel_width=3)

        # x: one req, 4 rows — phase0 [1,2], phase1 [3,4]
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

        def fake_all_gather(t: torch.Tensor, dim: int = 0):
            assert dim == 0
            assert t.shape[0] == mgr.cp_mome_local_suffix_len
            out = torch.zeros(8, 1, dtype=t.dtype)
            out[6:8] = torch.tensor([[200.0], [201.0]])
            return out

        mgr.sp_group.all_gather = Mock(side_effect=fake_all_gather)
        y = mgr.mome_suffix_exchange(x)

        expected = torch.tensor([[1.0], [2.0], [200.0], [201.0], [3.0], [4.0]])
        assert torch.allclose(y, expected)


class TestSPManagerAppendMomeReqGlobalTails:
    """Tests for SPManager.append_mome_req_global_tails (all_reduce mocked)."""

    def test_tail_appended_matches_phase1_tokens(self, monkeypatch):
        mgr = _sp_manager_for_mome(monkeypatch, sp_size=1, sp_rank=0)
        mgr._scheme_cp_mome(np.array([8], dtype=np.int32), mome_kernel_width=3)

        monkeypatch.setattr(
            torch.distributed,
            "all_reduce",
            lambda tensor, op=None, group=None: None,
        )

        # core layout len 10: phase0 [0:4], pad [4:6], phase1 [6:10]
        x = torch.zeros(10, 1)
        x[0:4, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        x[6:10, 0] = torch.tensor([100.0, 101.0, 102.0, 103.0])

        out = mgr.append_mome_req_global_tails(x)
        assert out.shape[0] == 15
        assert torch.allclose(out[10:15, 0], torch.tensor([4.0, 100.0, 101.0, 102.0, 103.0]))

    def test_no_tail_append_when_tail_len_zero(self, monkeypatch):
        """Branch req_tail_len==0: output equals input (no extra tail rows)."""
        mgr = _sp_manager_for_mome(monkeypatch, sp_size=1, sp_rank=0)
        mgr._scheme_cp_mome(np.array([8], dtype=np.int32), mome_kernel_width=3)
        mgr.cp_mome_req_tail_append_lens = (0,)
        monkeypatch.setattr(
            torch.distributed,
            "all_reduce",
            lambda tensor, op=None, group=None: None,
        )
        # sp_size=1 -> frag_num=2, cdiv(8,2)=4, core = 2*4+2 = 10 rows per request
        x = torch.randn(10, 4)
        out = mgr.append_mome_req_global_tails(x)
        assert torch.equal(out, x)


class TestSPManagerMomeSplitAndCat:
    """Tests for SPManager.mome_split_and_cat."""

    def test_restores_zigzag_chunks(self, monkeypatch):
        mgr = _sp_manager_for_mome(monkeypatch, sp_size=1, sp_rank=0)
        mgr._scheme_cp_mome(np.array([8], dtype=np.int32), mome_kernel_width=3)

        # merged per req: 15 rows — core regions [0:4], [4:6] pad, [6:10] phase1, [10:15] tail
        merged = torch.zeros(15, 1)
        merged[0:4, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        merged[6:10, 0] = torch.tensor([5.0, 6.0, 7.0, 8.0])
        merged[10:15, 0] = torch.tensor([9.0, 10.0, 11.0, 12.0, 13.0])

        out = mgr.mome_split_and_cat(merged)
        assert out.shape[0] == 8
        assert torch.allclose(
            out[:, 0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])