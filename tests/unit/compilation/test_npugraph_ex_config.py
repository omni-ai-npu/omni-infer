# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import pytest
from unittest.mock import MagicMock

from omni_npu.compilation.npugraph_ex_config import (
    AclGraphConfig,
    enable_sk_scope,
    get_aclgraph_config,
    init_aclgraph_config,
    _ACLGRAPH_CONFIG,
)


@pytest.fixture(autouse=True)
def reset_aclgraph_config():
    """Reset the global _ACLGRAPH_CONFIG before each test."""
    import omni_npu.compilation.npugraph_ex_config as config_module
    original_config = config_module._ACLGRAPH_CONFIG
    config_module._ACLGRAPH_CONFIG = None
    yield
    config_module._ACLGRAPH_CONFIG = original_config


class TestAclGraphConfig:
    def test_init_with_additional_config_containing_npugraph_ex_config(self):
        """Test AclGraphConfig initialization when additional_config contains npugraph_ex_config."""
        mock_vllm_config = MagicMock()
        mock_vllm_config.additional_config = {"npugraph_ex_config": {"enable": True, "enable_static_kernel": False}}

        config = AclGraphConfig(mock_vllm_config)

        assert config.additional_config == {"npugraph_ex_config": {"enable": True, "enable_static_kernel": False}}
        assert config.npugraph_ex_config == {"enable": True, "enable_static_kernel": False}

    def test_init_with_additional_config_without_npugraph_ex_config(self):
        """Test AclGraphConfig initialization when additional_config exists but doesn't contain npugraph_ex_config."""
        mock_vllm_config = MagicMock()
        mock_vllm_config.additional_config = {"other_config": {"key": "value"}}

        config = AclGraphConfig(mock_vllm_config)

        assert config.additional_config == {"other_config": {"key": "value"}}
        assert config.npugraph_ex_config == {}

    def test_init_with_none_additional_config(self):
        """Test AclGraphConfig initialization when additional_config is None."""
        mock_vllm_config = MagicMock()
        mock_vllm_config.additional_config = None

        config = AclGraphConfig(mock_vllm_config)

        assert config.additional_config == {}
        assert config.npugraph_ex_config == {}

    def test_init_rejects_invalid_npugraph_ex_config_type(self):
        mock_vllm_config = MagicMock()
        mock_vllm_config.additional_config = {"npugraph_ex_config": []}

        with pytest.raises(
            TypeError,
            match="additional_config.npugraph_ex_config must be dict",
        ):
            AclGraphConfig(mock_vllm_config)


class TestInitAclGraphConfig:
    def test_init_aclgraph_config_first_time(self):
        """Test init_aclgraph_config when called for the first time."""
        mock_vllm_config = MagicMock()
        mock_vllm_config.additional_config = {"npugraph_ex_config": {"enable": True}}

        config = init_aclgraph_config(mock_vllm_config)

        assert isinstance(config, AclGraphConfig)
        assert config.npugraph_ex_config == {"enable": True}
        assert get_aclgraph_config() is config

    def test_init_aclgraph_config_returns_existing_instance(self):
        """Test that init_aclgraph_config returns the same instance on subsequent calls."""
        mock_vllm_config1 = MagicMock()
        mock_vllm_config1.additional_config = {"npugraph_ex_config": {"enable": True}}

        mock_vllm_config2 = MagicMock()
        mock_vllm_config2.additional_config = {"npugraph_ex_config": {"enable": False}}

        config1 = init_aclgraph_config(mock_vllm_config1)
        config2 = init_aclgraph_config(mock_vllm_config2)

        assert config1 is config2
        # Should keep the first initialization
        assert config1.npugraph_ex_config == {"enable": True}


class TestGetAclGraphConfig:
    def test_get_aclgraph_config_success(self):
        """Test get_aclgraph_config when config is initialized."""
        mock_vllm_config = MagicMock()
        mock_vllm_config.additional_config = {"npugraph_ex_config": {"enable": True}}

        init_aclgraph_config(mock_vllm_config)
        config = get_aclgraph_config()

        assert isinstance(config, AclGraphConfig)
        assert config.npugraph_ex_config == {"enable": True}

    def test_get_aclgraph_config_not_initialized_raises_error(self):
        """Test that get_aclgraph_config raises RuntimeError when config is not initialized."""
        with pytest.raises(RuntimeError) as exc_info:
            get_aclgraph_config()

        assert "Ascend config is not initialized" in str(exc_info.value)
        assert "Please call init_aclgraph_config first" in str(exc_info.value)


class TestEnableSkScope:
    def test_enable_sk_scope_gates(self, monkeypatch):
        import omni_npu.compilation.npugraph_ex_config as config_module

        assert enable_sk_scope() is False

        mock_vllm_config = MagicMock()
        mock_vllm_config.additional_config = {
            "npugraph_ex_config": {"enable": True, "super_kernel_optimize": True}
        }
        monkeypatch.setattr(config_module, "on_ascend950", lambda: True)
        assert AclGraphConfig(mock_vllm_config).enable_sk_scope is False

        monkeypatch.setattr(config_module, "on_ascend950", lambda: False)
        extra = MagicMock()
        extra.operator_opt_config.enable_sk_scope = True
        monkeypatch.setattr(
            "omni_npu.model_config.config_loader.loader.model_extra_config",
            extra,
        )
        cfg = AclGraphConfig(mock_vllm_config)
        assert cfg.enable_sk_scope is True
        config_module._ACLGRAPH_CONFIG = cfg
        assert enable_sk_scope() is True
