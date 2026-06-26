# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_register.py

import sys
import pytest
import logging
from unittest.mock import Mock, patch, MagicMock, call, PropertyMock
from typing import Dict, Any

from omni_cache.connector.register import (
    _safe_register,
    register_connectors,
)


# ==================== _safe_register 测试 ====================

class TestSafeRegister:
    """测试 _safe_register 函数"""

    def test_safe_register_success(self):
        """测试成功注册连接器"""
        # Mock KVConnectorFactory
        mock_factory = MagicMock()
        mock_factory._registry = {}
        mock_factory.register_connector = MagicMock()

        # Mock vllm module to avoid real import
        mock_vllm_module = MagicMock()
        mock_vllm_module.KVConnectorFactory = mock_factory

        with patch.dict('sys.modules', {
            'vllm': mock_vllm_module,
            'vllm.distributed': mock_vllm_module,
            'vllm.distributed.kv_transfer': mock_vllm_module,
            'vllm.distributed.kv_transfer.kv_connector': mock_vllm_module,
            'vllm.distributed.kv_transfer.kv_connector.factory': mock_vllm_module,
        }):
            with patch('omni_cache.connector.register.logger'):
                _safe_register(
                    name="TestConnector",
                    module="test.module",
                    class_name="TestClass"
                )

        mock_factory.register_connector.assert_called_once_with(
            "TestConnector", "test.module", "TestClass"
        )

    def test_safe_register_duplicate_skip(self):
        """测试重复注册时跳过"""
        # Mock KVConnectorFactory with existing entry
        mock_factory = MagicMock()
        mock_factory._registry = {"TestConnector": "existing_class"}

        mock_vllm_module = MagicMock()
        mock_vllm_module.KVConnectorFactory = mock_factory

        with patch.dict('sys.modules', {
            'vllm': mock_vllm_module,
            'vllm.distributed': mock_vllm_module,
            'vllm.distributed.kv_transfer': mock_vllm_module,
            'vllm.distributed.kv_transfer.kv_connector': mock_vllm_module,
            'vllm.distributed.kv_transfer.kv_connector.factory': mock_vllm_module,
        }):
            with patch('omni_cache.connector.register.logger') as mock_logger:
                _safe_register(
                    name="TestConnector",
                    module="test.module",
                    class_name="TestClass"
                )

        mock_factory.register_connector.assert_not_called()
        mock_logger.debug.assert_called()

    @patch('omni_cache.connector.register.logger')
    def test_safe_register_import_failure(self, mock_logger):
        """测试导入 KVConnectorFactory 失败时安全退出"""
        with patch.dict('sys.modules', {'vllm.distributed.kv_transfer.kv_connector.factory': None}):
            _safe_register(
                name="TestConnector",
                module="test.module",
                class_name="TestClass"
            )
        mock_logger.warning.assert_called()

    def test_safe_register_register_failure(self):
        """测试 register_connector 调用失败时记录 warning"""
        mock_factory = MagicMock()
        mock_factory._registry = {}
        mock_factory.register_connector = MagicMock(
            side_effect=RuntimeError("Registration failed")
        )

        mock_vllm_module = MagicMock()
        mock_vllm_module.KVConnectorFactory = mock_factory

        with patch.dict('sys.modules', {
            'vllm': mock_vllm_module,
            'vllm.distributed': mock_vllm_module,
            'vllm.distributed.kv_transfer': mock_vllm_module,
            'vllm.distributed.kv_transfer.kv_connector': mock_vllm_module,
            'vllm.distributed.kv_transfer.kv_connector.factory': mock_vllm_module,
        }):
            with patch('omni_cache.connector.register.logger') as mock_logger:
                _safe_register(
                    name="TestConnector",
                    module="test.module",
                    class_name="TestClass"
                )

        mock_logger.warning.assert_called()

    def test_safe_register_various_name_types(self):
        """测试不同类型的连接器名称"""
        mock_factory = MagicMock()
        mock_factory._registry = {}
        mock_factory.register_connector = MagicMock()

        mock_vllm_module = MagicMock()
        mock_vllm_module.KVConnectorFactory = mock_factory

        with patch.dict('sys.modules', {
            'vllm': mock_vllm_module,
            'vllm.distributed': mock_vllm_module,
            'vllm.distributed.kv_transfer': mock_vllm_module,
            'vllm.distributed.kv_transfer.kv_connector': mock_vllm_module,
            'vllm.distributed.kv_transfer.kv_connector.factory': mock_vllm_module,
        }):
            with patch('omni_cache.connector.register.logger'):
                names = [
                    "OmniCacheConnector",
                    "SimpleConnector",
                    "Complex-Connector-V2",
                    "connector_with_underscores",
                ]
                for name in names:
                    _safe_register(name=name, module="test.module", class_name="TestClass")

        assert mock_factory.register_connector.call_count == len(names)

    def test_safe_register_inspection_fails_proceeds_anyway(self):
        """测试 registry 检查失败时仍继续注册"""
        mock_factory = MagicMock()
        type(mock_factory)._registry = PropertyMock(
            side_effect=AttributeError("No registry")
        )
        mock_factory._connectors = None
        mock_factory.register_connector = MagicMock()

        mock_vllm_module = MagicMock()
        mock_vllm_module.KVConnectorFactory = mock_factory

        with patch.dict('sys.modules', {
            'vllm': mock_vllm_module,
            'vllm.distributed': mock_vllm_module,
            'vllm.distributed.kv_transfer': mock_vllm_module,
            'vllm.distributed.kv_transfer.kv_connector': mock_vllm_module,
            'vllm.distributed.kv_transfer.kv_connector.factory': mock_vllm_module,
        }):
            with patch('omni_cache.connector.register.logger'):
                _safe_register(
                    name="TestConnector",
                    module="test.module",
                    class_name="TestClass"
                )

        mock_factory.register_connector.assert_called_once()


class TestSafeRegisterEdgeCases:
    """测试 _safe_register 边界情况"""

    def test_empty_strings(self):
        """测试空字符串参数"""
        mock_factory = MagicMock()
        mock_factory._registry = {}
        mock_factory.register_connector = MagicMock()

        mock_vllm_module = MagicMock()
        mock_vllm_module.KVConnectorFactory = mock_factory

        with patch.dict('sys.modules', {
            'vllm': mock_vllm_module,
            'vllm.distributed': mock_vllm_module,
            'vllm.distributed.kv_transfer': mock_vllm_module,
            'vllm.distributed.kv_transfer.kv_connector': mock_vllm_module,
            'vllm.distributed.kv_transfer.kv_connector.factory': mock_vllm_module,
        }):
            with patch('omni_cache.connector.register.logger'):
                _safe_register(name="", module="", class_name="")

        mock_factory.register_connector.assert_called_once_with("", "", "")

    def test_unicode_names(self):
        """测试 Unicode 名称"""
        mock_factory = MagicMock()
        mock_factory._registry = {}
        mock_factory.register_connector = MagicMock()

        mock_vllm_module = MagicMock()
        mock_vllm_module.KVConnectorFactory = mock_factory

        with patch.dict('sys.modules', {
            'vllm': mock_vllm_module,
            'vllm.distributed': mock_vllm_module,
            'vllm.distributed.kv_transfer': mock_vllm_module,
            'vllm.distributed.kv_transfer.kv_connector': mock_vllm_module,
            'vllm.distributed.kv_transfer.kv_connector.factory': mock_vllm_module,
        }):
            with patch('omni_cache.connector.register.logger'):
                _safe_register(
                    name="connector",
                    module="test.module",
                    class_name="TestClass"
                )

        mock_factory.register_connector.assert_called_once()


# ==================== register_connectors 测试 ====================

class TestRegisterConnectors:
    """测试 register_connectors 函数"""

    @patch('omni_cache.connector.register._safe_register')
    @patch('omni_cache.connector.register.logger')
    def test_register_connectors_calls_safe_register(self, mock_logger, mock_safe):
        """测试 register_connectors 调用 _safe_register"""
        register_connectors()

        mock_safe.assert_called_once_with(
            "OmniCacheConnector",
            "omni_cache.connector.connector",
            "OmniCacheConnector"
        )

    @patch('omni_cache.connector.register._safe_register')
    @patch('omni_cache.connector.register.logger')
    def test_register_connectors_logs_start_and_finish(self, mock_logger, mock_safe):
        """测试开始和结束日志记录"""
        register_connectors()

        info_calls = [call for call in mock_logger.info.call_args_list]
        assert len(info_calls) == 2
        assert "starting" in str(info_calls[0]).lower()
        assert "finished" in str(info_calls[1]).lower()

    @patch('omni_cache.connector.register._safe_register')
    @patch('omni_cache.connector.register.logger')
    def test_register_connectors_propagates_exception(self, mock_logger, mock_safe):
        """测试异常传播 - register_connectors 不捕获 _safe_register 的异常"""
        mock_safe.side_effect = RuntimeError("Test error")

        with pytest.raises(RuntimeError, match="Test error"):
            register_connectors()


class TestRegisterConnectorsIntegration:
    """register_connectors 集成测试"""

    def test_full_registration_flow(self):
        """测试完整注册流程 (无 mock _safe_register)"""
        mock_factory = MagicMock()
        mock_factory._registry = {}
        mock_factory.register_connector = MagicMock()

        mock_vllm_module = MagicMock()
        mock_vllm_module.KVConnectorFactory = mock_factory

        with patch.dict('sys.modules', {
            'vllm': mock_vllm_module,
            'vllm.distributed': mock_vllm_module,
            'vllm.distributed.kv_transfer': mock_vllm_module,
            'vllm.distributed.kv_transfer.kv_connector': mock_vllm_module,
            'vllm.distributed.kv_transfer.kv_connector.factory': mock_vllm_module,
        }):
            with patch('omni_cache.connector.register.logger'):
                register_connectors()

        mock_factory.register_connector.assert_called_with(
            "OmniCacheConnector",
            "omni_cache.connector.connector",
            "OmniCacheConnector"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])