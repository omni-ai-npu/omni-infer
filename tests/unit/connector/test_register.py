# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for register.py"""

from unittest.mock import MagicMock, patch, call
import pytest

from omni_npu.connector.register import _safe_register, register_connectors

# Path constants
KV_CONNECTOR_FACTORY_PATH = 'vllm.distributed.kv_transfer.kv_connector.factory.KVConnectorFactory'
LOGGER_PATH = 'omni_npu.connector.register.logger'
SAFE_REGISTER_PATH = 'omni_npu.connector.register._safe_register'


class TestSafeRegister:
    """Test cases for _safe_register function"""

    @pytest.fixture
    def mock_logger(self):
        with patch(LOGGER_PATH) as mock:
            yield mock

    @pytest.fixture
    def mock_factory(self):
        mock = MagicMock()
        return mock

    def test_safe_register_new_connector(self, mock_logger, mock_factory):
        """Test _safe_register with a new connector (not in registry)"""
        # Configure mock KVConnectorFactory
        mock_factory._registry = {}  # Empty registry

        with patch(KV_CONNECTOR_FACTORY_PATH, mock_factory):
            _safe_register("TestConnector", "test.module", "TestConnector")

            # Verify connector was registered
            mock_factory.register_connector.assert_called_once_with(
                "TestConnector", "test.module", "TestConnector"
            )

            # Verify info log was called
            mock_logger.info.assert_called_once_with(
                "connector: registered KV connector: %s -> %s.%s",
                "TestConnector", "test.module", "TestConnector"
            )

    def test_safe_register_existing_connector_in_registry(self, mock_logger, mock_factory):
        """Test _safe_register with existing connector in _registry"""
        # Configure mock KVConnectorFactory with existing connector
        mock_factory._registry = {"TestConnector": "existing_value"}

        with patch(KV_CONNECTOR_FACTORY_PATH, mock_factory):
            _safe_register("TestConnector", "test.module", "TestConnector")

            # Verify connector was NOT registered
            mock_factory.register_connector.assert_not_called()

            # Verify debug log was called
            mock_logger.debug.assert_called_once_with(
                "connector: '%s' already present in KVConnectorFactory registry, skip",
                "TestConnector"
            )

    def test_safe_register_existing_connector_in_connectors(self, mock_logger, mock_factory):
        """Test _safe_register with existing connector in _connectors"""
        # Configure mock KVConnectorFactory with existing connector in _connectors
        mock_factory._registry = None  # No _registry
        mock_factory._connectors = {"TestConnector": "existing_value"}

        with patch(KV_CONNECTOR_FACTORY_PATH, mock_factory):
            _safe_register("TestConnector", "test.module", "TestConnector")

            # Verify connector was NOT registered
            mock_factory.register_connector.assert_not_called()

            # Verify debug log was called
            mock_logger.debug.assert_called_once_with(
                "connector: '%s' already present in KVConnectorFactory registry, skip",
                "TestConnector"
            )

    def test_safe_register_no_registry_found(self, mock_logger, mock_factory):
        """Test _safe_register when no registry is found"""
        # Configure mock KVConnectorFactory with no registry attributes
        mock_factory._registry = None
        mock_factory._connectors = None

        with patch(KV_CONNECTOR_FACTORY_PATH, mock_factory):
            _safe_register("TestConnector", "test.module", "TestConnector")

            # Verify connector was registered (since no existing registry found)
            mock_factory.register_connector.assert_called_once_with(
                "TestConnector", "test.module", "TestConnector"
            )

            # Verify info log was called
            mock_logger.info.assert_called_once_with(
                "connector: registered KV connector: %s -> %s.%s",
                "TestConnector", "test.module", "TestConnector"
            )

    def test_safe_register_registry_not_dict(self, mock_logger, mock_factory):
        """Test _safe_register when registry exists but is not a dict"""
        # Configure mock KVConnectorFactory with non-dict registry
        mock_factory._registry = "not_a_dict"  # Not a dictionary

        with patch(KV_CONNECTOR_FACTORY_PATH, mock_factory):
            _safe_register("TestConnector", "test.module", "TestConnector")

            # Verify connector was registered (since registry is not a dict)
            mock_factory.register_connector.assert_called_once_with(
                "TestConnector", "test.module", "TestConnector"
            )

            # Verify info log was called
            mock_logger.info.assert_called_once_with(
                "connector: registered KV connector: %s -> %s.%s",
                "TestConnector", "test.module", "TestConnector"
            )


class TestRegisterConnectors:
    """Test cases for register_connectors function"""

    @pytest.fixture
    def mock_logger(self):
        with patch(LOGGER_PATH) as mock:
            yield mock

    @pytest.fixture
    def mock_safe_register(self):
        with patch(SAFE_REGISTER_PATH) as mock:
            yield mock

    def test_register_connectors_success(self, mock_logger, mock_safe_register):
        """Test register_connectors function"""
        register_connectors()

        # Verify start log
        mock_logger.info.assert_any_call("connector: starting KV connector registration")

        # Verify _safe_register was called with correct parameters
        mock_safe_register.assert_called_once_with(
            "LLMDataDistConnector",
            "omni_npu.connector.llmdatadist_connector_v1",
            "LLMDataDistConnector"
        )

        # Verify finish log
        mock_logger.info.assert_any_call("connector: KV connector registration finished")

    def test_register_connectors_multiple_calls(self, mock_logger, mock_safe_register):
        """Test register_connectors function called multiple times"""
        # Call register_connectors twice
        register_connectors()
        register_connectors()

        # Verify _safe_register was called twice
        assert mock_safe_register.call_count == 2

        # Verify logs were called twice
        assert mock_logger.info.call_count == 4  # 2 start + 2 finish


class TestModuleLevel:
    """Test module-level components"""

    def test_logger_initialization(self):
        """Test that logger is properly initialized"""
        from omni_npu.connector.register import logger

        # Verify logger exists and has expected attributes
        assert logger is not None
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')

    def test_import_statements(self):
        """Test that all required imports work correctly"""
        # Test that the module can be imported without errors
        try:
            from omni_npu.connector.register import _safe_register, register_connectors
            assert _safe_register is not None
            assert register_connectors is not None
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")


class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def mock_logger(self):
        with patch(LOGGER_PATH) as mock:
            yield mock

    @pytest.fixture
    def mock_factory(self):
        mock = MagicMock()
        return mock

    def test_safe_register_empty_strings(self, mock_logger, mock_factory):
        """Test _safe_register with empty strings"""
        mock_factory._registry = {}

        with patch(KV_CONNECTOR_FACTORY_PATH, mock_factory):
            _safe_register("", "", "")

            # Verify connector was registered even with empty strings
            mock_factory.register_connector.assert_called_once_with("", "", "")

    def test_safe_register_special_characters(self, mock_logger, mock_factory):
        """Test _safe_register with special characters"""
        mock_factory._registry = {}

        with patch(KV_CONNECTOR_FACTORY_PATH, mock_factory):
            _safe_register("Test-Connector_1", "test.module.v2", "TestConnector_v2")

            # Verify connector was registered with special characters
            mock_factory.register_connector.assert_called_once_with(
                "Test-Connector_1", "test.module.v2", "TestConnector_v2"
            )

    def test_safe_register_none_values(self, mock_logger, mock_factory):
        """Test _safe_register with None values"""
        mock_factory._registry = {}

        with patch(KV_CONNECTOR_FACTORY_PATH, mock_factory):
            # This should work since the function doesn't validate input types
            _safe_register(None, None, None)

            # Verify connector was registered with None values
            mock_factory.register_connector.assert_called_once_with(None, None, None)


# Mock KVConnectorFactory for testing
class MockKVConnectorFactory:
    """Mock KVConnectorFactory for testing"""

    def __init__(self, registry, connectors=None):
        self._registry = registry
        self._connectors = connectors

    def register_connector(self, name, module, class_name):
        self._registry[name] = (name, module, class_name)


def test_integration_with_mock_factory():
    """Integration test with mocked KVConnectorFactory"""
    # Create mock factory
    mock_factory = MockKVConnectorFactory(registry={})

    # Patch the factory
    with patch(KV_CONNECTOR_FACTORY_PATH, mock_factory):
        # Test registration
        register_info = "TestConnector", "test.module", "TestConnector"
        _safe_register(*register_info)

        # Verify registration
        assert len(mock_factory._registry) == 1
        assert mock_factory._registry[register_info[0]] == ("TestConnector", "test.module", "TestConnector")

        # Test duplicate registration
        _safe_register("TestConnector", "test.module", "TestConnector")

        # Should still only have one call (duplicate prevented)
        assert len(mock_factory._registry) == 1
