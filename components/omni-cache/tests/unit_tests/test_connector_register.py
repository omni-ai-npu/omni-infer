# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import builtins
import types
from unittest.mock import Mock, patch

import omni_cache.connector.register as register_module


def _install_factory_stub(install_stub_modules, factory):
    install_stub_modules({
        "vllm.distributed.kv_transfer.kv_connector.factory": {
            "KVConnectorFactory": factory,
        },
    })


class TestSafeRegister:
    def test_logs_warning_when_factory_import_fails(self, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "vllm.distributed.kv_transfer.kv_connector.factory":
                raise ImportError("factory missing")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with patch.object(register_module.logger, "warning") as mock_warning:
            register_module._safe_register("Demo", "demo.module", "DemoClass")

        mock_warning.assert_called_once()
        assert "cannot import KVConnectorFactory" in mock_warning.call_args[0][0]

    def test_skips_duplicate_registry_entry(self, install_stub_modules):
        class Factory:
            _registry = {"Demo": object()}
            register_connector = Mock()

        _install_factory_stub(install_stub_modules, Factory)

        register_module._safe_register("Demo", "demo.module", "DemoClass")

        Factory.register_connector.assert_not_called()

    def test_registers_connector_when_missing(self, install_stub_modules):
        class Factory:
            _registry = {}
            register_connector = Mock()

        _install_factory_stub(install_stub_modules, Factory)

        register_module._safe_register("Demo", "demo.module", "DemoClass")

        Factory.register_connector.assert_called_once_with("Demo", "demo.module", "DemoClass")

    def test_logs_warning_when_registration_raises(self, install_stub_modules):
        class Factory:
            _registry = {}
            register_connector = Mock(side_effect=RuntimeError("boom"))

        _install_factory_stub(install_stub_modules, Factory)

        with patch.object(register_module.logger, "warning") as mock_warning:
            register_module._safe_register("Demo", "demo.module", "DemoClass")

        mock_warning.assert_called_once()
        assert "registration of %s" in mock_warning.call_args[0][0]


class TestRegisterConnectors:
    def test_register_connectors_registers_omni_connector(self):
        with patch.object(register_module, "_safe_register") as mock_safe_register:
            register_module.register_connectors()

        mock_safe_register.assert_called_once_with(
            "OmniCacheConnector",
            "omni_cache.connector.connector",
            "OmniCacheConnector",
        )