# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import pytest
from unittest.mock import Mock, patch

from omni_npu.connector.mm_feature_transfer.config import (
    DiskConnectorConfig, 
    NetworkConnectorConfig, 
    MMFeatureTransferConfig, 
    GlobalMMFeatureConfig
)


class TestMMFeatureTransferConfig:
    def test_from_cli_valid_full(self):
        json_str = json.dumps({
            "connector_type": "DiskMMFeatureConnector",
            "storage_path": "/custom",
            "max_size_gb": 0.2,
            "exclude_fields": "a,b"
        })
        config = MMFeatureTransferConfig.from_cli(json_str).connectors["default"]
        assert config.type == "DiskMMFeatureConnector"
        assert config.storage_path == "/custom"
        assert config.max_size_gb == 0.2
        # exclude_fields automatically append address, monotonic_id
        assert config.exclude_fields == ("a", "b", "address", "monotonic_id")

    def test_from_cli_multiple_connectors(self):
        """使用 'connectors' 字典定义多个命名配置"""
        json_str = json.dumps({
            "connectors": {
                "disk": {
                    "connector_type": "DiskMMFeatureConnector",
                    "storage_path": "/path1"
                },
                "net": {
                    "connector_type": "NetworkMMFeatureConnector",
                    "local_endpoint": "tcp://127.0.0.1:6666"
                }
            }
        })
        config = MMFeatureTransferConfig.from_cli(json_str)
        
        assert "disk" in config.connectors
        assert "net" in config.connectors
        assert isinstance(config.connectors["disk"], DiskConnectorConfig)
        assert isinstance(config.connectors["net"], NetworkConnectorConfig)
        assert config.connectors["disk"].storage_path == "/path1"
        assert config.connectors["net"].local_endpoint == "tcp://127.0.0.1:6666"

    def test_from_cli_no_exclude(self):
        json_str = json.dumps({"connector_type": "DiskMMFeatureConnector"})
        config = MMFeatureTransferConfig.from_cli(json_str).connectors["default"]
        assert config.exclude_fields == ("address", "monotonic_id")

    def test_from_cli_empty_exclude(self):
        json_str = json.dumps({"connector_type": "DiskMMFeatureConnector", "exclude_fields": ""})
        config = MMFeatureTransferConfig.from_cli(json_str).connectors["default"]
        assert config.exclude_fields == ("address", "monotonic_id")

    def test_from_cli_no_type(self):
        json_str = json.dumps({"exclude_fields": ""})
        with pytest.raises(ValueError, match="Connector configuration missing 'connector_type'"):
            config = MMFeatureTransferConfig.from_cli(json_str).connectors["default"]

    def test_from_cli_none(self):
        assert MMFeatureTransferConfig.from_cli(None) is None

    def test_from_cli_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            MMFeatureTransferConfig.from_cli("{bad")


class TestGlobalMMFeatureConfig:
    def setup_method(self):
        GlobalMMFeatureConfig._instance = None

    def test_initialize_from_cli_first_time(self):
        GlobalMMFeatureConfig.initialize_from_cli('{"connector_type":"DiskMMFeatureConnector"}')
        config = GlobalMMFeatureConfig.get_config().connectors["default"]
        assert config is not None
        assert config.type == "DiskMMFeatureConnector"

    def test_initialize_from_cli_ignores_subsequent(self):
        GlobalMMFeatureConfig.initialize_from_cli('{"connector_type":"DiskMMFeatureConnector"}')
        GlobalMMFeatureConfig.initialize_from_cli('{"connector_type":"Unknown"}')
        config = GlobalMMFeatureConfig.get_config().connectors["default"]
        assert config.type == "DiskMMFeatureConnector"

    def test_initialize_with_none(self):
        GlobalMMFeatureConfig.initialize_from_cli(None)
        assert GlobalMMFeatureConfig.get_config() is None

    # def test_set_config_overrides(self):
    #     new_config = MMFeatureTransferConfig(connector_type="Manual")
    #     GlobalMMFeatureConfig.set_config(new_config)
    #     assert GlobalMMFeatureConfig.get_config() is new_config

    # def test_get_config_returns_none_if_not_initialized(self):
    #     GlobalMMFeatureConfig._instance = None
    #     assert GlobalMMFeatureConfig.get_config() is None