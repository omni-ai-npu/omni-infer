# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import pytest
from unittest.mock import Mock, patch

from omni_npu.connector.mm_feature_transfer.config import MMFeatureTransferConfig
from omni_npu.connector.mm_feature_transfer.mm_feature_connector import MMFeatureConnectorFactory


class TestMMFeatureConnectorFactory:
    @patch('omni_npu.connector.mm_feature_transfer.mm_feature_connector.factory.DiskMMFeatureConnector')
    @patch("omni_npu.connector.mm_feature_transfer.mm_feature_connector.factory.NetworkMMFeatureConnector")
    def test_create_connector_creates_new(self, mock_network_cls, mock_disk_cls):
        json_str = json.dumps({
            "connectors": {
                "local": {
                    "connector_type": "DiskMMFeatureConnector",
                    "storage_path": "/local/disk"
                },
                "remote": {
                    "connector_type": "NetworkMMFeatureConnector",
                    "remote_endpoints": "tcp://remote:7777"
                }
            }
        })
        config = MMFeatureTransferConfig.from_cli(json_str)

        # Mock 返回值
        mock_disk_instance = Mock(name="local_disk")
        mock_disk_cls.return_value = mock_disk_instance
        
        mock_network_instance = Mock(name="remote_network")
        mock_network_cls.return_value = mock_network_instance

        result = MMFeatureConnectorFactory.create_connector(config)

        # 断言 1：Disk 使用 local 配置实例化
        local_config = config.connectors["local"]
        mock_disk_cls.assert_called_once_with(local_config)

        # 断言 2：Network 使用 remote 配置和 disk 实例作为 local 参数
        remote_config = config.connectors["remote"]
        mock_network_cls.assert_called_once_with(remote_config, mock_disk_instance)

        # 断言 3：返回的是 remote (Network) 实例
        assert result is mock_network_instance

    @patch('omni_npu.connector.mm_feature_transfer.mm_feature_connector.factory.DiskMMFeatureConnector')
    def test_create_connector_reuses_instance(self, mock_disk_cls):
        json_str = json.dumps({
            "connector_type": "DiskMMFeatureConnector",
        })
        config = MMFeatureTransferConfig.from_cli(json_str)
        MMFeatureConnectorFactory._instances.clear()
        first = MMFeatureConnectorFactory.create_connector(config)
        second = MMFeatureConnectorFactory.create_connector(config)
        assert first is second
        mock_disk_cls.assert_called_once()  

