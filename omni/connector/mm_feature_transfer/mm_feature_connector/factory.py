# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from typing import Dict

from .. import is_api_server_process
from ..config import MMFeatureTransferConfig, ConnectorConfig
from .base import BaseMMFeatureConnector
from .disk_connector import DiskMMFeatureConnector
from .network_connector import NetworkMMFeatureConnector, DummyMMFeatureConnector


class MMFeatureConnectorFactory:
    """Factory for creating MM feature connectors."""
    _instances: Dict[str, BaseMMFeatureConnector] = {}
    _lock = threading.Lock()

    @staticmethod
    def create_connector(config: MMFeatureTransferConfig) -> BaseMMFeatureConnector:
        """
        Create an MM feature connector based on configuration.

        The connector type is determined by MMFeatureTransferConfig.connector_type.
        """
        if config is None or not is_api_server_process():
            return None
        config_hash = hash(str(config))

        with MMFeatureConnectorFactory._lock:
            if config_hash not in MMFeatureConnectorFactory._instances:
                connectors_config = config.connectors
                if "default" in connectors_config:
                    if connectors_config["default"].type == "NetworkMMFeatureConnector":
                        local_connector = DummyMMFeatureConnector(connectors_config["default"])
                        connector = NetworkMMFeatureConnector(connectors_config["default"], local_connector)
                    else:
                        connector = MMFeatureConnectorFactory._build_local_connector(connectors_config["default"])
                    MMFeatureConnectorFactory._instances[config_hash] = connector
                else:
                    connectors = {}

                    connectors["local"] = MMFeatureConnectorFactory._build_local_connector(connectors_config["local"])
                    connectors["remote"] = NetworkMMFeatureConnector(connectors_config["remote"], connectors["local"])

                    MMFeatureConnectorFactory._instances[config_hash] = connectors["remote"]

            return MMFeatureConnectorFactory._instances[config_hash]

    @staticmethod
    def _build_local_connector(config: ConnectorConfig) -> BaseMMFeatureConnector:
        if config.type == "DiskMMFeatureConnector":
            return DiskMMFeatureConnector(config)
        else:
            raise ValueError(f"Unknown connector type: {config.type}")

