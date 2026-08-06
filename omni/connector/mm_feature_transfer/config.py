# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass, field
import json
from typing import Optional, Tuple, Dict


@dataclass
class ConnectorConfig:
    type: str = "DiskMMFeatureConnector"  

    @staticmethod
    def _get_connector_class(type_name: str):
        mapping = {
            "DiskMMFeatureConnector": DiskConnectorConfig,
            "NetworkMMFeatureConnector": NetworkConnectorConfig,
        }
        try:
            return mapping[type_name]
        except KeyError as e:
            raise ValueError(f"Unknown connector type: {type_name}") from e
    

@dataclass
class DiskConnectorConfig(ConnectorConfig):
    type: str = "DiskMMFeatureConnector"
    storage_path: str = "/tmp/mm_features"
    metadata_store_type: str = "SQLiteMetadataStore"
    max_size_gb: float = 0.05
    high_water_ratio: float = 0.85
    low_water_ratio: float = 0.70
    ttl_seconds: int = 600
    cleanup_interval: int = 60
    is_producer: bool = True
    exclude_fields: Tuple[str, ...] = ("address", "monotonic_id")


@dataclass
class NetworkConnectorConfig(ConnectorConfig):
    type: str = "NetworkMMFeatureConnector"
    local_endpoint: str = "tcp://0.0.0.0:5555"
    remote_endpoints: str = "tcp://localhost:5555"
    is_producer: bool = True
    exclude_fields: Tuple[str, ...] = ("address", "monotonic_id")
    high_water_mark: int = 1000
    linger_ms: int = 0
    topic_filter: str = ""
    poll_timeout_ms: int = 100


@dataclass
class MMFeatureTransferConfig:
    connectors: Dict[str, ConnectorConfig] = field(default_factory=dict)

    @classmethod
    def from_cli(cls, json_str: Optional[str]) -> Optional["MMFeatureTransferConfig"]:
        if json_str is None:
            return None
        try:
            raw = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --mm-feature-transfer-config: {e}") from e

        connectors: Dict[str, ConnectorConfig] = {}
        if "connector_type" in raw:
            connectors["default"] = cls._build_single_config(raw)
        elif "connectors" in raw and isinstance(raw["connectors"], dict):
            conn_dict = raw.pop("connectors")
            for name, conn_data in conn_dict.items():
                if not isinstance(conn_data, dict):
                    raise ValueError(f"Connector {name} must be a JSON object: {conn_data}")
                connectors[name] = cls._build_single_config(conn_data)
        else:
            raise ValueError("Connector configuration missing 'connector_type'")

        return cls(connectors=connectors)
        
    @staticmethod
    def _build_single_config(data: dict) -> ConnectorConfig:
        """Build a single connector config from a JSON dict."""
        conn_type = data.pop("connector_type", None)
        if not conn_type:
            raise ValueError("Connector configuration missing 'connector_type'")

        if "exclude_fields" in data:
            data["exclude_fields"] = (
                tuple(f for f in data["exclude_fields"].split(",") if f) 
                + ("address", "monotonic_id")
            )

        cls = ConnectorConfig._get_connector_class(conn_type)   # see mapping below
        return cls(**data)


class GlobalMMFeatureConfig():
    _instance: Optional[MMFeatureTransferConfig] = None

    @classmethod
    def initialize_from_cli(cls, json_str: Optional[str] = None):
        if cls._instance is None:
            cls._instance = MMFeatureTransferConfig.from_cli(json_str)

    @classmethod
    def set_config(cls, config: MMFeatureTransferConfig):
        cls._instance = config

    @classmethod
    def get_config(cls) -> Optional[MMFeatureTransferConfig]:
        return cls._instance
