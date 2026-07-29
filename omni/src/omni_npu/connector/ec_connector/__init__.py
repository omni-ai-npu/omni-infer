# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from omni_npu.connector.ec_connector.network_connector import ECNetworkConnector
from omni_npu.connector.ec_connector.shared_memory_connector import ECSharedMemoryConnector
from vllm.logger import init_logger

logger = init_logger(__name__)

_CONNECTORS_TO_REGISTER = [
    ECSharedMemoryConnector,
    ECNetworkConnector,
]


def register_ec_connectors() -> None:
    from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory

    for connector_cls in _CONNECTORS_TO_REGISTER:
        name = connector_cls.__name__

        if name in ECConnectorFactory._registry:
            logger.info(f"Connector '{name}' is already registered, skipping")
            continue

        try:
            logger.info(f"Registering EC connector: {name}")
            ECConnectorFactory.register_connector(
                name,
                connector_cls.__module__,
                name,
            )
        except Exception as e:
            logger.error(f"Failed to register connector {name}: {e}")

    logger.info("All EC connectors registration process finished")
