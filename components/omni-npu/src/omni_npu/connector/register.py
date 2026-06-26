# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.logger import init_logger
import os
logger = init_logger(__name__)


def _safe_register(name: str, module: str, class_name: str) -> None:
    """Register a connector into KVConnectorFactory in a defensive way."""

    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    registry = getattr(KVConnectorFactory, "_registry", None) or getattr(
        KVConnectorFactory,
        "_connectors",
        None,
    )

    if isinstance(registry, dict) and name in registry:
        logger.debug(
            "connector: '%s' already present in KVConnectorFactory registry, "
            "skip",
            name,
        )
        return

    KVConnectorFactory.register_connector(name, module, class_name)
    logger.info(
        "connector: registered KV connector: %s -> %s.%s",
        name,
        module,
        class_name,
    )


def register_connectors() -> None:
    """Register LLMDataDsitConector as KV connector into vLLM."""
    logger.info("connector: starting KV connector registration")

    # support LLMDataDistConnector as the basic kv connector
    _safe_register(
        "LLMDataDistConnector",
        "omni_npu.connector.llmdatadist_connector_v1",
        "LLMDataDistConnector",
    )

    logger.info("connector: KV connector registration finished")