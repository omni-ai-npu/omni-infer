# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging

logger = logging.getLogger(__name__)


def _safe_register(name: str, module: str, class_name: str) -> None:
    """Register a connector into KVConnectorFactory in a defensive way."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory,
        )
    except Exception as e:
        logger.warning(
            "connector: cannot import KVConnectorFactory, skip %s: %s",
            name,
            e,
        )
        return

    # Best-effort check to avoid duplicate registration.
    try:
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
    except Exception:
        # If inspection fails, proceed to register anyway.
        pass

    try:
        KVConnectorFactory.register_connector(name, module, class_name)
        logger.info(
            "connector: registered KV connector: %s -> %s.%s",
            name,
            module,
            class_name,
        )
    except Exception as e:
        logger.warning(
            "connector: registration of %s (%s.%s) failed or already present: %s",
            name,
            module,
            class_name,
            e,
        )


def register_connectors() -> None:
    """Register all KV connectors provided by this project into vLLM."""
    logger.info("connector: starting KV connector registration")

    _safe_register(
        "OmniCacheConnector",
        "omni_cache.connector.connector",
        "OmniCacheConnector",
    )

    logger.info("connector: KV connector registration finished")