# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import sys
import multiprocessing
from typing import Optional

from vllm.logger import init_logger


logger = init_logger(__name__)


def is_api_server_process() -> bool:
    """Check if the current process is a vLLM API server process."""
    current_process = multiprocessing.current_process()
    process_name = current_process.name
    return "MainProcess" in process_name


def register_mm_feature_transfer():
    current_module = sys.modules[__name__]
    sys.modules["vllm.distributed.mm_feature_transfer"] = current_module


def register(mm_feature_transfer_config: Optional[str] = None):
    from vllm.distributed.mm_feature_transfer.config import GlobalMMFeatureConfig
    GlobalMMFeatureConfig.initialize_from_cli(mm_feature_transfer_config)

    from vllm.distributed.mm_feature_transfer.mm_feature_connector.factory import (
        MMFeatureConnectorFactory,
    )
    MMFeatureConnectorFactory.create_connector(
        GlobalMMFeatureConfig.get_config()
    )
