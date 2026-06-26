# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from logging.config import dictConfig

import vllm.envs as envs
from vllm.logger import DEFAULT_LOGGING_CONFIG, _DATE_FORMAT

VLLM_CONFIGURE_LOGGING = envs.VLLM_CONFIGURE_LOGGING
VLLM_LOGGING_CONFIG_PATH = envs.VLLM_LOGGING_CONFIG_PATH
VLLM_LOGGING_LEVEL = envs.VLLM_LOGGING_LEVEL
VLLM_LOGGING_PREFIX = envs.VLLM_LOGGING_PREFIX
VLLM_LOGGING_STREAM = envs.VLLM_LOGGING_STREAM

logger_initialized = False


def update_configure_vllm_root_logger() -> None:
    global logger_initialized

    if logger_initialized:
        return

    if VLLM_CONFIGURE_LOGGING and not VLLM_LOGGING_CONFIG_PATH:
        omni_npu_format = (f"{VLLM_LOGGING_PREFIX}%(levelname)s %(asctime)s "
                           "omni-npu[%(fileinfo)s:%(lineno)d] %(message)s")

        omni_npu_logging_config = {
            "formatters": {
                "omni_npu": {
                    "class": "vllm.logging_utils.NewLineFormatter",
                    "datefmt": _DATE_FORMAT,
                    "format": omni_npu_format,
                },
            },
            "handlers": {
                "omni_npu": {
                    "class": "logging.StreamHandler",
                    "formatter": "omni_npu",
                    "level": VLLM_LOGGING_LEVEL,
                    "stream": VLLM_LOGGING_STREAM,
                },
            },
            "loggers": {
                "omni_npu": {
                    "handlers": ["omni_npu"],
                    "level": "DEBUG",
                    "propagate": False,
                },
            }
        }

        logging_config = DEFAULT_LOGGING_CONFIG
        logging_config.update(omni_npu_logging_config)

        dictConfig(logging_config)

    logger_initialized = True
