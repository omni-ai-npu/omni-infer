# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OX_PATH = os.environ.get("OX_PATH", os.path.join(BASE_DIR, "backends/ox/ox"))
OX_LOG_PATH = os.environ.get("OX_LOG_PATH", os.path.join("/data/ox_log"))

# seconds, used to free blocks after a delay once the request is finished
BLOCK_RELEASE_DELAY = 3000
PER_REQUEST_CONNECTION = 8

BASE_PORT = int(os.environ.get("BASE_PORT", "15077"))
ZMQ_BASE_PORT = int(os.environ.get("ZMQ_BASE_PORT", "17555"))
P_SERVER_WAIT_TIMEOUT = int(os.environ.get("P_SERVER_WAIT_TIMEOUT", "600"))

P_NODE_PORT_LIST = os.environ.get("P_NODE_PORT_LIST", "")


__all__ = [
    "BASE_DIR",
    "OX_PATH",
    "OX_LOG_PATH",
    "BLOCK_RELEASE_DELAY",
    "PER_REQUEST_CONNECTION",
    "BASE_PORT",
    "ZMQ_BASE_PORT",
    "P_SERVER_WAIT_TIMEOUT",
    "P_NODE_PORT_LIST",
]
