# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OX_PATH = os.environ.get("OX_PATH", os.path.join(BASE_DIR, "backends/ox/ox"))
OX_LOG_PATH = os.environ.get("OX_LOG_PATH", os.path.join("/data/ox_log"))

# seconds, used to free blocks after a delay once the request is finished
BLOCK_RELEASE_DELAY = 3000
PER_REQUEST_CONNECTION = 8

P_NODE_LIST = os.environ.get("P_NODE_LIST", "<YOUR_IP>,<YOUR_IP>")
CLUSTER_LIST = [part.strip() for part in P_NODE_LIST.split(';') if part.strip()]
CLUSTER_SIZE = [len(part.split(',')) for part in CLUSTER_LIST][0]
NODE_IP_SPECS = [
    ip.strip()
    for segment in P_NODE_LIST.split(';')
    for ip in segment.split(',')
    if ip.strip()
]

BASE_PORT = int(os.environ.get("BASE_PORT", "15077"))
ZMQ_BASE_PORT = int(os.environ.get("ZMQ_BASE_PORT", "17555"))
P_SERVER_WAIT_TIMEOUT = int(os.environ.get("P_SERVER_WAIT_TIMEOUT", "600"))

P_NODE_PORT_LIST = os.environ.get("P_NODE_PORT_LIST") or ';'.join(
    ','.join(f"{host.strip()}:{BASE_PORT}" for host in group.split(',') if host.strip())
    for group in P_NODE_LIST.split(';') if group.strip()
)


__all__ = [
    "BASE_DIR",
    "OX_PATH",
    "OX_LOG_PATH",
    "BLOCK_RELEASE_DELAY",
    "PER_REQUEST_CONNECTION",
    "P_NODE_LIST",
    "CLUSTER_LIST",
    "CLUSTER_SIZE",
    "NODE_IP_SPECS",
    "BASE_PORT",
    "ZMQ_BASE_PORT",
    "P_SERVER_WAIT_TIMEOUT",
    "P_NODE_PORT_LIST",
]
