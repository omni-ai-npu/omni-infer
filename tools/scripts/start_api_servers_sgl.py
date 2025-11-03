import argparse
import copy
import logging
import multiprocessing as mp
import os
import random
import signal
import sys

import time
from typing import List

import omni.adaptors.sglang.patches.model_patch
import requests
from setproctitle import setproctitle

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_port_available

def setup_logger():
    logger = logging.getLogger("router")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[Router (Python)] %(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

logger = setup_logger()

def run_server(server_args, dp_rank):
    os.setpgrp()

    setproctitle("sglang::server")

    os.environ["SGLANG_DP_RANK"] = str(dp_rank)

    launch_server(server_args)

def launch_server_process(
        server_args: ServerArgs, worker_port: int, dp_id: int
) -> mp.Process:
    """Launch a single server process with the given args and port."""
    server_args = copy.deepcopy(server_args)
    server_args.port = server_args.port + dp_id
    proc = mp.Process(target=run_server, args=(server_args,dp_id))
    proc.start()
    return proc

def wait_for_server_health(host:str, port:int, timeout:int = 300) -> bool:
    """Wait for server to be healthy by checking /health endpoint."""
    start_time = time.perf_counter()
    url = f"http://{host}:{port}/health"

    while time.perf_counter() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False

def find_available_ports(base_port: int, count:int) -> List[int]:
    """Find consecutive available ports starting from base_port."""
    available_ports = []
    current_port = base_port

    while len(available_ports) < count:
        if is_port_available(current_port):
            available_ports.append(current_port)
        current_port += 1
    return available_ports

def cleanup_processes(processes: List[mp.Process]):
    for process in processes:
        logger.info(f"Terminating process group {process.pid}")
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            # Process group may already be terminated.
            pass

    logger.info("All process groups terminated.")

def main():
    # CUDA runtime isn't fork-safe, which can lead to subtle bugs or crashes
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Launch SGLang router and server processes"
    )

    ServerArgs.add_cli_args(parser)

    parser.add_argument(
        "--router-dp-worker-base-port",
        type=int,
        default=31000,
        help="Base port number for data parallel workers",
    )

    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)

    # Find available ports for workers
    worker_ports = find_available_ports(
        args.router_dp_worker_base_port, server_args.dp_size // server_args.nnodes
    )

    # Start server processes
    server_processes = []
    for i, worker_port in enumerate(worker_ports):
        server_args.api_server_id = i + (server_args.dp_size // server_args.nnodes)*server_args.node_rank

        proc = launch_server_process(server_args, worker_port, i)
        server_processes.append(proc)

    signal.signal(signal.SIGINT, lambda sig, frame: cleanup_processes(server_processes))
    signal.signal(
        signal.SIGTERM, lambda sig, frame: cleanup_processes(server_processes)
    )
    signal.signal(
        signal.SIGQUIT, lambda sig, frame: cleanup_processes(server_processes)
    )

if __name__ == "__main__":
    main()