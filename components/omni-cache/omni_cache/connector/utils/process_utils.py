# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
import queue
import socket
import subprocess
import sys
import time
from typing import Optional


def handle_exception(future, logger) -> None:
    error = future.exception()
    if error:
        logger.error("Exception occurred in future: %s", error)
        raise error


def stop_logged_process(proc, subprocess_module=subprocess, timeout: float = 5.0) -> None:
    try:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess_module.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        try:
            if proc.stderr:
                proc.stderr.close()
        except Exception:
            pass


def wait_ports(host_ports, timeout_sec=1000, interval=0.1, socket_module=socket, time_module=time):
    remaining = set(host_ports)
    deadline = time_module.time() + timeout_sec

    def can_connect(host, port, timeout):
        sock = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
            return True
        except Exception:
            return False
        finally:
            try:
                sock.close()
            except Exception:
                pass

    while remaining and time_module.time() < deadline:
        ready = [hp for hp in list(remaining) if can_connect(hp[0], hp[1], interval)]
        for hp in ready:
            remaining.discard(hp)
        if remaining:
            time_module.sleep(interval)
    return len(remaining) == 0, remaining


def stdout_reader(pipe, q):
    for line in iter(pipe.readline, ''):
        q.put(line)
    q.put(None)
    pipe.close()


def stdout_printer(q: queue.Queue, file_path: Optional[str] = None, os_module=os, sys_module=sys) -> None:
    file_obj = None
    try:
        if file_path is not None:
            os_module.makedirs(file_path, exist_ok=True)
            log_file = os_module.path.join(file_path, "ox_log_d_client.log")
            file_obj = open(log_file, 'w', buffering=1)

        while True:
            try:
                line = q.get(timeout=1)
            except queue.Empty:
                continue
            if line is None:
                break

            if file_obj:
                file_obj.write(line)
                file_obj.flush()
            else:
                sys_module.__stdout__.write(line)
                sys_module.__stdout__.flush()
    finally:
        if file_obj is not None:
            file_obj.close()


__all__ = [
    "handle_exception",
    "stop_logged_process",
    "wait_ports",
    "stdout_reader",
    "stdout_printer",
]
