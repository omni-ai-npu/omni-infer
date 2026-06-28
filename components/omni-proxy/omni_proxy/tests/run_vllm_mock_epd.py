# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
EPD Mock Server Runner for omni-proxy integration tests.

Launches encode, prefill, and decode mock servers using vllm_mock_server_epd.py.
Mock servers implement PD/EPD separation modes for testing.

Usage:
    python run_vllm_mock_epd.py <encode_num> <prefill_num> <decode_num>   # Start
    python run_vllm_mock_epd.py stop                                        # Stop
"""

import os
import sys
import subprocess
import time
import socket
from pathlib import Path
import port_manager


def is_port_listening(port):
    """Check if a port is listening."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(("127.0.0.1", port))
            return result == 0
    except Exception:
        return False

# Configuration
LOG_FILE_PREFIX = "server"
APP_START_MARKER = "Application startup complete."
STARTUP_TIMEOUT = 120  # seconds
tp = 1
dp = 1

CUR_DIR = Path(__file__).parent
EPD_MOCK_SERVER = str(CUR_DIR / "vllm_mock_server_epd.py")
COVRC_DIR = os.path.abspath(os.path.dirname(__file__) + "/../../")
TOP_DIR = os.path.abspath(os.path.dirname(__file__) + "/../../../")


def graceful_kill_mock(timeout=10):
    """
    Gracefully kill mock server processes.
    """
    print("Sending SIGTERM to mock server processes...")
    subprocess.run(
        ["pkill", "-f", "-15", "vllm_mock_server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(timeout):
        result = subprocess.run(["pgrep", "-f", "vllm_mock_server"], capture_output=True)
        if result.returncode != 0:
            print("All mock server processes exited gracefully.")
            return
        time.sleep(1)

    print(f"Mock server processes did not exit within {timeout}s. Sending SIGKILL...")
    subprocess.run(
        ["pkill", "-f", "-9", "vllm_mock_server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def start_mock_server(encode_num, prefill_num, decode_num, prefill_exec_time_multiplier=1.0):
    """
    Start encode, prefill, and decode mock servers.

    Args:
        encode_num: Number of encode endpoints
        prefill_num: Number of prefill endpoints
        decode_num: Number of decode endpoints

    Returns:
        List of subprocess.Popen objects for all started processes
    """
    if encode_num == 0 and prefill_num == 0 and decode_num == 0:
        ports = port_manager.get_ports_from_file_epd()
    else:
        ports = port_manager.load_ports_epd(encode_num, prefill_num, decode_num)

    encode_port_list = ports.get("encode", [])
    prefill_port_list = ports.get("prefill", [])
    decode_port_list = ports.get("decode", [])

    all_ports = encode_port_list + prefill_port_list + decode_port_list

    # Check if servers are already running on these ports
    all_ready = all(is_port_listening(port) for port in all_ports)
    if all_ready:
        print(f"[SETUP] EPD mock servers already running on all {len(all_ports)} ports.")
        # Return a dummy process list to indicate success
        return [None]  # None indicates servers were already running

    port_str = ",".join(f"127.0.0.1:{p}" for p in all_ports)

    cmd = [
        "python3", EPD_MOCK_SERVER,
        "--encode-endpoints", ",".join(f"127.0.0.1:{p}" for p in encode_port_list),
        "--prefill-endpoints", ",".join(f"127.0.0.1:{p}" for p in prefill_port_list),
        "--decode-endpoints", ",".join(f"127.0.0.1:{p}" for p in decode_port_list),
        "--prefill-exec-time-multiplier", str(prefill_exec_time_multiplier),
    ]

    if encode_num > 0:
        print(f"\n[SETUP] Starting EPD mock server with {encode_num} encode, {prefill_num} prefill, {decode_num} decode")
        print(f"[SETUP] Command: {' '.join(cmd)}")

    log_file = CUR_DIR / "epd_mock_server.log"
    if log_file.exists():
        log_file.unlink()

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "123"
    env["PYTHONPATH"] = str(CUR_DIR) + ":" + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    print(f"[SETUP] EPD mock server started, pid {proc.pid}")

    # Wait for server to be ready by checking ports
    # Note: vllm_mock_server_epd.py spawns child processes and parent exits with code 0
    # So we need to check if ports are listening instead of checking process status
    start_time = time.time()
    all_ports = encode_port_list + prefill_port_list + decode_port_list

    while time.time() - start_time < STARTUP_TIMEOUT:
        # Check if all ports are listening
        all_ready = True
        for port in all_ports:
            if not is_port_listening(port):
                all_ready = False
                break

        if all_ready:
            print(f"[SETUP] EPD mock server is ready (all {len(all_ports)} ports listening).")
            return [proc]

        # Check if process crashed
        if proc.poll() is not None and proc.returncode != 0:
            print(f"[ERROR] Mock server exited early with code {proc.returncode}. Check {log_file}")
            return None

        time.sleep(1)

    # Final check - if ports are ready even though loop exited
    all_ready = all(is_port_listening(port) for port in all_ports)
    if all_ready:
        print(f"[SETUP] EPD mock server is ready (ports confirmed after timeout).")
        return [proc]

    print(f"[ERROR] Mock server did not start within {STARTUP_TIMEOUT}s. Check {log_file}")
    proc.terminate()
    return None


def cleanup_subprocess(processes):
    """Stop all started mock server processes."""
    if not processes:
        return
    for i, proc in enumerate(processes):
        if proc is None:
            # Server was already running, don't kill it
            continue
        if proc.poll() is None:
            print(f"[TEARDOWN] Terminating mock server instance {i} pid {proc.pid}...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"[TEARDOWN] Killing mock server instance {i} pid {proc.pid}...")
                proc.kill()
                proc.wait()
        else:
            print(f"[TEARDOWN] Instance {i} pid {proc.pid} already exited (code: {proc.returncode}).")

    # Use the standard stop command to clean up remaining child processes
    print("[TEARDOWN] Stopping EPD mock server via run_vllm_mock_epd.py stop...")
    subprocess.run(["python3", str(CUR_DIR / "run_vllm_mock_epd.py"), "stop"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[TEARDOWN] EPD mock server cleanup complete.")


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        # Default: start with 1 encode, 2 prefill, 2 decode
        processes = start_mock_server(1, 2, 2)
        if processes is None:
            print("Failed to start mock server")
            sys.exit(1)
        print("Mock server started. Press Ctrl+C to stop...")
        try:
            processes[0].wait()
        except KeyboardInterrupt:
            graceful_kill_mock()
    elif len(args) == 1 and args[0] == "stop":
        graceful_kill_mock()
    elif len(args) == 3:
        try:
            encode_num = int(args[0])
            prefill_num = int(args[1])
            decode_num = int(args[2])
            processes = start_mock_server(encode_num, prefill_num, decode_num)
            if processes is None:
                print("Failed to start mock server")
                sys.exit(1)
            print("Mock server started successfully.")
        except ValueError as e:
            print(f"Error: All arguments must be valid numbers. Got: {args}")
            print("Usage: python run_vllm_mock_epd.py <encode_num> <prefill_num> <decode_num>")
            sys.exit(1)
    else:
        print(f"Error: Invalid arguments: {args}")
        print("Usage:")
        print("  python run_vllm_mock_epd.py <encode_num> <prefill_num> <decode_num>   # Start")
        print("  python run_vllm_mock_epd.py stop                                        # Stop")
        sys.exit(1)