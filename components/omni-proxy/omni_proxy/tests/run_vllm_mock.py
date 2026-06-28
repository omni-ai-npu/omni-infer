# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import sys
import subprocess
import time
from pathlib import Path
import port_manager


# Configuration
LOG_FILE_PREFIX = "server"
APP_START_MARKER = "Application startup complete."
STARTUP_TIMEOUT = 120  # seconds
tp = 1
dp = 1

CUR_DIR = Path(__file__).parent
model_path = f"{CUR_DIR}/mock_model/"
COVRC_DIR = os.path.abspath(os.path.dirname(__file__) + "/../../")
TOP_DIR = os.path.abspath(os.path.dirname(__file__) + "/../../../")


def graceful_kill_vllm(timeout=10):
    """
    Gracefully kill vllm processes:
    1. Send SIGTERM (pkill -15 vllm)
    2. Wait up to `timeout` seconds for processes to exit
    3. If still running, send SIGKILL (pkill -9 vllm)
    """
    # Step 1: Send SIGTERM
    print("Sending SIGTERM to vllm processes...")
    subprocess.run(["pkill", "-f", "-15", "vllm"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Step 2: Wait for processes to exit
    for _ in range(timeout):
        # Check if any vllm process is still running
        result = subprocess.run(["pgrep", "-f", "vllm"], capture_output=True)
        if result.returncode != 0:
            print("All vllm processes exited gracefully.")
            return
        time.sleep(1)

    # Step 3: Timeout reached, force kill
    print(f"vllm processes did not exit within {timeout} seconds. Sending SIGKILL...")
    subprocess.run(["pkill", "-f", "-9", "vllm"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Optional: Verify they're gone
    time.sleep(1)
    result = subprocess.run(["pgrep", "-f", "vllm"], capture_output=True)
    if result.returncode == 0:
        print("Warning: vllm processes still running (may require manual kill)")
    else:
        print("All vllm processes killed forcefully.")


def setup_vllm(is_prefill, port_list, log_file_prefix=None):
    if log_file_prefix is None:
        log_file_prefix = LOG_FILE_PREFIX
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "1"
    env["VLLM_CPU_KVCACHE_SPACE"] = "48"
    env['RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES'] = "1"
    env['PYTHONHASHSEED'] = '123'
    env["COVERAGE_PROCESS_START"] = f"{COVRC_DIR}/.coveragerc"
    env["PYTHONPATH"] = f"{CUR_DIR}" + ":" + env.get("PYTHONPATH", "")

    env["NO_NPU_MOCK"] = "1"
    env["RANDOM_MODE"] = "1"
    env["KV_CACHE_MODE"] = "1"

    node_type = "decode"
    if is_prefill:
        node_type = "prefill"
        os.environ["PREFILL_PROCESS"] = "1"

    proc_list = []
    log_list = []
    idx = 0
    # env["VLLM_PLUGINS"] = "omni-npu,omni_npu_patches,omni_custom_models" # use npu
    env["VLLM_PLUGINS"] = "omni_custom_models" # use cpu
    for port in port_list:
        KV_EVENTS_CONFIG='{"enable_kv_cache_events":true,"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:' + str(port + 100) + '"}'
        cmd = [
            "vllm", "serve", model_path,
            "--port", f"{port}",
            "--max-num-seqs", "128",
            "--max_model_len", "8000", 
            "--tensor-parallel-size", f"{tp}",
            "--data_parallel_size", f"{dp}",
            "--gpu_memory_utilization", "0.15",
            "--trust_remote_code",
            "--served-model-name", "qwen",
            "--dtype", "float32",
            "--distributed-executor-backend", "mp",
            "--block_size", "128",
            "--kv-events-config", f"{KV_EVENTS_CONFIG}",
            "--enable-chunked-prefill",
            "--enable-prefix-caching",
            "--enforce-eager",
        ]
        log_file = CUR_DIR / Path(f"{node_type}_{log_file_prefix}_{idx}.log")
        log_list.append(log_file)

        # Clean existing log
        if log_file.exists():
            log_file.unlink()
        
        if idx == 0:
            print(f"\n[SETUP] Starting {len(port_list)} {node_type} background processes with command: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        print(f"[SETUP] Starting {node_type} instance {idx}, pid {proc.pid}, port {port} -> {log_file}")
        proc_list.append(proc)
        idx += 1
    return proc_list, log_list


def cleanup_subprocess(processes):
    for i, proc in enumerate(processes):
        if proc.poll() is None:
            print(f"[TEARDOWN] Terminating instance {i} pid {proc.pid}...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"[TEARDOWN] Killing instance {i} pid {proc.pid}...")
                proc.kill()
                proc.wait()
        else:
            print(f"[TEARDOWN] Instance {i} pid {proc.pid} already exited (code: {proc.returncode}).")
    # Optional: auto-cleanup logs
    # for lf in log_files:
    #     if lf.exists():
    #         lf.unlink()


def start_vllm_mock(prefill_num=0, decode_num=0):
    processes = []
    log_files = []

    if prefill_num == 0 and decode_num == 0:
        ports = port_manager.get_ports_from_file()
    else:
        ports = port_manager.load_ports(prefill_num, decode_num)
    prefill_port_list = ports["prefill"]
    decode_port_list = ports["decode"]
    prefill_num = len(prefill_port_list)
    decode_num = len(decode_port_list)

    proc_list, log_list = setup_vllm(True, prefill_port_list)
    processes.extend(proc_list)
    log_files.extend(log_list)
    proc_list, log_list = setup_vllm(False, decode_port_list)
    processes.extend(proc_list)
    log_files.extend(log_list)

    total_node_num = prefill_num + decode_num
    # --- Wait for all instances to emit "APP started" ---
    start_time = time.time()
    ready_status = [False] * total_node_num

    while time.time() - start_time < STARTUP_TIMEOUT:
        all_ready = True
        for i, (proc, log_file) in enumerate(zip(processes, log_files)):
            if ready_status[i]:
                continue

            # Check if process crashed
            if proc.poll() is not None:
                cleanup_subprocess(processes)
                print(f"Instance {i} pid {proc.pid} exited early with code {proc.returncode}. Check {log_file}")
                return None

            # Check log for start marker
            if log_file.exists():
                with open(log_file, "r") as f:
                    if APP_START_MARKER in f.read():
                        ready_status[i] = True
                        print(f"[SETUP] Instance {i} pid {proc.pid} started successfully.")

            all_ready = all_ready and ready_status[i]

        if all_ready:
            print(f"time cost:{time.time() - start_time}")
            break

        time.sleep(1)

    # --- Handle timeout ---
    if not all(ready_status):
        failed = [i for i, r in enumerate(ready_status) if not r]
        cleanup_subprocess(processes)
        print(
            f"{len(failed)} instance(s) did not start within {STARTUP_TIMEOUT}s: {failed}. "
            f"Check logs: {[str(log_files[i]) for i in failed]}"
        )
        return None

    print(f"[SETUP] All {total_node_num} instances are ready.")
    return processes


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        start_vllm_mock()
    elif len(args) == 2:
        try:
            numeric_args = []
            for arg in args:
                numeric_args.append(int(arg))
            start_vllm_mock(*numeric_args)
        except ValueError as e:
            print(f"Error: All four arguments must be valid numbers. Got: {args}")
            print("Usage: python script.py <prefill_num> <decode_num>")
            sys.exit(1)
    elif len(args) == 1 and args[0] == "stop":
        graceful_kill_vllm()
    else:
        print(f"Error: Invalid arguments: {args}")
        print("Usage:")
        print("  python run_vllm_mock.py <prefill_num> <decode_num>   # Start with 2 numbers")
        print("  python run_vllm_mock.py stop                         # Stop/cleanup")
        sys.exit(1)
        
