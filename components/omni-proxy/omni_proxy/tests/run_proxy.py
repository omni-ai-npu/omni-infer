# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
import sys
import signal
import subprocess
from pathlib import Path
import port_manager

CUR_DIR = Path(__file__).parent
proxy_script_path = "../omni_proxy.sh"

def generate_proxy_endpoints(port_list) -> str:
    return ",".join(f"127.0.0.1:{port}" for port in port_list)

def setup_proxy(proxy_port=7000, prefill_port_list=None, decode_port_list=None,
                encode_port_list=None,
                prefill_groups=None, decode_groups=None, dry_run=False,
                stream_ops="add",
                max_request_slots=None,
                worker_processes=1):
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = '123'
    if '/usr/sbin' not in env.get('PATH', ''):
        env['PATH'] = '/usr/sbin:' + env.get('PATH', '')

    prefill_list = generate_proxy_endpoints(prefill_port_list)
    decode_list = generate_proxy_endpoints(decode_port_list)
    try:
        cmd = [
            "bash", proxy_script_path,
            "--nginx-conf-file", f"{CUR_DIR}/nginx.conf",
            "--core-num", str(worker_processes),
            "--listen-port", f"{proxy_port}",
            "--prefill-endpoints", prefill_list,
            "--decode-endpoints", decode_list,
            "--log-file", f"{CUR_DIR}/nginx_error.log",
            "--log-level", "info",
            "--access-log-file", f"{CUR_DIR}/nginx_access.log",
            "--stream-ops", stream_ops,
            "--no-reuseport"
        ]
        if encode_port_list:
            cmd += ["--encode-endpoints", generate_proxy_endpoints(encode_port_list)]
        # else:
            # cmd += ["--omni-proxy-model-path", f"{CUR_DIR}/mock_model"]
        if prefill_groups:
            cmd.extend(["--omni-proxy-prefill-groups", prefill_groups])
        if decode_groups:
            cmd.extend(["--omni-proxy-decode-groups", decode_groups])
        if dry_run:
            cmd.extend(["--dry-run"])
        if max_request_slots is not None:
            cmd.extend(["--omni-proxy-max-request-slots", str(max_request_slots)])
        print(f"\n[SETUP] Starting proxy with command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[SETUP] Script succeeded. Output:\n{result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Setup script failed with exit code {e.returncode}.\n"
            f"STDERR: {e.stderr}\n"
            f"STDOUT: {e.stdout}"
        )
        print(error_msg)
        return -1

def teardown_proxy():
    try:
        cmd = [
            "bash", proxy_script_path,
            "--stop",
        ]
        print(f"\n[TEARDOWN] Stopping proxy with command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[TEARDOWN] Script succeeded. Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Teardown script failed with exit code {e.returncode}.\n"
            f"STDERR: {e.stderr}\n"
            f"STDOUT: {e.stdout}"
        )
        print(error_msg)

def graceful_quit_proxy(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError("PID must be positive")

    try:
        os.kill(pid, signal.SIGQUIT)
        logger.info(f"send SIGQUIT to PID {pid}, graceful quit nginx")
        return True
    except PermissionError:
        logger.error(f"no permission to send SIGQUIT to PID {pid}")
        return False
    except ProcessLookupError:
        logger.warning(f"PID {pid} has already exited")
        return False
    except Exception as e:
        logger.error(f"Exception when sending SIGQUIT to PID {pid}: {e}")
        return False

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:
        # Auto-detect: try EPD file first, fall back to PD file
        epd_file = os.path.join(CUR_DIR, "shared_ports_epd.json")
        if os.path.exists(epd_file):
            ports = port_manager.get_ports_from_file_epd()
        else:
            ports = port_manager.get_ports_from_file()
        proxy_port = ports["proxy_port"]
        if ports.get("encode"):
            # EPD mode
            encode_port_list = ports.get("encode", [])
            prefill_port_list = ports.get("prefill", [])
            decode_port_list = ports.get("decode", [])
            setup_proxy(proxy_port, prefill_port_list, decode_port_list, encode_port_list)
        else:
            # PD mode
            prefill_port_list = ports["prefill"]
            decode_port_list = ports["decode"]
            setup_proxy(proxy_port, prefill_port_list, decode_port_list)
    elif len(args) == 1 and args[0] == "stop":
        teardown_proxy()
    elif len(args) == 3:
        # EPD mode: <encode_num> <prefill_num> <decode_num>
        try:
            encode_num = int(args[0])
            prefill_num = int(args[1])
            decode_num = int(args[2])
            ports = port_manager.load_ports_epd(encode_num, prefill_num, decode_num)
            proxy_port = ports["proxy_port"]
            encode_port_list = ports.get("encode", [])
            prefill_port_list = ports.get("prefill", [])
            decode_port_list = ports.get("decode", [])
            setup_proxy(proxy_port, prefill_port_list, decode_port_list, encode_port_list)
        except ValueError as e:
            print(f"Error: All arguments must be valid numbers. Got: {args}")
            print("Usage: python run_proxy.py <encode_num> <prefill_num> <decode_num>")
            sys.exit(1)
    elif len(args) == 2:
        # PD mode: <prefill_num> <decode_num>
        try:
            prefill_num = int(args[0])
            decode_num = int(args[1])
            ports = port_manager.load_ports(prefill_num, decode_num)
            proxy_port = ports["proxy_port"]
            prefill_port_list = ports["prefill"]
            decode_port_list = ports["decode"]
            setup_proxy(proxy_port, prefill_port_list, decode_port_list)
        except ValueError as e:
            print(f"Error: All arguments must be valid numbers. Got: {args}")
            print("Usage: python run_proxy.py <prefill_num> <decode_num>")
            sys.exit(1)
    else:
        print(f"Error: Invalid arguments: {args}")
        print("Usage:")
        print("  python run_proxy.py <encode_num> <prefill_num> <decode_num>   # EPD mode")
        print("  python run_proxy.py <prefill_num> <decode_num>                # PD mode")
        print("  python run_proxy.py stop                                     # Stop")
        sys.exit(1)
