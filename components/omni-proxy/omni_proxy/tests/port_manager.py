# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import json
import os
import socket
import errno
from typing import Optional, Tuple

PORTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "shared_ports.json"
)

PORTS_FILE_EPD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "shared_ports_epd.json"
)


def _get_ephemeral_port_range() -> Optional[Tuple[int, int]]:
    path = "/proc/sys/net/ipv4/ip_local_port_range"
    try:
        with open(path, "r") as f:
            parts = f.read().strip().split()
        if len(parts) != 2:
            return None
        start, end = int(parts[0]), int(parts[1])

        if 1 <= start <= end <= 65535:
            return start, end
    except Exception:
        return None
    return None


def _is_in_ephemeral_range(port: int, ep_range: Optional[Tuple[int, int]]) -> bool:
    if not ep_range:
        return False
    start, end = ep_range
    return start <= port <= end


def find_free_port(used = None):
    ep_range = _get_ephemeral_port_range()
    start_port = (ep_range[0] - 1) if ep_range else 60999
    lower_bound = 7000

    def _can_bind(port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
            return True
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            return False

    for port1 in range(start_port, lower_bound, -1):
        port2 = port1 + 100
        if port2 > 65535:
            continue
        if _is_in_ephemeral_range(port1, ep_range) or _is_in_ephemeral_range(port2, ep_range):
            continue
        if used and (port1 in used or port2 in used):
            continue
        if _can_bind(port1) and _can_bind(port2):
            return port1

    raise RuntimeError("[WARNING] cannot find free port, please try again")


def find_n_free_ports(num_ports, used = None):
    ports = []
    used_set = used if used is not None else set()
    for _ in range(num_ports):
        port = find_free_port(used_set)
        ports.append(port)
        used_set.add(port)
        used_set.add(port+100)
    return ports


def find_free_port_excluding_existing(used = None):
    if not os.path.exists(PORTS_FILE):
        return find_free_port()
    
    try:
        with open(PORTS_FILE, "r") as f:
            ports_config = json.load(f)
        
        exclude_ports = []
        if "prefill" in ports_config:
            exclude_ports.extend(ports_config["prefill"])
        if "decode" in ports_config:
            exclude_ports.extend(ports_config["decode"])
        if "proxy_port" in ports_config:
            exclude_ports.append(ports_config["proxy_port"])
        exclude_set = set(exclude_ports)

        if used:
            exclude_set.add(used)
        exclude_set.update(p + 100 for p in exclude_ports)
        max_attempts = 100
        
        for _ in range(max_attempts):
            port = find_free_port(exclude_set)
            if port not in exclude_set:
                return port
        
        raise RuntimeError(f"cannot find port in {max_attempts} times")
    
    except (json.JSONDecodeError, KeyError, TypeError):
        return find_free_port()


def ensure_ports_file(prefill_num=1, decode_num=1):
    if os.path.exists(PORTS_FILE):
        try:
            with open(PORTS_FILE, "r") as f:
                ports = json.load(f)
            
            if (isinstance(ports, dict) and 
                "proxy_port" in ports and
                "prefill" in ports and "decode" in ports and
                len(ports["prefill"]) == prefill_num and
                len(ports["decode"]) == decode_num):
                return ports
            print(f"port num does not match in {PORTS_FILE}, regenerating...")

        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"port conf file {PORTS_FILE} format invalid, regenerating...")
    used = set()
    proxy_port = find_free_port(used)
    used.add(proxy_port)
    prefill_ports = find_n_free_ports(prefill_num, used)
    used.update(set(prefill_ports))
    used.update(p + 100 for p in prefill_ports)
    decode_ports = find_n_free_ports(decode_num, used)
    
    ports = {
        "proxy_port": proxy_port,
        "prefill": prefill_ports,
        "decode": decode_ports
    }
    print(f"generate {ports=} and store in {PORTS_FILE}")

    temp_file = PORTS_FILE + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(ports, f)
    os.replace(temp_file, PORTS_FILE)
    
    return ports


def load_ports(prefill_num=1, decode_num=1):
    return ensure_ports_file(prefill_num, decode_num)


def get_ports_from_file():
    if not os.path.exists(PORTS_FILE):
        raise FileNotFoundError(f"Port configuration file does not exist: {PORTS_FILE}")

    try:
        with open(PORTS_FILE, "r") as f:
            ports = json.load(f)

        required_fields = ["proxy_port", "prefill", "decode"]
        for field in required_fields:
            if field not in ports:
                raise KeyError(
                    f"Port configuration file is missing required field: {field}"
                )

        return ports

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Port configuration file has invalid format (JSON parsing error): {e}"
        )
    except KeyError as e:
        raise KeyError(
            f"Port configuration file is missing required field: {e}"
            )
    except Exception as e:
        raise RuntimeError(
            f"Unknown error occurred while reading port configuration file: {e}"
        )


def get_ports_from_file_epd():
    """Read EPD ports from the dedicated shared_ports_epd.json file."""
    if not os.path.exists(PORTS_FILE_EPD):
        raise FileNotFoundError(f"EPD port configuration file does not exist: {PORTS_FILE_EPD}")

    try:
        with open(PORTS_FILE_EPD, "r") as f:
            ports = json.load(f)

        required_fields = ["proxy_port", "encode", "prefill", "decode"]
        for field in required_fields:
            if field not in ports:
                raise KeyError(
                    f"EPD port configuration file is missing required field: {field}"
                )

        return ports

    except json.JSONDecodeError as e:
        raise ValueError(
            f"EPD port configuration file has invalid format (JSON parsing error): {e}"
        )
    except KeyError as e:
        raise KeyError(
            f"EPD port configuration file is missing required field: {e}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Unknown error occurred while reading EPD port configuration file: {e}"
        )


def ensure_ports_file_epd(encode_num=1, prefill_num=1, decode_num=1):
    """
    Ensure ports file exists with encode + prefill + decode ports.
    Same file as PD, but adds 'encode' field.
    """
    if os.path.exists(PORTS_FILE_EPD):
        try:
            with open(PORTS_FILE_EPD, "r") as f:
                ports = json.load(f)

            if (
                isinstance(ports, dict)
                and "proxy_port" in ports
                and "encode" in ports
                and "prefill" in ports
                and "decode" in ports
                and len(ports.get("encode", [])) == encode_num
                and len(ports["prefill"]) == prefill_num
                and len(ports["decode"]) == decode_num
            ):
                return ports
            print(f"port num does not match in {PORTS_FILE_EPD}, regenerating...")

        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"port conf file {PORTS_FILE_EPD} format invalid, regenerating...")

    used = set()
    proxy_port = find_free_port(used)
    used.add(proxy_port)
    encode_ports = find_n_free_ports(encode_num, used) if encode_num > 0 else []
    used.update(encode_ports)
    used.update(p + 100 for p in encode_ports)
    prefill_ports = find_n_free_ports(prefill_num, used)
    used.update(set(prefill_ports))
    used.update(p + 100 for p in prefill_ports)
    decode_ports = find_n_free_ports(decode_num, used)

    ports = {
        "proxy_port": proxy_port,
        "encode": encode_ports,
        "prefill": prefill_ports,
        "decode": decode_ports,
    }
    print(f"generate {ports=} and store in {PORTS_FILE_EPD}")

    temp_file = PORTS_FILE_EPD + ".tmp"
    with open(temp_file, "w") as f:
        json.dump(ports, f)
    os.replace(temp_file, PORTS_FILE_EPD)

    return ports


def load_ports_epd(encode_num=1, prefill_num=1, decode_num=1):
    """Load or create ports file with encode + prefill + decode ports."""
    return ensure_ports_file_epd(encode_num, prefill_num, decode_num)
