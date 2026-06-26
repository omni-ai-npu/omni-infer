# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import subprocess

import requests

# Parse log into line style
def parse_log_line(line):

    line = line.strip()
    if not line.startswith("{") or not line.endswith("}"):
        return None  

    line = line[1:-1]  
    parts = []
    current = ""
    in_string = False 

    for char in line:
        if char == '"' and not in_string:
            in_string = True 
        elif char == '"' and in_string:
            in_string = False 
        elif char == "," and not in_string:
            parts.append(current)
            current = ""
        else:
            current += char

    if current:
        parts.append(current)

    result = {}
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.replace(".", "", 1).isdigit():
            value = float(value) if "." in value else int(value)
        result[key] = value

    return result


def fetch_post(url, headers, data):
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)  
        return {
            "url": url,
            "status": response.status_code,
            "text": response.text[:200] + "..." if len(response.text) > 200 else response.text
        }
    except requests.exceptions.RequestException as e:
        return {
            "url": url,
            "error": str(e)
        }

def get_ngx_pid():
    try:
        cmd = "ps -ef --sort=-lstart | grep 'nginx: master' | grep -v grep | head -n1 | awk '{print $2}'"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )

        if result.returncode == 0:
            ngx_mstr_pid = result.stdout.strip()
            print(f"[NGX ID] Script succeeded. Output:\n{result.stdout}")
            return ngx_mstr_pid
        
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Get nginx pid failed with exit code {e.returncode}.\n"
            f"STDERR: {e.stderr}\n"
            f"STDOUT: {e.stdout}"
        )
        print(error_msg)


def teardown_proxy_balance(ngx_pid):
    try:
        cmd = f"kill -QUIT {ngx_pid}"
        
        print(f"\n[TEARDOWN] Stopping proxy with command: {cmd}")
        result = subprocess.run(
            cmd,
            shell=True,
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