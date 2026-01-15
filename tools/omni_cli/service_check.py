"""
vLLM C/P/D Instances Status Checker
Checks proxy, and separated prefill and decode instances via SSH
"""

import subprocess
import requests
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple
import threading
from pathlib import Path
import yaml
import curses


def print_node_list(inventory_path: str) -> None:
    """
    Print the current node list including role, name, IP address, and status.

    Args:
        inventory_path: Path to the inventory YAML file
    """
    inv_file = Path(inventory_path).expanduser().resolve()
    with open(inv_file, "r", encoding="utf-8") as f:
        inv = yaml.safe_load(f)

    children = inv.get("all", {}).get("children", {})

    def curses_run(stdscr):

        screen_lock = threading.Lock()
        #stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        _height, width = stdscr.getmaxyx()
        stdscr.addstr(0, 0, f"{'Role':<5} | {'Name':<5} | {'IP Address':<15} | {'ascend_rt_visible_devices':<40} | {'status':<8}")
        stdscr.addstr(1, 0, "-" * width)

        threads = []
        row = 2

        for role, role_data in children.items():
            if role not in ["C", "D", "P"]:
                continue

            hosts = role_data.get("hosts", {})

            for host_name, host_data in hosts.items():
                ip_address = host_data.get("ansible_host", "N/A")
                ascend_rt_visible_devices = host_data.get("ascend_rt_visible_devices", "N/A")

                env = host_data.get("env", {})

                log_path = env.get("LOG_PATH")
                sub_path = host_name + "/server_0.log" if host_name != "c0" else "nginx/nginx_error.log"
                log_file = log_path + "/" + sub_path

                port = env.get("API_PORT")

                ssh_user = host_data.get("ansible_user", "")
                ssh_private_key = host_data.get("ansible_ssh_private_key_file", "")
                ssh_password = host_data.get("ansible_ssh_pass","")

                status_checker = StatusChecker(ip_address, port, log_file, ssh_user, ssh_password, ssh_private_key)
                thread = status_checker.check_instance_health_and_print(screen_lock, row, stdscr)
                threads.append(thread)
                # time.sleep(0.5)

                stdscr.addstr(row, 0, f"{role:<5} | {host_name:<5} | {ip_address:<15} | {ascend_rt_visible_devices:<40} | LOADING...")

                row += 1


        stdscr.addstr(row, 0, "-" * width)
        stdscr.refresh()

        for thread in threads:
            thread.join()

        curses.nocbreak()
        # curses.echo()

        message = "Press Enter to exit..."
        stdscr.addstr(row+1, 0, message)
        stdscr.refresh()

        curses.flushinp()
        stdscr.getch()
        #curses.endwin()

    curses.wrapper(curses_run)

class ServiceStatus(Enum):
    RUNNING = "running"
    PENDING = "pending"
    FAILED = "failed"
    UNKNOWN = "unknown"

class StatusChecker:
    def __init__(self, 
                 host: str,
                 port: int,
                 log_file: str,
                 ssh_user: str = "root",
                 ssh_pass: Optional[str] = None,
                 ssh_key_path: Optional[str] = None):
        
        self.host = host
        self.port = port
        self.log_file = log_file

        self.ssh_user = ssh_user
        self.ssh_pass = ssh_pass
        self.ssh_key_path = ssh_key_path
    
    def run_ssh_command(self, command: str) -> Tuple[bool, str]:
        """Execute command on remote host via SSH"""
        try:
            if self.ssh_pass:
                ssh_cmd = ["sshpass", "-p", self.ssh_pass, "ssh"]
            else:
                ssh_cmd = ["ssh"]
                if self.ssh_key_path:
                    ssh_cmd.extend(["-i", self.ssh_key_path])
            
            ssh_cmd.extend([f"{self.ssh_user}@{self.host}", command])
            
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return True, result.stdout.strip()

            else:
                return False, result.stderr.strip()
                
        except subprocess.TimeoutExpired:
            return False, "SSH connection timeout"
        except Exception as e:
            return False, f"SSH error: {e}"

    def check_log_error(self) -> Tuple[bool, str]:
        command = f"tail -100 {self.log_file} | grep -E -i -q '^error|\[error\]|exception|fail|timeout|oom|segfault' 2>/dev/null && exit 1 || exit 0 "
        success, output = self.run_ssh_command(command)

        return success, output
    
    def check_remote_api(self) -> Tuple[bool, str]:
        """Check remote API endpoints"""
        # Try direct connection if possible, otherwise use SSH tunnel
        endpoints = [
            # "/health",
            "/v1/models", 
            "/docs",
            "/metrics"
        ]
        
        for endpoint in endpoints:
            try:
                # First try direct connection
                url = f"http://{self.host}:{self.port}{endpoint}"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    if endpoint == "/v1/models":
                        data = response.json()
                        model_count = len(data.get('data', []))
                        return True, f"API healthy - {model_count} models loaded"
                    return True, f"API healthy - {endpoint}"
                    
            except requests.exceptions.RequestException:
                continue
        
        # If direct connection fails, try via SSH command
        command = f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{self.port}/v1/models || echo 'FAILED'"
        success, output = self.run_ssh_command(command)
        
        if success and output.strip() == "200":
            return True, "API healthy (via local check)"
        
        return False, "API not responding"
    
    def check_instance_health_and_print(self, screen_lock: threading.Lock, row: int, stdscr) -> Dict:
        """Comprehensive health check for a single instance"""
        # print(f"Checking {instance_type.value} instance on {host}:{port}...")

        def run_update():
            status = {
                "host": self.host,
                "port": self.port,
                "api_healthy": False,
                "api_info": "",
                "no_log_error": False,
                "log_info": "",
                "final_status": ServiceStatus.UNKNOWN
            }

            # Check log
            status["no_log_error"], status["log_info"] = self.check_log_error()
            
            # Check API
            status["api_healthy"], status["api_info"] = self.check_remote_api()
            
            # Determine final status
            if status["api_healthy"]:
                status["final_status"] = ServiceStatus.RUNNING
            elif not status["no_log_error"]:
                status["final_status"] = ServiceStatus.FAILED
            else:
                status["final_status"] = ServiceStatus.PENDING

            with screen_lock:
                stdscr.addstr(row, 77, " " * 10)
                stdscr.addstr(row, 77, f'{status["final_status"].name:<8}')
                stdscr.refresh()
            
        thread = threading.Thread(target=run_update)
        thread.daemon = True
        thread.start()
        return thread
        