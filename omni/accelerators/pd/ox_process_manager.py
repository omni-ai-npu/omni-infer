import subprocess
import threading
import queue
import time
import os
from pathlib import Path
from typing import List, Optional

from vllm.logger import init_logger
logger = init_logger(__name__)

class OxProcessManager:
    def __init__(self, log_prefix: str = "ox"):
        self.process: Optional[subprocess.Popen] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.log_prefix = log_prefix
        self._startup_wait_seconds = 0.5

    def start(self, cmd: List[str], log_file_path: Optional[str] = None) -> subprocess.Popen:
        self._check_executable(cmd[0])
        proc = self._lauch_process(cmd)
        self._check_immediate_exit(proc)
        self._start_log_reader(proc, log_file_path)
        self._start_monitor(proc)
        self.process = proc
        logger.info(f"[{self.log_prefix}] process started successfully, PID: {proc.pid}")
        return proc

    def _check_executable(self, executable: str):
        import shutil
        exe_path = Path(executable)
        if not exe_path.is_absolute():
            exe_path = shutil.which(executable)
            if exe_path is None:
                raise FileNotFoundError(f"Executable files are not in the PATH: {executable}")
            exe_path = Path(exe_path)
        if not exe_path.exists():
            raise FileNotFoundError(f"Executable file does not exist: {exe_path}")
        if not os.access(exe_path, os.X_OK):
            raise PermissionError(f"Executable file does not have the execute permission: {exe_path}")

    def _lauch_process(self, cmd: List[str]) -> subprocess.Popen:
        try:
            return subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
        except Exception as e:
            logger.error(f"[{self.log_prefix}] startup failed: {e}")
            raise RuntimeError(f"OX process startup failed") from e

    def _check_immediate_exit(self, proc: subprocess.Popen):
        time.sleep(self._startup_wait_seconds)
        if proc.poll() is not None:
            stdout, _ = proc.communicate()
            logger.error(f"[{self.log_prefix}] process exit immediately after being started. Exit code: {proc.returncode}")
            if stdout:
                logger.error(f"[{self.log_prefix}] OX proc output:\n{stdout}")
            raise RuntimeError(f"OX process exit immediately after being started.")

    def _start_log_reader(self, proc: subprocess.Popen, log_file_path: Optional[str] = None):
        q = queue.Queue()
        t_read = threading.Thread(target=self._stdout_reader, args=(proc.stdout, q), daemon=True)
        if log_file_path:
            t_print = threading.Thread(target=self._stdout_printer_with_file, args=(q, log_file_path), daemon=True)
        else:
            t_print = threading.Thread(target=self._stdout_printer, args=(q,), daemon=True)
        t_read.start()
        t_print.start()
    
    def _start_monitor(self, proc: subprocess.Popen):
        def monitor():
            return_code = proc.wait()
            if return_code < 0:
                logger.error(f"[{self.log_prefix}] process terminated by signal (signal {-return_code})")
            else:
                logger.error(f"[{self.log_prefix}] process exit unexpectedly: {return_code}")
            os._exit(1)

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    @staticmethod
    def _stdout_reader(pipe, q):
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    q.put(line)
        except:
            pass
        finally:
            q.put(None)
    
    @staticmethod
    def _stdout_printer(q):
        while True:
            line = q.get()
            if line is None:
                break
            print(line, end='')
    
    @staticmethod
    def _stdout_printer_with_file(q, log_file_path):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, 'a') as f:
            while True:
                line = q.get()
                if line is None:
                    break
                print(line, end='')
                f.write(line)
                f.flush()
    
    def stop(self):
        if self.process and self.process.poll() is None:
            logger.info(f"[{self.log_prefix}] Stopping the process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()