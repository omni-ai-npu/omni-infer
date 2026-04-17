#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Unit tests for OxProcessManager — verifies connector can detect ox failures.
#
# Usage:
#   python test_ox_monitor.py                     # run all tests
#   python test_ox_monitor.py startup_fail        # run specific test
#   python test_ox_monitor.py runtime_crash
#   python test_ox_monitor.py signal_crash
#
# Available test cases:
#   startup_fail   — ox exits immediately after launch
#                    Expected: OxProcessManager.start() raises RuntimeError
#   runtime_crash  — ox runs for a while then exits with non-zero code
#                    Expected: process is killed by os._exit(1) from monitor thread
#   signal_crash   — ox runs then kills itself with SIGABRT
#                    Expected: process is killed by os._exit(1) from monitor thread

import logging
import os
import sys
import time
import subprocess
import argparse

# ---------------------------------------------------------------------------
# Paths (needed early for module loading)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OX_DIR = os.path.join(BASE_DIR, "ox")

# ---------------------------------------------------------------------------
# Mock vllm.logger so we can import OxProcessManager without a real vllm install.
# We use importlib to load ox_process_manager directly by file path to avoid
# triggering the pd/__init__.py which has heavy vllm dependencies.
# ---------------------------------------------------------------------------
import importlib.util

def _mock_init_logger(name):
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)
    if not _logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"))
        _logger.addHandler(_handler)
    return _logger

# Inject mock vllm.logger before loading the module
import types
sys.modules["vllm"] = types.ModuleType("vllm")
_vllm_logger = types.ModuleType("vllm.logger")
_vllm_logger.init_logger = _mock_init_logger
sys.modules["vllm.logger"] = _vllm_logger

# Load OxProcessManager directly from file to bypass pd/__init__.py
_mod_path = os.path.join(BASE_DIR, "ox_process_manager.py")
_spec = importlib.util.spec_from_file_location("ox_process_manager", _mod_path)
_ox_pm_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ox_pm_mod)
OxProcessManager = _ox_pm_mod.OxProcessManager

MOCK_OX = {
    "startup_fail":   os.path.join(OX_DIR, "mock_ox_startup_fail"),
    "runtime_crash":  os.path.join(OX_DIR, "mock_ox_runtime_crash"),
    "signal_crash":   os.path.join(OX_DIR, "mock_ox_signal_crash"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

# Subprocess env — no special PYTHONPATH needed since we load by file path
_SUBPROCESS_ENV = os.environ.copy()


def run_test_subprocess(test_name: str, timeout: int = 15):
    """Run a single test case in a subprocess so os._exit() doesn't kill us."""
    result = subprocess.run(
        [sys.executable, __file__, "--inner", test_name],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_SUBPROCESS_ENV,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def test_startup_fail():
    """
    ox exits immediately → OxProcessManager.start() should raise RuntimeError
    in _check_immediate_exit.
    """
    mgr = OxProcessManager(log_prefix="test-startup-fail")
    cmd = [MOCK_OX["startup_fail"]]
    try:
        mgr.start(cmd)
        print("  start() returned without exception — unexpected")
        return False
    except RuntimeError as e:
        if "exit immediately" in str(e):
            print(f"  Caught expected RuntimeError: {e}")
            return True
        else:
            print(f"  RuntimeError message mismatch: {e}")
            return False
    except Exception as e:
        print(f"  Unexpected exception type: {type(e).__name__}: {e}")
        return False


def test_runtime_crash():
    """
    ox runs for ~2s then exits with code 1 → monitor thread should call
    os._exit(1), killing this process. We should never reach the return.
    """
    mgr = OxProcessManager(log_prefix="test-runtime-crash")
    cmd = [MOCK_OX["runtime_crash"]]
    try:
        mgr.start(cmd)
        # Wait long enough for the mock ox to crash (it sleeps 2s).
        # If monitor works, os._exit(1) kills us before this sleep finishes.
        time.sleep(5)
        print("  Process survived 5s after ox crash — monitor did NOT kill us")
        return False
    except Exception as e:
        print(f"  Unexpected exception (should have been os._exit): {e}")
        return False


def test_signal_crash():
    """
    ox runs for ~2s then raises SIGABRT → monitor thread should call
    os._exit(1), killing this process. We should never reach the return.
    """
    mgr = OxProcessManager(log_prefix="test-signal-crash")
    cmd = [MOCK_OX["signal_crash"]]
    try:
        mgr.start(cmd)
        time.sleep(5)
        print("  Process survived 5s after ox signal crash — monitor did NOT kill us")
        return False
    except Exception as e:
        print(f"  Unexpected exception (should have been os._exit): {e}")
        return False


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------
TESTS = {
    "startup_fail":  test_startup_fail,
    "runtime_crash": test_runtime_crash,
    "signal_crash":  test_signal_crash,
}

# For runtime_crash and signal_crash, the monitor calls os._exit(1) which
# kills the entire process. We detect success by the subprocess exiting
# with code 1 (not 0, and not killed by our timeout).
EXPECTED_EXIT_CODE = {
    "startup_fail":  0,   # test function returns True/False normally
    "runtime_crash": 1,   # os._exit(1) from monitor thread
    "signal_crash":  1,   # os._exit(1) from monitor thread
}


def run_outer():
    """Outer runner: each test runs in a subprocess so os._exit() is isolated."""
    parser = argparse.ArgumentParser(description="OxProcessManager unit tests")
    parser.add_argument("test_name", nargs="?", default="all",
                        choices=list(TESTS.keys()) + ["all"],
                        help="Which test to run (default: all)")
    args = parser.parse_args()

    tests_to_run = list(TESTS.keys()) if args.test_name == "all" else [args.test_name]
    results = {}

    for name in tests_to_run:
        print(f"\n{'='*60}")
        print(f"  TEST: {name}")
        print(f"{'='*60}")

        expected_code = EXPECTED_EXIT_CODE[name]

        if expected_code == 1:
            # These tests end with os._exit(1) — run in subprocess
            try:
                proc_result = run_test_subprocess(name, timeout=15)
                actual_code = proc_result.returncode
                if actual_code == 1:
                    print(f"  Subprocess exited with code 1 (os._exit from monitor) — as expected")
                    results[name] = True
                else:
                    print(f"  Subprocess exited with code {actual_code}, expected 1")
                    results[name] = False
            except subprocess.TimeoutExpired:
                print(f"  Subprocess timed out — monitor did NOT kill the process")
                results[name] = False
        else:
            # Normal tests — run in-process
            try:
                ok = TESTS[name]()
                results[name] = ok
            except Exception as e:
                print(f"  Test threw unexpected exception: {e}")
                results[name] = False

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    return 0 if all_pass else 1


def run_inner(test_name: str):
    """Inner runner: invoked with --inner flag, runs a single test in-process."""
    if test_name not in TESTS:
        print(f"Unknown test: {test_name}")
        sys.exit(2)
    ok = TESTS[test_name]()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--inner":
        run_inner(sys.argv[2])
    else:
        sys.exit(run_outer())