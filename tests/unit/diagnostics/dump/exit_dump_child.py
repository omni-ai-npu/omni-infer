# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Child process entry used by the exit_dump integration tests.

Usage: python exit_dump_child.py <role> <dump_dir> <scenario> <stop_file>
"""
import asyncio
import os
import sys
import time

from omni.diagnostics.dump import exit_dump


def _block_until(stop_file):
    while not os.path.exists(stop_file):
        time.sleep(0.02)


def main():
    role, dump_dir, scenario, stop_file = sys.argv[1:5]
    rank = 0 if role == "worker" else None

    if scenario == "api":

        async def amain():
            exit_dump.install(
                role="api", dump_dir=dump_dir, stats_fn=lambda: {"in_flight_reqs": 2}
            )
            print("READY", flush=True)
            while not os.path.exists(stop_file):
                await asyncio.sleep(0.02)

        asyncio.run(amain())
        return

    if scenario == "getsignal":
        import signal

        before = signal.getsignal(signal.SIGTERM)
        exit_dump.install(role=role, dump_dir=dump_dir, rank=rank)
        after = signal.getsignal(signal.SIGTERM)
        print(f"HANDLER_UNCHANGED={before is after}", flush=True)
        print("READY", flush=True)
        _block_until(stop_file)
        return

    stats_fn = (lambda: {"engine_step_count": 5}) if role == "engine" else None
    exit_dump.install(role=role, dump_dir=dump_dir, rank=rank, stats_fn=stats_fn)
    print("READY", flush=True)

    if scenario == "exit0":
        sys.exit(0)
    if scenario == "block":
        _block_until(stop_file)
        sys.exit(0)
    if scenario == "segv":
        _block_until(stop_file)
        import ctypes

        ctypes.string_at(0)


if __name__ == "__main__":
    main()
