# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""One-shot installer that wires all exit-forensics machinery into a process.

install() must be called from the main thread (set_wakeup_fd requirement).
SIGTERM is deliberately left untouched: healthy exits are covered by atexit,
stuck-process forensics go through the SIGUSR1 path only.
"""
import atexit
import faulthandler
import logging
import os
import signal
from pathlib import Path

from omni_npu.diagnostics.dump import constants, dump_thread, forensic, hardware_probe, stats_flush

logger = logging.getLogger(__name__)

_installed = None


def _noop(signum, frame):
    # Enables the CPython signal machinery that writes the signum byte
    # to the wakeup fd; the real work happens in the dump thread.
    pass


# starttime is field 22 in /proc/<pid>/stat (proc(5)); after splitting off
# "<pid> (<comm>)" the remainder starts at field 3, so its index is 22-3=19.
_PROC_STAT_STARTTIME_INDEX = 19

# comm can itself contain "(" / ")" / spaces, so only the *last* ")" reliably
# delimits it (proc(5)); split once from the right and keep what follows.
_COMM_FIELD_MAXSPLIT = 1
_AFTER_COMM_PART = 1


def _proc_start():
    """Process start time (jiffies) from /proc/self/stat; empty off Linux."""
    try:
        stat = Path("/proc/self/stat").read_text()
        after_comm = stat.rsplit(")", _COMM_FIELD_MAXSPLIT)[_AFTER_COMM_PART]
        return after_comm.split()[_PROC_STAT_STARTTIME_INDEX]
    except (OSError, IndexError):
        return ""


def _write_pidfile(pids_dir, role, rank, pid):
    path = Path(pids_dir) / f"omni_npu_{role}_{pid}.pid"
    lines = [
        f"pid={pid}",
        f"role={role}",
        f"rank={'' if rank is None else rank}",
        f"proc_start={_proc_start()}",
    ]
    forensic.atomic_write_text(path, "\n".join(lines) + "\n")
    return path


class _Installation:
    def __init__(self, role, rank, dump_dir, guard, collector, flusher, pidfile):
        self.role = role
        self.rank = rank
        self.dump_dir = dump_dir
        self.guard = guard
        self.collector = collector
        self.flusher = flusher
        self.pidfile = pidfile


def install(role, dump_dir, rank=None, stats_fn=None):
    """Install signal handlers, collection threads, atexit hook and pidfile.

    stats_fn is an optional zero-arg callable composed by the role hooks; its
    result becomes the stats section of the dump package.
    """
    global _installed
    if _installed is not None:
        logger.warning("omni-dump: already installed, ignoring second install")
        return _installed

    dump_dir = Path(dump_dir)
    pids_dir = dump_dir / "pids"
    dump_dir.mkdir(parents=True, exist_ok=True)
    pids_dir.mkdir(parents=True, exist_ok=True)

    pid = os.getpid()
    forensic.cleanup_stale(dump_dir, pid)

    stack_path = forensic.stack_file_path(dump_dir)
    stack_path.parent.mkdir(parents=True, exist_ok=True)
    # stack_file deliberately stays open for the rest of the process lifetime:
    # faulthandler keeps writing crash stacks to it up to and including
    # interpreter teardown, so it is reclaimed by the OS at exit. It is closed
    # explicitly only when install fails partway below.
    stack_file = open(stack_path, "a", encoding="utf-8")
    wakeup_r = wakeup_w = None
    wakeup_fd_armed = False
    api_loop = None
    collector = None
    try:
        faulthandler.enable(file=stack_file, all_threads=True)

        guard = forensic.OnceGuard()

        def _do_collect(trigger):
            stack_text = None
            if trigger == constants.TRIGGER_SIGNAL:
                try:
                    stack_text = stack_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    stack_text = ""
            stats = None
            if stats_fn is not None:
                try:
                    stats = stats_fn()
                except Exception as e:  # noqa: BLE001 - stats must not void the dump
                    stats = {"error": f"{type(e).__name__}:{e}"}
            hardware = hardware_probe.collect_process_hw(
                include_device_mem=(trigger == constants.TRIGGER_SIGNAL)
            )
            hardware["node_ref"] = hardware_probe.find_hw_reference(dump_dir)
            forensic.merge_dump(
                dump_dir,
                role=role,
                trigger=trigger,
                rank=rank,
                stack_text=stack_text,
                stats=stats,
                hardware=hardware,
            )

        def _collect(trigger):
            guard.enter(trigger, lambda: _do_collect(trigger))

        wakeup_r, wakeup_w = os.pipe()
        os.set_blocking(wakeup_w, False)

        if role == constants.ROLE_API:
            import asyncio

            loop = asyncio.get_running_loop()

            def _wake():
                try:
                    os.write(wakeup_w, bytes([constants.DUMP_SIGNAL]))
                except OSError:
                    pass

            loop.add_signal_handler(constants.DUMP_SIGNAL, _wake)
            api_loop = loop
            # chain=True keeps asyncio's own Python-level handler working.
            faulthandler.register(
                constants.DUMP_SIGNAL, all_threads=True, chain=True, file=stack_file
            )
        else:
            signal.signal(constants.DUMP_SIGNAL, _noop)
            signal.set_wakeup_fd(wakeup_w)
            wakeup_fd_armed = True
            faulthandler.register(
                constants.DUMP_SIGNAL, all_threads=True, chain=True, file=stack_file
            )

        collector = dump_thread.DumpThread(
            wakeup_r, lambda: _collect(constants.TRIGGER_SIGNAL)
        )
        collector.start()

        flusher = None
        if role == constants.ROLE_ENGINE and stats_fn is not None:
            flusher = stats_flush.StatsFlushThread(dump_dir, stats_fn)
            flusher.start()

        pidfile = _write_pidfile(pids_dir, role, rank, pid)

        def _on_atexit():
            try:
                _collect(constants.TRIGGER_ATEXIT)
            finally:
                try:
                    pidfile.unlink()
                except OSError:
                    pass

        atexit.register(_on_atexit)

        _installed = _Installation(
            role, rank, dump_dir, guard, collector, flusher, pidfile
        )
        logger.info(
            "omni-dump: installed for role=%s rank=%s dir=%s", role, rank, dump_dir
        )
        return _installed
    except BaseException:
        # Roll back everything wired so far so a failed install leaks neither
        # the stack file nor the wakeup pipe (G.PRM.03).
        faulthandler.unregister(constants.DUMP_SIGNAL)
        faulthandler.disable()
        if wakeup_fd_armed:
            signal.set_wakeup_fd(-1)
        if api_loop is not None:
            api_loop.remove_signal_handler(constants.DUMP_SIGNAL)
        if collector is not None:
            collector.stop()
        if wakeup_r is not None:
            for fd in (wakeup_r, wakeup_w):
                try:
                    os.close(fd)
                except OSError:
                    pass
        stack_file.close()
        raise
