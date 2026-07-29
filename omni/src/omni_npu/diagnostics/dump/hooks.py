# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Role hooks for the three vLLM process roles plus their shared plumbing.

Each on_<role>_init entry point is vllm-free and mounts exit forensics for one
process role:

* worker: mounted at the end of NPUWorker.init_device; carries no stats
  section (progress bookkeeping lives in EngineCore, the worker's unique
  contribution is its stack plus process-level hardware).
* engine: wraps executor.execute_model for the micro tick and reads live
  scheduler state at dump time.
* api: needs a running asyncio event loop (loop.add_signal_handler); a missing
  loop is swallowed like any other install failure and only degrades forensics
  for this process. Stats section is the in-flight request count.
"""
import logging
import os

from omni_npu.diagnostics.dump import constants, engine_stats, exit_dump

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------

def enabled():
    return os.environ.get(constants.ENV_ENABLE, "1") == "1"


def default_dump_dir():
    return os.environ.get(constants.ENV_DUMP_DIR, constants.DEFAULT_DUMP_DIR)


def guarded_install(role, dump_dir=None, rank=None, stats_fn=None):
    """Install exit forensics; a failure must never break the host process."""
    try:
        exit_dump.install(
            role=role,
            dump_dir=dump_dir or default_dump_dir(),
            rank=rank,
            stats_fn=stats_fn,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("omni-dump: install failed, forensics disabled: %r", e)


# --------------------------------------------------------------------------
# worker role
# --------------------------------------------------------------------------

def worker_rank(worker):
    for attr in ("local_rank", "rank"):
        value = getattr(worker, attr, None)
        if value is not None:
            return value
    for env in ("LOCAL_RANK", "RANK", "RANK_ID"):
        value = os.environ.get(env)
        if value:
            try:
                return int(value)
            except ValueError:
                return value
    return None


def on_worker_init(worker, dump_dir=None):
    if not enabled():
        return
    guarded_install(
        role=constants.ROLE_WORKER, dump_dir=dump_dir, rank=worker_rank(worker)
    )


# --------------------------------------------------------------------------
# engine role
# --------------------------------------------------------------------------

def wrap_executor(executor):
    """Wrap executor.execute_model on the instance to record dispatch/complete.

    When execute_model returns a future (async scheduling / batch queue mode),
    the completion timestamp is recorded by its done-callback instead.
    """
    orig = executor.execute_model

    def wrapped(scheduler_output, *args, **kwargs):
        engine_stats.on_dispatch(scheduler_output)
        out = orig(scheduler_output, *args, **kwargs)
        if hasattr(out, "add_done_callback"):
            out.add_done_callback(lambda _f: engine_stats.on_complete())
        else:
            engine_stats.on_complete()
        return out

    executor.execute_model = wrapped
    return executor


def build_engine_stats_fn(scheduler):
    def stats_fn():
        stats = engine_stats.snapshot()
        stats["scheduler"] = engine_stats.collect_scheduler_state(scheduler)
        return stats

    return stats_fn


def on_engine_init(engine_core, dump_dir=None):
    if not enabled():
        return
    try:
        executor = getattr(engine_core, "model_executor", None)
        if executor is not None:
            wrap_executor(executor)
    except Exception as e:  # noqa: BLE001
        logger.warning("omni-dump: executor wrap failed, stats degraded: %r", e)
    scheduler = getattr(engine_core, "scheduler", None)
    guarded_install(
        role=constants.ROLE_ENGINE,
        dump_dir=dump_dir,
        stats_fn=build_engine_stats_fn(scheduler),
    )


# --------------------------------------------------------------------------
# api role
# --------------------------------------------------------------------------

def build_api_stats_fn(async_llm):
    def stats_fn():
        try:
            return {"in_flight_reqs": len(async_llm.output_processor.request_states)}
        except Exception as e:  # noqa: BLE001
            return {"in_flight_reqs": f"ERROR:{type(e).__name__}"}

    return stats_fn


def on_api_init(async_llm, dump_dir=None):
    if not enabled():
        return
    guarded_install(
        role=constants.ROLE_API, dump_dir=dump_dir, stats_fn=build_api_stats_fn(async_llm)
    )
