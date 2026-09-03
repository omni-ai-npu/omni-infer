# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OMNI-DUMP mounts for the three process roles (api / engine / worker).

Every hook runs after the state it reads exists. The api one additionally needs
a running event loop (AsyncLLM is normally built inside the server's); a missing
one is swallowed by guarded_install, leaving that process without forensics.

The engine mount is run_busy_loop, NOT EngineCore.__init__: vLLM loads general
plugins -- and so runs omni's apply_patches() -- from *inside* that __init__
(0.25.1 core.py:110), so under spawn the patch is installed while the very call
it patches is already on the stack, and the hook at its end is never reached.
Nothing raises; the role is silently missing. Under fork it happens to work,
because the child inherits the parent's already-patched __init__. run_busy_loop
is entered after construction returns, on the main thread (set_wakeup_fd needs
that) and before the first step, so wrap_executor still brackets every
execute_model.

DPEngineCoreProc reimplements run_busy_loop instead of calling super(), so
data-parallel deployments need their own mount.

The OMNI_DUMP_ENABLE gate sits at registration time: disabled means no wrapper
exists at all, not a wrapper that short-circuits.
"""
from vllm.logger import init_logger
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc

from omni_npu import envs
from omni_npu.diagnostics.dump import hooks
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.worker.npu_worker import NPUWorker

logger = init_logger(__name__)

if envs.OMNI_DUMP_ENABLE:
    _orig_api_init = AsyncLLM.__init__

    @register_patch("ExitDumpApiPatch", AsyncLLM)
    class ExitDumpApiPatch(VLLMPatch):
        _attr_names_to_apply = ["__init__"]

        def __init__(self, *args, **kwargs):
            _orig_api_init(self, *args, **kwargs)
            hooks.on_api_init(self)

    # VLLMPatch.apply() finds its "already patched" bookkeeping with hasattr(),
    # which walks the MRO, so DPEngineCoreProc would write into EngineCoreProc's
    # dict and be rejected as a double patch -- a ValueError that
    # PatchManager.apply_patch swallows into one log line. Its own dict up front
    # is order-independent, unlike relying on which mount applies first.
    if "_omni_npu_applied_patches" not in DPEngineCoreProc.__dict__:
        DPEngineCoreProc._omni_npu_applied_patches = {}

    _orig_engine_loop = EngineCoreProc.run_busy_loop

    @register_patch("ExitDumpEnginePatch", EngineCoreProc)
    class ExitDumpEnginePatch(VLLMPatch):
        _attr_names_to_apply = ["run_busy_loop"]

        def run_busy_loop(self):
            hooks.on_engine_init(self)
            return _orig_engine_loop(self)

    _orig_dp_engine_loop = DPEngineCoreProc.run_busy_loop

    @register_patch("ExitDumpDPEnginePatch", DPEngineCoreProc)
    class ExitDumpDPEnginePatch(VLLMPatch):
        _attr_names_to_apply = ["run_busy_loop"]

        def run_busy_loop(self):
            hooks.on_engine_init(self)
            return _orig_dp_engine_loop(self)

    _orig_init_device = NPUWorker.init_device

    @register_patch("ExitDumpWorkerPatch", NPUWorker)
    class ExitDumpWorkerPatch(VLLMPatch):
        _attr_names_to_apply = ["init_device"]

        def init_device(self):
            _orig_init_device(self)
            hooks.on_worker_init(self)
