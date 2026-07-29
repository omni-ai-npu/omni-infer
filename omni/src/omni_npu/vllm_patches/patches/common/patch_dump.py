# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OMNI-DUMP mounts for all three process roles (api / engine / worker).

Each role gets its forensics installed at the point its process is fully
initialised:

- api (AsyncLLM): install() for the api role needs a running event loop;
  AsyncLLM is normally constructed inside the server's loop. If not, the
  failure is swallowed and this process simply runs without forensics
  (mount point pending real-machine verification).
- engine (EngineCore): installs at the end of EngineCore.__init__ (runs on
  the engine process main thread, which set_wakeup_fd requires) and wraps
  the executor instance so dispatch/complete timestamps bracket
  execute_model.
- worker (NPUWorker): installs at the end of init_device.

The OMNI_DUMP_ENABLE gate sits at registration time: when disabled, no
method wrapper is installed at all and the hot path stays untouched.
"""
import os

from vllm.logger import init_logger
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore

from omni_npu.diagnostics.dump import hooks
from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.worker.npu_worker import NPUWorker

logger = init_logger(__name__)

if os.environ.get("OMNI_DUMP_ENABLE", "1") == "1":
    _orig_api_init = AsyncLLM.__init__

    @register_patch("ExitDumpApiPatch", AsyncLLM)
    class ExitDumpApiPatch(VLLMPatch):
        _attr_names_to_apply = ["__init__"]

        def __init__(self, *args, **kwargs):
            _orig_api_init(self, *args, **kwargs)
            hooks.on_api_init(self)

    _orig_engine_init = EngineCore.__init__

    @register_patch("ExitDumpEnginePatch", EngineCore)
    class ExitDumpEnginePatch(VLLMPatch):
        _attr_names_to_apply = ["__init__"]

        def __init__(self, *args, **kwargs):
            _orig_engine_init(self, *args, **kwargs)
            hooks.on_engine_init(self)

    _orig_init_device = NPUWorker.init_device

    @register_patch("ExitDumpWorkerPatch", NPUWorker)
    class ExitDumpWorkerPatch(VLLMPatch):
        _attr_names_to_apply = ["init_device"]

        def init_device(self):
            _orig_init_device(self)
            hooks.on_worker_init(self)
