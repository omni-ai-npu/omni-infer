# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright contributors to the vLLM project.

import os
import gc
from typing import Optional, Union, List
from types import NoneType

import torch
import torch_npu

import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.utils.torch_utils import set_random_seed
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.parallel_state import (
    get_tp_group,
    get_pp_group,
)
from vllm.model_executor.layers.batch_invariant import init_batch_invariance
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
    parallel_state,
    GroupCoordinator,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.worker.workspace import init_workspace_manager

from .npu_model_runner import NPUModelRunner
from omni_npu.worker.npu_mem_pool import NpuMemAllocator
from omni_npu.model_config.config_loader.loader import load_model_extra_config
from omni_npu.plugin_decorators import load_model_decorator
from omni_npu.compilation.acl_graph import set_aclgraph_recapture

logger = init_logger(__name__)


class NPUWorker(WorkerBase):
    """An NPU worker class using torch_npu and HCCL backend."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        **kwargs,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        # RL support: Store world_ranks if provided
        self.world_ranks = kwargs.get("world_ranks", None)
        logger.info(f"[NPUWorker.init] {self.rank=}, {self.local_rank=}, {self.world_ranks=}")

        device_config = self.device_config
        assert device_config.device_type == "npu"
        assert current_platform.device_type == "npu"
        self.profiler = None
        current_platform.pre_register_and_update()

    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """Get KV connector metadata from this worker if available."""

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        tp_rank = get_tp_group().rank_in_group
        return {tp_rank: metadata}

    def init_device(self):
        if self.device_config.device.type == "npu" and current_platform.device_type == "npu":
            self.device = torch.device(f"npu:{self.local_rank}")
            current_platform.set_device(self.device)
            torch.npu.empty_cache()
            # Initialize distributed before measuring memory
            backend = getattr(current_platform, "dist_backend", "hccl")
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                backend,
                world_ranks=self.world_ranks,
            )

            # Initialize the model best practice configs.
            load_model_extra_config(self.model_config, self.vllm_config, self.scheduler_config)

            # OMNI-CONF worker-scope snapshot (solution doc §3.3): full dump
            # on local_rank 0, hash-only elsewhere (cross-rank drift signal).
            # Exception-isolated + once-guarded + OMNI_CONFIG_SUMMARY=0 gated inside.
            from omni_npu.diagnostics.config_summary import emit_config_summary
            emit_config_summary(
                vllm_config=self.vllm_config,
                scope="worker",
                rank=self.rank,
                local_rank=self.local_rank,
                include_omni=True,
                hash_only=(self.local_rank != 0),
            )

            # Only initialize the custom layer-parallel communication domain when
            # explicitly enabled by the high-performance launcher script.
            if "omni_custom_models" in os.environ.get("VLLM_PLUGINS", ""):
                # Initialize the model best practice configs.
                from omni_npu.v1.distributed.parallel_state_ext import ( 
                    ensure_layer_parallel_initialized,
                )

                ensure_layer_parallel_initialized(backend=backend)

            # Set random seed
            set_random_seed(self.model_config.seed)
            # Snapshot available memory
            free, total = torch.npu.mem_get_info()
            self.init_snapshot = type("_Snap", (), {"free_memory": free, "total_memory": total})()
            self.requested_memory = total * self.cache_config.gpu_memory_utilization
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # Construct the model runner
        self.model_runner = NPUModelRunner(self.vllm_config, self.device)  # type: ignore

        if self.rank == 0:
            from vllm.v1.utils import report_usage_stats
            report_usage_stats(self.vllm_config)
        self.profiler = self._init_profiler()
        from omni_placement.utils import _init_omni_eplb_configs
        _init_omni_eplb_configs(self.vllm_config, self.local_rank)

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profile to determine memory available for KV cache on NPU."""

        def GiB(b):
            return b / (1 << 30)

        if self.cache_config.kv_cache_memory_bytes:
            # still do compile/profile run to initialize kernels
            self.model_runner.profile_run()
            logger.info(
                "Reserved %.2f GiB for KV cache as specified; skipping profiling.",
                GiB(self.cache_config.kv_cache_memory_bytes),
            )
            return self.cache_config.kv_cache_memory_bytes

        torch.npu.empty_cache()
        try:
            torch.npu.reset_peak_memory_stats()
        except Exception:
            pass

        # Profile run compiles and warms kernels
        self.model_runner.profile_run()

        free_after, total = torch.npu.mem_get_info()
        try:
            peak = torch.npu.max_memory_allocated()
        except Exception:
            # Fallback: estimate by delta from init snapshot
            peak = max(0, self.init_snapshot.free_memory - free_after)

        available = int(total * self.cache_config.gpu_memory_utilization - peak)
        logger.info(
            "Available KV cache memory: %.2f GiB (total=%.2f, util=%.2f, peak=%.2f)",
            GiB(available), GiB(total), self.cache_config.gpu_memory_utilization, GiB(peak)
        )
        return max(available, 0)

    def get_kv_cache_spec(self):
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate NPU KV cache with the specified kv_cache_config."""
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        if self.model_config.enable_sleep_mode:
            allocator = NpuMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            self.model_runner.initialize_kv_cache(kv_cache_config)


    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        # NOP: KV caches are fully initialized in initialize_from_config.
        # vLLM calls this with (num_gpu_blocks, num_cpu_blocks);
        # for NPU we don't need additional allocation here.
        return None

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if getattr(self, '_use_token_for_profile', False):
            logger.info("origin profiler is disabled because PROFILER_TOKEN_THRESHOLD is set.")
            return
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    def compile_or_warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()
        set_random_seed(self.model_config.seed)

    def get_model(self):
        return self.model_runner.get_model()

    @load_model_decorator
    def load_model(self) -> None:
        if self.model_config.enable_sleep_mode:
            allocator = NpuMemAllocator.get_instance()
            if allocator.get_current_usage() != 0:
                raise RuntimeError("Sleep mode can only be used for one instance per process.")
            context = allocator.use_memory_pool(tag="weights")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context, set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1, uniform_decode=True, force_attention=True)

    def add_lora(self, lora_request) -> bool:  # type: ignore[no-untyped-def]
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",  # type: ignore[name-defined]
    ) -> Optional[Union[ModelRunnerOutput, AsyncModelRunnerOutput]]:
        if envs.VLLM_TORCH_PROFILER_DIR and self._use_token_for_profile:
            new_reqs = getattr(scheduler_output, "scheduled_new_reqs", None) or []
            if len(new_reqs) > 0 and not self.profile_already_start:
                self._requests_seen += len(new_reqs)
                if self._requests_seen > self.profiler_skip_requests:
                    logger.info(
                        "Profiler armed on request #%d (skipped %d)",
                        self._requests_seen, self.profiler_skip_requests,
                    )
            if not self.profile_already_start:
                num_tokens = scheduler_output.total_num_scheduled_tokens
                if not self.profile_already_start:
                    num_tokens = scheduler_output.total_num_scheduled_tokens
                    if ((self.enable_prefill_profiler
                            and self._requests_seen > self.profiler_skip_requests and num_tokens > self.profiler_token_threshold)
                            or (self._requests_seen > self.profiler_skip_requests and num_tokens == self.profiler_token_threshold)):
                    # Prefill phase: only when ENABLE_PREFILL_PROFILER is set
                    # Decode phase: always profile
                    # Prefill phase: only when ENABLE_PREFILL_PROFILER is set
                    # Decode phase: always profile
                        self.profiler.start()
                        self.profile_already_start = True
                        self.profile_step = 0
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0

        if forward_pass and not get_pp_group().is_first_rank:
            tensor_dict = get_pp_group().recv_tensor_dict(
                all_gather_group=get_tp_group(),
                all_gather_tensors={"hidden_states": False, "residual": False},
            )
            intermediate_tensors = IntermediateTensors(tensor_dict)
        else:
            intermediate_tensors = None
        output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)

        if isinstance(
            output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType
        ):
            res = output
        else:
            assert isinstance(output, IntermediateTensors)
            get_pp_group().send_tensor_dict(
                output.tensors,
                all_gather_group=get_tp_group(),
                all_gather_tensors={"hidden_states": False, "residual": False},
            )
            res = None
        if envs.VLLM_TORCH_PROFILER_DIR and self._use_token_for_profile:
            if self.profile_already_start and not self.profile_finished:
                self.profile_step += 1
            if not self.profile_finished and self.profile_step > self.profiler_stop_step:
                self.profiler.stop()
                self.profile_finished = True
        return res

    def _init_profiler(self):
        self.profile_already_start = False
        self.profile_step = 0
        self.profile_finished = False
        self._requests_seen = 0
        self._use_token_for_profile = os.getenv("PROFILER_TOKEN_THRESHOLD") is not None

        if envs.VLLM_TORCH_PROFILER_DIR:
            self.profiler_token_threshold = int(os.environ.get('PROFILER_TOKEN_THRESHOLD', "1"))
            self.profiler_stop_step = int(os.environ.get('PROFILER_STOP_STEP', "5"))
            self.enable_prefill_profiler = (os.environ.get('ENABLE_PREFILL_PROFILER', 'FALSE').lower() == 'true')
            self.profiler_skip_requests = int(os.environ.get('PROFILER_SKIP_REQUESTS', "0"))
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            )
            self.profile_already_start = False
            self.profile_finished = False
            return torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir))
        else:
            return None

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        return self.model_runner.take_draft_token_ids()

    @torch.inference_mode()
    def sample_tokens(self, grammar_output):
        return self.model_runner.sample_tokens(grammar_output)

    def sleep(self, level: int = 1) -> None:
        def GiB(b):
            return b / (1 << 30)

        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()

        free_bytes_before_sleep = torch.npu.mem_get_info()[0]

        self.model_runner.unregister_kv_caches()

        self.model_runner.get_model().to("cpu")
        if hasattr(self.model_runner, "drafter") and self.model_runner.drafter:
            self.model_runner.get_drafter_model().to("cpu")

        self.kv_nbytes:List[List[int]] = [
            [t.untyped_storage().nbytes() for t in row]
            for row in self.model_runner.kv_caches
        ]

        for i, kv_caches_i in enumerate(self.model_runner.kv_caches):
            for j, kv_caches_i_j in enumerate(kv_caches_i):
                kv_caches_i_j.untyped_storage().resize_(0)

        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()

        free_bytes_after_sleep, total = torch.npu.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", GiB(freed_bytes), GiB(used_bytes)
        )

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        def GiB(b):
            return b / (1 << 30)

        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()
        
        free_bytes_before = torch.npu.mem_get_info()[0]

        if (tags == ["weights"]):
            self.model_runner.get_model().to("npu")
            if hasattr(self.model_runner, "drafter") and self.model_runner.drafter:
                self.model_runner.get_drafter_model().to("npu")

        if (tags == ["kv_cache"]):
            for i, kv_caches_i in enumerate(self.model_runner.kv_caches):
                for j, kv_caches_i_j in enumerate(kv_caches_i):
                    kv_caches_i_j.untyped_storage().resize_(self.kv_nbytes[i][j])

            logger.info(f"re-register kv caches now")
            self.model_runner.reregister_kv_caches()
            if not self.model_config.enforce_eager:
                set_aclgraph_recapture(True)
                self.model_runner.capture_model()

        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()

        free_bytes_after = torch.npu.mem_get_info()[0]
        use_bytes = free_bytes_before - free_bytes_after
        logger.info(f"wake_up {tags=} use %.2f GiB memory.", GiB(use_bytes))


def init_worker_distributed_environment(
    vllm_config,
    rank,
    distributed_init_method=None,
    local_rank=-1,
    backend="hccl",
    world_ranks=None,
) -> None:
    """Initialize the distributed environment."""
    attention_config = vllm_config.attention_config
    parallel_config = vllm_config.parallel_config
    init_batch_invariance(attention_config.backend)
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)
    if world_ranks is None:
        init_method = distributed_init_method or "env://"
        init_distributed_environment(
            parallel_config.world_size, rank, init_method, local_rank, backend)
    else:
        init_world_group(world_ranks, local_rank, backend)
    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )
    # Init ec connector here before KV caches caches init
    # NOTE: We do not init KV caches for Encoder-only instance in EPD disagg mode
    ensure_ec_transfer_initialized(vllm_config)

    
def init_world_group(ranks: list[int], local_rank: int, backend: str):
    """Initialize world group for RL scenarios where ranks are externally provided."""
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if parallel_state._WORLD is not None:
        raise RuntimeError("_WORLD must not be initialized")
    world_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    logger.debug(f"worker init world group {ranks=}, {local_rank=}, {backend=}, {world_rank=}, {world_size=}")
    if len(ranks) != world_size:
        GroupCoordinator.use_local_synchronization = True
    world_group = parallel_state.init_world_group(
        ranks,
        local_rank,
        backend,
    )
    parallel_state._WORLD = world_group
    logger.debug(f"worker init world group done {ranks=}, {local_rank=}, {backend=}, {world_rank=}")