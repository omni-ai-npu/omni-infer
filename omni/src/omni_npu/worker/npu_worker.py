# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import gc
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Optional, Union, List
from types import NoneType

import torch
import torch_npu

import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.worker.utils import request_memory
from vllm.v1.worker.worker_base import CompilationTimes
from vllm.utils.torch_utils import set_random_seed
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.tracing import instrument
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized
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
from vllm.distributed.weight_transfer import (
    WeightTransferEngineFactory,
)
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
)
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.workspace import init_workspace_manager

from omni_npu.model_config.config_loader.loader import load_model_extra_config
from omni_npu.plugin_decorators import load_model_decorator, determine_memory_decorator
from omni_npu.compilation.acl_graph import (
    consume_aclgraph_recapture,
    set_aclgraph_recapture,
)
from omni_npu.v1.utils import on_ascend950, switch_torch_device

logger = init_logger(__name__)


def _env_to_bool(value) -> bool:
    """Interpret "1"/"true" (case-insensitive) as True, anything else as False."""
    return str(value).strip().lower() in ("1", "true")


@dataclass
class NPUMemorySnapshot(MemorySnapshot):

    def measure(self) -> None:
        device = self.device_

        # we measure the torch peak memory usage via allocated_bytes,
        # rather than `torch.accelerator.memory_reserved()` .
        # After `torch.accelerator.reset_peak_memory_stats()`,
        # `torch.accelerator.memory_reserved()` will keep growing, and only shrink
        # when we call `torch.accelerator.empty_cache()` or OOM happens.
        self.torch_peak = torch.npu.memory_stats(device).get(
            "allocated_bytes.all.peak", 0
        )
        self.free_memory, self.total_memory = torch.npu.mem_get_info(device)
        self.cuda_memory = self.total_memory - self.free_memory

        # torch.accelerator.memory_reserved() is how many bytes
        # PyTorch gets from cuda (by calling cudaMalloc, etc.)
        # this is used to measure the non-torch memory usage
        self.torch_memory = torch.npu.memory_reserved(device)
        self.non_torch_memory = self.cuda_memory - self.torch_memory
        self.timestamp = time.time()


class NPUWorker(Worker):
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
        additional_config = vllm_config.additional_config or {}
        self.is_full_async_rl = bool(
            additional_config.get("enable_full_async_rl", False))
        device_config = self.device_config
        if device_config.device_type != "npu":
            raise ValueError(f"Expected device_type 'npu', got '{device_config.device_type}'")
        if current_platform.device_type != "npu":
            raise ValueError(f"Expected platform device_type 'npu', got '{current_platform.device_type}'")
        self.profiler = None
        current_platform.pre_register_and_update()

    @instrument(span_name="Init device")
    def init_device(self):
        if self.device_config.device.type == "npu" and current_platform.device_type == "npu":
            parallel_config = self.parallel_config
            if (
                parallel_config.distributed_executor_backend
                not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and parallel_config.nnodes_within_dp == 1
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_index

                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size
                    * self.parallel_config.tensor_parallel_size
                )

                # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                self.local_rank += dp_local_rank * tp_pp_world_size
            
            visible_device_index = current_platform.logical_device_id_to_visible_device_id(self.local_rank)
            self.device = torch.device(f"npu:{visible_device_index}")
            torch.npu.set_device(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
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

                ensure_layer_parallel_initialized(backend=current_platform.dist_backend)

            if self.use_v2_model_runner:
                logger.info_once("Using V2 Model Runner")

            # Set random seed
            set_random_seed(self.model_config.seed)

            # Snapshot available memory after distributed init
            # NOTE: HCCL buffers are NOT allocated yet (lazy allocation on first use)
            # NOTE: gc.collect() aligns with GPU worker's init_device() to ensure
            # accurate memory measurement by releasing cyclic references before snapshot.
            gc.collect()
            torch.npu.empty_cache()

            # take current memory snapshot
            self.init_snapshot = init_snapshot = NPUMemorySnapshot(device=self.device)
            self.requested_memory = request_memory(init_snapshot, self.cache_config)
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)
            logger.debug(
                "worker requested memory: %sGiB", format_gib(self.requested_memory)
            )

            # with switch_torch_device():
            #     self.init_snapshot = MemorySnapshot(device=self.device)
            # self.requested_memory = self.init_snapshot.total_memory * self.cache_config.gpu_memory_utilization
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        # Construct the model runner
        if self.use_v2_model_runner:
            raise NotImplementedError("V2 Model Runner is not supported for NPU")
        else:
            from omni_npu.worker.npu_model_runner import NPUModelRunner

            self.model_runner = NPUModelRunner(self.vllm_config, self.device)

        if self.rank == 0:
            report_usage_stats(self.vllm_config)

        # TODO: profiler feature need to adapt vllm 0.25.1
        # self.profiler = self._init_profiler()

        # TODO: eplb feature need to adapt vllm 0.25.1
        # from omni_placement.utils import _init_omni_eplb_configs
        # _init_omni_eplb_configs(self.vllm_config, self.local_rank)

    @torch.inference_mode()
    @determine_memory_decorator
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
            torch.npu.reset_peak_memory_stats(self.device)
        except Exception as e:
            logger.debug("Failed to reset peak memory stats: %s", e)

        # Record snapshot before profile_run (to measure non-torch increase)
        with switch_torch_device():
            before_profile = NPUMemorySnapshot(device=self.device)

        # profile_run triggers lazy HCCL allocation
        self.model_runner.profile_run()

        with switch_torch_device():
            after_profile = NPUMemorySnapshot(device=self.device)

        non_torch_increase = max(
            0, after_profile.non_torch_memory - self.init_snapshot.non_torch_memory
        )

        weights_memory = getattr(self.model_runner, 'model_memory_usage', 0)
        # MemorySnapshot.torch_peak is measured 
        # via method torch.npu.memory_stats(device).get("allocated_bytes.all.peak", 0)
        peak_activation = after_profile.torch_peak - before_profile.torch_peak

        total = after_profile.total_memory
        available = int(
            total * self.cache_config.gpu_memory_utilization
            - weights_memory
            - peak_activation
            - non_torch_increase
        )

        logger.info(
            f"Available KV cache memory: {GiB(available):.2f} GiB "
            f"(total={GiB(total):.2f}, util={self.cache_config.gpu_memory_utilization}, "
            f"weights={GiB(weights_memory):.2f}, peak_torch_activation_during_profile_run={GiB(peak_activation):.2f}, "
            f"non_torch_memory_increase_during_profile_run={GiB(non_torch_increase):.2f})"
        )
        return max(available, 0)

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

    def compile_or_warm_up_model(self) -> CompilationTimes:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()
        set_random_seed(self.model_config.seed)
        return CompilationTimes(
            language_model=self.vllm_config.compilation_config.compilation_time,
            encoder=self.vllm_config.compilation_config.encoder_compilation_time,
        )

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if current_platform.device_type == "npu" and not self.vllm_config.model_config.enable_cumem_allocator:
            return nullcontext()
        return super()._maybe_get_memory_pool_context(tag)

    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    @load_model_decorator
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        with (
            self._maybe_get_memory_pool_context(tag="weights"),
            set_current_vllm_config(self.vllm_config),
            # 20 MiB is the minimum PyTorch allows for max_split_size_mb.
            self._scoped_allocator_max_split(max_split_size_mb=20),
        ):
            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)

        if self.vllm_config.weight_transfer_config is not None:
            self.weight_transfer_engine = WeightTransferEngineFactory.create_engine(
                self.vllm_config.weight_transfer_config,
                self.vllm_config,
                self.device,
                self.model_runner.get_model(),
            )

    def execute_dummy_batch(self) -> None:
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        self.model_runner._dummy_run(num_tokens, uniform_decode=True, force_attention=True)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",  # type: ignore[name-defined]
    ) -> Optional[Union[ModelRunnerOutput, AsyncModelRunnerOutput]]:
        if self.profiler is not None and self._use_token_for_profile:
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
                requests_exceeded = (
                    self._requests_seen > self.profiler_skip_requests
                )
                prefill_cond = (
                    self.enable_prefill_profiler
                    and requests_exceeded
                    and num_tokens > self.profiler_token_threshold
                )
                decode_cond = (
                    requests_exceeded
                    and len(new_reqs) == 0
                    and num_tokens == self.profiler_token_threshold
                )
                if prefill_cond or decode_cond:
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
            if not isinstance(output, IntermediateTensors):
                raise TypeError(f"Expected IntermediateTensors, got {type(output)}")
            get_pp_group().send_tensor_dict(
                output.tensors,
                all_gather_group=get_tp_group(),
                all_gather_tensors={"hidden_states": False, "residual": False},
            )
            res = None
        if self.profiler is not None and self._use_token_for_profile:
            if self.profile_already_start and not self.profile_finished:
                self.profile_step += 1
            if not self.profile_finished and self.profile_step > self.profiler_stop_step:
                self.profiler.stop()
                self.profile_finished = True
        return res

    def _init_profiler(self):
        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        self.profile_already_start = False
        self.profile_step = 0
        self.profile_finished = False
        self._requests_seen = 0
        self._use_token_for_profile = os.getenv("PROFILER_TOKEN_THRESHOLD") is not None

        if self.profiler is not None:
            self.profiler_token_threshold = int(os.environ.get('PROFILER_TOKEN_THRESHOLD', "1"))
            self.profiler_stop_step = int(os.environ.get('PROFILER_STOP_STEP', "5"))
            self.enable_prefill_profiler = (
                os.environ.get('ENABLE_PREFILL_PROFILER', 'FALSE').lower() == 'true'
            )
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
                record_shapes=_env_to_bool(envs.VLLM_TORCH_PROFILER_RECORD_SHAPES),
                profile_memory=_env_to_bool(envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY),
                with_stack=_env_to_bool(envs.VLLM_TORCH_PROFILER_WITH_STACK),
                with_flops=_env_to_bool(envs.VLLM_TORCH_PROFILER_WITH_FLOPS),
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir))
        else:
            return None

    def sleep(self, level: int = 1) -> None:
        if self.is_full_async_rl:
            raise RuntimeError(
                "sleep is forbidden under full-async RL scenario "
                "(enable_full_async_rl=True)")

        def GiB(b):
            return b / (1 << 30)

        def cast_module_to_nd(module: torch.nn.Module, tag: str) -> None:
            """when using .to('cpu'), the ND format is required. First, 
            convert the model weights back to ND format uniformly."""
            format_counter: dict = {}

            def _cast_one(kind: str, name: str, tensor: torch.Tensor) -> torch.Tensor:
                cur_fmt = torch_npu.get_npu_format(tensor)
                format_counter[cur_fmt] = format_counter.get(cur_fmt, 0) + 1
                logger.debug(
                    "[%s] %s=%s shape=%s dtype=%s current format=%s -> ND",
                    tag, kind[:-1], name, tuple(tensor.shape), tensor.dtype,
                    _npu_format_name(cur_fmt),
                )
                return torch_npu.npu_format_cast(tensor, torch_npu.Format.ND)

            for name, param in module.named_parameters(recurse=True):
                if param.device.type == "npu":
                    param.data = _cast_one("params", name, param.data)
            for name, buf in module.named_buffers(recurse=True):
                if buf.device.type == "npu":
                    buf.data = _cast_one("buffers", name, buf.data)

            if format_counter:
                sorted_formats = sorted(
                    format_counter.items(), key=lambda item: int(item[0])
                )
                summary = ", ".join(
                    f"{_npu_format_name(fmt)}={cnt}"
                    for fmt, cnt in sorted_formats
                )
                logger.debug("[%s] NPU tensor format distribution before ND cast: %s",
                            tag, summary)
            else:
                logger.debug("[%s] no NPU tensors found to cast.", tag)

        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()

        free_bytes_before_sleep = torch.npu.mem_get_info()[0]

        self.model_runner.unregister_kv_caches()

        # On A5, the format must be explicitly converted to ND before calling .to("cpu")
        need_cast_to_nd = on_ascend950()

        if need_cast_to_nd:
            cast_module_to_nd(self.model_runner.get_model(), tag="main_model")
        self.model_runner.get_model().to("cpu")
        if hasattr(self.model_runner, "drafter") and self.model_runner.drafter:
            if need_cast_to_nd:
                cast_module_to_nd(self.model_runner.get_drafter_model(), tag="drafter_model")
            self.model_runner.get_drafter_model().to("cpu")

        self.kv_nbytes: List[List[int]] = [
            [t.untyped_storage().nbytes() for t in row]
            for row in self.model_runner.kv_caches
        ]

        for kv_caches_i in self.model_runner.kv_caches:
            for kv_caches_i_j in kv_caches_i:
                kv_caches_i_j.untyped_storage().resize_(0)

        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()

        free_bytes_after_sleep, total = torch.npu.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        if freed_bytes < 0:
            raise RuntimeError("Memory usage increased after sleeping.")
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", GiB(freed_bytes), GiB(used_bytes)
        )

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        if self.is_full_async_rl:
            raise RuntimeError(
                "wake_up is forbidden under full-async RL scenario "
                "(enable_full_async_rl=True)")

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


    def recapture_model(self) -> None:
        """Re-capture ACLGraph against the current model weights."""
        if not self.is_full_async_rl:
            return
        if self.model_config.enforce_eager:
            return
        logger.info("recapture aclgraph on /resume")
        set_aclgraph_recapture(True)
        try:
            self.model_runner.capture_model()
        except Exception as e:
            logger.exception("recapture_model: capture_model failed")
            raise RuntimeError(
                "ACLGraph recapture failed during resume"
            ) from e
        finally:
            # capture_model() normally consumes the flag on entry; clear any
            # residual if it failed before consume or was bypassed.
            consume_aclgraph_recapture()


def _npu_format_name(fmt) -> str:
    try:
        return torch_npu.Format(int(fmt)).name
    except (ValueError, AttributeError):
        return f"UNKNOWN({fmt})"


def init_worker_distributed_environment(
    vllm_config,
    rank,
    distributed_init_method=None,
    local_rank=-1,
    backend="hccl",
    world_ranks=None,
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    init_batch_invariance()
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
