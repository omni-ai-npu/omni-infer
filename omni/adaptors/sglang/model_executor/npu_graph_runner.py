# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run the model with npu graph engine and torch.compile."""

from __future__ import annotations

import bisect
import inspect
import os
import types
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Generator,
    Optional,
    Tuple,
    Union,
)

import torch
import tqdm
from networkx.utils.backends import backends
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.cuda_graph_runner import (
    DeviceRunnerBase,
    _to_torch,
    get_batch_sizes_to_capture,
    patch_model,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
    enable_num_token_non_padded,
)
from sglang.srt.speculative.eagle_utils import (
    EagleDraftInput
)
from sglang.srt.utils import (
    get_available_gpu_memory,
    get_compiler_backend,
    get_device,
    get_device_memory_capacity,
    rank0_log,
)

import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class NpuGraphRunner(DeviceRunnerBase):
    """A NpuGraphRunner runs the forward pass of a model with npu graph engine and torch.compile."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.enable_cache = os.getenv("ENABLE_TORCH_COMPILE_CACHE", "0") == "1"
        try:
            self.warm_up()
        except RuntimeError as e:
            raise Exception(
                f"compile npu graph failed: {e}\n"
                "Possible solutions:\n"
                "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
                "2. set --torch-compile-max-bs to a smaller value (e.g., 16)\n"
                "3. disable torch compile by not using --enable-torch-compile\n"
                "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
            )

    def get_spec_info(self, num_tokens: int):
        spec_info = None
        if self.model_runner.spec_algorithm.is_mtp():
            from sglang.srt.speculative.eagle_utils import EagleVerifyInput, EagleDraftInput

            # Target Model
            if not self.model_runner.is_draft_worker:
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    custom_mask=self.custom_mask,
                    positions=None,
                    retrive_index=None,
                    retrive_next_token=None,
                    retrive_next_sibling=None,
                    retrive_cum_len=None,
                    spec_steps=self.model_runner.server_args.speculative_num_steps,
                    topk=self.model_runner.server_args.speculative_eagle_topk,
                    draft_token_num=self.model_runner.server_args.speculative_num_draft_tokens,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    seq_lens_sum=None,
                    seq_lens_cpu=None,
                )
            # Draft Model
            else:
                with torch.device(self.model_runner.device):
                    self.topk = self.model_runner.server_args.speculative_eagle_topk
                    self.speculative_num_steps = self.model_runner.server_args.speculative_num_steps
                    self.topk_p = torch.zeros((self.max_bs, self.topk), dtype=torch.float32)
                    self.topk_index = torch.zeros((self.max_bs, self.topk), dtype=torch.int64)
                    self.hidden_states = torch.zeros(
                        (self.max_bs, self.model_runner.model_config.hidden_size),
                        dtype=self.model_runner.dtype,
                    )
                topk_p = self.topk_p[:num_tokens]
                topk_index = self.topk_index[:num_tokens]
                hidden_states = self.hidden_states[:num_tokens]

                spec_info = EagleDraftInput(
                    topk_p=None,
                    topk_index=None,
                    hidden_states=hidden_states,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                )

        return spec_info

    def prepare_forward_batch(self, bs: int, num_tokens: int) -> ForwardBatch:
        # Graph inputs
        with torch.device(self.model_runner.device):
            input_ids = torch.zeros((num_tokens,), dtype=torch.int64)
            req_pool_indices = torch.zeros((bs,), dtype=torch.int64)
            seq_lens = torch.full((bs,), self.seq_len_fill_value, dtype=torch.int64)
            out_cache_loc = torch.zeros((num_tokens,), dtype=torch.int32)
            positions = torch.zeros((num_tokens,), dtype=torch.int64)
            gathered_buffer = None
            if self.require_gathered_buffer:
                gathered_buffer = torch.zeros(
                    (
                        num_tokens,
                        self.model_runner.model_config.hidden_size,
                    ),
                    dtype=self.model_runner.dtype,
                )
            num_token_non_padded = torch.tensor(num_tokens, dtype=torch.int32)

        if self.is_encoder_decoder:
            encoder_lens = self.encoder_lens[:bs]
        else:
            encoder_lens = None
        mrope_positions = None

        # pipeline parallelism
        if self.pp_size > 1:
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:num_tokens] for k, v in self.pp_proxy_tensors.items()}
            )

        if self.require_mlp_tp_gather:
            global_num_tokens_cpu = [num_tokens] * self.dp_size
            global_num_tokens = torch.tensor(
                [num_tokens] * self.dp_size,
                dtype=torch.int64,
                device=input_ids.device,
            )
        elif self.require_attn_tp_gather:
            global_num_tokens_cpu = [num_tokens]
            global_num_tokens = torch.tensor(
                [num_tokens], dtype=torch.int64, device=input_ids.device
            )
        else:
            global_num_tokens_cpu = None
            global_num_tokens = None
            gathered_buffer = None

        spec_info = self.get_spec_info(num_tokens)

        if self.capture_hidden_mode != CaptureHiddenMode.FULL:
            self.capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            encoder_lens=encoder_lens,
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=global_num_tokens,
            gathered_buffer=gathered_buffer,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            num_token_non_padded=num_token_non_padded,
            global_forward_mode=None,
            mm_inputs=[None] * bs,
            lora_ids=[None] * bs,
            global_num_tokens_cpu=global_num_tokens_cpu,
            global_num_tokens_for_logprob_cpu=global_num_tokens_cpu.copy(),
            global_num_tokens_for_logprob_gpu=global_num_tokens.clone(),
            can_run_graph=True,
        )
        forward_batch.is_decode_or_idle = True
        return forward_batch

    def mark_static(
        self, forward_batch: ForwardBatch, pp_proxy_tensors: PPProxyTensors = None
    ):
        def mark_tensor_static(model_input, is_cache=False):
            if model_input is not None:
                if isinstance(model_input, torch.Tensor):
                    torch._dynamo.mark_static(model_input)
                elif is_cache:
                    for buffer_per_layer in model_input:
                        torch._dynamo.mark_static(buffer_per_layer)
                elif isinstance(model_input, PPProxyTensors):
                    for pp_out in model_input.tensors.items():
                        torch._dynamo.mark_static(pp_out)
                elif isinstance(model_input, tuple):
                    for value in model_input:
                        torch._dynamo.mark_static(value)
                else:
                    raise ValueError(
                        f"Unsupported type with mark static: {type(model_input)}"
                    )

        mark_tensor_static(pp_proxy_tensors)
        mark_tensor_static(forward_batch.input_ids)
        mark_tensor_static(forward_batch.positions)
        mark_tensor_static(forward_batch.input_embeds)
        mark_tensor_static(forward_batch.out_cache_loc)
        mark_tensor_static(forward_batch.gathered_buffer)
        mark_tensor_static(forward_batch.attn_backend.forward_metadata.block_kv_indices)
        mark_tensor_static(forward_batch.attn_backend.forward_metadata.cos)
        mark_tensor_static(forward_batch.attn_backend.forward_metadata.sin)
        mark_tensor_static(forward_batch.attn_backend.forward_metadata.seq_lens)
        mark_tensor_static(forward_batch.attn_backend.forward_metadata.actual_seq_lengths)
        mark_tensor_static(forward_batch.attn_backend.forward_metadata.norm_res)
        try:
            mark_tensor_static(forward_batch.token_to_kv_pool.k_buffer, is_cache=True)
            mark_tensor_static(forward_batch.token_to_kv_pool.v_buffer, is_cache=True)
        except AttributeError as e:
            mark_tensor_static(forward_batch.token_to_kv_pool.kv_buffer, is_cache=True)

    def warm_up(self):
        if not self.enable_torch_compile:
            rank0_log(
                "enable_torch_compile is False, model will run eagerly on npu, this may cause performance loss."
                "please set --enable-torch-compile"
            )
            return

        rank0_log("Warming up npu graph")
        backend = get_compiler_backend()
        if not self.enable_cache:
            self.model_runner.model.compile_forward = torch.compile(
                torch.no_grad()(self.model_runner.model.forward),
                fullgraph=True,
                dynamic=False,
                backend=backend,
            )

        compile_range = (
            tqdm.tqdm(list(reversed(self.compile_bs)))
            if get_tensor_model_parallel_rank() == 0
            else reversed(self.compile_bs)
        )

        def build_method(method_name):
            method_code = f"""
def {method_name}(self, input_ids, positions, forward_batch, **kwargs):
    return self.forward(input_ids, positions, forward_batch, **kwargs)
"""
            exec(method_code)
            return locals()[method_name]

        for bs in compile_range:
            if get_tensor_model_parallel_rank() == 0:
                avail_mem = get_available_gpu_memory(
                    self.model_runner.device,
                    self.model_runner.gpu_id,
                    empty_cache=False,
                )
                compile_range.set_description(
                    f"Capturing batches ({avail_mem=:.2f} GB)"
                )
            num_tokens = bs * self.num_tokens_per_bs
            forward_batch = self.prepare_forward_batch(bs, num_tokens)
            if forward_batch.global_num_tokens_cpu is not None:
                forward_batch.prepare_mlp_sync_batch(self.model_runner)
            forward_batch.attn_backend.init_forward_metadata(forward_batch)

            if self.enable_cache:
                import torchair

                method_name = f"forward_{bs}bs"
                compile_method_name = f"compile_forward_{bs}bs"
                setattr(
                    self.model_runner.model,
                    method_name,
                    types.MethodType(
                        build_method(method_name), self.model_runner.model
                    ),
                )
                setattr(
                    self.model_runner.model,
                    compile_method_name,
                    torchair.inference.cache_compile(
                        getattr(self.model_runner.model, method_name), backend=backend
                    ),
                )

            # Run and capture
            def run_once():
                # Clean intermediate result cache for DP attention
                forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = (
                    None
                )

                kwargs = {}
                if (
                    self.pp_size > 1
                    and "pp_proxy_tensors"
                    in inspect.signature(
                        self.model_runner.model.compile_forward
                    ).parameters
                ):
                    kwargs["pp_proxy_tensors"] = forward_batch.pp_proxy_tensors
                # if use dynamic graph, should mark static for inputs
                # self.mark_static(forward_batch, kwargs.get("pp_proxy_tensors"))

                compile_forward = (
                    getattr(self.model_runner.model, compile_method_name)
                    if self.enable_cache
                    else self.model_runner.model.compile_forward
                )

                with torch.no_grad():
                    logits_output_or_pp_proxy_tensors = compile_forward(
                        forward_batch.input_ids,
                        forward_batch.positions,
                        forward_batch,
                        **kwargs,
                    )
                    return logits_output_or_pp_proxy_tensors

            for _ in range(2):
                torch.npu.synchronize()
                self.model_runner.tp_group.barrier()
                run_once()
        return

    @contextmanager
    def get_runner_context(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
    ) -> Generator[Callable[[bool, PPProxyTensors | None], Any], Any, None]:
        def runner_fn(pp_proxy_tensors: Optional[PPProxyTensors]):
            kwargs = {}
            if pp_proxy_tensors is not None:
                kwargs["pp_proxy_tensors"] = pp_proxy_tensors

            compile_method_name = f"compile_forward_{forward_batch.input_ids.size(0)}bs"
            compile_forward = (
                getattr(self.model_runner.model, compile_method_name)
                if self.enable_cache
                else self.model_runner.model.compile_forward
            )
            with torch.no_grad():
                if self.model_runner.profile_save_dir != "":
                    if not self.model_runner.profile_already_start and forward_batch.max_tokens >= self.model_runner.profiler_bs_threshold:
                        logger.info(f"******* start graph profiling . . .")
                        self.model_runner.profiler.start()
                        self.model_runner.profile_already_start = True
                        self.model_runner.profile_step = 0
                ret = compile_forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )
                if self.model_runner.profile_save_dir != "":
                    if self.model_runner.profile_already_start and not self.model_runner.profile_finished:
                        self.model_runner.profile_step += 1
                    if not self.model_runner.profile_finished and self.model_runner.profile_step > self.model_runner.profiler_stop_step:
                        self.model_runner.profiler.stop()
                        self.model_runner.profile_finished = True
                return ret

        if not skip_attn_backend_init:
            forward_batch.attn_backend.init_forward_metadata(forward_batch)

        yield runner_fn

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        graph_bs = forward_batch.batch_size
        if self.require_mlp_tp_gather:
            graph_bs = (
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                else max(forward_batch.global_num_tokens_cpu)
            )
        return bool(
            forward_batch.forward_mode.is_decode_or_idle()
            and self.enable_torch_compile
            and not forward_batch.is_extend_in_batch
            and graph_bs <= max(self.compile_bs)
        )
