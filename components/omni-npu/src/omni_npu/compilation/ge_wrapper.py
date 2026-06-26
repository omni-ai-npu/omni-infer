# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from abc import abstractmethod
import types
from typing import Union

import torch
import torch.nn as nn
import torchair
from vllm.config import VllmConfig
from vllm.logger import init_logger

from omni_npu.compilation.ge_compile_config import get_torchair_config

from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.forward_context import get_forward_context

logger = init_logger(__name__)
torch._dynamo.config.inline_inbuilt_nn_modules=False

def get_tp_pad_size(num_seqs: int):
    tp_size = get_tensor_model_parallel_world_size()
    return (tp_size - num_seqs % tp_size) % tp_size

def GE_graph_padding(
    graph_pad_size: int,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
):
    if graph_pad_size < 0:
        return input_ids, positions

    attn_metadata_dict = get_forward_context().attn_metadata

    if attn_metadata_dict is None:
        return input_ids, positions

    attn_metadata = attn_metadata_dict[next(iter(attn_metadata_dict))]

    slot_mapping = attn_metadata.slot_mapping
    block_table = attn_metadata.block_tables
    seq_lens_tensor = attn_metadata.seq_lens_tensor

    pad_slots = torch.full(
        (graph_pad_size,),
        PAD_SLOT_ID,
        dtype=slot_mapping.dtype,
        device=slot_mapping.device,
    )
    pad_positions = torch.zeros(
        (graph_pad_size,),
        dtype=positions.dtype,
        device=positions.device,
    )
    pad_input_ids = torch.zeros(
        (graph_pad_size,),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )

    # concat 1D vectors
    slot_mapping = torch.cat([slot_mapping, pad_slots], dim=0)
    input_ids = torch.cat([input_ids, pad_input_ids], dim=0)
    positions = torch.cat([positions, pad_positions], dim=0)

    if attn_metadata.num_prefills > 0:
        pad_block_table = torch.zeros(
            (1,) + block_table.shape[1:],
            dtype=block_table.dtype,
            device=block_table.device,
        )
        block_table = torch.cat([block_table, pad_block_table], dim=0)

        # write back
        attn_metadata.slot_mapping = slot_mapping
        attn_metadata.block_tables = block_table
        attn_metadata.seq_lens.append(graph_pad_size)
        attn_metadata.query_cumlens.append(input_ids.shape[0])
    else:
        pad_seq_lens_tensor = torch.ones(
            (graph_pad_size,),
            dtype=seq_lens_tensor.dtype,
            device=seq_lens_tensor.device,
        )
        pad_block_table = torch.zeros(
            (graph_pad_size,) + block_table.shape[1:],
            dtype=block_table.dtype,
            device=block_table.device,
        )
        seq_lens_tensor = torch.cat([seq_lens_tensor, pad_seq_lens_tensor], dim=0)
        block_table = torch.cat([block_table, pad_block_table], dim=0)

        # write back
        attn_metadata.slot_mapping = slot_mapping
        attn_metadata.block_tables = block_table
        attn_metadata.seq_lens_tensor = seq_lens_tensor
        attn_metadata.seq_lens += [1] * graph_pad_size

    return input_ids, positions

class TorchNpuCompilerWrapperWithCustomDispatcher:

    def __init__(self, vllm_config: VllmConfig, dynamic_arg_dims: dict[str, Union[int, list[int]]]):
        self.compiled_model = None
        self.cached_compiled_models = {}
        self.vllm_config = vllm_config
        self.dynamic_arg_dims = dynamic_arg_dims
        self.do_not_compile = not vllm_config.npu_compilation_config.use_gegraph
        if self.do_not_compile:
            return
        self.compile_dispatcher()

    def compile_dispatcher(self):
        backend = self.vllm_config.npu_compilation_config.init_backend(self.vllm_config)
        if not self.vllm_config.npu_compilation_config.use_ge_graph_cached:
            logger.debug("not use ge cache graph")
            self.compiled_model = torch.compile(
                self.forward,
                dynamic=False,
                fullgraph=True,
                backend=backend)

        elif self.vllm_config.npu_compilation_config.use_ge_graph_cached:
            logger.debug("use ge cache graph")
            for gear_size in self.vllm_config.npu_compilation_config.decode_gear_list:
                new_forward_proxy_name = f"{self.__class__.__name__}_forward_with_gear_size_{gear_size}"
                code = self.forward.__code__
                new_code = code.replace(co_name=new_forward_proxy_name, )
                new_func = types.FunctionType(new_code, self.forward.__globals__,
                                              name=new_forward_proxy_name,
                                              argdefs=self.forward.__defaults__)
                self.__dict__[new_forward_proxy_name] = new_func.__get__(self, nn.Module)
                config = get_torchair_config(self.vllm_config)
                self.cached_compiled_models[gear_size] = torchair.inference.cache_compile(
                    self.__dict__[new_forward_proxy_name],
                    config=config,
                    dynamic=False,
                    ge_cache=True,
                    fullgraph=True,
                )
                logger.debug(f"[use cache npu graph], the method name = {new_forward_proxy_name}")

    def call_dispatcher(self, *args, **kwargs):
        use_eager_model = self.should_use_eager_mode(*args, **kwargs)
        if self.do_not_compile or use_eager_model:
            logger.debug(f"[ge graph] call_dispatcher do_not_compile:{self.do_not_compile},use_eager_model:{use_eager_model}")
            return self.forward(*args, **kwargs)

        if not self.vllm_config.npu_compilation_config.use_ge_graph_cached:
            return self.compiled_model(*args, **kwargs)

        elif self.vllm_config.npu_compilation_config.use_ge_graph_cached:
            gear_size = args[0].shape[0]
            return self.cached_compiled_models[gear_size](*args, **kwargs)

        logger.error(f"encountered a missed scene")
        return None

    def __call__(self, *args, **kwargs):
        logger.debug(f"[ge graph], call enter")
        if len(args) > 0:
            inputs = args[0]
            positions = args[1]
            args = args[2:]
        elif len(kwargs) > 0:
            if kwargs["input_ids"] is not None:
                inputs = kwargs["input_ids"]
            else:
                inputs = kwargs["inputs_embeds"]
            positions = kwargs["positions"]
        else:
            raise ValueError("No input_ids or inputs_embeds found in kwargs")

        attn_metadata = get_forward_context().attn_metadata
        batch_descriptor = get_forward_context().batch_descriptor
        uniform = batch_descriptor.uniform if batch_descriptor is not None else False
        is_prefill = attn_metadata is None or attn_metadata[next(iter(attn_metadata))].num_prefills > 0

        gear_size = inputs.shape[0]
        if attn_metadata is not None and attn_metadata[next(iter(attn_metadata))].num_prefills > 0:
            graph_pad_size = get_tp_pad_size(gear_size)
        else:
            graph_pad_size = self.vllm_config.scheduler_config.max_num_seqs - gear_size
        inputs, positions = GE_graph_padding(graph_pad_size, inputs, positions)
        kwargs["input_ids"] = inputs
        kwargs["positions"] = positions

        if is_prefill or not uniform:
            logger.debug(f"<<< [ge graph]use original forward")
            return self.forward(*args, **kwargs)

        return self.call_dispatcher(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    @abstractmethod
    def should_use_eager_mode(self, *args, **kwargs):
        ...