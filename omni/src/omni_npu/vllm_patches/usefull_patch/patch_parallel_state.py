# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This file is based on vLLM implementation:
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# https://github.com/vllm-project/vllm/blob/v0.16.0/vllm/distributed/parallel_state.py

import os
import torch
import torch_npu
from torch.distributed import Backend, ProcessGroup

from vllm.distributed import (
    parallel_state,
    init_model_parallel_group,
    get_world_group,
    get_pp_group,
    get_ep_group
)

from vllm.distributed.parallel_state import (
    GroupCoordinator,
    _register_group,
    _get_unique_name
)
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.system_utils import suppress_stdout
from vllm.logger import init_logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.v1.distributed.parallel_state_ext import (
    initialize_round_swap_comm_group_list,
    initialize_local_comm_group_list,
    initialize_cross_comm_group_list
)

logger = init_logger(__name__)


@register_patch("ParallelStatePatch", parallel_state)
class ParallelStatePatch(VLLMPatch):
    _attr_names_to_apply = ["initialize_model_parallel"]

    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        prefill_context_model_parallel_size: int = 1,
        decode_context_model_parallel_size: int | None = 1,
        backend: str | None = None,
    ) -> None:
        """
        Initialize model parallel groups.

        Arguments:
            tensor_model_parallel_size: number of GPUs used for tensor model
                parallelism.
            pipeline_model_parallel_size: number of GPUs used for pipeline model
                parallelism.
            backend: name of torch distributed communication backend.

        Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
        use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
        the model pipeline. The present function will
        create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
            4 tensor model-parallel groups:
                [g0, g1], [g2, g3], [g4, g5], [g6, g7]
            2 pipeline model-parallel groups:
                [g0, g2, g4, g6], [g1, g3, g5, g7]
        Note that for efficiency, the caller should make sure adjacent ranks
        are on the same DGX box. For example if we are using 2 DGX-1 boxes
        with a total of 16 GPUs, rank 0 to 7 belong to the first box and
        ranks 8 to 15 belong to the second box.
        """
        # Get world size and rank. Ensure some consistencies.
        if not torch.distributed.is_initialized():
            raise RuntimeError("torch.distributed is not initialized")
        world_size: int = get_world_group().world_size
        rank = torch.distributed.get_rank()
        backend = backend or torch.distributed.get_backend(get_world_group().device_group)

        data_parallel_size = 1
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if config is not None:
            data_parallel_size = config.parallel_config.data_parallel_size

        # Use world_group.ranks for rank list
        # the layout order is: ExternalDP x DP x PP x TP
        # ExternalDP is the data parallel group that is not part of the model,
        # every dp rank can generate independently (in verl integration).
        # DP is the data parallel group that is part of the model,
        # all the ranks in the same DP group should generate simultaneously,
        # i.e. the `generate` call in the same DP group should be called together,
        # otherwise it will cause deadlock.
        # to get group_ranks for each dimension, transpose that dimension to the
        # last dimension, then reshape to 2D, then unbind the last dimension
        all_ranks = torch.tensor(get_world_group().ranks).reshape(
            -1,
            data_parallel_size,
            pipeline_model_parallel_size,
            prefill_context_model_parallel_size,
            tensor_model_parallel_size,
        )  # noqa

        # Build the tensor model-parallel groups
        if parallel_state._TP is not None:
            raise RuntimeError("tensor model parallel group is already initialized")
        group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]

        # message queue broadcaster is only used in tensor model parallel group
        parallel_state._TP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=True,
            group_name="tp",
        )

        # Build the DCP model-parallel groups.
        if parallel_state._DCP is not None:
            raise RuntimeError("decode context model parallel group is already initialized")
        # Note(hc): In the current implementation of decode context parallel,
        # dcp_size must not exceed tp_size, because the world size does not
        # change by DCP, it simply reuses the GPUs of TP group, and split one
        # TP group into tp_size//dcp_size DCP groups.
        group_ranks = all_ranks.reshape(-1, decode_context_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        parallel_state._DCP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=True,
            group_name="dcp",
        )

        if parallel_state._PCP is not None:
            raise RuntimeError("prefill context parallel group is already initialized")
        group_ranks = (
            all_ranks.transpose(3, 4)
            .reshape(-1, prefill_context_model_parallel_size)
            .unbind(0)
        )
        group_ranks = [x.tolist() for x in group_ranks]
        parallel_state._PCP = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="pcp"
        )

        # Build the pipeline model-parallel groups
        if parallel_state._PP is not None:
            raise RuntimeError("pipeline model parallel group is already initialized")
        group_ranks = (
            all_ranks.transpose(2, 4).reshape(-1, pipeline_model_parallel_size).unbind(0)
        )
        group_ranks = [x.tolist() for x in group_ranks]
        parallel_state._PP = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="pp"
        )

        if parallel_state._DP is not None:
            raise RuntimeError("data parallel group is already initialized")
        group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        parallel_state._DP = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="dp"
        )

        if parallel_state._EP is not None:
            raise RuntimeError("expert parallel group is already initialized")
        if getattr(parallel_state, "_MOE_DISPATCH_EP", None) is not None:
            raise RuntimeError("moe dispatch expert parallel group is already initialized")
        # Don't create EP group for dense models.
        if config is not None and config.model_config is not None and not config.model_config.is_moe:
            return
        
        group_ranks = (
            all_ranks.transpose(1, 2)
            .reshape(
                -1,
                data_parallel_size
                * prefill_context_model_parallel_size
                * tensor_model_parallel_size,
            )
            .unbind(0)
        )
        group_ranks = [x.tolist() for x in group_ranks]
        parallel_state._EP = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="ep"
        )
        parallel_state._MOE_DISPATCH_EP = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="moe_dispatch_ep"
        )
        # If no EP group needed, _EP remains None

        logger.info_once(
            "rank %s in world size %s is assigned as "
            "DP rank %s, PP rank %s, PCP rank %s, "
            "TP rank %s, EP rank %s",
            rank,
            world_size,
            parallel_state._DP.rank_in_group,
            parallel_state._PP.rank_in_group,
            parallel_state._PCP.rank_in_group,
            parallel_state._TP.rank_in_group,
            parallel_state._EP.rank_in_group if parallel_state._EP is not None else "N/A",
        )
        device_name = torch_npu.npu.get_device_name(0)

        if device_name.startswith("Ascend910B"):
            is_a2_device = True
        else:
            is_a2_device = False

        if is_a2_device:
            num_nodes = torch.distributed.get_world_size() // 8
            if num_nodes > 0:
                initialize_local_comm_group_list(backend)
            if num_nodes >= 2 and num_nodes % 2 == 0:
                initialize_round_swap_comm_group_list(backend)
                initialize_cross_comm_group_list(backend)


@register_patch("GroupCoordinatorPatch", GroupCoordinator)
class GroupCoordinatorPatch(VLLMPatch):

    _attr_names_to_apply = ['__init__', 'use_local_synchronization', 'swap', 'all_gather_into_tensor']

    use_local_synchronization = False

    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str | Backend,
        use_device_communicator: bool,  # whether to use device communicator
        use_message_queue_broadcaster: bool = False,
        group_name: str | None = None,
    ):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank

        self_device_group = None
        self_cpu_group = None
        logger.info(f"{self.use_local_synchronization=}")

        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        if group_name == "moe_dispatch_ep":
            hccl_buffer_size = os.getenv('HCCL_MOE_DISPATCH_EP_BUFFSIZE', os.getenv('HCCL_BUFFSIZE', 200))
            hccl_config = {
                "group_name": self.unique_name,
                "hccl_buffer_size": int(hccl_buffer_size)
            }
        else:
            hccl_config = {
                "group_name": self.unique_name,
            }
        if "HCCL_OP_EXPANSION_MODE" not in os.environ:
            hccl_config["hccl_op_expansion_mode"] = 3
        options.hccl_config = hccl_config

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks,
                backend=torch_distributed_backend,
                use_local_synchronization=self.use_local_synchronization,
                pg_options=options,
            )
            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            with suppress_stdout():
                cpu_group = torch.distributed.new_group(
                    ranks,
                    backend="gloo",
                    use_local_synchronization=self.use_local_synchronization,
                )
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self_device_group = device_group
                self_cpu_group = cpu_group

        if self_cpu_group is None:
            raise RuntimeError("self_cpu_group is not initialized")
        if self_device_group is None:
            raise RuntimeError("self_device_group is not initialized")

        self.cpu_group = self_cpu_group
        self.device_group = self_device_group

        from vllm.platforms import current_platform

        if current_platform.is_cuda_alike():
            self.device = torch.device(f"cuda:{local_rank}")
        elif current_platform.is_xpu():
            self.device = torch.device(f"xpu:{local_rank}")
        elif current_platform.is_out_of_tree():
            self.device = torch.device(f"{current_platform.device_name}:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self.use_device_communicator = use_device_communicator
        self.device_communicator = None
        if use_device_communicator and self.world_size > 1:
            device_comm_cls = resolve_obj_by_qualname(
                current_platform.get_device_communicator_cls()
            )
            self.device_communicator = device_comm_cls(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
            )

        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        self.mq_broadcaster: MessageQueue | None = None
        if use_message_queue_broadcaster and self.world_size > 1:
            self.mq_broadcaster = MessageQueue.create_from_process_group(
                self.cpu_group, 1 << 22, 6
            )


        self.use_custom_op_call = (
            current_platform.is_cuda_alike() or current_platform.is_tpu()
        )

        self.use_cpu_custom_send_recv = current_platform.is_cpu() and hasattr(
            torch.ops._C, "init_shm_manager"
        )

    def swap(self, input_: torch.Tensor, method="all2allv") -> torch.Tensor:
        if len(self.ranks) != 2:
            return input_

        if method == "all2allv":
            rank_0 = self.ranks[0]
            rank_1 = self.ranks[1]
            input_shape = input_.shape
            input_ = input_.view(-1)
            output = torch.empty_like(input_, dtype=input_.dtype, device=input_.device)

            if self.rank == rank_0:
                split_sizes = [0, input_.shape[0]]
            elif self.rank == rank_1:
                split_sizes = [input_.shape[0], 0]

            torch.distributed.all_to_all_single(output, input_,
                                                output_split_sizes=split_sizes,
                                                input_split_sizes=split_sizes,
                                                group=self.device_group)
            return output.view(input_shape)

        if method == "allgather":
            rank_0 = self.ranks[0]
            rank_1 = self.ranks[1]
            output = torch.empty_like(input_, dtype=input_.dtype, device=input_.device)
            input_size = input_.size()
            output_size = (input_size[0] * 2, ) + input_size[1:]
            output_tensor = torch.empty(output_size, dtype=input_.dtype, device=input_.device)
            torch.distributed.all_gather_into_tensor(output_tensor, input_, group=self.device_group)

            if self.rank == rank_1:
                output, _ = torch.split(output_tensor, output_tensor.shape[0] // 2, dim=0)
            elif self.rank == rank_0:
                _, output = torch.split(output_tensor, output_tensor.shape[0] // 2, dim=0)

            return output
        return input_
    
    def all_gather_into_tensor(self, output_: torch.Tensor, input_: torch.Tensor) -> torch.Tensor:
        return self.device_communicator.all_gather_into_tensor(output_, input_)
