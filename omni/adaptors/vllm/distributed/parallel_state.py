# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
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

from typing import Optional, List, Any, Union

import torch
import torch.distributed
from torch.distributed import Backend
from vllm.distributed import GroupCoordinator as GroupCoordinatorGPU
from vllm.logger import logger
from vllm.distributed import (
    parallel_state,
    init_model_parallel_group,
    get_world_group,
    get_pp_group,
    get_ep_group,
    get_tp_group
)
from vllm.logger import logger
from vllm.config import get_current_vllm_config
from omni.models.config_loader.loader import model_extra_config
import os
import torch_npu

_DIE_PER_NODE_910C = 16
_DIE_PER_NODE_910B = 8

is_device_a2 = os.getenv("ASCEND_PLATFORM", "A3") == "A2"


def get_npu_device_count():
    if is_device_a2:
        return _DIE_PER_NODE_910B
    else:
        return _DIE_PER_NODE_910C


class GroupCoordinator(GroupCoordinatorGPU):
    use_local_synchronization = False

    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_device_communicator: bool,  # whether to use device communicator
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,
    ):
        group_name = group_name or "anonymous"
        self.unique_name = parallel_state._get_unique_name(group_name)
        parallel_state._register_group(self)

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None

        torch_distributed_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        if group_name != "ep":
            hccl_buffer_size = os.getenv('HCCL_BASIC_BUFFSIZE', os.getenv('HCCL_BUFFSIZE', 200))
            torch_distributed_options.hccl_config = {"hccl_buffer_size":int(hccl_buffer_size)}

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend, pg_options=torch_distributed_options,
                use_local_synchronization=self.use_local_synchronization)
            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            cpu_group = torch.distributed.new_group(ranks, backend="gloo", use_local_synchronization=self.use_local_synchronization)
            if self.rank in ranks:
                from omni.adaptors.vllm.token_recovery.ha_patches import register_process_group
                register_process_group(device_group)
                register_process_group(cpu_group)
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.cpu_group = cpu_group

        assert self.cpu_group is not None
        assert self.device_group is not None

        from vllm.platforms import current_platform

        if current_platform.is_cuda_alike():
            self.device = torch.device(f"cuda:{local_rank}")
        elif current_platform.is_out_of_tree():
            self.device = torch.device(
                f"{current_platform.device_name}:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self.use_device_communicator = use_device_communicator

        self.device_communicator = None
        if use_device_communicator and self.world_size > 1:
            from vllm.utils import resolve_obj_by_qualname
            device_comm_cls = resolve_obj_by_qualname(
                current_platform.get_device_communicator_cls())
            self.device_communicator = device_comm_cls(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
            )

        from vllm.distributed.device_communicators.shm_broadcast import (
            MessageQueue)
        self.mq_broadcaster: Optional[MessageQueue] = None
        if use_message_queue_broadcaster and self.world_size > 1:
            self.mq_broadcaster = MessageQueue.create_from_process_group(
                self.cpu_group, 1 << 22, 6)

        from vllm.platforms import current_platform
        self.use_custom_op_call = (current_platform.is_cuda_alike()
                                   or current_platform.is_tpu())

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: Optional[List[int]] = None,
        gather_sizes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        return self.device_communicator.all_to_all(input_, scatter_dim, gather_dim, scatter_sizes, gather_sizes)

    def swap(self, input: torch.Tensor, method="all2allv") -> torch.Tensor:
        if len(self.ranks) != 2:
            return input

        if method == "all2allv":
            rank_0 = self.ranks[0]
            rank_1 = self.ranks[1]
            input_shape = input.shape
            input = input.view(-1)
            output = torch.empty_like(input, dtype=input.dtype, device=input.device)

            if self.rank == rank_0:
                split_sizes = [0, input.shape[0]]
            elif self.rank == rank_1:
                split_sizes = [input.shape[0], 0]

            torch.distributed.all_to_all_single(output, input,
                                                output_split_sizes=split_sizes,
                                                input_split_sizes=split_sizes,
                                                group=self.device_group)
            return output.view(input_shape)

        if method == "allgather":
            rank_0 = self.ranks[0]
            rank_1 = self.ranks[1]
            output = torch.empty_like(input, dtype=input.dtype, device=input.device)
            input_size = input.size()
            output_size= (input_size[0] * 2, ) + input_size[1:]
            output_tensor = torch.empty(output_size, dtype=input.dtype, device=input.device)
            torch.distributed.all_gather_into_tensor(output_tensor, input, group=self.device_group)

            if self.rank == rank_1:
                output, _ = torch.split(output_tensor, output_tensor.shape[0] // 2, dim=0)
            elif self.rank == rank_0:
                _, output = torch.split(output_tensor, output_tensor.shape[0] // 2, dim=0)

            return output
        return input

    def all_reduce_async(self, input_: torch.Tensor):
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_, None

        return self.device_communicator.all_reduce_async(input_)

    def all_gather_async(self, input_: torch.Tensor, dim: int = -1, output_tensor: Optional[torch.tensor] = None):
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        # assert -input_.dim() <= dim < input_.dim(), (
        #     f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")

        return self.device_communicator.all_gather_async(input_, dim, output_tensor)

    def reduce_scatter_async(self, input_: torch.Tensor, output_tensor: Optional[torch.tensor] = None) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        return self.device_communicator.reduce_scatter_async(input_, output_tensor)

    def all_gather_v(self, output_list_: List[torch.Tensor], input_: torch.Tensor) -> None:
        self.device_communicator.all_gather_v(output_list_, input_)

    def reduce_scatter_v(self, output_: torch.Tensor, input_list_: List[torch.Tensor]) -> None:
        self.device_communicator.reduce_scatter_v(output_, input_list_)

_NUM_COMM_GROUP = 2
_LOCAL_COMM_LIST = None
_CROSS_COMM_LIST = None
_GLOBAL_COMM_LIST = None
_CROSS_FAR_COMM_LIST = None
_CROSS_NEAR_COMM_LIST = None
_CROSS_ROUND_COMM_LIST = None
# kept for backward compatibility
_LOCAL_WORLD: Optional[GroupCoordinator] = None
_MLP_TP: Optional[GroupCoordinator] = None
_STREAM1_ATTN_GROUP: Optional[GroupCoordinator] = None
_STREAM1_MLP_GROUP: Optional[GroupCoordinator] = None
_STREAM1_MOE_GROUP: Optional[GroupCoordinator] = None
_SCALE_PARALLEL_GROUP: Optional[GroupCoordinator] = None
GROUP_STREAM1_ATTN = "stream1_attn" # p侧使能双micro batch为第二个流创建 attention 层通信域
GROUP_STREAM1_MLP = "stream1_mlp" # p侧使能双micro batch为第二个流创建 mlp 层通信域
GROUP_STREAM1_MOE = "stream1_moe" # p侧使能双micro batch为第二个流创建 moe 层通信域
_O_PROJ_TP: Optional[GroupCoordinator] = None
_O_PROJ_DP: Optional[GroupCoordinator] = None
_EH_PROJ_TP: Optional[GroupCoordinator] = None
_MLA_CP: Optional[GroupCoordinator] = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    backend: Optional[str] = None,
) -> None:
    # TP、PP、EP、DP
    # NOTE: avoid reshape error in parallel_state.initialize_model_parallel with RL,
    #       such as torch.arange(96).reshape(-1, 64, 1, 1), where world_size is 96 and 64 dies for D
    initialize_model_parallel_default(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        backend,
    )
    initialize_mlp_tp_group(backend)
    initialize_local_world_group(backend)
    initialize_eh_proj_tp_group(backend)

    if model_extra_config.operator_opt_config.enable_prefill_micro_batch:
        initialize_stream1_attn_group(backend)
        initialize_stream1_mlp_group(backend)
        initialize_stream1_moe_group(backend)

    if model_extra_config.parall_config.o_proj_tp_size > 1:
        initialize_o_proj_tp_group(backend)
        initialize_o_proj_dp_group(backend)

    if model_extra_config.operator_opt_config.use_dcp:
        initialize_mla_cp_group(backend)
    
    if is_device_a2 or not model_extra_config.operator_opt_config.prefill_moe_all_to_all:

        if not model_extra_config.operator_opt_config.two_stage_comm:
            initialize_world_comm_group_list(backend)
        
        initialize_cross_comm_group_list(backend)
        initialize_local_comm_group_list(backend)

        num_nodes = torch.distributed.get_world_size() // get_npu_device_count()
        if num_nodes == 4 and model_extra_config.operator_opt_config.enable_round_pipeline_comm:
            initialize_round_cross_comm_group_list(backend)

        if model_extra_config.operator_opt_config.enable_pipeline_comm:
            initialize_far_cross_comm_group_list(backend)
            initialize_near_cross_comm_group_list(backend)

    scale_parallel = model_extra_config.operator_opt_config.enable_scale_parallel
    if scale_parallel:
        initial_scale_parallel_group(backend)

# called when world_ranks is built by RL
def init_world_group(ranks: list[int], local_rank: int, backend: str):
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if parallel_state._WORLD is not None:
        raise RuntimeError("_WORLD must not be initialized")
    world_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    logger.info(f"worker init world group {ranks=}, {local_rank=}, {backend=}, {world_rank=}, {world_size=}")
    if len(ranks) != world_size:
        GroupCoordinator.use_local_synchronization = True
    world_group = parallel_state.init_world_group(
        ranks,
        local_rank,
        backend,
    )
    parallel_state._WORLD = world_group
    logger.info(f"worker init world group done {ranks=}, {local_rank=}, {backend=}, {world_rank=}")

def initialize_model_parallel_default(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.

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
    assert torch.distributed.is_initialized()
    world_size: int = get_world_group().world_size
    rank = torch.distributed.get_rank()
    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)

    data_parallel_size = 1
    from vllm.config import get_current_vllm_config
    config = get_current_vllm_config()
    if config is not None:
        data_parallel_size = config.parallel_config.data_parallel_size

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
        -1, data_parallel_size, pipeline_model_parallel_size,
        tensor_model_parallel_size)  # noqa

    # Build the tensor model-parallel groups.
    assert parallel_state._TP is None, ("tensor model parallel group is already initialized")
    group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]

    # message queue broadcaster is only used in tensor model parallel group
    parallel_state._TP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    use_message_queue_broadcaster=True,
                                    group_name="tp")

    # Build the pipeline model-parallel groups.
    assert parallel_state._PP is None, (
        "pipeline model parallel group is already initialized")
    group_ranks = all_ranks.transpose(2, 3).reshape(
        -1, pipeline_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    parallel_state._PP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="pp")

    assert parallel_state._DP is None, ("data parallel group is already initialized")
    group_ranks = all_ranks.transpose(1,
                                      3).reshape(-1,
                                                 data_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    parallel_state._DP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="dp")

    assert parallel_state._EP is None, ("expert parallel group is already initialized")
    group_ranks = all_ranks.transpose(1, 2).reshape(
        -1, data_parallel_size * tensor_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    parallel_state._EP = init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="ep")

    logger.info(
        "rank %s in world size %s is assigned as "
        "DP rank %s, PP rank %s, TP rank %s, EP rank %s", rank, world_size,
        parallel_state._DP.rank_in_group, parallel_state._PP.rank_in_group, parallel_state._TP.rank_in_group,
        parallel_state._EP.rank_in_group)


def initial_scale_parallel_group(backend: Optional[str] = None):
    config = get_current_vllm_config()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    data_parallel_size = config.parallel_config.data_parallel_size
    pipeline_model_parallel_size = config.parallel_config.pipeline_parallel_size
    tensor_model_parallel_size = config.parallel_config.tensor_parallel_size

    all_ranks = torch.arange(world_size).reshape(-1, data_parallel_size, pipeline_model_parallel_size,
                tensor_model_parallel_size)
    group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    global _SCALE_PARALLEL_GROUP
    _SCALE_PARALLEL_GROUP = init_model_parallel_group(group_ranks,
                                    parallel_state.get_world_group().local_rank,
                                    backend,
                                    use_message_queue_broadcaster=False,
                                    group_name="scale_parallel")


def get_mlp_world_group():
    return get_local_group_from_list(0)


def _init_parallel_group_factory(
        group_name: str,
        local_size: int,
        backend: Optional[str] = None,
) -> Any:
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    if local_size <= 0:
        raise RuntimeError(f"local_size must be positive, got {local_size}")
    world_size = torch.distributed.get_world_size()
    if world_size % local_size != 0:
        raise RuntimeError(f"world_size[{world_size}] must be divisible by local_size[{local_size}]")

    num_groups = world_size // local_size
    group_ranks = [
        list(range(i * local_size, (i + 1) * local_size))
        for i in range(num_groups)
    ]
    return init_model_parallel_group(
        group_ranks=group_ranks,
        local_rank=get_world_group().local_rank,
        backend=backend or torch.distributed.get_backend(),
        group_name=f'{group_name}'
    )


def  initialize_stream1_attn_group(backend: Optional[str] = None) -> None:
    global _STREAM1_ATTN_GROUP
    if _STREAM1_ATTN_GROUP is not None:
        raise RuntimeError("stream1 attn group already initialized")
    _STREAM1_ATTN_GROUP = _init_parallel_group_factory(
        group_name=GROUP_STREAM1_ATTN,
        local_size=torch.distributed.get_world_size(),
        backend=backend,
    )


def initialize_stream1_mlp_group(backend: Optional[str] = None) -> None:
    global _STREAM1_MLP_GROUP
    if _STREAM1_MLP_GROUP is not None:
        raise RuntimeError("stream1 mlp group already initialized")
    _STREAM1_MLP_GROUP = _init_parallel_group_factory(
        group_name=GROUP_STREAM1_MLP,
        local_size=16,
        backend=backend,
    )


def initialize_stream1_moe_group(backend: Optional[str] = None) -> None:
    global _STREAM1_MOE_GROUP
    if _STREAM1_MOE_GROUP is not None:
        raise RuntimeError("stream1 moe group already initialized")
    _STREAM1_MOE_GROUP = _init_parallel_group_factory(
        group_name=GROUP_STREAM1_MOE,
        local_size=get_ep_group().world_size,
        backend=backend,
    )


def get_stream1_attn_group() -> Any:
    global _STREAM1_ATTN_GROUP
    if _STREAM1_ATTN_GROUP is None:
        raise RuntimeError("stream1 attn group not initialized")
    return _STREAM1_ATTN_GROUP


def get_stream1_mlp_group() -> Any:
    global _STREAM1_MLP_GROUP
    if _STREAM1_MLP_GROUP is None:
        raise RuntimeError("stream1 mlp group not initialized")
    return _STREAM1_MLP_GROUP


def get_stream1_moe_group() -> Any:
    global _STREAM1_MOE_GROUP
    if _STREAM1_MOE_GROUP is None:
        raise RuntimeError("stream1 moe group not initialized")
    return _STREAM1_MOE_GROUP


def calculate_effective_local_size(local_size: int, world_size: int) -> int:
    """
    Calculate the effective local size based on available devices and world size.

    Args:
        local_size (int): Number of available NPU devices.
        world_size (int): Total number of processes in the distributed setup.

    Returns:
        int: The effective local size (minimum of local_size and world_size).

    Notes:
        - Logs a warning if not all devices are used.
        - Ensures world_size is divisible by the effective local size (raises AssertionError otherwise).
    """
    effective_local_size = min(local_size, world_size)
    if effective_local_size < local_size:
        logger.info(f"Note: Using only {effective_local_size} of {local_size} available NPU devices")

    if world_size % effective_local_size != 0:
        raise AssertionError(
            f"world_size ({world_size}) must be divisible by effective_local_size ({effective_local_size})"
        )
    return effective_local_size


def initialize_mlp_tp_group(backend) -> None:
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = get_world_group().world_size
    mtp_tp_size = model_extra_config.parall_config.dense_mlp_tp_size
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    if world_size % mtp_tp_size != 0:
        raise RuntimeError(f"Dense MLP TP Size ({mtp_tp_size}) should be divisible by world size ({world_size})")
    global _MLP_TP
    if _MLP_TP is not None:
        raise RuntimeError("_MLP_TP must be None")
    group_ranks = [get_world_group().ranks[i : i + mtp_tp_size] for i in range(0, world_size, mtp_tp_size)]

    # message queue broadcaster is only used in tensor model parallel group
    _MLP_TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="mlp_tp_group",
    )


def initialize_o_proj_tp_group(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = get_world_group().world_size
    o_proj_tp_size = model_extra_config.parall_config.o_proj_tp_size
    if world_size % o_proj_tp_size != 0:
        raise RuntimeError(f"o_proj TP Size ({o_proj_tp_size}) should be divisible by world size ({world_size})")
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    if model_extra_config.operator_opt_config.use_dcp and not model_extra_config.task_config.is_prefill_node:
        attn_tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if attn_tp_size >= o_proj_tp_size:
            if attn_tp_size % o_proj_tp_size != 0:
                raise RuntimeError("attn_tp_size must be divided by o_proj_tp_size")
            o_proj_tp_size = attn_tp_size // o_proj_tp_size   #merge_size
        else:
            raise RuntimeError("not support when attn_tp_size < o_proj_tp_size")

    global _O_PROJ_TP
    if _O_PROJ_TP is not None:
        raise RuntimeError("_O_PROJ_TP must be None")
    group_ranks = [get_world_group().ranks[i : i + o_proj_tp_size] for i in range(0, world_size, o_proj_tp_size)]

    # message queue broadcaster is only used in tensor model parallel group
    _O_PROJ_TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=False,
        group_name="o_proj_tp_group",
    )


def initialize_o_proj_dp_group(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = get_world_group().world_size
    o_proj_tp_size = model_extra_config.parall_config.o_proj_tp_size
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    
    if model_extra_config.operator_opt_config.use_dcp and not model_extra_config.task_config.is_prefill_node:
        attn_tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if attn_tp_size >= o_proj_tp_size:
            if attn_tp_size % o_proj_tp_size != 0:
                raise RuntimeError("attn_tp_size must be divided by o_proj_tp_size")
            merge_size = attn_tp_size // o_proj_tp_size   #merge_size
            o_proj_tp_size = merge_size
        else:
            raise RuntimeError("not support when attn_tp_size < o_proj_tp_size")
        
    dp_size: int = world_size // o_proj_tp_size
    global _O_PROJ_DP
    if _O_PROJ_DP is not None:
        raise RuntimeError("_O_PROJ_DP must be None")
    all_ranks = torch.tensor(get_world_group().ranks).reshape(dp_size, o_proj_tp_size)
    group_ranks = all_ranks.transpose(0, 1)
    
    if model_extra_config.operator_opt_config.use_dcp and not model_extra_config.task_config.is_prefill_node:
        group_ranks = group_ranks.reshape(-1, attn_tp_size // merge_size)

    group_ranks = [x.tolist() for x in group_ranks]
    # message queue broadcaster is only used in tensor model parallel group
    _O_PROJ_DP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=False,
        group_name="o_proj_dp_group",
    )

def initialize_mla_cp_group(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = get_world_group().world_size
    mla_cp_size = get_current_vllm_config().parallel_config.tensor_parallel_size
    if world_size % mla_cp_size != 0:
        raise RuntimeError(f"mla CP Size ({mla_cp_size}) should be divisible by world size ({world_size})")
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    global _MLA_CP
    if _MLA_CP is not None:
        raise RuntimeError("_MLA_CP must be None")
    group_ranks = torch.tensor(get_world_group().ranks).reshape(-1, mla_cp_size)
    group_ranks = [x.tolist() for x in group_ranks]
    # message queue broadcaster is only used in tensor model parallel group
    _MLA_CP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=False,
        group_name="mla_cp_group",
    ) 

def initialize_eh_proj_tp_group(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = get_world_group().world_size
    eh_proj_tp_size = model_extra_config.parall_config.eh_proj_tp_size
    if world_size % eh_proj_tp_size != 0:
        raise RuntimeError(f"eh_proj TP Size ({eh_proj_tp_size}) should be divisible by world size ({world_size})")
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    global _EH_PROJ_TP
    if _EH_PROJ_TP is not None:
        raise RuntimeError("_EH_PROJ_TP must be None")
    group_ranks = [get_world_group().ranks[i : i + eh_proj_tp_size] for i in range(0, world_size, eh_proj_tp_size)]

    # message queue broadcaster is only used in tensor model parallel group
    _EH_PROJ_TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=False,
        group_name="eh_proj_tp_group",
    )

def initialize_local_world_group(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = get_world_group().world_size
    local_size = calculate_effective_local_size(torch.npu.device_count() if not int(os.getenv("NO_NPU_MOCK", "0")) \
        else len(os.getenv("ASCEND_RT_VISIBLE_DEVICES").split(",")), world_size)

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    global _LOCAL_WORLD
    if _LOCAL_WORLD is not None:
        raise RuntimeError("_LOCAL_WORLD must be None")
    group_ranks = [get_world_group().ranks[i : i + local_size] for i in range(0, world_size, local_size)]

    # message queue broadcaster is only used in tensor model parallel group
    _LOCAL_WORLD = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="world_local",
    )


def initialize_local_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    local_size = calculate_effective_local_size(torch.npu.device_count() if not int(os.getenv("NO_NPU_MOCK", "0")) \
        else len(os.getenv("ASCEND_RT_VISIBLE_DEVICES").split(",")), world_size)

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    num_local_groups: int = world_size // local_size
    global _LOCAL_COMM_LIST
    if _LOCAL_COMM_LIST is not None:
        raise RuntimeError("_LOCAL_COMM_LIST must be None")
    _LOCAL_COMM_LIST = list()
    group_ranks = []
    for i in range(num_local_groups):
        ranks = list(range(i * local_size, (i + 1) * local_size))
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    comm_group_per_server = 1
    total_comm_groups = comm_group_per_server * num_local_groups + _NUM_COMM_GROUP # one group for topk and the other is redundant
    for i in range(total_comm_groups):
        _LOCAL_COMM_LIST.append(
            init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                use_message_queue_broadcaster=True,
                group_name="world_local",
            )
        )


def initialize_cross_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    local_size = calculate_effective_local_size(torch.npu.device_count() if not int(os.getenv("NO_NPU_MOCK", "0")) \
        else len(os.getenv("ASCEND_RT_VISIBLE_DEVICES").split(",")), world_size)

    server_size = world_size // local_size

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    # Build the pipeline model-parallel groups.
    num_cross_groups: int = world_size // server_size
    global _CROSS_COMM_LIST
    if _CROSS_COMM_LIST is not None:
        raise RuntimeError("pipeline model parallel group is already initialized")
    _CROSS_COMM_LIST = list()
    group_ranks = []
    for i in range(num_cross_groups):
        ranks = list(range(i, world_size, num_cross_groups))
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce

    for i in range(_NUM_COMM_GROUP):
        _CROSS_COMM_LIST.append(
            init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                group_name="world_cross",
            )
        )


def initialize_world_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    global _GLOBAL_COMM_LIST
    if _GLOBAL_COMM_LIST is not None:
        raise RuntimeError("_GLOBAL_COMM_LIST must be None")
    _GLOBAL_COMM_LIST = list()
    group_ranks = [range(world_size)]
    for i in range(_NUM_COMM_GROUP):
        _GLOBAL_COMM_LIST.append(
            init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                use_message_queue_broadcaster=True,
                group_name="world_local",
            )
        )


def get_mlp_tp_group() -> GroupCoordinator:
    return _MLP_TP


def get_o_proj_tp_group() -> GroupCoordinator:
    return _O_PROJ_TP


def get_o_proj_dp_group() -> GroupCoordinator:
    return _O_PROJ_DP

def get_eh_proj_tp_group() -> GroupCoordinator:
    return _EH_PROJ_TP

def get_mla_cp_group() -> GroupCoordinator:
    return _MLA_CP

def get_local_world_group() -> GroupCoordinator:
    return _LOCAL_WORLD

def get_scale_parallel_group() -> GroupCoordinator:
    return _SCALE_PARALLEL_GROUP


def get_local_group_from_list(idx: int) -> GroupCoordinator:
    return _LOCAL_COMM_LIST[idx]


def get_cross_group_from_list(idx: int) -> GroupCoordinator:
    return _CROSS_COMM_LIST[idx]


def get_world_group_from_list(idx: int) -> GroupCoordinator:
    return _GLOBAL_COMM_LIST[idx]


def get_local_group_world_size_from_list(idx: int):
    return _LOCAL_COMM_LIST[idx].world_size


def get_local_group_rank_from_list(idx: int):
    return _LOCAL_COMM_LIST[idx].rank_in_group


def get_near_cross_group_from_list(idx: int) -> GroupCoordinator:
    return _CROSS_NEAR_COMM_LIST[idx]


def get_far_cross_group_from_list(idx: int) -> GroupCoordinator:
    return _CROSS_FAR_COMM_LIST[idx]


def get_pipeline_parallel_world_size():
    return get_pp_group().world_size


def get_local_group_size():
    return get_local_group_from_list(idx=0).world_size


def get_local_group_rank():
    return get_local_group_from_list(idx=0).rank_in_group


def initialize_round_cross_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    local_size = get_npu_device_count()
    assert world_size % local_size == 0

    server_size = world_size // local_size

    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)

    num_cross_groups: int = (world_size // server_size)
    global _CROSS_ROUND_COMM_LIST
    assert _CROSS_ROUND_COMM_LIST is None, (
        "pipeline model parallel group is already initialized")
    _CROSS_ROUND_COMM_LIST = list()

    group_ranks_round0 = []
    group_ranks_round1 = []
    group_ranks_round2 = []
    for i in range(num_cross_groups):
        ranks = [[i + 0 * num_cross_groups, i + 1 * num_cross_groups], \
                [i + 2 * num_cross_groups, i + 3 * num_cross_groups]]
        group_ranks_round0.extend(ranks)

        ranks = [[i + 0 * num_cross_groups, i + 2 * num_cross_groups], \
                [i + 1 * num_cross_groups, i + 3 * num_cross_groups]]
        group_ranks_round1.extend(ranks)

        ranks = [[i + 0 * num_cross_groups, i + 3 * num_cross_groups], \
                [i + 1 * num_cross_groups, i + 2 * num_cross_groups]]
        group_ranks_round2.extend(ranks)


    _CROSS_ROUND_COMM_LIST.append(init_model_parallel_group(group_ranks_round0,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="world_round0_cross"))

    _CROSS_ROUND_COMM_LIST.append(init_model_parallel_group(group_ranks_round1,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="world_round1_cross"))

    _CROSS_ROUND_COMM_LIST.append(init_model_parallel_group(group_ranks_round2,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="world_round2_cross"))


def get_round_cross_group_from_list(round: int) -> GroupCoordinator:
    return _CROSS_ROUND_COMM_LIST[round]


def initialize_far_cross_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    local_size  = get_npu_device_count()
    assert world_size % local_size == 0

    server_size = world_size // local_size

    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)

    num_cross_groups: int = (world_size // server_size)
    global _CROSS_FAR_COMM_LIST
    assert _CROSS_FAR_COMM_LIST is None, (
        "pipeline model parallel group is already initialized")
    _CROSS_FAR_COMM_LIST = list()
    group_ranks = []
    for i in range(num_cross_groups):
        for j in range(server_size // 2):
            ranks = list(range(i + j * num_cross_groups, world_size, world_size // 2))
            group_ranks.append(ranks)

    for i in range(_NUM_COMM_GROUP):
        _CROSS_FAR_COMM_LIST.append(init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="world_far_cross"))


def initialize_near_cross_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    local_size = get_npu_device_count()
    assert world_size % local_size == 0

    server_size = world_size // local_size

    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)

    num_cross_groups: int = (world_size // server_size)
    global _CROSS_NEAR_COMM_LIST
    assert _CROSS_NEAR_COMM_LIST is None, (
        "pipeline model parallel group is already initialized")
    _CROSS_NEAR_COMM_LIST = list()
    group_ranks = []
    for i in range(num_cross_groups):
        ranks = list(range(i, world_size // 2, num_cross_groups))
        group_ranks.append(ranks)

        ranks = list(range(world_size // 2 + i, world_size, num_cross_groups))
        group_ranks.append(ranks)

    for i in range(_NUM_COMM_GROUP):
        _CROSS_NEAR_COMM_LIST.append(init_model_parallel_group(group_ranks,
                                    get_world_group().local_rank,
                                    backend,
                                    group_name="world_near_cross"))