# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py

from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_local_world_group


def tensor_model_local_world_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across local world group."""
    return get_local_world_group().all_reduce(input_)
