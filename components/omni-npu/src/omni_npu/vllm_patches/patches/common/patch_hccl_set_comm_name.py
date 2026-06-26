# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from datetime import timedelta
from typing import Any, Optional, Union

import torch
from torch._C._distributed_c10d import Store

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


_original_new_group = torch.distributed.new_group
@register_patch("NewGroupHCCLPatch", torch.distributed)
class NewGroupHCCLPatch(VLLMPatch):
    """
    Patch torch.distributed.new_group for HCCL backend support.
    """
    _attr_names_to_apply = ["new_group"]

    @staticmethod
    def new_group(
        ranks=None,
        timeout=None,
        backend=None,
        pg_options=None,
        use_local_synchronization=False,
        group_desc=None,
        device_id: Optional[torch.device] = None,
    ):
        #####patch start
        if backend == "hccl":
            import torch_npu
            options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
            options.hccl_config = {"group_name": "GroupCoordinator_new_group"}
            group= _original_new_group(
                ranks=ranks,
                timeout=timeout,
                backend=backend,
                pg_options=options,
                use_local_synchronization=use_local_synchronization,
                group_desc=group_desc,
                device_id=device_id
            )
        #####patch end
        else:
            group= _original_new_group(
                ranks=ranks,
                timeout=timeout,
                backend=backend,
                pg_options=pg_options,
                use_local_synchronization=use_local_synchronization,
                group_desc=group_desc,
                device_id=device_id
            )
        return group


_original_init_process_group = torch.distributed.init_process_group
@register_patch("InitProcessGroupHCCLPatch", torch.distributed)
class InitProcessGroupHCCLPatch(VLLMPatch):
    """
    Patch torch.distributed.init_process_group for HCCL backend support.
    """
    _attr_names_to_apply = ["init_process_group"]

    @staticmethod
    def init_process_group(
        backend: Optional[str] = None,
        init_method: Optional[str] = None,
        timeout: Optional[timedelta] = None,
        world_size: int = -1,
        rank: int = -1,
        store: Optional[Store] = None,
        group_name: str = "",
        pg_options: Optional[Any] = None,
        device_id: Optional[Union[torch.device, int]] = None,
    ) -> None:
        #####patch start
        if backend == "hccl":
            import torch_npu
            options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
            options.hccl_config = {"group_name": "init_distributed_environment"}
            _original_init_process_group(
                backend=backend,
                init_method=init_method,
                timeout=timeout,
                world_size=world_size,
                rank=rank,
                store=store,
                group_name=group_name,
                pg_options=options,
                device_id=device_id
            )
        #####patch end
        else:
            _original_init_process_group(
                backend=backend,
                init_method=init_method,
                timeout=timeout,
                world_size=world_size,
                rank=rank,
                store=store,
                group_name=group_name,
                pg_options=pg_options,
                device_id=device_id
            )
