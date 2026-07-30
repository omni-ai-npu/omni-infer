# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from typing import TYPE_CHECKING

from omni_npu.configs import OmniAdditionalConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class AclGraphConfig:
    """
    Configuration Object for additional_config from vllm.configs.
    """

    def __init__(self, vllm_config: "VllmConfig"):
        self.additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}
        omni_additional_config = OmniAdditionalConfig.from_vllm_config(
            vllm_config)
        self.npugraph_ex_config = omni_additional_config.npugraph_ex_config
        
_ACLGRAPH_CONFIG: AclGraphConfig | None = None  


def init_aclgraph_config(vllm_config):

    global _ACLGRAPH_CONFIG
    if _ACLGRAPH_CONFIG is not None:
        return _ACLGRAPH_CONFIG
    _ACLGRAPH_CONFIG = AclGraphConfig(vllm_config)
    return _ACLGRAPH_CONFIG


def get_aclgraph_config():
    global _ACLGRAPH_CONFIG
    if _ACLGRAPH_CONFIG is None:
        raise RuntimeError("Ascend config is not initialized. Please call init_aclgraph_config first.")
    return _ACLGRAPH_CONFIG