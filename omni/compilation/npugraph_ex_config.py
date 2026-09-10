# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from typing import TYPE_CHECKING

from omni_npu.configs import OmniAdditionalConfig
from omni_npu.v1.utils import on_ascend950

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
        self.enable_sk_scope = self._should_enable_sk_scope()

    def _should_enable_sk_scope(self) -> bool:
        if not (
            self.npugraph_ex_config.get("enable", False)
            and self.npugraph_ex_config.get("super_kernel_optimize", False)
        ):
            return False
        if on_ascend950():
            return False
        from omni_npu.model_config.config_loader.loader import model_extra_config
        return bool(model_extra_config.operator_opt_config.enable_sk_scope)


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


def enable_sk_scope() -> bool:
    """Whether named ``sk_scope`` wrappers may fire.

    Master SK switch is npugraph_ex ``super_kernel_optimize`` (also needs
    ``enable``). This flag only turns on the profitable scopes.
    """
    if _ACLGRAPH_CONFIG is None:
        return False
    return _ACLGRAPH_CONFIG.enable_sk_scope