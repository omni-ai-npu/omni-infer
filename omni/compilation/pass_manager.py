# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from torch import fx
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.compilation.passes.inductor_pass import get_pass_context
from vllm.compilation.passes.pass_manager import PostGradPassManager
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass

from omni_npu.configs import OmniAdditionalConfig
from omni_npu.logger import update_configure_vllm_root_logger


update_configure_vllm_root_logger()
logger = init_logger(__name__)


class GraphPassManager(PostGradPassManager):
    """
    NPU-specific pass manager that extends PostGradPassManager.
    This manager supports configuring graph optimization passes through
    `additional_config["npugraph_ex_config"]`, including operator fusion
    (e.g. merge_dynamic_quant) and other NPU-specific optimizations.
    The __call__ signature matches torchair's post_grad_custom_post_pass
    contract: (graph, example_inputs, config) -> graph.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, graph: fx.Graph, example_inputs, config) -> fx.Graph:
        compile_range = get_pass_context().compile_range
        for pass_ in self.passes:
            if pass_.is_applicable_for_range(compile_range):
                pass_(graph)
        graph.recompile()
        return graph

    def add(self, pass_: VllmInductorPass):
        self.passes.append(pass_)

    def configure(self, config: VllmConfig):
        self.npugraph_ex_config: dict = (
            OmniAdditionalConfig.from_vllm_config(
                config
            ).npugraph_ex_config
        )
        if self.npugraph_ex_config.get("merge_dynamic_quant", False):
            from omni_npu.compilation.passes.merge_dynamic_quant_pass import (
                MergeDynamicQuantPass,
            )

            self.passes.append(MergeDynamicQuantPass(config))
        if self.npugraph_ex_config.get("enable_moe_multistream", False):
            pass
