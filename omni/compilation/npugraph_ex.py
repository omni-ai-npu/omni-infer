# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from collections.abc import Callable
from typing import Any

import torch
import torch._inductor.compile_fx
import torch.fx as fx


from vllm.compilation.compiler_interface import CompilerInterface
from vllm.logger import init_logger

logger = init_logger(__name__)


def graph_output_is_tuple(graph: fx.GraphModule) -> bool:
    """Return whether the FX graph output is already a tuple."""
    output_node = graph.graph.output_node()
    if not output_node.args:
        return False

    return_value = output_node.args[0]
    if isinstance(return_value, tuple):
        return True
    return (
        isinstance(return_value, fx.Node)
        and return_value.op == "call_function"
        and return_value.target is tuple
    )


class NpuGraphExAdaptor(CompilerInterface):
    name = "npugraph_ex"

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        runtime_shape: int | None = None,
        key: str | None = None,
    ) -> tuple[Callable | None, Any | None]:

        from omni_npu.compilation.npugraph_ex_config import get_aclgraph_config
        npugraph_ex_config = get_aclgraph_config().npugraph_ex_config
        if not npugraph_ex_config.get("enable", False):
            return graph, None
        from torch._inductor.compile_fx import graph_returns_tuple
        fx_graph = graph.graph
        if not graph_returns_tuple(graph) and not graph_output_is_tuple(graph):
            output_node = fx_graph.output_node()
            return_value = output_node.args[0]
            with fx_graph.inserting_before(output_node):
                tuple_node = fx_graph.create_node("call_function", tuple, args=([return_value], ))
            output_node.args = (tuple_node, )
            graph.recompile()

        import npugraph_ex
        config = npugraph_ex.CompilerConfig()
        config.mode = "npugraph_ex"
        # execute FX graph in eager mode before graph mode to optimize FX graph.
        config.force_eager = True
        # static kernel switch, suitable for static shapes or scenes with less shape changes.
        if npugraph_ex_config.get(
            "static_kernel_compile", False
        ) or npugraph_ex_config.get("super_kernel_optimize", False):
            config.static_kernel_compile = True
            # Control whether to enable super kernel optimize
            if npugraph_ex_config.get("super_kernel_optimize", False):
                config.super_kernel_optimize= True
                config.super_kernel_optimize_options = {
                    "dcci_before_kernel_start": [
                        ".*GroupedMatmul.*",
                        ".*AiInfraSparseFlashAttentionPioneer.*",
                        ".*AiInfraKvRmsNormRopeCache.*",
                        ".*AiInfraScatterBlockUpdate.*",
                        ".*AiInfraFusedInferAttentionSink.*"
                    ],
                    "dcci_disable_on_kernel": [".*"]
                }
        # Acquisition count limit configuration
        if npugraph_ex_config.get("capture_limit", 64) != 64:
            config.debug.aclgraph.static_capture_size_limit = npugraph_ex_config.get("capture_limit")
        # Control whether to clone inputs
        if not npugraph_ex_config.get("clone_input", True):
            config.clone_input = False
        # Control whether to clone outputs
        if npugraph_ex_config.get("clone_output", False):
            config.clone_output = True 
        # Redundant operator elimination configuration
        if not npugraph_ex_config.get("remove_noop_ops", True):
            config.remove_noop_ops = False
        # Replace non-in-place operators in the middle of FX graph with in-place operators
        if not npugraph_ex_config.get("inplace_pass", True):
            config.inplace_pass = False
        # Convert operators that were converted to non-in-place operations by dynamo Functionalize in inputs back to
        # in-place operators
        if not npugraph_ex_config.get("input_inplace_pass", True):
            config.input_inplace_pass = False
        # Configuration for enabling operator fusion Pass in FX graph
        if not npugraph_ex_config.get("pattern_fusion_pass", True):
            config.pattern_fusion_pass = False
        # Freeze model weights and computation graph to prevent dynamic changes
        if npugraph_ex_config.get("frozen_parameter", False):
            config.frozen_parameter = True
        # insert the GraphPassManager into the npugraph_ex  config
        if "post_grad_custom_post_pass" in compiler_config:
            config.post_grad_custom_post_pass = compiler_config["post_grad_custom_post_pass"]
        if "post_grad_custom_pre_pass" in compiler_config:
            config.post_grad_custom_pre_pass = compiler_config["post_grad_custom_pre_pass"]
        
        logger.debug(f"static_kernel_compile: {config.static_kernel_compile}")
        logger.debug(f"super_kernel_optimize: {config.super_kernel_optimize}")
        logger.debug(f"capture_limit: {config.capture_limit}")
        logger.debug(f"clone_input: {config.clone_input}")
        logger.debug(f"clone_output: {config.clone_output}")
        logger.debug(f"remove_noop_ops: {config.remove_noop_ops}")
        logger.debug(f"inplace_pass: {config.inplace_pass}")
        logger.debug(f"input_inplace_pass: {config.input_inplace_pass}")
        logger.debug(f"pattern_fusion_pass: {config.pattern_fusion_pass}")
        logger.debug(f"frozen_parameter: {config.frozen_parameter}")
        npugraph_ex_compile = npugraph_ex.get_npu_backend(compiler_config=config)
        compile_graph = npugraph_ex_compile(graph, example_inputs)
        return compile_graph, None