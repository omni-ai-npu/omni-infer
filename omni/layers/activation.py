# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import nullcontext
from typing import Any

import torch
import torch_npu
from vllm.model_executor.layers.activation import SiluAndMul

from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.v1.utils import on_ascend950


@SiluAndMul.register_oot
class NPUSiluAndMul(SiluAndMul):
    def __init__(self):
        super().__init__()
        self.should_limit_core = (
            model_extra_config.task_config.graph_mode == "acl_graph"
            and on_ascend950()
        )

    def forward_oot(
        self,
        x: torch.Tensor | dict[str, Any],
        quant_symbol: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        if quant_symbol and isinstance(x, dict):
            kwargs = {
                "x": x.get("x_int8"),
                "weight_scale": x.get("out_scale").to(torch.float32),
                "quant_scale": x.get("in_scale", None),
                "activation_scale": x.get("pertoken_scale", None),
                "bias": None,
                "quant_offset": None,
                "group_index": None,
                "activate_left": True,
                "quant_mode": 1,
            }
            h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(**kwargs)
            return {"x_int8": h, "pertoken_scale": pertoken_scale}

        core_limit_ctx = (
            torch.npu.npugraph_ex.scope.limit_core_num(0, 8)
            if self.should_limit_core
            else nullcontext()
        )
        with core_limit_ctx:
            return torch_npu.npu_swiglu(x)
