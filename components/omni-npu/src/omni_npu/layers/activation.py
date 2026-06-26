# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import torch
import torch_npu
from vllm.model_executor.layers.activation import SiluAndMul


@SiluAndMul.register_oot
class NPUSiluAndMul(SiluAndMul):
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

        return torch_npu.npu_swiglu(x)