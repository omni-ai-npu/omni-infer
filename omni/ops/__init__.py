#
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
# This file is a part of the vllm-ascend project.
#

import torch
import omni.ops.activation  # noqa
import omni.ops.common_fused_moe  # noqa
import omni.ops.fused_moe  # noqa
import omni.ops.layernorm  # noqa
import omni.ops.rotary_embedding  # noqa
import omni.ops.vocab_parallel_embedding  # noqa


class dummyFusionOp:
    default = None

    def __init__(self, name=""):
        self.name = name


def register_dummy_fusion_op() -> None:
    torch.cuda.CUDAGraph = torch.npu.NPUGraph
    torch.ops._C.rms_norm = dummyFusionOp(name="rms_norm")
    torch.ops._C.fused_add_rms_norm = dummyFusionOp(name="fused_add_rms_norm")
    torch.ops._C.static_scaled_fp8_quant = dummyFusionOp(
        name="static_scaled_fp8_quant")
    torch.ops._C.dynamic_scaled_fp8_quant = dummyFusionOp(
        name="dynamic_scaled_fp8_quant")
    torch.ops._C.dynamic_per_token_scaled_fp8_quant = dummyFusionOp(
        name="dynamic_per_token_scaled_fp8_quant")
    torch.ops._C.rms_norm_static_fp8_quant = dummyFusionOp(
        name="rms_norm_static_fp8_quant")
    torch.ops._C.fused_add_rms_norm_static_fp8_quant = dummyFusionOp(
        name="fused_add_rms_norm_static_fp8_quant")
    torch.ops._C.rms_norm_dynamic_per_token_quant = dummyFusionOp(
        name="rms_norm_dynamic_per_token_quant")
