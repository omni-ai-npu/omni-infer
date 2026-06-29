# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm.logger import init_logger
logger = init_logger(__name__)

from omni_npu.layers.attention.mm_encoder_attention import NPUMMEncoderAttention
from omni_npu.layers.quantization.compressed_tensors.compressed_tensors import NPUCompressedTensorsConfig
from omni_npu.layers.quantization.hifloat8 import Hifloat8Config
from omni_npu.layers.quantization.mxfp8 import Mxfp8Config
from omni_npu.layers.fused_moe.layer import NPUUnquantizedFusedMoEMethod, NPUFusedMoE
from omni_npu.layers.attention.npu_mla_wrapper import NPUMultiHeadLatentAttentionWrapper
from omni_npu.layers.npu_rms_norm import NPURMSNorm, NPUMiniMaxText01RMSNormTP
from omni_npu.layers.activation import NPUSiluAndMul
from omni_npu.layers.rotary_embedding.rotary_embedding_torch_npu import NPURotaryEmbedding
try: # Avoiding UT case ImportError, for which its running process won't include all the patches
    from omni_npu.layers.rotary_embedding.mrope_interleaved_torch_npu import NPUMRotaryEmbeddingInterleaved
except ImportError:
    logger.warning("MRotaryEmbeddingInterleaved has not being defined, skipping...")
    pass
from omni_npu.layers.rotary_embedding.linear_scaling_rope import NPULinearScalingRotaryEmbedding
from omni_npu.layers.rotary_embedding.llama3_rope import NPULlama3RotaryEmbedding
from omni_npu.layers.rotary_embedding.deepseek_scaling_rope import NPUDeepseekScalingRotaryEmbedding
from omni_npu.layers.rotary_embedding.mrope import NPUMRotaryEmbedding
from omni_npu.layers.rotary_embedding.yarn_scaling_rope import NPUYaRNScalingRotaryEmbedding
