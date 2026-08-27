# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU sampling ops for the V2 runner. Implementation: omni/worker/npu/sampler.py.

gumbel_sample: upstream reaches tldevice.log1p, a returning-None stub on
triton-ascend 3.2.2. apply_top_k_top_p: the upstream Qrita Triton kernel fails
to compile there, so the PyTorch sort path is taken unconditionally.

Only V2 consumers are replaced. The defining module
vllm/v1/sample/ops/topk_topp_sampler.py is left alone because MRv1's
TopKTopPSampler uses it. Bindings are listed one by one: patches are applied
after the consumers imported the names, so replacing the defining module alone
covers nothing. Add a class when upstream adds a consumer.
"""

from vllm.logger import init_logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.worker.npu.sampler import apply_top_k_top_p, gumbel_sample

logger = init_logger(__name__)

try:
    import vllm.v1.worker.gpu.sample.gumbel as up_gumbel
    import vllm.v1.worker.gpu.sample.sampler as up_sampler
    import vllm.v1.worker.gpu.sample.states as up_states
    import vllm.v1.worker.gpu.spec_decode.speculator as up_speculator
except ImportError:
    # Log and skip: this directory is imported wholesale by the plugin and
    # exec_module has no try/except, so raising here takes down MRv1 too.
    logger.error(
        "[omni-npu/mrv2] sample patch targets unavailable; not registered",
        exc_info=True,
    )
else:

    @register_patch("MRv2GumbelPatch", up_gumbel)
    class MRv2GumbelPatch(VLLMPatch):
        """Defining module."""

        _attr_names_to_apply = ["gumbel_sample"]

        gumbel_sample = gumbel_sample

    @register_patch("MRv2SamplerPatch", up_sampler)
    class MRv2SamplerPatch(VLLMPatch):
        """Both names Sampler.sample resolves at call time."""

        _attr_names_to_apply = ["gumbel_sample", "apply_top_k_top_p"]

        gumbel_sample = gumbel_sample
        apply_top_k_top_p = apply_top_k_top_p

    @register_patch("MRv2SamplerStatesPatch", up_states)
    class MRv2SamplerStatesPatch(VLLMPatch):
        """SamplingStates.apply_top_k_top_p, reached by the rejection sampler."""

        _attr_names_to_apply = ["apply_top_k_top_p"]

        apply_top_k_top_p = apply_top_k_top_p

    @register_patch("MRv2SpeculatorGumbelPatch", up_speculator)
    class MRv2SpeculatorGumbelPatch(VLLMPatch):
        """Draft sampling, shared by the MTP/EAGLE speculators."""

        _attr_names_to_apply = ["gumbel_sample"]

        gumbel_sample = gumbel_sample


try:
    import vllm.v1.worker.gpu.spec_decode.dspark.speculator as up_dspark_speculator
except ImportError:
    logger.warning(
        "[omni-npu/mrv2] dspark speculator unavailable; gumbel_sample not patched there"
    )
else:

    @register_patch("MRv2DsparkSpeculatorGumbelPatch", up_dspark_speculator)
    class MRv2DsparkSpeculatorGumbelPatch(VLLMPatch):
        _attr_names_to_apply = ["gumbel_sample"]

        gumbel_sample = gumbel_sample
