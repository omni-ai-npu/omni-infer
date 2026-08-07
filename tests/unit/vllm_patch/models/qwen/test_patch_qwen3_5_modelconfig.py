# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import torch

from omni_npu.vllm_patches.patches.models.qwen3_5 import patch_mamba_abstract
from omni_npu.vllm_patches.patches.models.qwen3_5 import patch_modelconfig


def test_qwen35_mtp_convertor_uses_text_config_mtp_layers():
    convertor = patch_modelconfig.Qwen3_5MTPModelArchConfigConvertor.__new__(
        patch_modelconfig.Qwen3_5MTPModelArchConfigConvertor
    )
    convertor.hf_text_config = SimpleNamespace(mtp_num_hidden_layers=2)

    assert convertor.get_num_hidden_layers() == 2
    assert (
        patch_modelconfig.MODEL_ARCH_CONFIG_CONVERTORS["qwen3_5_mtp"]
        is patch_modelconfig.Qwen3_5MTPModelArchConfigConvertor
    )


def test_mamba_spec_patch_counts_base_and_speculative_blocks():
    spec = patch_mamba_abstract.MambaSpec(
        block_size=1,
        shapes=((2, 2),),
        dtypes=(torch.float16,),
        num_speculative_blocks=3,
    )

    assert patch_mamba_abstract.MambaSpecPatch.max_memory_usage_bytes(spec, None) == (
        spec.page_size_bytes * 5
    )


def test_mamba_base_patch_builds_spec_from_vllm_config():
    base = patch_mamba_abstract.MambaBasePatch.__new__(
        patch_mamba_abstract.MambaBasePatch
    )
    base.get_state_shape = lambda: ((2, 2),)
    base.get_state_dtype = lambda: (torch.float16,)
    base.mamba_type = "gdn"
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_block_size=1,
            mamba_page_size_padded=4096,
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=2),
    )

    spec = base.get_kv_cache_spec(vllm_config)

    assert spec.block_size == 1
    assert spec.shapes == ((2, 2),)
    assert spec.dtypes == (torch.float16,)
    assert spec.page_size_padded == 4096
    assert spec.num_speculative_blocks == 2
