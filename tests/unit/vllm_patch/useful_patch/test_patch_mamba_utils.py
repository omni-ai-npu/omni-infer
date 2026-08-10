import torch

from omni_npu.vllm_patches.usefull_patch import patch_mamba_utils as patch_mod


def test_mome_state_dtype_reuses_resolved_cache_dtype(monkeypatch):
    calls = []

    def resolve(cache_dtype, model_dtype):
        calls.append((cache_dtype, model_dtype))
        return torch.bfloat16

    monkeypatch.setattr(patch_mod, "get_kv_cache_torch_dtype", resolve)

    result = patch_mod.MambaStateDtypeCalculatorMomePatch.mome_state_dtype(
        torch.float16, "auto"
    )

    assert result == (torch.bfloat16, torch.bfloat16, torch.bfloat16)
    assert calls == [("auto", torch.float16)]


def test_mome_state_shape_uses_kernel_history_and_spec_tokens():
    result = patch_mod.MambaStateShapeCalculatorMomePatch.mome_state_shape(
        q_lora_rank=8,
        kv_lora_rank=16,
        num_heads=4,
        v_head_dim=32,
        kernel_size=5,
        num_spec=3,
    )

    assert result == ((7, 8), (7, 16), (7, 128))


def test_mome_state_shape_supports_zero_history():
    assert patch_mod.MambaStateShapeCalculatorMomePatch.mome_state_shape(
        q_lora_rank=2,
        kv_lora_rank=3,
        num_heads=1,
        v_head_dim=4,
    ) == ((0, 2), (0, 3), (0, 4))


def test_mamba_patch_registration_targets():
    assert patch_mod.MambaStateDtypeCalculatorMomePatch._target is (
        patch_mod.MambaStateDtypeCalculator
    )
    assert patch_mod.MambaStateShapeCalculatorMomePatch._target is (
        patch_mod.MambaStateShapeCalculator
    )
