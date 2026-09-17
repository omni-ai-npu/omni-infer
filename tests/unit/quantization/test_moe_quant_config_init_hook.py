# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Pin the vLLM hook name that the A5 quant methods call after weight loading.

`process_weights_after_loading()` in mxfp8 / hifloat8 / w4a8_mxfp finishes by
asking the experts layer to build its MoE quant config. The layer those methods
receive at runtime is a real `RoutedExperts`, but every existing test passes a
plain `MagicMock`, which answers to any attribute name at all. That is why the
0.25 rename from `ensure_moe_quant_config_init` to `_ensure_moe_quant_config_init`
kept the whole quantization suite green and only surfaced on hardware, as
`AttributeError: 'NPURoutedExperts' object has no attribute ...` at model load.

These tests close that gap: they assert the name against the class vLLM actually
provides, so the next rename fails in CI instead of at model load.
"""

import inspect

import pytest

HOOK_NAME = "_ensure_moe_quant_config_init"

QUANT_MODULES = [
    "omni_npu.layers.quantization.mxfp8",
    "omni_npu.layers.quantization.hifloat8",
    "omni_npu.layers.quantization.w4a8_mxfp",
]


def test_vllm_routed_experts_exposes_the_hook():
    """The name the quant methods call must exist on vLLM's RoutedExperts."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    assert hasattr(RoutedExperts, HOOK_NAME), (
        f"vLLM no longer exposes RoutedExperts.{HOOK_NAME}; "
        "process_weights_after_loading() in the A5 quant methods calls it and "
        "will raise AttributeError at model load. Update both together."
    )
    assert callable(getattr(RoutedExperts, HOOK_NAME))


@pytest.mark.parametrize("module_path", QUANT_MODULES)
def test_quant_methods_call_the_hook_vllm_exposes(module_path):
    """Each quant module must call the hook under the name vLLM exposes."""
    import importlib

    module = importlib.import_module(module_path)
    source = inspect.getsource(module)

    assert f"layer.{HOOK_NAME}()" in source, (
        f"{module_path} does not call layer.{HOOK_NAME}() -- either the hook was "
        "dropped or it is still using the pre-0.25 name."
    )
    # The old public name must be gone; a leftover call raises at model load.
    assert f"layer.ensure_moe_quant_config_init()" not in source, (
        f"{module_path} still calls the pre-0.25 public name "
        "layer.ensure_moe_quant_config_init()."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
