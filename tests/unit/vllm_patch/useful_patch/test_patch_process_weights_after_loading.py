from contextlib import contextmanager

import torch

from omni_npu.vllm_patches.usefull_patch import (
    patch_process_weights_after_loading as patch_mod,
)


def test_process_weights_calls_upstream_then_npu_specific_modules(monkeypatch):
    events = []

    class SupportedModule:
        def __init__(self, name):
            self.name = name

        def process_weights_after_loading(self):
            events.append(("process", self.name))

    class UnsupportedModule:
        def process_weights_after_loading(self):
            events.append(("unexpected",))

    first = SupportedModule("attention")
    second = SupportedModule("rms_norm")
    unsupported = UnsupportedModule()
    model = type(
        "Model",
        (),
        {"named_modules": lambda self: [
            ("first", first),
            ("unsupported", unsupported),
            ("second", second),
        ]},
    )()
    model_config = object()
    target_device = torch.device("npu:0")

    def original(model_arg, config_arg, device_arg):
        events.append(("upstream", model_arg, config_arg, device_arg))

    @contextmanager
    def loading_context(module, device):
        events.append(("enter", module.name, device))
        yield
        events.append(("exit", module.name, device))

    monkeypatch.setattr(
        patch_mod, "_ORIGINAL_PROCESS_WEIGHTS_AFTER_LOADING", original
    )
    monkeypatch.setattr(patch_mod, "NPUPanguSparseAttention", SupportedModule)
    monkeypatch.setattr(patch_mod, "NPUmHC", SupportedModule)
    monkeypatch.setattr(patch_mod, "NPURMSNorm", SupportedModule)
    monkeypatch.setattr(
        patch_mod.model_loader_utils,
        "device_loading_context",
        loading_context,
    )

    patch_mod._patched_process_weights_after_loading(
        model, model_config, target_device
    )

    assert events == [
        ("upstream", model, model_config, target_device),
        ("enter", "attention", target_device),
        ("process", "attention"),
        ("exit", "attention", target_device),
        ("enter", "rms_norm", target_device),
        ("process", "rms_norm"),
        ("exit", "rms_norm", target_device),
    ]


def test_process_weights_patch_registration_targets_both_import_sites():
    assert patch_mod.PanguV2MoeProcessWeightsUtilsPatch._target is (
        patch_mod.model_loader_utils
    )
    assert patch_mod.PanguV2MoeProcessWeightsBaseLoaderPatch._target is (
        patch_mod.base_loader_module
    )
    assert (
        patch_mod.PanguV2MoeProcessWeightsUtilsPatch.process_weights_after_loading
        is patch_mod._patched_process_weights_after_loading
    )
    assert (
        patch_mod.PanguV2MoeProcessWeightsBaseLoaderPatch.process_weights_after_loading
        is patch_mod._patched_process_weights_after_loading
    )
