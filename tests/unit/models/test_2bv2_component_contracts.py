# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Contract tests pinning the 2BV2 component-route API locations on vLLM 0.25.1 / transformers 5.16.1.

Following test_multimodal_migration_contracts.py's approach: when an upstream
symbol a fix depends on moves or is removed, these tests fail before runtime,
and the failure point is the migration point. This file pins the symbol contracts
touched by our PR-1 (omni-models component repo) + PR-2 (omniinfer patches):

  1. All vllm symbols top-level imported by openpangu.py (new locations after 0.25.1 reorg)
  2. models/__init__ ModelRegistry registration chain (lazy-load fix + 2BV2 registration)
  3. imageprocessor transformers 5.x symbols (after the Fast processor merge)

All run on CPU (import-time asserts, no model instances constructed).
"""

import importlib.util

import pytest

if importlib.util.find_spec("omni_models") is None:
    pytest.skip(
        "optional omni_models package is not installed",
        allow_module_level=True,
    )


class TestVLLM025SymbolContracts:
    """Symbol-location contracts for openpangu.py's module-level 0.25.1 imports."""

    def test_attention_imports_resolve(self):
        """New locations after the vllm.attention package removal (PR-1 fix point 1)."""
        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.layers.attention.static_sink_attention import (
            StaticSinkAttention,
        )
        from vllm.v1.attention.backend import AttentionType

        assert callable(Attention)
        assert StaticSinkAttention is not None
        assert AttentionType.DECODER is not None

    def test_moe_factory_symbols_resolve(self):
        """Symbols after SharedFusedMoE removal and FusedMoE becoming a factory (fix point 3)."""
        from vllm.model_executor.layers.fused_moe import (
            FusedMoE,
            fused_moe_make_expert_params_mapping,
        )

        assert FusedMoE is not None
        assert callable(fused_moe_make_expert_params_mapping)

    def test_mla_wrapper_symbols_require_patches(self):
        """Upstream mla lacks the sink wrapper openpangu.py's MLA branch lazy-imports."""
        from vllm.model_executor.layers import mla

        assert not hasattr(mla, "StaticSinkMultiHeadLatentAttentionWrapper")

    def test_single_type_kv_cache_manager_signature(self):
        """Required scheduler_block_size arg contract.

        The 5th param of the 0.25.1 upstream signature is required scheduler_block_size;
        the official pangu_v2_base/patch_single_type_kv_cache_manager rewrites the
        manager selection on top of this signature. If upstream changes it again,
        that patch must be updated.
        """
        import inspect
        from vllm.v1.core.single_type_kv_cache_manager import (
            SingleTypeKVCacheManager,
            SinkFullAttentionManager,
        )

        sig = inspect.signature(SingleTypeKVCacheManager.__init__)
        params = list(sig.parameters)
        assert params[5] == "scheduler_block_size"
        assert sig.parameters["scheduler_block_size"].default is inspect.Parameter.empty

        # The subclass having its own __init__ (not forwarding scheduler_block_size)
        # is exactly what the official patch overrides.
        assert "__init__" in SinkFullAttentionManager.__dict__


class TestComponentModelRegistry:
    """models/__init__.py registration-chain contracts (lazy-load fix + 2BV2 registration)."""

    def test_model_registry_imports_from_submodule(self):
        """Lazy-load fix: must import from the vllm.model_executor.models submodule.

        A top-level `from vllm import ModelRegistry` during plugin registration
        gets a partially-initialized vllm package (MODULE_ATTRS lazy load) and
        raises "cannot import name". This assert pins that the fixed import path
        itself resolves.
        """
        from vllm.model_executor.models import ModelRegistry
        assert hasattr(ModelRegistry, "register_model")

    def test_register_model_registers_2bv2_arch(self):
        """After register_model(), PanguEmbeddedForCausalLM is in the registry.

        register_model triggers register_processor/register_configuration, whose
        import chain includes all omni_models huggingface processors -- so this
        test also guards the imageprocessor transformers 5.x fix at import time.
        """
        from omni_models.models import register_model
        from vllm.model_executor.models import ModelRegistry

        register_model()

        archs = ModelRegistry.get_supported_archs()
        assert "PanguEmbeddedForCausalLM" in archs
        assert "PanguProMoEV2ForCausalLM" in archs

    def test_registered_class_resolves_to_component_impl(self, load_patch):
        """The registered entry resolves to the component impl (component-route switch).

        openpangu.py's top-level import only requires patch_static_sink_attention
        (high_throughout); AggregateConv / StaticSinkMLA are lazy, in-branch imports.
        """
        patch_static_sink = load_patch("patch_static_sink_attention", group="high_throughout")
        # Clear the applied-marker: a prior test may have applied this patch
        # and the residue makes a second apply raise "already patched".
        import vllm.model_executor.layers.attention.static_sink_attention as ssa
        ssa._omni_npu_applied_patches = {}
        patch_static_sink.StaticSinkAttentionPatch.apply()

        import omni_models.models.pangu.openpangu as component_openpangu  # noqa: F401
        from omni_models.models import register_model
        from vllm.model_executor.models import ModelRegistry

        register_model()
        entry = ModelRegistry.models.get("PanguEmbeddedForCausalLM")
        assert entry is not None
        # 0.25.1 registry value is _LazyRegisteredModel(module_name, class_name)
        assert entry.module_name == "omni_models.models.pangu.openpangu"
        assert entry.class_name == "PanguEmbeddedForCausalLM"


class TestImageProcessorTransformers5Contracts:
    """imageprocessor_openpangu_vl.py transformers 5.16.1 symbol contracts."""

    def test_qwen2_vl_fast_processor_module_merged(self):
        """Qwen2VL Fast processor symbol-layout contract on transformers 5.16.1.

        5.x merged layout: Qwen2VLImageProcessor / Kwargs are importable from both
        image_processing_qwen2_vl (the post-merge slow location, PR-1's import point)
        and image_processing_qwen2_vl_fast (PIL variant). The fixed imageprocessor
        imports the slow base from the merged location and subclasses it -- this
        assert pins that the import point still resolves after a transformers upgrade.
        """
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor,
            Qwen2VLImageProcessorKwargs,
        )
        from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
            Qwen2VLImageProcessor as FastVariant,
        )

        assert Qwen2VLImageProcessor is not None
        assert Qwen2VLImageProcessorKwargs is not None
        # The fast module is currently the PIL variant; it also exports Qwen2VLImageProcessor.
        assert FastVariant is not None

    def test_component_imageprocessor_imports(self):
        """Component imageprocessor full import chain (incl. register_transformers decorator)."""
        from omni_models.models.openpangu_vl.huggingface.imageprocessor_openpangu_vl import (
            OpenPanguVLImageProcessorFast,
        )
        assert OpenPanguVLImageProcessorFast.image_use_fast is True
        assert OpenPanguVLImageProcessorFast.pixel_stride == 1
