# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest


def _stub_module(monkeypatch, name: str, *, is_package: bool = False):
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent_name, attr = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            # Use monkeypatch so the parent package's attribute is restored after
            # the test; a raw setattr here leaks (e.g. rebinding the real
            # vllm.attention.layer to a bare stub) and pollutes later tests.
            monkeypatch.setattr(parent, attr, module, raising=False)
    return module


def _install_core_stubs(monkeypatch):
    _stub_module(monkeypatch, "vllm", is_package=True)
    logger_mod = _stub_module(monkeypatch, "vllm.logger")

    class Logger:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    logger_mod.init_logger = lambda *_a, **_kw: Logger()

    utils = _stub_module(monkeypatch, "vllm.utils", is_package=True)
    math_utils = _stub_module(monkeypatch, "vllm.utils.math_utils")
    math_utils.cdiv = lambda a, b: -(-a // b)
    torch_utils = _stub_module(monkeypatch, "vllm.utils.torch_utils")
    torch_utils.get_dtype_size = lambda dtype: {
        "bf16": 2,
        "fp32": 4,
        "int8": 1,
    }.get(dtype, 2)
    torch_utils.STR_DTYPE_TO_TORCH_DTYPE = {"auto": "bf16", "bf16": "bf16"}

    _stub_module(monkeypatch, "vllm.v1", is_package=True)
    _stub_module(monkeypatch, "vllm.v1.core", is_package=True)
    kv_iface = _stub_module(monkeypatch, "vllm.v1.kv_cache_interface")

    @dataclass(frozen=True, kw_only=True)
    class KVCacheSpec:
        block_size: int

    @dataclass(frozen=True, kw_only=True)
    class FullAttentionSpec(KVCacheSpec):
        num_kv_heads: int = 1
        head_size: int = 1
        dtype: object = "bf16"
        page_size_padded: int | None = None
        sliding_window: int | None = None

        @property
        def page_size_bytes(self):
            return self.block_size * self.num_kv_heads * self.head_size * 2

    @dataclass(frozen=True, kw_only=True)
    class CrossAttentionSpec(KVCacheSpec):
        pass

    @dataclass(frozen=True, kw_only=True)
    class SlidingWindowSpec(FullAttentionSpec):
        sliding_window: int = 1

    @dataclass(frozen=True, kw_only=True)
    class ChunkedLocalAttentionSpec(KVCacheSpec):
        attention_chunk_size: int = 1

    @dataclass(frozen=True, kw_only=True)
    class MambaSpec(KVCacheSpec):
        shapes: tuple = ()
        dtypes: tuple = ()
        page_size_padded: int | None = None
        mamba_type: str = "mamba"
        num_speculative_blocks: int = 0

    class UniformTypeKVCacheSpecs:
        pass

    for name, value in {
        "KVCacheSpec": KVCacheSpec,
        "FullAttentionSpec": FullAttentionSpec,
        "CrossAttentionSpec": CrossAttentionSpec,
        "SlidingWindowSpec": SlidingWindowSpec,
        "ChunkedLocalAttentionSpec": ChunkedLocalAttentionSpec,
        "MambaSpec": MambaSpec,
        "UniformTypeKVCacheSpecs": UniformTypeKVCacheSpecs,
        "EncoderOnlyAttentionSpec": type("EncoderOnlyAttentionSpec", (), {}),
        "AttentionSpec": FullAttentionSpec,
        "SinkMLAAttentionSpec": FullAttentionSpec,
    }.items():
        setattr(kv_iface, name, value)

    return kv_iface


def _load_module(monkeypatch, module_name: str):
    package_name = (
        "omni.vllm_patches.patches.models.pangu_v2_hybrid")
    if module_name.startswith(package_name + ".") and package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(Path(__file__).resolve().parents[5] / "src" /
                                "omni_npu" / "vllm_patches" / "patches" /
                                "models" / "pangu_v2_hybrid")]
        monkeypatch.setitem(sys.modules, package_name, package)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.unit
def test_pangu_kv_cache_specs_page_sizes_and_validation(monkeypatch):
    _install_core_stubs(monkeypatch)
    mod = _load_module(
        monkeypatch,
        "omni.vllm_patches.patches.models.pangu_v2_hybrid."
        "patch_kv_cache_interface",
    )

    assert mod.DSAAttentionSpec(
        block_size=2, num_kv_heads=1, head_size=10, dtype="bf16",
        cache_dtype_str="fp8_ds_mla").real_page_size_bytes == 2 * (656 + 128 + 4)
    assert mod.DSAAttentionSpec(
        block_size=2, num_kv_heads=1, head_size=10, dtype="bf16",
        cache_dtype_str="li_int8_ds_mla").real_page_size_bytes == (
            2 * (576 * 2 + 128 + 2))
    assert mod.DSAAttentionSpec(
        block_size=2, num_kv_heads=1, head_size=10, dtype="bf16",
    ).real_page_size_bytes == 40

    merged = mod.DSAAttentionSpec.merge([
        mod.DSAAttentionSpec(block_size=2, num_kv_heads=1, head_size=10,
                             dtype="bf16", cache_dtype_str="int8_ds_mla"),
        mod.DSAAttentionSpec(block_size=2, num_kv_heads=1, head_size=10,
                             dtype="bf16", cache_dtype_str="int8_ds_mla"),
    ])
    assert merged.cache_dtype_str == "int8_ds_mla"

    with pytest.raises(AssertionError):
        mod.DSAAttentionSpec(block_size=1, num_kv_heads=2, head_size=10,
                             dtype="bf16")

    sliding = mod.ShareKVSlidingWindowSpec(
        block_size=2, num_kv_heads=1, head_size=512, dtype="bf16",
        sliding_window=128, num_extra_reserved_blocks=3)
    assert sliding.real_page_size_bytes == 2048
    assert sliding.num_extra_reserved_blocks == 3

    mome = mod.MomeSpec(
        block_size=4,
        shapes=((2,), (3,), (4,)),
        dtypes=("bf16", "bf16", "fp32"),
        kernel_size=5,
        num_spec_tokens=2,
    )
    assert mome.num_total_tokens == 6
    assert mome.page_size_bytes == (2 * 2 + 3 * 2 + 4 * 4) * 6
    assert mome.max_memory_usage_bytes(
        types.SimpleNamespace(
            model_config=types.SimpleNamespace(max_model_len=9))) == (
                3 * mome.page_size_bytes)

    with pytest.raises(ValueError, match="3 components"):
        mod.MomeSpec(block_size=1, shapes=((1,),), dtypes=("bf16",),
                     kernel_size=1)


@pytest.mark.unit
def test_pangu_uniform_type_supports_mome_specs(monkeypatch):
    _install_core_stubs(monkeypatch)
    mod = _load_module(
        monkeypatch,
        "omni.vllm_patches.patches.models.pangu_v2_hybrid."
        "patch_kv_cache_interface",
    )
    spec_a = mod.MomeSpec(
        block_size=4, shapes=((1,), (1,), (1,)), dtypes=("bf16",) * 3,
        kernel_size=4, num_spec_tokens=1)
    spec_b = mod.MomeSpec(
        block_size=4, shapes=((1,), (1,), (1,)), dtypes=("bf16",) * 3,
        kernel_size=4, num_spec_tokens=1)
    spec_c = mod.MomeSpec(
        block_size=4, shapes=((1,), (1,), (1,)), dtypes=("bf16",) * 3,
        kernel_size=5, num_spec_tokens=1)

    assert mod.UniformTypeKVCacheSpecsPatch.is_uniform_type({"a": spec_a, "b": spec_b})
    assert not mod.UniformTypeKVCacheSpecsPatch.is_uniform_type({
        "a": spec_a,
        "b": spec_c,
    })


def _install_manager_stubs(monkeypatch):
    _install_core_stubs(monkeypatch)
    core = sys.modules["vllm.v1.core"]
    block_pool = _stub_module(monkeypatch, "vllm.v1.core.block_pool")
    block_pool.BlockPool = type("BlockPool", (), {})
    kv_coord = _stub_module(monkeypatch, "vllm.v1.core.kv_cache_coordinator")
    kv_coord.HybridKVCacheCoordinator = type("HybridKVCacheCoordinator", (), {})
    kv_utils = _stub_module(monkeypatch, "vllm.v1.core.kv_cache_utils")
    kv_utils.BlockHash = object
    kv_utils.BlockHashList = list
    kv_utils.KVCacheBlock = object

    class BlockHashListWithBlockSize(list):
        def __init__(self, hashes, old_block_size, new_block_size):
            super().__init__(hashes)
            self.old_block_size = old_block_size
            self.new_block_size = new_block_size

    kv_utils.BlockHashListWithBlockSize = BlockHashListWithBlockSize

    single = _stub_module(monkeypatch, "vllm.v1.core.single_type_kv_cache_manager")

    class SingleTypeKVCacheManager:
        def __init__(self, kv_cache_spec, block_pool, enable_caching=False,
                     kv_cache_group_id=0, **kwargs):
            self.kv_cache_spec = kv_cache_spec
            self.block_pool = block_pool
            self.block_size = kv_cache_spec.block_size

    class SlidingWindowManager(SingleTypeKVCacheManager):
        def __init__(self, kv_cache_spec, block_pool, **kwargs):
            super().__init__(kv_cache_spec, block_pool, **kwargs)
            self.sliding_window = kv_cache_spec.sliding_window

    single.SingleTypeKVCacheManager = SingleTypeKVCacheManager
    single.SlidingWindowManager = SlidingWindowManager
    single.FullAttentionManager = type("FullAttentionManager", (), {})
    single.spec_manager_map = {}
    core.single_type_kv_cache_manager = single


@pytest.mark.unit
def test_pangu_mome_and_share_kv_managers(monkeypatch):
    _install_manager_stubs(monkeypatch)
    iface_mod = _load_module(
        monkeypatch,
        "omni.vllm_patches.patches.models.pangu_v2_hybrid."
        "patch_kv_cache_interface",
    )
    mgr_mod = _load_module(
        monkeypatch,
        "omni.vllm_patches.patches.models.pangu_v2_hybrid."
        "patch_single_type_kv_cache_manager",
    )

    spec = iface_mod.MomeSpec(
        block_size=4, shapes=((1,), (1,), (1,)), dtypes=("bf16",) * 3,
        kernel_size=5, num_extra_reserved_blocks=1)
    manager = mgr_mod.MomeManager(spec, block_pool=object())
    assert manager.get_num_skipped_tokens(3) == 0
    assert manager.get_num_skipped_tokens(12) == 4
    assert manager.get_num_common_prefix_blocks("req") == 0

    sw_spec = iface_mod.ShareKVSlidingWindowSpec(
        block_size=4, num_kv_heads=1, head_size=512, dtype="bf16",
        sliding_window=8, num_extra_reserved_blocks=1)
    sw_manager = mgr_mod.ShareKVSlidingWindowManager(sw_spec, block_pool=object())
    assert sw_manager.get_num_skipped_tokens(20) == 9

    class Pool:
        null_block = "null"

        def get_cached_block(self, block_hash, group_ids):
            return ["cached"] if block_hash == "h1" else None

    hits = mgr_mod.MomeManager.find_longest_cache_hit(
        block_hashes=["h0", "h1", "h2"],
        max_length=12,
        kv_cache_group_ids=[0],
        block_pool=Pool(),
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=4,
    )
    assert hits == (["null", "cached"],)


@pytest.mark.unit
def test_pangu_hybrid_coordinator_converges_to_shorter_hit(monkeypatch):
    _install_manager_stubs(monkeypatch)
    iface_mod = _load_module(
        monkeypatch,
        "omni.vllm_patches.patches.models.pangu_v2_hybrid."
        "patch_kv_cache_interface",
    )
    mgr_mod = _load_module(
        monkeypatch,
        "omni.vllm_patches.patches.models.pangu_v2_hybrid."
        "patch_single_type_kv_cache_manager",
    )

    class FullManager:
        calls = 0

        @classmethod
        def find_longest_cache_hit(cls, **kwargs):
            cls.calls += 1
            return (["f0", "f1", "f2"],)

    class ShortManager:
        @classmethod
        def find_longest_cache_hit(cls, **kwargs):
            return (["s0"],)

    full_spec = iface_mod.DSAAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=8, dtype="bf16")
    short_spec = iface_mod.MomeSpec(
        block_size=4, shapes=((1,), (1,), (1,)), dtypes=("bf16",) * 3,
        kernel_size=2)
    coordinator = types.SimpleNamespace(
        hash_block_size=4,
        kv_cache_config=types.SimpleNamespace(kv_cache_groups=[0, 1]),
        attention_groups=[
            (full_spec, [0], FullManager),
            (short_spec, [1], ShortManager),
        ],
        block_pool=object(),
        use_eagle=False,
        lcm_block_size=4,
    )

    blocks, hit_length = (
        mgr_mod.HybridKVCacheCoordinatorPatch.find_longest_cache_hit(
            coordinator, ["h0", "h1", "h2"], 12))
    assert hit_length == 4
    assert blocks == (["f0"], ["s0"])


@pytest.mark.unit
def test_pangu_kv_cache_group_size_override(monkeypatch):
    _install_core_stubs(monkeypatch)
    kv_utils = _stub_module(monkeypatch, "vllm.v1.core.kv_cache_utils")
    calls = {}
    kv_utils.logger = types.SimpleNamespace(warning=lambda *a, **kw: None)
    kv_utils.create_kv_cache_group_specs = (
        lambda specs, grouped: calls.setdefault("grouped", grouped) or grouped)

    mod = _load_module(
        monkeypatch,
        "omni.vllm_patches.patches.models.pangu_v2_hybrid."
        "patch_kv_cache_utils",
    )
    monkeypatch.setenv("HYBRID_ATTN_GROUP_SIZE", "2")

    class Spec:
        def __init__(self, block_size):
            self.block_size = block_size

        def __hash__(self):
            return id(self)

    spec_a = Spec(block_size=1)
    spec_b = Spec(block_size=1)
    mod._get_kv_cache_groups_uniform_page_size_patched({
        "a0": spec_a,
        "a1": spec_a,
        "a2": spec_a,
        "b0": spec_b,
    })

    assert calls["grouped"] == [["a0", "a2"], ["a1"], ["b0"]]


@pytest.mark.unit
def test_pangu_model_arch_convertor_and_speculative_mapping(monkeypatch):
    _install_core_stubs(monkeypatch)
    transformers_utils = _stub_module(monkeypatch, "vllm.transformers_utils",
                                      is_package=True)
    convertor_mod = _stub_module(
        monkeypatch, "vllm.transformers_utils.model_arch_config_convertor")

    class BaseConvertor:
        def __init__(self, hf_text_config):
            self.hf_text_config = hf_text_config

        def is_deepseek_mla(self):
            return False

    convertor_mod.ModelArchConfigConvertorBase = BaseConvertor
    convertor_mod.MODEL_ARCH_CONFIG_CONVERTORS = {}
    transformers_utils.model_arch_config_convertor = convertor_mod

    modelconfig = _load_module(
        monkeypatch,
        "omni.vllm_patches.patches.models.pangu_v2_hybrid."
        "patch_modelconfig",
    )
    cfg = types.SimpleNamespace(model_type="openpangu_v2", kv_lora_rank=128,
                                num_nextn_predict_layers=3)
    assert modelconfig.OpenpanguMTPModelArchConfigConvertor(
        cfg).get_num_hidden_layers() == 3
    assert BaseConvertor(cfg).is_deepseek_mla()
    assert "openpangu_mtp" in convertor_mod.MODEL_ARCH_CONFIG_CONVERTORS

    pydantic_dc = _stub_module(monkeypatch, "pydantic.dataclasses")
    pydantic_dc.dataclass = lambda cls=None, **kw: (
        dataclass(cls) if cls is not None else lambda c: dataclass(c))
    config_utils = _stub_module(monkeypatch, "vllm.config.utils")
    config_utils.config = lambda cls: cls
    speculative_pkg = _stub_module(monkeypatch, "vllm.config.speculative")

    class SpeculativeConfig:
        @staticmethod
        def hf_config_override(hf_config):
            hf_config.fallback = True
            return hf_config

    speculative_pkg.SpeculativeConfig = SpeculativeConfig
    vllm_config = _stub_module(monkeypatch, "vllm.config")
    vllm_config.speculative = speculative_pkg
    import_utils = _stub_module(monkeypatch, "vllm.utils.import_utils")
    import_utils.LazyLoader = lambda *a, **kw: object()

    spec_mod = _load_module(
        monkeypatch,
        "omni.vllm_patches.patches.models.pangu_v2_hybrid."
        "patch_speculative",
    )

    class HFConfig:
        def __init__(self, model_type):
            self.model_type = model_type
            self.num_nextn_predict_layers = 2

        def update(self, values):
            self.__dict__.update(values)

    openpangu = spec_mod.PanguV2MoeSpeculativeConfigPatch.hf_config_override(
        HFConfig("openpangu_v2"))
    assert openpangu.model_type == "openpangu_mtp"
    assert openpangu.architectures == ["OpenPanguMTPModel"]

    pangu_moe = spec_mod.PanguV2MoeSpeculativeConfigPatch.hf_config_override(
        HFConfig("pangu_v2_moe"))
    assert pangu_moe.model_type == "mtp"
    assert pangu_moe.architectures == ["PanguV2MTPModel"]

    fallback = spec_mod.PanguV2MoeSpeculativeConfigPatch.hf_config_override(
        HFConfig("other"))
    assert fallback.fallback is True


@pytest.mark.unit
def test_pangu_models_config_aligns_hybrid_page_size(monkeypatch):
    _install_core_stubs(monkeypatch)
    config_mod = types.ModuleType("vllm.config")
    config_mod.VllmConfig = type("VllmConfig", (), {})
    monkeypatch.setitem(sys.modules, "vllm.config", config_mod)
    runai_mod = _stub_module(monkeypatch, "vllm.transformers_utils.runai_utils")
    runai_mod.is_runai_obj_uri = lambda value: str(value).startswith("s3://")

    models_pkg = _stub_module(monkeypatch, "vllm.model_executor", is_package=True)
    _stub_module(monkeypatch, "vllm.model_executor.models", is_package=True)
    models_config = _stub_module(monkeypatch,
                                 "vllm.model_executor.models.config")
    models_config.MODELS_CONFIG_MAP = {}

    class MambaModelConfig:
        calls = []

        @classmethod
        def verify_and_update_config(cls, vllm_config):
            cls.calls.append(vllm_config.model_config.architecture)

    class HybridAttentionMambaModelConfig:
        @classmethod
        def verify_and_update_config(cls, vllm_config):
            vllm_config.hybrid_called = True

    models_config.MambaModelConfig = MambaModelConfig
    models_config.HybridAttentionMambaModelConfig = HybridAttentionMambaModelConfig
    models_pkg.models = types.SimpleNamespace(config=models_config)

    mod = _load_module(
        monkeypatch,
        "omni.vllm_patches.patches.models.pangu_v2_hybrid."
        "patch_models_config",
    )

    hf_config = types.SimpleNamespace(
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        q_lora_rank=3,
        num_attention_heads=2,
        v_head_dim=5,
        use_mome=True,
        router_sliding_window=4,
    )
    vllm_config = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(
            cache_dtype="auto",
            block_size=16,
            mamba_page_size_padded=None,
        ),
        model_config=types.SimpleNamespace(
            architecture="PanguUltraMoEForCausalLM",
            dtype="bf16",
            hf_config=hf_config,
            config_updated=False,
            convert_type=None,
        ),
        kv_transfer_config=None,
        speculative_config=types.SimpleNamespace(num_speculative_tokens=2),
    )

    mod.PanguV2HybridForCausalLMConfig.verify_and_update_config(vllm_config)
    assert vllm_config.cache_config.mamba_page_size_padded == 16 * 12

    wrapper = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            architecture="PanguUltraMoEForCausalLM",
            config_updated=True,
            convert_type=None,
        ),
        cache_config=types.SimpleNamespace(mamba_page_size_padded=None),
        load_config=types.SimpleNamespace(load_format="auto"),
    )
    called = {}
    monkeypatch.setattr(
        mod.PanguV2HybridForCausalLMConfig,
        "verify_and_update_config",
        classmethod(lambda cls, cfg: called.setdefault("cfg", cfg)),
    )
    mod.models_config_module.MODELS_CONFIG_MAP["PanguUltraMoEForCausalLM"] = (
        mod.PanguV2HybridForCausalLMConfig)
    mod.PanguV2HybridVllmConfigPatch.try_verify_and_update_config(wrapper)
    assert called["cfg"] is wrapper

    skip_cfg = types.SimpleNamespace(
        model_config=types.SimpleNamespace(architecture="PanguV2MoEForCausalLM"))
    mod.PanguV2MoEHybridAttentionMambaConfigPatch.verify_and_update_config(
        skip_cfg)
    assert MambaModelConfig.calls[-1] == "PanguV2MoEForCausalLM"


@pytest.mark.unit
def test_pangu_models_config_dsa_and_runai_edges(monkeypatch):
    _install_core_stubs(monkeypatch)
    config_mod = types.ModuleType("vllm.config")
    config_mod.VllmConfig = type("VllmConfig", (), {})
    monkeypatch.setitem(sys.modules, "vllm.config", config_mod)
    runai_mod = _stub_module(monkeypatch, "vllm.transformers_utils.runai_utils")
    runai_mod.is_runai_obj_uri = lambda value: str(value).startswith("s3://")

    _stub_module(monkeypatch, "vllm.model_executor", is_package=True)
    _stub_module(monkeypatch, "vllm.model_executor.models", is_package=True)
    models_config = _stub_module(monkeypatch,
                                 "vllm.model_executor.models.config")
    models_config.MODELS_CONFIG_MAP = {}

    class MambaModelConfig:
        @classmethod
        def verify_and_update_config(cls, vllm_config):
            pass

    class HybridAttentionMambaModelConfig:
        @classmethod
        def verify_and_update_config(cls, vllm_config):
            vllm_config.hybrid_called = True

    models_config.MambaModelConfig = MambaModelConfig
    models_config.HybridAttentionMambaModelConfig = HybridAttentionMambaModelConfig

    kv_iface = sys.modules["vllm.v1.kv_cache_interface"]

    class DSAAttentionSpec(kv_iface.FullAttentionSpec):
        def __init__(self, *args, cache_dtype_str=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.cache_dtype_str = cache_dtype_str

    kv_iface.DSAAttentionSpec = DSAAttentionSpec

    mod = _load_module(
        monkeypatch,
        "omni.vllm_patches.patches.models.pangu_v2_hybrid."
        "patch_models_config",
    )

    hf_config = types.SimpleNamespace(
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        q_lora_rank=3,
        num_attention_heads=2,
        v_head_dim=5,
        use_mome=False,
        index_topk=1,
        index_head_dim=2,
    )
    vllm_config = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(
            cache_dtype="bf16",
            block_size=None,
            mamba_page_size_padded=None,
        ),
        model_config=types.SimpleNamespace(
            architecture="PanguUltraMoEForCausalLM",
            dtype="bf16",
            hf_config=hf_config,
            config_updated=False,
            convert_type=None,
        ),
        kv_transfer_config=object(),
        speculative_config=None,
    )
    mod.PanguV2HybridForCausalLMConfig.verify_and_update_config(vllm_config)
    assert vllm_config.cache_config.block_size == 128
    assert vllm_config.cache_config.mamba_page_size_padded == 128 * 16

    early_return = types.SimpleNamespace(model_config=None)
    assert mod.PanguV2HybridVllmConfigPatch.try_verify_and_update_config(
        early_return) is None

    runai_cfg = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            architecture="Unknown",
            config_updated=False,
            convert_type=None,
            model_weights="s3://bucket/model",
            model="model",
        ),
        cache_config=types.SimpleNamespace(mamba_page_size_padded=None),
        load_config=types.SimpleNamespace(load_format="auto"),
    )
    mod.PanguV2HybridVllmConfigPatch.try_verify_and_update_config(runai_cfg)
    assert runai_cfg.load_config.load_format == "runai_streamer"

    bad_runai_cfg = types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            architecture="Unknown",
            config_updated=False,
            convert_type=None,
            model_weights="s3://bucket/model",
            model="model",
        ),
        cache_config=types.SimpleNamespace(mamba_page_size_padded=None),
        load_config=types.SimpleNamespace(load_format="bad"),
    )
    with pytest.raises(ValueError, match="must be 'runai_streamer'"):
        mod.PanguV2HybridVllmConfigPatch.try_verify_and_update_config(
            bad_runai_cfg)

    fallback_cfg = types.SimpleNamespace(
        model_config=types.SimpleNamespace(architecture="Other"))
    mod.PanguV2MoEHybridAttentionMambaConfigPatch.verify_and_update_config(
        fallback_cfg)
    assert fallback_cfg.hybrid_called is True


@pytest.mark.unit
def test_pangu_worker_utils_bind_kv_cache(monkeypatch):
    _install_core_stubs(monkeypatch)
    attention_layer = _stub_module(monkeypatch, "vllm.attention.layer")
    attention_layer.Attention = type("Attention", (), {})
    _stub_module(monkeypatch, "vllm.attention", is_package=True)
    _stub_module(monkeypatch, "vllm.v1.worker", is_package=True)
    gpu_runner = _stub_module(monkeypatch, "vllm.v1.worker.gpu_model_runner")
    models_utils = _stub_module(monkeypatch, "vllm.model_executor.models.utils")
    models_utils.extract_layer_index = (
        lambda name, num_attn_module=1: int(name.split(".")[1]))

    mod = _load_module(
        monkeypatch,
        "omni.vllm_patches.patches.models.pangu_v2_hybrid."
        "patch_worker_utils",
    )

    runner_kv_caches = []
    forward_context = {
        "layers.1.attn": types.SimpleNamespace(),
        "layers.0.attn": types.SimpleNamespace(),
    }
    mod.bind_kv_cache_patched(
        {"layers.1.attn": "kv1", "layers.0.attn": "kv0"},
        forward_context,
        runner_kv_caches,
    )

    assert runner_kv_caches == ["kv0", "kv1"]
    assert forward_context["layers.0.attn"].kv_cache == ["kv0"]
    assert gpu_runner.WorkerUtilsPatch if hasattr(gpu_runner, "WorkerUtilsPatch") else True


@pytest.mark.unit
def test_pangu_scheduler_updates_output_with_speculative_margin(monkeypatch):
    _install_core_stubs(monkeypatch)
    scheduler_mod = _stub_module(monkeypatch, "vllm.v1.core.sched.scheduler")
    scheduler_mod.Scheduler = type("Scheduler", (), {})
    request_mod = _stub_module(monkeypatch, "vllm.v1.request")
    request_mod.Request = type("Request", (), {})
    stop_mod = _stub_module(
        monkeypatch,
        "omni.vllm_patches.patches.common.patch_user_repetition_detection")

    calls = []

    def check_stop(request, max_len):
        calls.append((list(request.output_token_ids), max_len))
        return len(request.output_token_ids) >= 2

    stop_mod.check_stop = check_stop
    mod = _load_module(
        monkeypatch,
        "omni.vllm_patches.patches.models.pangu_v2_hybrid."
        "patch_scheduler",
    )

    request = types.SimpleNamespace(output_token_ids=[])
    request.append_output_token_ids = request.output_token_ids.append
    scheduler = types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(
            speculative_config=types.SimpleNamespace(num_speculative_tokens=2)),
        max_model_len=100,
    )

    new_tokens, stopped = mod.PanguV2SchedulerPatch._update_request_with_output(
        scheduler, request, [11, 12, 13])

    assert new_tokens == [11, 12]
    assert stopped is True
    assert calls[-1] == ([11, 12], 94)
