# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import types

import omni_cache.cache as cache_module
from omni_cache.cache.omni_attention.runtime_config import (
    check_omni_attn_cmd_arg,
    _apply_sink_config,
    _apply_beta_config,
    _apply_pattern_config,
    _apply_recent_config,
    _apply_runtime_config,
)
from omni_cache.cache.omni_attention.runtime_patch import (
    _apply_attention_runtime_patch,
    _apply_cache_manager_runtime_patch,
    _load_patch_dependencies,
)


def _install_cache_runtime_stubs(install_stub_modules):
    sentinel_block_table = type("SentinelBlockTable", (), {})
    sentinel_blocks = type("SentinelBlocks", (), {})
    sentinel_manager = type("SentinelManager", (), {})

    def sentinel_get_kv_cache_config(*args, **kwargs):
        return (args, kwargs)

    modules = install_stub_modules({
        "vllm.v1.core.kv_cache_utils": {
            "get_kv_cache_config": object(),
        },
        "vllm.v1.engine.core": {
            "get_kv_cache_config": object(),
        },
        "vllm.v1.worker.block_table": {
            "MultiGroupBlockTable": object(),
        },
        "vllm.v1.worker.gpu_input_batch": {
            "MultiGroupBlockTable": object(),
        },
        "vllm.v1.core.kv_cache_manager": {
            "KVCacheBlocks": object(),
            "KVCacheManager": object(),
        },
        "vllm.v1.core.sched.scheduler": {
            "KVCacheBlocks": object(),
            "KVCacheManager": object(),
        },
        "omni_cache.cache.utils": {
            "to_bool_or_raise": lambda value: value in (1, True, "true", "TRUE"),
        },
        "omni_cache.attention.kv_cache_interface": {
            "SINK": 0,
            "RECENT": 0,
            "BETA": 0.5,
            "PATTERN": [0, 1],
            "get_kv_cache_config_omni_type": sentinel_get_kv_cache_config,
            "OmniMultiGroupBlockTable": sentinel_block_table,
        },
        "omni_cache.attention.kv_managers": {
            "OmniKVCacheBlocks": sentinel_blocks,
            "OmniKVCacheManager": sentinel_manager,
        },
    })
    return modules, sentinel_get_kv_cache_config, sentinel_block_table, sentinel_blocks, sentinel_manager


class TestCheckOmniAttnCmdArg:
    def test_returns_false_for_missing_flag(self):
        assert check_omni_attn_cmd_arg(None) is False
        assert check_omni_attn_cmd_arg({}) is False

    def test_uses_converter_for_present_flag(self, install_stub_modules):
        install_stub_modules({
            "omni_cache.cache.omni_attention.utils": {
                "to_bool_or_raise": lambda value: value == "enabled",
            },
        })

        assert check_omni_attn_cmd_arg({"enable_omni_attn": "enabled"}) is True

    def test_propagates_converter_errors(self, install_stub_modules):
        def _raise_on_invalid(_value):
            raise ValueError("invalid bool value")

        install_stub_modules({
            "omni_cache.cache.omni_attention.utils": {
                "to_bool_or_raise": _raise_on_invalid,
            },
        })

        try:
            check_omni_attn_cmd_arg({"enable_omni_attn": "invalid"})
            raised = None
        except ValueError as exc:
            raised = exc

        assert raised is not None
        assert "invalid bool value" in str(raised)


class TestPrivateConfigHelpers:
    def test_apply_sink_config_updates_only_when_present(self):
        itfc = types.SimpleNamespace(SINK=0)

        _apply_sink_config({"sink": 12}, itfc)

        assert itfc.SINK == 12

    def test_apply_sink_config_rejects_non_int(self):
        itfc = types.SimpleNamespace(SINK=0)

        try:
            _apply_sink_config({"sink": "12"}, itfc)
            raised = None
        except ValueError as exc:
            raised = exc

        assert raised is not None
        assert "sink should be int" in str(raised)

    def test_apply_recent_config_missing_key_is_noop(self):
        itfc = types.SimpleNamespace(RECENT=7)

        _apply_recent_config({}, itfc)

        assert itfc.RECENT == 7

    def test_apply_beta_config_rejects_int_even_if_in_range(self):
        itfc = types.SimpleNamespace(BETA=0.5)

        try:
            _apply_beta_config({"beta": 0}, itfc)
            raised = None
        except ValueError as exc:
            raised = exc

        assert raised is not None
        assert "beta should be float in (0,1)" in str(raised)

    def test_apply_pattern_config_accepts_binary_list(self):
        itfc = types.SimpleNamespace(PATTERN=[1, 1])

        _apply_pattern_config({"pattern": [0, 1, 0, 1]}, itfc)

        assert itfc.PATTERN == [0, 1, 0, 1]

    def test_apply_runtime_config_none_is_noop(self):
        itfc = types.SimpleNamespace(SINK=1, RECENT=2, BETA=0.5, PATTERN=[0, 1])

        _apply_runtime_config(None, itfc)

        assert itfc.SINK == 1
        assert itfc.RECENT == 2
        assert itfc.BETA == 0.5
        assert itfc.PATTERN == [0, 1]

    def test_apply_runtime_config_updates_only_specified_fields(self):
        itfc = types.SimpleNamespace(SINK=1, RECENT=2, BETA=0.5, PATTERN=[0, 1])

        _apply_runtime_config({"recent": 9}, itfc)

        assert itfc.SINK == 1
        assert itfc.RECENT == 9
        assert itfc.BETA == 0.5
        assert itfc.PATTERN == [0, 1]


class TestPrivatePatchHelpers:
    def test_apply_attention_runtime_patch_updates_all_targets(self):
        sentinel_get_kv_cache_config = object()
        sentinel_block_table = type("SentinelBlockTable", (), {})
        deps = {
            "kv_cache_utils": types.SimpleNamespace(get_kv_cache_config=object()),
            "engine_core": types.SimpleNamespace(get_kv_cache_config=object()),
            "block_table": types.SimpleNamespace(MultiGroupBlockTable=object()),
            "gpu_input_batch": types.SimpleNamespace(MultiGroupBlockTable=object()),
            "get_kv_cache_config_omni_type": sentinel_get_kv_cache_config,
            "OmniMultiGroupBlockTable": sentinel_block_table,
        }

        _apply_attention_runtime_patch(deps)

        assert deps["kv_cache_utils"].get_kv_cache_config is sentinel_get_kv_cache_config
        assert deps["engine_core"].get_kv_cache_config is sentinel_get_kv_cache_config
        assert deps["block_table"].MultiGroupBlockTable is sentinel_block_table
        assert deps["gpu_input_batch"].MultiGroupBlockTable is sentinel_block_table

    def test_apply_cache_manager_runtime_patch_updates_all_targets(self):
        sentinel_blocks = type("SentinelBlocks", (), {})
        sentinel_manager = type("SentinelManager", (), {})
        deps = {
            "orig_manager": types.SimpleNamespace(KVCacheBlocks=object(), KVCacheManager=object()),
            "scheduler": types.SimpleNamespace(KVCacheBlocks=object(), KVCacheManager=object()),
            "OmniKVCacheBlocks": sentinel_blocks,
            "OmniKVCacheManager": sentinel_manager,
        }

        _apply_cache_manager_runtime_patch(deps)

        assert deps["orig_manager"].KVCacheBlocks is sentinel_blocks
        assert deps["orig_manager"].KVCacheManager is sentinel_manager
        assert deps["scheduler"].KVCacheBlocks is sentinel_blocks
        assert deps["scheduler"].KVCacheManager is sentinel_manager

    def test_load_patch_dependencies_exposes_expected_keys(self, install_stub_modules):
        _install_cache_runtime_stubs(install_stub_modules)

        deps = _load_patch_dependencies()

        expected_keys = {
            "kv_cache_utils",
            "engine_core",
            "block_table",
            "gpu_input_batch",
            "orig_manager",
            "scheduler",
            "itfc",
            "get_kv_cache_config_omni_type",
            "OmniMultiGroupBlockTable",
            "OmniKVCacheBlocks",
            "OmniKVCacheManager",
        }
        assert set(deps) == expected_keys


class TestApplyOmniAttnPatch:
    def test_returns_without_flags(self):
        assert cache_module.apply_omni_attn_patch() is None

    def test_updates_runtime_modules_and_config(self, install_stub_modules):
        modules, sentinel_get_kv_cache_config, sentinel_block_table, sentinel_blocks, sentinel_manager = _install_cache_runtime_stubs(install_stub_modules)

        cache_module.apply_omni_attn_patch(
            enable_attn=True,
            enable_cache=True,
            is_kv_consumer=True,
            config={
                "sink": 8,
                "recent": 16,
                "beta": 0.25,
                "pattern": [0, 1, 0],
            },
        )

        kv_interface = modules["omni_cache.attention.kv_cache_interface"]
        assert kv_interface.SINK == 8
        assert kv_interface.RECENT == 16
        assert kv_interface.BETA == 0.25
        assert kv_interface.PATTERN == [0, 1, 0]

        assert modules["vllm.v1.core.kv_cache_utils"].get_kv_cache_config is sentinel_get_kv_cache_config
        assert modules["vllm.v1.engine.core"].get_kv_cache_config is sentinel_get_kv_cache_config
        assert modules["vllm.v1.worker.block_table"].MultiGroupBlockTable is sentinel_block_table
        assert modules["vllm.v1.worker.gpu_input_batch"].MultiGroupBlockTable is sentinel_block_table
        assert modules["vllm.v1.core.kv_cache_manager"].KVCacheBlocks is sentinel_blocks
        assert modules["vllm.v1.core.kv_cache_manager"].KVCacheManager is sentinel_manager
        assert modules["vllm.v1.core.sched.scheduler"].KVCacheBlocks is sentinel_blocks
        assert modules["vllm.v1.core.sched.scheduler"].KVCacheManager is sentinel_manager

    def test_enable_cache_only_patches_managers_not_attention_bindings(self, install_stub_modules):
        modules, _sentinel_get_kv_cache_config, sentinel_block_table, sentinel_blocks, sentinel_manager = _install_cache_runtime_stubs(install_stub_modules)
        original_core_fn = modules["vllm.v1.core.kv_cache_utils"].get_kv_cache_config
        original_block_table = modules["vllm.v1.worker.block_table"].MultiGroupBlockTable

        cache_module.apply_omni_attn_patch(enable_attn=False, enable_cache=True, is_kv_consumer=True)

        assert modules["vllm.v1.core.kv_cache_utils"].get_kv_cache_config is original_core_fn
        assert modules["vllm.v1.worker.block_table"].MultiGroupBlockTable is original_block_table
        assert modules["vllm.v1.core.kv_cache_manager"].KVCacheBlocks is sentinel_blocks
        assert modules["vllm.v1.core.kv_cache_manager"].KVCacheManager is sentinel_manager
        assert modules["vllm.v1.core.sched.scheduler"].KVCacheBlocks is sentinel_blocks
        assert modules["vllm.v1.core.sched.scheduler"].KVCacheManager is sentinel_manager

    def test_rejects_invalid_beta_boundaries(self, install_stub_modules):
        _install_cache_runtime_stubs(install_stub_modules)

        for beta_val in (0.0, 1.0, -0.1, 1.2):
            try:
                cache_module.apply_omni_attn_patch(enable_attn=True, config={"beta": beta_val})
                raised = None
            except ValueError as exc:
                raised = exc

            assert raised is not None
            assert "beta should be float in (0,1)" in str(raised)

    def test_rejects_invalid_pattern_values(self, install_stub_modules):
        _install_cache_runtime_stubs(install_stub_modules)

        try:
            cache_module.apply_omni_attn_patch(enable_attn=True, config={"pattern": [0, 2, 1]})
            raised = None
        except ValueError as exc:
            raised = exc

        assert raised is not None
        assert "pattern should be a list of 0s and 1s" in str(raised)

    def test_skips_consumer_patches_when_not_kv_consumer(self, install_stub_modules):
        modules, sentinel_get_kv_cache_config, sentinel_block_table, sentinel_blocks, sentinel_manager = _install_cache_runtime_stubs(install_stub_modules)
        original_core_fn = modules["vllm.v1.core.kv_cache_utils"].get_kv_cache_config
        original_block_table = modules["vllm.v1.worker.block_table"].MultiGroupBlockTable

        cache_module.apply_omni_attn_patch(enable_attn=True, enable_cache=True, is_kv_consumer=False)

        assert modules["vllm.v1.core.kv_cache_utils"].get_kv_cache_config is original_core_fn
        assert modules["vllm.v1.worker.block_table"].MultiGroupBlockTable is original_block_table
        assert modules["vllm.v1.core.kv_cache_manager"].KVCacheBlocks is sentinel_blocks
        assert modules["vllm.v1.core.kv_cache_manager"].KVCacheManager is sentinel_manager
        assert modules["vllm.v1.core.sched.scheduler"].KVCacheBlocks is sentinel_blocks
        assert modules["vllm.v1.core.sched.scheduler"].KVCacheManager is sentinel_manager

    def test_validates_config_types(self, install_stub_modules):
        _install_cache_runtime_stubs(install_stub_modules)

        try:
            cache_module.apply_omni_attn_patch(enable_attn=True, config={"sink": "bad"})
            raised = None
        except ValueError as exc:
            raised = exc

        assert raised is not None
        assert "sink should be int" in str(raised)

    def test_set_omni_cache_updates_global(self):
        cache_value = types.SimpleNamespace(name="demo")

        cache_module.set_omni_cache(cache_value)

        assert cache_module.omni_cache is cache_value
        assert "apply_omni_attn_patch" in cache_module.__all__
        assert "apply_omni_cache_patch" not in cache_module.__all__