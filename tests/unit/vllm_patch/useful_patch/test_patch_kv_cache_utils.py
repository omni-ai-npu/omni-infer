from types import SimpleNamespace

from omni_npu.vllm_patches.usefull_patch import patch_kv_cache_utils as patch_mod


class Spec:
    pass


def _run(monkeypatch, specs, override):
    captured = {}

    def create(all_specs, grouped_layers):
        captured["specs"] = all_specs
        captured["groups"] = grouped_layers
        return "created-groups"

    monkeypatch.setattr(patch_mod, "create_kv_cache_group_specs", create)
    monkeypatch.setattr(
        patch_mod.envs,
        "OMNI_HYBRID_ATTN_GROUP_SIZE",
        override,
        raising=False,
    )
    monkeypatch.setattr(
        patch_mod, "logger", SimpleNamespace(warning=lambda *args, **kwargs: None)
    )
    result = patch_mod._get_kv_cache_groups_uniform_page_size_patched(specs)
    return result, captured


def test_group_size_override_splits_each_attention_type(monkeypatch):
    full = Spec()
    sliding = Spec()
    specs = {
        "full.0": full,
        "full.1": full,
        "full.2": full,
        "sw.0": sliding,
    }

    result, captured = _run(monkeypatch, specs, override=2)

    assert result == "created-groups"
    assert captured["specs"] is specs
    assert captured["groups"] == [
        ["full.0", "full.2"],
        ["full.1"],
        ["sw.0"],
    ]


def test_close_group_sizes_use_larger_group_to_reduce_padding(monkeypatch):
    full = Spec()
    sliding = Spec()
    specs = {
        **{f"full.{index}": full for index in range(3)},
        **{f"sw.{index}": sliding for index in range(4)},
    }

    _, captured = _run(monkeypatch, specs, override=0)

    assert captured["groups"] == [
        ["full.0", "full.1", "full.2"],
        ["sw.0", "sw.1", "sw.2", "sw.3"],
    ]


def test_kv_cache_utils_patch_registration():
    cls = patch_mod.OverrideGroupSizePatch
    assert cls._target is patch_mod.kv_cache_utils
    assert cls._get_kv_cache_groups_uniform_page_size is (
        patch_mod._get_kv_cache_groups_uniform_page_size_patched
    )
