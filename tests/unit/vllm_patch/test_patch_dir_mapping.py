# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from pathlib import Path

from omni import vllm_patches
from omni.vllm_patches import patches as _patches_mod  # explicit submodule load


def test_get_patch_dir_names_for_openpangu_v2():
    assert vllm_patches._get_patch_dir_names("openpangu_v2") == [
        "pangu_v2_base",
        "pangu_sink_swa_mla",
    ]


def test_get_patch_dir_names_for_manual_legacy_alias():
    assert vllm_patches._get_patch_dir_names("pangu_sink_swa_mla") == [
        "pangu_v2_base",
        "pangu_sink_swa_mla",
    ]


def test_get_patch_dir_names_for_minimax_m2():
    assert vllm_patches._get_patch_dir_names("minimax_m2") == ["minimax"]


def test_find_patch_dir_exact_supports_multiple_manual_dirs():
    models_root = Path(_patches_mod.__file__).parent / "models"

    patch_dirs = vllm_patches._find_patch_dir_exact("pangu_sink_swa_mla", models_root)

    assert [path.name for path in patch_dirs] == [
        "pangu_v2_base",
        "pangu_sink_swa_mla",
    ]


def test_find_patch_dir_fuzzy_supports_multiple_auto_dirs():
    models_root = Path(_patches_mod.__file__).parent / "models"

    patch_dirs = vllm_patches._find_patch_dir_fuzzy("openpangu_ultra_omni", models_root)

    assert [path.name for path in patch_dirs] == [
        "pangu_v2_base",
        "pangu_sink_swa_mla",
        "openpangu_v1_vl",
    ]


def test_find_patch_dir_exact_supports_bench_aligned_decode_manual_dir():
    models_root = Path(_patches_mod.__file__).parent / "models"

    patch_dirs = vllm_patches._find_patch_dir_exact(
        "pd_bench_aligned_decode", models_root
    )

    assert [path.name for path in patch_dirs] == ["pd_bench_aligned_decode"]
