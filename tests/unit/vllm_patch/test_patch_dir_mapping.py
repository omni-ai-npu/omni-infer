# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from pathlib import Path

from omni import vllm_patches


def test_get_patch_dir_names_for_high_throughout():
    assert vllm_patches._get_patch_dir_names("high_throughout") == [
        "pangu_v2_base",
        "high_throughout",
    ]


def test_get_patch_dir_names_for_low_latency():
    assert vllm_patches._get_patch_dir_names("low_latency") == [
        "pangu_v2_base",
        "low_latency",
    ]


def test_get_patch_dir_names_legacy_pangu_v2_hybrid_maps_to_high_throughout():
    assert vllm_patches._get_patch_dir_names("pangu_v2_hybrid") == [
        "pangu_v2_base",
        "high_throughout",
    ]


def test_get_patch_dir_names_legacy_pangu_v2_moe_maps_to_low_latency():
    assert vllm_patches._get_patch_dir_names("pangu_v2_moe") == [
        "pangu_v2_base",
        "low_latency",
    ]


def test_get_patch_dir_names_legacy_comma_separated_aliases_are_expanded():
    assert vllm_patches._get_patch_dir_names("pangu_v2_hybrid, pangu_v2_moe") == [
        "pangu_v2_base",
        "high_throughout",
        "low_latency",
    ]


def test_find_patch_dir_exact_high_throughout_includes_pangu_v2_base():
    models_root = Path(vllm_patches.__file__).parent / "usefull_patch" / "models"

    patch_dirs = vllm_patches._find_patch_dir_exact("high_throughout", models_root)

    assert [path.name for path in patch_dirs] == [
        "pangu_v2_base",
        "high_throughout",
    ]


def test_find_patch_dir_exact_low_latency_includes_pangu_v2_base():
    models_root = Path(vllm_patches.__file__).parent / "usefull_patch" / "models"

    patch_dirs = vllm_patches._find_patch_dir_exact("low_latency", models_root)

    assert [path.name for path in patch_dirs] == [
        "pangu_v2_base",
        "low_latency",
    ]


def test_find_patch_dir_exact_legacy_pangu_v2_hybrid():
    models_root = Path(vllm_patches.__file__).parent / "usefull_patch" / "models"

    patch_dirs = vllm_patches._find_patch_dir_exact("pangu_v2_hybrid", models_root)

    assert [path.name for path in patch_dirs] == [
        "pangu_v2_base",
        "high_throughout",
    ]


def test_find_patch_dir_exact_legacy_pangu_v2_moe():
    models_root = Path(vllm_patches.__file__).parent / "usefull_patch" / "models"

    patch_dirs = vllm_patches._find_patch_dir_exact("pangu_v2_moe", models_root)

    assert [path.name for path in patch_dirs] == [
        "pangu_v2_base",
        "low_latency",
    ]
