# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MRv2 sampling injection: bindings, MRv1 isolation, missing-target fallback."""

from __future__ import annotations

import subprocess
import sys

import pytest

from omni_npu.vllm_patches.usefull_patch.common import patch_mrv2_sampler as patch_mod
from omni_npu.worker.npu import sampler as npu_sample
from omni_npu.worker.npu.ops import rejection_sampler_utils as npu_rejection_sampler_utils
from unit.vllm_patch.useful_patch.patch_test_utils import applied_patches


def _patch_classes(mod):
    classes = []
    for name in (
        "MRv2GumbelPatch",
        "MRv2SamplerPatch",
        "MRv2SamplerStatesPatch",
        "MRv2SpeculatorGumbelPatch",
        "MRv2DsparkSpeculatorGumbelPatch",
        "MRv2RejectionSamplerPatch",
        "MRv2RejectionSamplerUtilsPatch",
        "MRv2V1RejectionSamplerTopKTopPPatch",
    ):
        if hasattr(mod, name):
            classes.append(getattr(mod, name))
    return classes


@pytest.fixture
def applied():
    """Apply this file's patches, restoring symbols and ownership on exit."""
    classes = _patch_classes(patch_mod)
    with applied_patches(classes) as applied_classes:
        yield applied_classes


def test_gumbel_binding_covers_the_speculators(applied):
    """Draft sampling is the easy one to miss: each speculator from-imports it."""
    import vllm.v1.worker.gpu.spec_decode.speculator as up_speculator

    assert up_speculator.gumbel_sample is npu_sample.gumbel_sample

    dspark = sys.modules.get("vllm.v1.worker.gpu.spec_decode.dspark.speculator")
    if dspark is not None:
        assert dspark.gumbel_sample is npu_sample.gumbel_sample


def test_top_k_top_p_definition_module_is_left_to_mrv1(applied):
    """MRv1 guard: TopKTopPSampler uses the defining module, which stays upstream."""
    from vllm.v1.sample.ops import topk_topp_sampler as up_topk_topp

    assert up_topk_topp.apply_top_k_top_p is not npu_sample.apply_top_k_top_p
    assert (
        up_topk_topp.apply_top_k_top_p.__module__
        == "vllm.v1.sample.ops.topk_topp_sampler"
    )


def test_rejection_sampler_bindings_are_patched(applied):
    """MTP rejection sampling keeps from-imported bindings in several modules."""
    import vllm.v1.sample.rejection_sampler as up_v1_rejection_sampler
    import vllm.v1.worker.gpu.spec_decode.rejection_sampler as up_rejection_sampler
    import vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils as up_rejection_sampler_utils

    assert (
        up_rejection_sampler.rejection_sample
        is npu_rejection_sampler_utils.rejection_sample
    )
    assert (
        up_rejection_sampler_utils.rejection_sample
        is npu_rejection_sampler_utils.rejection_sample
    )
    assert up_v1_rejection_sampler.apply_top_k_top_p is npu_sample.apply_top_k_top_p


def test_missing_target_degrades_to_an_error_log():
    """An unimportable target must log and skip registration, never raise.

    The plugin imports this directory wholesale and exec_module has no
    try/except, so raising here would take MRv1 down with it.

    Runs in a subprocess: sibling tests stub entries in sys.modules, and the
    module is loaded fresh rather than reloaded (reload keeps the old namespace).
    """
    code = (
        "import importlib.util, sys\n"
        "import omni_npu.vllm_patches.usefull_patch.common.patch_mrv2_sampler as m\n"
        "sys.modules['vllm.v1.worker.gpu.sample.states'] = None\n"
        "spec = importlib.util.spec_from_file_location('probe', m.__file__)\n"
        "probe = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(probe)  # must not raise\n"
        "leaked = [n for n in ('MRv2GumbelPatch', 'MRv2SamplerPatch',\n"
        "                      'MRv2SamplerStatesPatch', 'MRv2SpeculatorGumbelPatch')\n"
        "          if hasattr(probe, n)]\n"
        "assert not leaked, f'registered despite a missing target: {leaked}'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )

    assert proc.returncode == 0, (
        f"fallback failed:\nstdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
    )
    assert "not registered" in (proc.stdout + proc.stderr), "expected an error log"
