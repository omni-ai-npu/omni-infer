# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MRv2 capture-time graph signals: the mode rewrite, and both patch bindings."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from omni_npu.vllm_patches.usefull_patch.common import patch_mrv2_capture_mode as patch_mod
from omni_npu.worker.npu import aclgraph_utils as cudagraph


def _desc(cg_mode):
    return SimpleNamespace(cg_mode=cg_mode)


def _upstream_capture_loop(self, create_forward_fn, descs, tag="unused"):
    """复刻 CudaGraphManager.capture 的循环（cudagraph_utils.py:317-365）。

    三个细节必须保留，它们正是被测行为的前提：FULL 的工厂在 warmup 与录制各
    调一次而 PIECEWISE 复用 warmup 那个闭包、warmup 用关键字传参、FULL 的录制
    那一遍传的是字面量 NONE。
    """
    from vllm.config import CUDAGraphMode

    for desc in descs:
        forward_fn, _ = create_forward_fn(desc, warmup=True)
        forward_fn(CUDAGraphMode.NONE)
        if desc.cg_mode == CUDAGraphMode.PIECEWISE:
            forward_fn(CUDAGraphMode.PIECEWISE)
        else:
            forward_fn, _ = create_forward_fn(desc, warmup=False)
            forward_fn(CUDAGraphMode.NONE)
    return "captured"


def _recording_factory(seen, attn_state="attn-state"):
    """工厂：把每次 forward_fn 收到的 mode 连同 (cg_mode, warmup) 记下来。"""

    def create_forward_fn(desc, warmup):
        def forward_fn(cg_mode):
            seen.append((desc.cg_mode.name, "warmup" if warmup else "record",
                         cg_mode.name))

        return forward_fn, attn_state

    return create_forward_fn


def _flag_recording_factory(seen):
    """工厂：记下每次 forward_fn 期间 capturing 标志的取值。"""

    def create_forward_fn(desc, warmup):
        def forward_fn(cg_mode):
            seen.append((desc.cg_mode.name, "warmup" if warmup else "record",
                         cudagraph._capturing_default()))

        return forward_fn, "attn-state"

    return create_forward_fn


def test_only_the_full_recorded_pass_is_rewritten():
    """FULL 的录制那一遍改成 FULL；warmup 与 PIECEWISE 一律不动。"""
    from vllm.config import CUDAGraphMode

    seen = []
    capture = cudagraph.build_capture_with_full_mode(_upstream_capture_loop)
    capture(
        object(),
        _recording_factory(seen),
        [_desc(CUDAGraphMode.FULL), _desc(CUDAGraphMode.PIECEWISE)],
    )

    assert seen == [
        ("FULL", "warmup", "NONE"),            # 图外，保持 NONE
        ("FULL", "record", "FULL"),            # ← 唯一被改写的一处
        ("PIECEWISE", "warmup", "NONE"),
        ("PIECEWISE", "warmup", "PIECEWISE"),  # 复用 warmup 闭包，真值不误伤
    ]


def test_dual_configs_are_resolved_before_reaching_the_wrapper():
    """FULL_DECODE_ONLY 之类的双模式在建 desc 时就被拆开，这里只会见到单模式。

    上游用 decode_mode()/mixed_mode() 拆（cudagraph_utils.py:189-190），
    所以包装器按 desc.cg_mode 判断即可，新增双模式组合也不用改。
    """
    from vllm.config import CUDAGraphMode

    decode_only = CUDAGraphMode.FULL_DECODE_ONLY
    assert decode_only.decode_mode() is CUDAGraphMode.FULL
    assert decode_only.mixed_mode() is CUDAGraphMode.NONE

    seen = []
    capture = cudagraph.build_capture_with_full_mode(_upstream_capture_loop)
    # mixed_mode 为 NONE 时上游不建 mixed 表，只剩 decode 侧的 FULL desc
    capture(object(), _recording_factory(seen), [_desc(decode_only.decode_mode())])

    assert seen == [("FULL", "warmup", "NONE"), ("FULL", "record", "FULL")]


def test_a_non_none_recorded_mode_passes_through():
    """只把 NONE 换成 FULL。上游若哪天在录制那一遍传了别的值，原样透传。"""
    from vllm.config import CUDAGraphMode

    seen = []

    def loop(self, create_forward_fn, descs):
        forward_fn, _ = create_forward_fn(descs[0], warmup=False)
        forward_fn(CUDAGraphMode.PIECEWISE)

    capture = cudagraph.build_capture_with_full_mode(loop)
    capture(object(), _recording_factory(seen), [_desc(CUDAGraphMode.FULL)])

    assert seen == [("FULL", "record", "PIECEWISE")]


def test_factory_second_return_value_and_extra_args_are_forwarded():
    """attn_state 原样返回；capture 的其余参数透传给上游。"""
    from vllm.config import CUDAGraphMode

    states, forwarded = [], {}

    def loop(self, create_forward_fn, descs, tag):
        forwarded["tag"] = tag
        _, attn_state = create_forward_fn(descs[0], warmup=False)
        states.append(attn_state)
        return "returned"

    capture = cudagraph.build_capture_with_full_mode(loop)
    result = capture(
        object(),
        _recording_factory([], attn_state="the-state"),
        [_desc(CUDAGraphMode.FULL)],
        tag="progress-bar",
    )

    assert states == ["the-state"]
    assert forwarded == {"tag": "progress-bar"}
    assert result == "returned"


def test_capture_mode_patch_replaces_only_capture():
    cls = patch_mod.MRv2CaptureModePatch
    target = cls._target
    saved = target.capture
    owners = dict(getattr(target, "_omni_npu_applied_patches", {}))
    target._omni_npu_applied_patches = {}
    try:
        cls.apply()
        assert target.capture is cls.__dict__["capture"]
        assert target.capture is not saved
    finally:
        target.capture = saved
        target._omni_npu_applied_patches = owners


def test_capturing_flag_patch_installs_the_descriptor():
    cls = patch_mod.MRv2CapturingFlagPatch
    target = cls._target
    saved = target.__dict__.get("capturing")
    owners = dict(getattr(target, "_omni_npu_applied_patches", {}))
    target._omni_npu_applied_patches = {}
    try:
        cls.apply()
        assert isinstance(
            target.__dict__["capturing"], cudagraph._CapturingDescriptor
        )
    finally:
        if saved is None:
            del target.capturing
        else:
            target.capturing = saved
        target._omni_npu_applied_patches = owners


def test_capturing_flag_is_raised_only_for_the_full_recorded_pass():
    """标志与 mode 改写同范围：只有 FULL 的录制那一遍抬起。

    warmup 跑在 torch.cuda.graph 之外，抬起会让 attention 在没有图的情况下
    注册 ACL task；PIECEWISE 由内层 wrapper 自己管。标志挂在这次调用上而非
    torch.cuda.graph 这个公共名字上，别处进图也就不会误触。
    """
    from vllm.config import CUDAGraphMode

    cudagraph.reset_for_testing()
    seen = []
    capture = cudagraph.build_capture_with_full_mode(_upstream_capture_loop)
    capture(
        object(),
        _flag_recording_factory(seen),
        [_desc(CUDAGraphMode.FULL), _desc(CUDAGraphMode.PIECEWISE)],
    )

    assert seen == [
        ("FULL", "warmup", False),
        ("FULL", "record", True),          # ← 唯一抬起的一处
        ("PIECEWISE", "warmup", False),
        ("PIECEWISE", "warmup", False),   # PIECEWISE 复用 warmup 闭包，全程不抬
    ]
    assert cudagraph._capturing_default() is False   # 每次调用后都落回


def test_capturing_flag_is_restored_when_the_recorded_pass_raises():
    """录制那一遍抛异常，标志也必须落回，否则后续步骤会一直以为在捕获。"""
    from vllm.config import CUDAGraphMode

    cudagraph.reset_for_testing()

    def create_forward_fn(desc, warmup):
        def forward_fn(cg_mode):
            if warmup:                       # 上游先跑 warmup，那一遍不该抬标志
                assert cudagraph._capturing_default() is False
                return
            assert cudagraph._capturing_default() is True
            raise RuntimeError("capture blew up")

        return forward_fn, "attn-state"

    capture = cudagraph.build_capture_with_full_mode(_upstream_capture_loop)
    with pytest.raises(RuntimeError, match="capture blew up"):
        capture(object(), create_forward_fn, [_desc(CUDAGraphMode.FULL)])

    assert cudagraph._capturing_default() is False
