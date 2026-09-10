# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MomeModelState：把本步 prompt 长度交给 MoME builder。

MRv1 走 _prepare_inputs -> _refresh_mome_num_prompt_tokens，MRv2 的对应位置是
模型自己挑的 ModelState —— runner 里不再有任何 MoME 相关的覆写。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _make_state(max_num_reqs=4):
    import torch

    from omni_npu.worker.npu.mome_model_state import MomeModelState

    state = object.__new__(MomeModelState)
    state.max_num_reqs = max_num_reqs
    state.device = torch.device("cpu")
    state.prompt_len = torch.zeros(max_num_reqs, dtype=torch.int32)
    state.num_prompt_tokens = torch.zeros(max_num_reqs, dtype=torch.int32)
    state._real_batch = False
    state._align_mode = False  # 让上游 preprocess_state 早退，真调 super()
    return state


def test_add_request_records_the_prompt_length(monkeypatch):
    """长度取 prompt_token_ids，和上游 req_states.prompt_len 同源。"""
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    from omni_npu.worker.npu.mome_model_state import MomeModelState

    seen = []
    monkeypatch.setattr(
        MambaHybridModelState, "add_request",
        lambda self, req_index, new_req_data: seen.append(req_index))
    state = _make_state()

    state.add_request(2, SimpleNamespace(prompt_token_ids=[1, 2, 3, 4, 5]))

    assert state.prompt_len.tolist() == [0, 0, 5, 0]
    assert seen == [2]  # 上游那一半（num_accepted_tokens 等）必须照跑
    assert MomeModelState.add_request is not MambaHybridModelState.add_request


def test_prepare_attn_binds_this_step_lengths_and_clears_the_tail(monkeypatch):
    import torch
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    from omni_npu.attention.backends import mome

    bound = []
    monkeypatch.setattr(mome, "bind_num_prompt_tokens",
                        lambda groups, value: bound.append((groups, value)))
    forwarded = {}
    metadata = object()

    def fake_super(self, *args, **kwargs):
        # 绑定必须发生在上游 build 之前，否则 builder 读到的是上一步的
        forwarded["at_call"] = None if bound[-1][1] is None else bound[-1][1].tolist()
        forwarded["args"] = args
        forwarded["kwargs"] = kwargs
        return metadata

    monkeypatch.setattr(MambaHybridModelState, "prepare_attn", fake_super)
    state = _make_state()
    state.prompt_len = torch.tensor([3, 5, 7, 9], dtype=torch.int32)
    state.num_prompt_tokens = torch.full((4,), 99, dtype=torch.int32)
    state._real_batch = True  # preprocess_state 立的
    groups = [[object()]]
    batch = SimpleNamespace(
        num_reqs=2, idx_mapping=torch.tensor([2, 0], dtype=torch.int64))

    result = state.prepare_attn(batch, "cg", "tables", "slots", groups, "kv")

    assert result is metadata
    assert bound[-1][0] is groups
    assert bound[-1][1] is state.num_prompt_tokens
    # 按本步行序 gather，尾部清零而不是留着上一步的 99
    assert forwarded["at_call"] == [7, 3, 0, 0]
    assert forwarded["args"] == (batch, "cg", "tables", "slots", groups, "kv")
    assert forwarded["kwargs"] == {"for_capture": False}


def test_prepare_attn_unbinds_for_capture(monkeypatch):
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    from omni_npu.attention.backends import mome

    bound = []
    monkeypatch.setattr(mome, "bind_num_prompt_tokens",
                        lambda groups, value: bound.append(value))
    monkeypatch.setattr(
        MambaHybridModelState, "prepare_attn",
        lambda self, *args, **kwargs: None)
    state = _make_state()

    # 捕获期 seq_lens 是编造的；input_batch 故意不给属性，碰它就该炸
    state.prepare_attn(object(), "cg", "tables", "slots", [[object()]], "kv",
                       for_capture=True)

    assert bound == [None]


def test_preprocess_state_marks_the_batch_real(monkeypatch):
    """上游保证它只在真实批上跑，且早于 prepare_attn。"""
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    seen = []
    monkeypatch.setattr(
        MambaHybridModelState, "preprocess_state",
        lambda self, *args, **kwargs: seen.append(args))
    state = _make_state()

    state.preprocess_state("batch", "tables", "kv", "computed")

    assert state._real_batch is True
    assert seen == [("batch", "tables", "kv", "computed")]  # 上游那半照跑


def test_a_dummy_batch_unbinds_and_cannot_inherit_the_token(monkeypatch):
    """dummy 批不经过 preprocess_state，令牌取用即清，下一批拿不到。"""
    import torch
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    from omni_npu.attention.backends import mome

    bound = []
    monkeypatch.setattr(mome, "bind_num_prompt_tokens",
                        lambda groups, value: bound.append(value))
    monkeypatch.setattr(
        MambaHybridModelState, "prepare_attn",
        lambda self, *args, **kwargs: None)
    state = _make_state()
    state.prompt_len = torch.tensor([3, 5, 7, 9], dtype=torch.int32)
    groups = [[object()]]
    batch = SimpleNamespace(
        num_reqs=2, idx_mapping=torch.tensor([2, 0], dtype=torch.int64))

    # 1) 真实批：preprocess_state 立令牌，prepare_attn 取用
    state.preprocess_state(batch, "tables", "kv", "computed")
    state.prepare_attn(batch, "cg", "tables", "slots", groups, "kv")
    assert bound[-1] is state.num_prompt_tokens

    # 2) 紧接着的 dummy 批没有 preprocess_state —— input_batch 故意不给属性，
    #    走到取值就该炸，而不是拿伪造的 idx_mapping 去 gather
    state.prepare_attn(object(), "cg", "tables", "slots", groups, "kv")
    assert bound[-1] is None
    assert state._real_batch is False


def test_capture_does_not_consume_a_pending_token(monkeypatch):
    """捕获不置也不清令牌，否则它后面那一批会被当成 dummy。"""
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    from omni_npu.attention.backends import mome

    monkeypatch.setattr(mome, "bind_num_prompt_tokens",
                        lambda groups, value: None)
    monkeypatch.setattr(
        MambaHybridModelState, "prepare_attn",
        lambda self, *args, **kwargs: None)
    state = _make_state()
    state._real_batch = True

    state.prepare_attn(object(), "cg", "tables", "slots", [[object()]], "kv",
                       for_capture=True)

    assert state._real_batch is True


def test_the_model_selects_this_state():
    """上游正式扩展点：模型自己声明，runner 里不留 patch。"""
    from omni_npu.v1.models.pangu.pangu_v2_moe import OpenPanguV2ForCausalLM
    from omni_npu.worker.npu.mome_model_state import MomeModelState

    assert OpenPanguV2ForCausalLM.get_model_state_cls() is MomeModelState


def test_the_runner_no_longer_binds_mome():
    """绑定只能有一处；runner 里再出现就是又打了一层 patch。

    prepare_dummy_attn 的注释里提 MoME 是解释块表为什么要清零，不是绑定。
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[4]
              / "omni/worker/npu/model_runner.py").read_text(encoding="utf-8")

    assert "bind_num_prompt_tokens" not in source
    assert "_bind_mome_prompt_lens" not in source
    assert "def load_model" not in source  # 曾经只为包 prepare_attn 而存在
