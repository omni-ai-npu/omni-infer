# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from vllm.v1.kv_cache_interface import MambaSpec

import omni.worker.npu_model_runner as runner_module
from omni.vllm_patches.patches.models.qwen import qwen_hybrid_common
from omni.worker.npu_model_runner import NPUModelRunner


def test_build_conv_context_removes_stale_request_cache(monkeypatch):
    runner = object.__new__(NPUModelRunner)
    forward_context = SimpleNamespace()
    runner.req_cache_map = {"stale": 9, "keep": 7}
    runner.input_batch = SimpleNamespace(req_ids=["keep", "new"], num_reqs=2)
    runner.cache_slot_id = torch.full((4,), -1, dtype=torch.int64)
    monkeypatch.setattr(
        runner_module,
        "get_forward_context",
        lambda: forward_context,
    )

    runner._build_conv_context()

    assert runner.req_cache_map == {"keep": 1, "new": 2}
    assert runner.cache_slot_id.tolist() == [7, 0, 0, 0]
    assert forward_context.cache_slot_id is runner.cache_slot_id


def test_qwen_hybrid_input_batch_patch_adjusts_mamba_block_size():
    runner = object.__new__(NPUModelRunner)
    runner.cache_config = SimpleNamespace(block_size=16, cpu_offload_gb=0)
    runner._init_npu_input_batch = MagicMock()
    mamba_spec = MambaSpec(
        block_size=8,
        shapes=((2, 2),),
        dtypes=(torch.float32,),
    )
    attention_spec = SimpleNamespace(block_size=32)
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=mamba_spec),
            SimpleNamespace(kv_cache_spec=attention_spec),
        ]
    )

    qwen_hybrid_common.QwenHybridInputBatchPatch.may_reinitialize_input_batch(
        runner, kv_cache_config, [8, 32]
    )

    runner._init_npu_input_batch.assert_called_once_with([16, 32], [16, 32])


def test_kv_cache_after_wake_up_handles_sink_attention_without_static_mla(monkeypatch):
    class FakeSinkAttention:
        pass

    module = FakeSinkAttention()
    runner = object.__new__(NPUModelRunner)
    runner.compilation_config = SimpleNamespace(static_forward_context={"layer": module})
    runner.model_config = SimpleNamespace(enable_sleep_mode=True)
    runner._kv_cache_sink_attn_after_wake_up = MagicMock()
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.static_sink_attention.StaticSinkAttention",
        FakeSinkAttention,
    )
    monkeypatch.delattr(
        "vllm.model_executor.layers.attention.static_sink_attention.StaticSinkMLAAttention",
        raising=False,
    )
    monkeypatch.setattr(runner_module.logger, "warning", MagicMock())

    runner.kv_cache_after_wake_up()

    runner_module.logger.warning.assert_called_once()
    runner._kv_cache_sink_attn_after_wake_up.assert_called_once_with(module)


def test_kv_cache_sink_attn_after_wake_up_populates_cache_and_reinits_builders():
    runner = object.__new__(NPUModelRunner)
    builder = SimpleNamespace(reinit_block_table_with_sink=MagicMock())
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    runner.attn_groups = [[SimpleNamespace(metadata_builders=[builder])]]
    sink_cache = (torch.ones(1), torch.ones(1))
    module = SimpleNamespace(
        kv_cache=(sink_cache,),
        maybe_populate_sink_kv_after_wakeup=MagicMock(),
    )

    runner._kv_cache_sink_attn_after_wake_up(module)

    module.maybe_populate_sink_kv_after_wakeup.assert_called_once_with(
        sink_cache[0], sink_cache[1]
    )
    builder.reinit_block_table_with_sink.assert_called_once()


def test_take_draft_token_ids_filters_discarded_requests(monkeypatch):
    runner = object.__new__(NPUModelRunner)
    runner.num_spec_tokens = 2
    runner._draft_token_req_ids = ["req0", "req1", "req2"]
    runner.discard_request_mask = SimpleNamespace(
        cpu=torch.tensor([False, True, False])
    )
    runner._get_draft_token_ids_cpu = lambda: (
        [[1, 2], [3, 4], [5, 6]],
        ["req0", "req1", "req2"],
    )
    monkeypatch.setattr(
        runner_module,
        "DraftTokenIds",
        lambda req_ids, token_ids: (req_ids, token_ids),
    )

    assert runner.take_draft_token_ids() == (
        ["req0", "req2"],
        [[1, 2], [5, 6]],
    )


def test_take_draft_token_ids_returns_none_when_all_requests_discarded(monkeypatch):
    runner = object.__new__(NPUModelRunner)
    runner.num_spec_tokens = 2
    runner._draft_token_req_ids = ["req0"]
    runner.discard_request_mask = SimpleNamespace(cpu=torch.tensor([True]))
    runner._get_draft_token_ids_cpu = lambda: ([[1, 2]], ["req0"])
    monkeypatch.setattr(
        runner_module,
        "DraftTokenIds",
        lambda req_ids, token_ids: (req_ids, token_ids),
    )

    assert runner.take_draft_token_ids() is None
