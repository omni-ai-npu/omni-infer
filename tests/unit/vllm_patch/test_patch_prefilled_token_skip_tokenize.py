# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for patch_prefilled_token_skip_tokenize."""

from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from omni_npu.vllm_patches.patches.common import (
    patch_prefilled_token_skip_tokenize as patch_mod,
)


class TestSpeculativeMarginEngineCapture(unittest.TestCase):
    def setUp(self) -> None:
        patch_mod._ENGINE_VLLM_CONFIG = None

    def tearDown(self) -> None:
        patch_mod._ENGINE_VLLM_CONFIG = None

    def test_real_config_wins_over_captured(self) -> None:
        patch_mod._ENGINE_VLLM_CONFIG = SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=99))
        cfg = SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=3))
        self.assertEqual(patch_mod._speculative_margin(cfg), 9)

    def test_captured_config_used_when_arg_missing(self) -> None:
        # Simulates the serving-side call: its own config resolves to None, but
        # the engine's vllm_config was remembered on a prior generate().
        patch_mod._remember_engine_vllm_config(
            SimpleNamespace(
                speculative_config=SimpleNamespace(num_speculative_tokens=3)))
        self.assertEqual(patch_mod._speculative_margin(None), 9)

    def test_zero_when_nothing_captured(self) -> None:
        patch_mod._ENGINE_VLLM_CONFIG = None
        self.assertEqual(patch_mod._speculative_margin(None), 0)

    def test_remember_is_sticky_and_ignores_none(self) -> None:
        patch_mod._remember_engine_vllm_config(
            SimpleNamespace(
                speculative_config=SimpleNamespace(num_speculative_tokens=3)))
        patch_mod._remember_engine_vllm_config(None)  # must not clobber
        self.assertEqual(patch_mod._speculative_margin(None), 9)


class _FakeKvConnectorOutput:
    def __init__(self, finished_recving=None, finished_sending=None):
        self.finished_recving = finished_recving
        self.finished_sending = finished_sending


class _FakeScheduler:
    def __init__(self, requests: dict[str, Request], max_model_len: int = 8):
        self.requests = requests
        self.max_model_len = max_model_len
        self.finish_requests_calls: list[tuple] = []
        self.finished_recving_kv_req_ids: set[str] = set(requests)
        self.failed_recving_kv_req_ids: set[str] = set()

    def finish_requests(self, request_ids, finished_status) -> None:
        self.finish_requests_calls.append((request_ids, finished_status))


def _make_request(
    prompt_len: int,
    max_tokens: int,
    prefilled_token: list[int] | None,
) -> Request:
    extra_args = {}
    if prefilled_token is not None:
        extra_args["kv_transfer_params"] = {"prefilled_token": list(prefilled_token)}
    sampling_params = SamplingParams(max_tokens=max_tokens, extra_args=extra_args or None)
    return Request(
        request_id="req-1",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=sampling_params,
        pooling_params=None,
        eos_token_id=None,
    )


def _run(scheduler: _FakeScheduler, kv_connector_output: _FakeKvConnectorOutput) -> None:
    with mock.patch.object(patch_mod, "_original_update_from_kv_xfer_finished"):
        patch_mod.SchedulerKvXferFinishedPatch._update_from_kv_xfer_finished(
            scheduler, kv_connector_output
        )


class TestUpdateFromKvXferFinished(unittest.TestCase):
    def setUp(self) -> None:
        patch_mod._ENGINE_VLLM_CONFIG = None  # isolate captured-config global
        os.environ["OMNI_REUSE_PREFILLED_TOKENS"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("OMNI_REUSE_PREFILLED_TOKENS", None)

    def test_feature_flag_off_is_noop(self) -> None:
        os.environ["OMNI_REUSE_PREFILLED_TOKENS"] = "0"
        request = _make_request(prompt_len=4, max_tokens=1, prefilled_token=[99])
        scheduler = _FakeScheduler({request.request_id: request})
        _run(scheduler, _FakeKvConnectorOutput(finished_recving=[request.request_id]))

        self.assertEqual(request.num_output_tokens, 0)
        self.assertEqual(scheduler.finish_requests_calls, [])

    def test_no_prefilled_token_is_noop(self) -> None:
        request = _make_request(prompt_len=4, max_tokens=1, prefilled_token=None)
        scheduler = _FakeScheduler({request.request_id: request})
        _run(scheduler, _FakeKvConnectorOutput(finished_recving=[request.request_id]))

        self.assertEqual(request.num_output_tokens, 0)
        self.assertEqual(scheduler.finish_requests_calls, [])

    def test_prefilled_token_exhausting_budget_finishes_request(self) -> None:
        request = _make_request(prompt_len=4, max_tokens=1, prefilled_token=[99])
        scheduler = _FakeScheduler({request.request_id: request}, max_model_len=8)
        _run(scheduler, _FakeKvConnectorOutput(finished_recving=[request.request_id]))

        self.assertEqual(list(request.output_token_ids), [99])
        self.assertEqual(len(scheduler.finish_requests_calls), 1)
        req_ids, status = scheduler.finish_requests_calls[0]
        self.assertEqual(req_ids, request.request_id)
        self.assertEqual(status, RequestStatus.FINISHED_LENGTH_CAPPED)
        self.assertNotIn(
            "prefilled_token",
            request.sampling_params.extra_args["kv_transfer_params"],
        )
        self.assertNotIn(
            request.request_id, scheduler.finished_recving_kv_req_ids)
        pending = scheduler._omni_pending_finish_outputs
        self.assertEqual(len(pending), 1)
        client_index, engine_core_output = pending[0]
        self.assertEqual(client_index, request.client_index)
        self.assertEqual(engine_core_output.request_id, request.request_id)
        self.assertEqual(
            engine_core_output.finish_reason,
            RequestStatus.get_finished_reason(RequestStatus.FINISHED_LENGTH_CAPPED),
        )
        self.assertEqual(engine_core_output.new_token_ids, [])
        self.assertEqual(request.status, RequestStatus.WAITING_FOR_REMOTE_KVS)

    def test_context_exhausted_before_budget_finishes_as_error(self) -> None:
        request = _make_request(prompt_len=7, max_tokens=5, prefilled_token=[99])
        scheduler = _FakeScheduler({request.request_id: request}, max_model_len=8)
        _run(scheduler, _FakeKvConnectorOutput(finished_recving=[request.request_id]))

        self.assertEqual(len(scheduler.finish_requests_calls), 1)
        _, status = scheduler.finish_requests_calls[0]
        self.assertEqual(status, RequestStatus.FINISHED_ERROR)
        pending = scheduler._omni_pending_finish_outputs
        self.assertEqual(len(pending), 1)
        self.assertEqual(
            pending[0][1].finish_reason,
            RequestStatus.get_finished_reason(RequestStatus.FINISHED_ERROR),
        )
        self.assertEqual(request.status, RequestStatus.WAITING_FOR_REMOTE_KVS)

    def test_speculative_context_boundary_finishes_as_error(self) -> None:
        request = _make_request(prompt_len=13, max_tokens=5, prefilled_token=[99])
        scheduler = _FakeScheduler({request.request_id: request}, max_model_len=20)
        scheduler.vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=2)
        )
        _run(scheduler, _FakeKvConnectorOutput(finished_recving=[request.request_id]))

        self.assertEqual(len(scheduler.finish_requests_calls), 1)
        _, status = scheduler.finish_requests_calls[0]
        self.assertEqual(status, RequestStatus.FINISHED_ERROR)

    def test_failed_recving_request_is_left_for_retry(self) -> None:
        request = _make_request(prompt_len=4, max_tokens=1, prefilled_token=[99])
        scheduler = _FakeScheduler({request.request_id: request}, max_model_len=8)
        scheduler.failed_recving_kv_req_ids.add(request.request_id)
        _run(scheduler, _FakeKvConnectorOutput(finished_recving=[request.request_id]))

        self.assertEqual(request.num_output_tokens, 0)
        self.assertIn(
            "prefilled_token",
            request.sampling_params.extra_args["kv_transfer_params"],
        )
        self.assertEqual(scheduler.finish_requests_calls, [])

    def test_prefilled_token_matching_eos_finishes_as_stopped(self) -> None:
        extra_args = {"kv_transfer_params": {"prefilled_token": [7]}}
        sampling_params = SamplingParams(max_tokens=5, extra_args=extra_args)
        request = Request(
            request_id="req-1",
            prompt_token_ids=list(range(4)),
            sampling_params=sampling_params,
            pooling_params=None,
            eos_token_id=7,
        )
        scheduler = _FakeScheduler({request.request_id: request}, max_model_len=8)
        _run(scheduler, _FakeKvConnectorOutput(finished_recving=[request.request_id]))

        self.assertEqual(len(scheduler.finish_requests_calls), 1)
        _, status = scheduler.finish_requests_calls[0]
        self.assertEqual(status, RequestStatus.FINISHED_STOPPED)
        pending = scheduler._omni_pending_finish_outputs
        self.assertEqual(len(pending), 1)
        self.assertEqual(
            pending[0][1].finish_reason,
            RequestStatus.get_finished_reason(RequestStatus.FINISHED_STOPPED),
        )
        self.assertEqual(request.status, RequestStatus.WAITING_FOR_REMOTE_KVS)

    def test_prefilled_token_under_budget_keeps_request_running(self) -> None:
        request = _make_request(prompt_len=4, max_tokens=3, prefilled_token=[99])
        scheduler = _FakeScheduler({request.request_id: request}, max_model_len=8)
        _run(scheduler, _FakeKvConnectorOutput(finished_recving=[request.request_id]))

        self.assertEqual(list(request.output_token_ids), [99])
        self.assertEqual(scheduler.finish_requests_calls, [])

    def test_unknown_or_already_finished_request_is_skipped(self) -> None:
        finished_request = _make_request(prompt_len=4, max_tokens=1, prefilled_token=[99])
        finished_request.status = RequestStatus.FINISHED_STOPPED
        scheduler = _FakeScheduler({finished_request.request_id: finished_request})
        _run(
            scheduler,
            _FakeKvConnectorOutput(
                finished_recving=["missing-req", finished_request.request_id]
            ),
        )

        self.assertEqual(finished_request.num_output_tokens, 0)
        self.assertEqual(scheduler.finish_requests_calls, [])


class _FakeSchedulerForFallback:
    class _CacheManager:
        def cache_blocks(self, request, num_tokens) -> None:
            pass

        def free(self, request) -> None:
            raise AssertionError("free should not be called")

    def __init__(self, request: Request, max_model_len: int = 8):
        self.connector = object()
        self.max_model_len = max_model_len
        self.finished_recving_kv_req_ids = {request.request_id}
        self.failed_recving_kv_req_ids: set[str] = set()
        self.kv_cache_manager = self._CacheManager()


class TestSchedulerFallbackAppend(unittest.TestCase):
    def setUp(self) -> None:
        patch_mod._ENGINE_VLLM_CONFIG = None  # isolate captured-config global
        os.environ["OMNI_REUSE_PREFILLED_TOKENS"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("OMNI_REUSE_PREFILLED_TOKENS", None)

    def test_fallback_appends_below_boundary(self) -> None:
        request = _make_request(prompt_len=4, max_tokens=3, prefilled_token=[99])
        scheduler = _FakeSchedulerForFallback(request, max_model_len=8)

        ready = patch_mod.SchedulerPatch._update_waiting_for_remote_kv(
            scheduler, request)

        self.assertTrue(ready)
        self.assertEqual(list(request.output_token_ids), [99])
        self.assertEqual(len(request.prompt_token_ids), 5)
        self.assertNotIn(
            "prefilled_token",
            request.sampling_params.extra_args["kv_transfer_params"],
        )

    def test_fallback_drops_token_at_context_boundary(self) -> None:
        request = _make_request(prompt_len=7, max_tokens=3, prefilled_token=[99])
        scheduler = _FakeSchedulerForFallback(request, max_model_len=8)

        ready = patch_mod.SchedulerPatch._update_waiting_for_remote_kv(
            scheduler, request)

        self.assertTrue(ready)
        self.assertEqual(request.num_output_tokens, 0)
        self.assertEqual(len(request.prompt_token_ids), 7)
        self.assertNotIn(
            "prefilled_token",
            request.sampling_params.extra_args["kv_transfer_params"],
        )

    def test_fallback_uses_speculative_context_boundary(self) -> None:
        request = _make_request(prompt_len=13, max_tokens=1, prefilled_token=[99])
        scheduler = _FakeSchedulerForFallback(request, max_model_len=20)
        scheduler.vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=2)
        )

        ready = patch_mod.SchedulerPatch._update_waiting_for_remote_kv(
            scheduler, request)

        self.assertTrue(ready)
        self.assertEqual(request.num_output_tokens, 0)
        self.assertNotIn(
            "prefilled_token",
            request.sampling_params.extra_args["kv_transfer_params"],
        )


class TestContextLimits(unittest.TestCase):
    def test_speculative_margin_and_effective_limit(self) -> None:
        vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=2)
        )

        self.assertEqual(patch_mod._speculative_margin(vllm_config), 6)
        self.assertEqual(
            patch_mod._effective_max_model_len(20, vllm_config), 14)
        self.assertTrue(
            patch_mod._fits_effective_max_model_len(
                13, 20, vllm_config, reserve_tokens=1))
        self.assertFalse(
            patch_mod._has_decode_room(14, 20, vllm_config))


class _FakeSchedulerForAddRequest:
    def __init__(self, max_model_len: int = 8):
        self.max_model_len = max_model_len
        self.finish_requests_calls: list[tuple] = []

    def finish_requests(self, request_ids, finished_status) -> None:
        self.finish_requests_calls.append((request_ids, finished_status))


class TestSchedulerAddRequestGuard(unittest.TestCase):
    def setUp(self) -> None:
        patch_mod._ENGINE_VLLM_CONFIG = None  # isolate captured-config global
        os.environ["OMNI_REUSE_PREFILLED_TOKENS"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("OMNI_REUSE_PREFILLED_TOKENS", None)
        os.environ.pop("OMNI_SKIP_DECODE_TOKENIZE", None)

    def test_guard_is_noop_when_pd_flags_off(self) -> None:
        os.environ.pop("OMNI_REUSE_PREFILLED_TOKENS", None)
        request = _make_request(prompt_len=8, max_tokens=1, prefilled_token=None)
        scheduler = _FakeSchedulerForAddRequest(max_model_len=8)
        with mock.patch.object(patch_mod, "_original_add_request") as mock_add:
            patch_mod.SchedulerAddRequestGuardPatch.add_request(scheduler, request)

        mock_add.assert_called_once_with(scheduler, request)
        self.assertEqual(scheduler.finish_requests_calls, [])

    def test_pooling_request_is_admitted_at_full_context(self) -> None:
        request = SimpleNamespace(
            request_id="req-pool",
            sampling_params=None,
            max_tokens=1,
            num_tokens=8,
        )
        scheduler = _FakeSchedulerForAddRequest(max_model_len=8)
        with mock.patch.object(patch_mod, "_original_add_request") as mock_add:
            patch_mod.SchedulerAddRequestGuardPatch.add_request(scheduler, request)

        mock_add.assert_called_once_with(scheduler, request)
        self.assertEqual(scheduler.finish_requests_calls, [])

    def test_request_within_budget_is_admitted_normally(self) -> None:
        request = _make_request(prompt_len=4, max_tokens=3, prefilled_token=None)
        scheduler = _FakeSchedulerForAddRequest(max_model_len=8)
        with mock.patch.object(patch_mod, "_original_add_request") as mock_add:
            patch_mod.SchedulerAddRequestGuardPatch.add_request(scheduler, request)

        mock_add.assert_called_once_with(scheduler, request)
        self.assertEqual(scheduler.finish_requests_calls, [])

    def test_request_leaving_exact_room_is_admitted(self) -> None:
        request = _make_request(prompt_len=7, max_tokens=1, prefilled_token=None)
        scheduler = _FakeSchedulerForAddRequest(max_model_len=8)
        with mock.patch.object(patch_mod, "_original_add_request") as mock_add:
            patch_mod.SchedulerAddRequestGuardPatch.add_request(scheduler, request)

        mock_add.assert_called_once_with(scheduler, request)
        self.assertEqual(scheduler.finish_requests_calls, [])

    def test_request_at_exact_boundary_is_rejected_as_error(self) -> None:
        request = _make_request(prompt_len=8, max_tokens=1, prefilled_token=None)
        scheduler = _FakeSchedulerForAddRequest(max_model_len=8)
        with mock.patch.object(patch_mod, "_original_add_request") as mock_add:
            patch_mod.SchedulerAddRequestGuardPatch.add_request(scheduler, request)

        mock_add.assert_called_once_with(scheduler, request)
        self.assertEqual(len(scheduler.finish_requests_calls), 1)
        req_id, status = scheduler.finish_requests_calls[0]
        self.assertEqual(req_id, request.request_id)
        self.assertEqual(status, RequestStatus.FINISHED_ERROR)
        pending = scheduler._omni_pending_finish_outputs
        self.assertEqual(len(pending), 1)
        _, engine_core_output = pending[0]
        self.assertEqual(engine_core_output.request_id, request.request_id)
        self.assertIsNotNone(engine_core_output.finish_reason)

    def test_falsy_max_tokens_still_requires_one_token_of_room(self) -> None:
        request = _make_request(prompt_len=8, max_tokens=1, prefilled_token=None)
        request.max_tokens = 0
        scheduler = _FakeSchedulerForAddRequest(max_model_len=8)
        with mock.patch.object(patch_mod, "_original_add_request") as mock_add:
            patch_mod.SchedulerAddRequestGuardPatch.add_request(scheduler, request)

        mock_add.assert_called_once_with(scheduler, request)
        self.assertEqual(len(scheduler.finish_requests_calls), 1)

    def test_speculative_margin_rejects_request_inside_drafter_window(self) -> None:
        # max_model_len=20, num_speculative_tokens=2 -> margin 6, effective 14.
        # prompt(15)+max_tokens(1)=16 <= 20 would pass the plain check, but it
        # exceeds effective 14, landing in the reserved drafter window.
        request = _make_request(prompt_len=15, max_tokens=1, prefilled_token=None)
        scheduler = _FakeSchedulerForAddRequest(max_model_len=20)
        scheduler.vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=2)
        )
        with mock.patch.object(patch_mod, "_original_add_request") as mock_add:
            patch_mod.SchedulerAddRequestGuardPatch.add_request(scheduler, request)

        mock_add.assert_called_once_with(scheduler, request)
        self.assertEqual(len(scheduler.finish_requests_calls), 1)
        _, status = scheduler.finish_requests_calls[0]
        self.assertEqual(status, RequestStatus.FINISHED_ERROR)

    def test_speculative_margin_admits_request_below_drafter_window(self) -> None:
        # Same config (effective 14); prompt(12)+max_tokens(1)=13 < 14 -> admit.
        request = _make_request(prompt_len=12, max_tokens=1, prefilled_token=None)
        scheduler = _FakeSchedulerForAddRequest(max_model_len=20)
        scheduler.vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=2)
        )
        with mock.patch.object(patch_mod, "_original_add_request") as mock_add:
            patch_mod.SchedulerAddRequestGuardPatch.add_request(scheduler, request)

        mock_add.assert_called_once_with(scheduler, request)
        self.assertEqual(scheduler.finish_requests_calls, [])

    def test_spec_margin_guards_even_when_pd_flags_off(self) -> None:
        # Regression for the drafter-boundary hang: with passthrough OFF but
        # speculative decoding ON, the guard must still reject inside [M-3N, M).
        os.environ.pop("OMNI_REUSE_PREFILLED_TOKENS", None)
        os.environ.pop("OMNI_SKIP_DECODE_TOKENIZE", None)
        request = _make_request(prompt_len=15, max_tokens=1, prefilled_token=None)
        scheduler = _FakeSchedulerForAddRequest(max_model_len=20)
        scheduler.vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=2)
        )  # margin 6, effective 14; prompt(15)+1 > 14 -> reject
        with mock.patch.object(patch_mod, "_original_add_request") as mock_add:
            patch_mod.SchedulerAddRequestGuardPatch.add_request(scheduler, request)

        mock_add.assert_called_once_with(scheduler, request)
        self.assertEqual(len(scheduler.finish_requests_calls), 1)
        _, status = scheduler.finish_requests_calls[0]
        self.assertEqual(status, RequestStatus.FINISHED_ERROR)

    def test_no_spec_no_pd_is_noop(self) -> None:
        # Neither passthrough nor speculative decoding -> guard stays inert.
        os.environ.pop("OMNI_REUSE_PREFILLED_TOKENS", None)
        os.environ.pop("OMNI_SKIP_DECODE_TOKENIZE", None)
        request = _make_request(prompt_len=8, max_tokens=1, prefilled_token=None)
        scheduler = _FakeSchedulerForAddRequest(max_model_len=8)  # no vllm_config
        with mock.patch.object(patch_mod, "_original_add_request") as mock_add:
            patch_mod.SchedulerAddRequestGuardPatch.add_request(scheduler, request)

        mock_add.assert_called_once_with(scheduler, request)
        self.assertEqual(scheduler.finish_requests_calls, [])


class TestPendingFinishNotifications(unittest.TestCase):
    def test_drain_moves_outputs_to_client_bucket_and_clears(self) -> None:
        import collections

        request = _make_request(prompt_len=4, max_tokens=1, prefilled_token=None)
        scheduler = SimpleNamespace()
        patch_mod._queue_finish_notification(
            scheduler, request, RequestStatus.FINISHED_ERROR)

        outputs = collections.defaultdict(list)
        patch_mod.drain_pending_finish_outputs(scheduler, outputs)

        self.assertEqual(len(outputs[request.client_index]), 1)
        engine_core_output = outputs[request.client_index][0]
        self.assertEqual(engine_core_output.request_id, request.request_id)
        self.assertEqual(engine_core_output.new_token_ids, [])
        self.assertIsNotNone(engine_core_output.finish_reason)
        self.assertEqual(scheduler._omni_pending_finish_outputs, [])

        patch_mod.drain_pending_finish_outputs(scheduler, outputs)
        self.assertEqual(len(outputs[request.client_index]), 1)

    def test_drain_without_queue_attribute_is_noop(self) -> None:
        import collections

        outputs = collections.defaultdict(list)
        patch_mod.drain_pending_finish_outputs(SimpleNamespace(), outputs)
        self.assertEqual(len(outputs), 0)

    def test_multiple_queued_notifications_all_delivered(self) -> None:
        import collections

        request_a = _make_request(prompt_len=4, max_tokens=1, prefilled_token=None)
        request_b = _make_request(prompt_len=4, max_tokens=1, prefilled_token=None)
        request_b.request_id = "req-2"
        scheduler = SimpleNamespace()
        patch_mod._queue_finish_notification(
            scheduler, request_a, RequestStatus.FINISHED_ERROR)
        patch_mod._queue_finish_notification(
            scheduler, request_b, RequestStatus.FINISHED_LENGTH_CAPPED)

        outputs = collections.defaultdict(list)
        patch_mod.drain_pending_finish_outputs(scheduler, outputs)

        delivered = outputs[request_a.client_index]
        self.assertEqual(
            [o.request_id for o in delivered], ["req-1", "req-2"])
        self.assertEqual(scheduler._omni_pending_finish_outputs, [])


class TestGetRequestMaxTokens(unittest.TestCase):
    def test_chat_completion_request_prefers_max_completion_tokens(self) -> None:
        request = ChatCompletionRequest.model_construct(
            max_tokens=5, max_completion_tokens=2)
        self.assertEqual(patch_mod._get_request_max_tokens(request), 2)

    def test_chat_completion_request_falls_back_to_max_tokens(self) -> None:
        request = ChatCompletionRequest.model_construct(
            max_tokens=5, max_completion_tokens=None)
        self.assertEqual(patch_mod._get_request_max_tokens(request), 5)

    def test_non_chat_request_reads_max_tokens_attribute(self) -> None:
        request = SimpleNamespace(max_tokens=4)
        self.assertEqual(patch_mod._get_request_max_tokens(request), 4)

    def test_request_without_max_tokens_attribute_returns_none(self) -> None:
        self.assertIsNone(patch_mod._get_request_max_tokens(SimpleNamespace()))


class TestIsKvProducer(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.pop("ROLE", None)

    def tearDown(self) -> None:
        os.environ.pop("ROLE", None)

    def test_kv_role_producer_is_true(self) -> None:
        serving = SimpleNamespace(engine_client=SimpleNamespace(vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                is_kv_transfer_instance=True, kv_role="kv_producer"))))
        self.assertTrue(patch_mod._is_kv_producer(serving))

    def test_kv_role_consumer_is_false(self) -> None:
        serving = SimpleNamespace(engine_client=SimpleNamespace(vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                is_kv_transfer_instance=True, kv_role="kv_consumer"))))
        self.assertFalse(patch_mod._is_kv_producer(serving))

    def test_is_kv_producer_flag_true(self) -> None:
        serving = SimpleNamespace(engine_client=SimpleNamespace(vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                is_kv_transfer_instance=True, kv_role=None, is_kv_producer=True))))
        self.assertTrue(patch_mod._is_kv_producer(serving))

    def test_no_kv_transfer_config_falls_back_to_role_env(self) -> None:
        os.environ["ROLE"] = "prefill"
        self.assertTrue(patch_mod._is_kv_producer(SimpleNamespace()))

    def test_no_kv_transfer_config_and_no_role_env_is_false(self) -> None:
        self.assertFalse(patch_mod._is_kv_producer(SimpleNamespace()))

    def test_kv_transfer_config_present_but_not_instance_falls_back_to_role_env(
        self,
    ) -> None:
        os.environ["ROLE"] = "prefill"
        serving = SimpleNamespace(engine_client=SimpleNamespace(vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                is_kv_transfer_instance=False, kv_role="kv_producer"))))
        self.assertTrue(patch_mod._is_kv_producer(serving))


class TestRejectIfPromptOverflowsMaxModelLen(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("ROLE", None)
        os.environ.pop("OMNI_REUSE_PREFILLED_TOKENS", None)

    def test_serving_config_takes_priority_over_engine_client_config(self) -> None:
        serving_config = SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=2)
        )
        engine_client_config = SimpleNamespace(speculative_config=None)
        serving = SimpleNamespace(
            max_model_len=20,
            vllm_config=serving_config,
            engine_client=SimpleNamespace(vllm_config=engine_client_config),
        )

        with self.assertRaises(ValueError) as ctx:
            patch_mod._reject_if_prompt_overflows_max_model_len(
                serving, SimpleNamespace(max_tokens=1), list(range(15)))

        self.assertEqual(ctx.exception.parameter, "input_tokens")
        self.assertIs(
            patch_mod._get_serving_vllm_config(serving), serving_config)

    def test_current_config_is_used_when_engine_client_config_is_partial(self) -> None:
        current_config = SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=2)
        )
        serving = SimpleNamespace(
            max_model_len=20,
            engine_client=SimpleNamespace(
                vllm_config=SimpleNamespace(speculative_config=None)
            ),
        )

        with mock.patch(
            "vllm.config.get_current_vllm_config_or_none",
            return_value=current_config,
        ):
            with self.assertRaises(ValueError) as ctx:
                patch_mod._reject_if_prompt_overflows_max_model_len(
                    serving, SimpleNamespace(max_tokens=1), list(range(15)))

        self.assertEqual(ctx.exception.parameter, "input_tokens")

    def test_prefill_node_rejects_prompt_that_would_fit_without_overhead(self) -> None:
        os.environ["ROLE"] = "prefill"
        os.environ["OMNI_REUSE_PREFILLED_TOKENS"] = "1"
        serving = SimpleNamespace(max_model_len=8)
        request = SimpleNamespace(max_tokens=1)
        with self.assertRaises(ValueError):
            patch_mod._reject_if_prompt_overflows_max_model_len(
                serving, request, list(range(7)))

    def test_prefill_node_allows_prompt_with_enough_room(self) -> None:
        os.environ["ROLE"] = "prefill"
        os.environ["OMNI_REUSE_PREFILLED_TOKENS"] = "1"
        serving = SimpleNamespace(max_model_len=8)
        request = SimpleNamespace(max_tokens=1)
        patch_mod._reject_if_prompt_overflows_max_model_len(
            serving, request, list(range(5)))

    def test_prefill_node_without_reuse_has_no_overhead(self) -> None:
        os.environ["ROLE"] = "prefill"
        serving = SimpleNamespace(max_model_len=8)
        request = SimpleNamespace(max_tokens=1)
        patch_mod._reject_if_prompt_overflows_max_model_len(
            serving, request, list(range(7)))

    def test_empty_prompt_is_noop(self) -> None:
        serving = SimpleNamespace(max_model_len=8)
        patch_mod._reject_if_prompt_overflows_max_model_len(
            serving, SimpleNamespace(max_tokens=1), [])

    def test_prompt_already_at_max_model_len_raises(self) -> None:
        serving = SimpleNamespace(max_model_len=8)
        request = SimpleNamespace(max_tokens=1)
        with self.assertRaises(ValueError) as ctx:
            patch_mod._reject_if_prompt_overflows_max_model_len(
                serving, request, list(range(8)))
        self.assertEqual(ctx.exception.parameter, "input_tokens")
        self.assertEqual(ctx.exception.value, 8)

    def test_prompt_plus_max_tokens_overflow_raises(self) -> None:
        serving = SimpleNamespace(max_model_len=8)
        request = SimpleNamespace(max_tokens=3)
        with self.assertRaises(ValueError) as ctx:
            patch_mod._reject_if_prompt_overflows_max_model_len(
                serving, request, list(range(6)))  # 6 + 3 = 9 > 8
        self.assertEqual(ctx.exception.parameter, "max_tokens")
        self.assertEqual(ctx.exception.value, 3)

    def test_prefilled_token_is_counted_in_decode_admission(self) -> None:
        serving = SimpleNamespace(max_model_len=8)
        request = SimpleNamespace(max_tokens=1)
        with self.assertRaises(ValueError) as ctx:
            patch_mod._reject_if_prompt_overflows_max_model_len(
                serving,
                request,
                list(range(7)),
                prefilled_token_ids=[42],
            )

        self.assertEqual(ctx.exception.parameter, "input_tokens")
        self.assertEqual(ctx.exception.value, 8)

    def test_prompt_within_budget_is_noop(self) -> None:
        serving = SimpleNamespace(max_model_len=8)
        request = SimpleNamespace(max_tokens=1)
        patch_mod._reject_if_prompt_overflows_max_model_len(
            serving, request, list(range(7)))

    def test_missing_max_model_len_is_noop(self) -> None:
        serving = SimpleNamespace()
        patch_mod._reject_if_prompt_overflows_max_model_len(
            serving, SimpleNamespace(max_tokens=1), list(range(100)))

    def test_request_without_max_tokens_only_checks_bare_length(self) -> None:
        serving = SimpleNamespace(max_model_len=8)
        request = SimpleNamespace()
        patch_mod._reject_if_prompt_overflows_max_model_len(
            serving, request, list(range(7)))


def _run_coro(coro):
    return asyncio.run(coro)


class TestPreprocessChatMaxModelLenGuard(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.pop("OMNI_REUSE_PREFILLED_TOKENS", None)

    def tearDown(self) -> None:
        os.environ.pop("OMNI_REUSE_PREFILLED_TOKENS", None)

    @staticmethod
    def _fake_original(engine_prompt):
        async def _original(*args, **kwargs):
            return [], [engine_prompt]
        return _original

    @staticmethod
    def _fake_tokenizer(eos_token_id=999):
        return SimpleNamespace(
            convert_ids_to_tokens=lambda tid: "tok",
            convert_tokens_to_string=lambda toks: "tok",
            eos_token_id=eos_token_id,
        )

    @staticmethod
    def _reuse_kv_params(prefilled_token):
        return {
            "prefilled_token": list(prefilled_token),
            "stop_reasons": [None],
            "prefilled_logprobs": None,
            "prefilled_cumulative_logprob": None,
        }

    def test_chat_preprocess_attaches_prefilled_token(self) -> None:
        os.environ["OMNI_REUSE_PREFILLED_TOKENS"] = "1"
        serving = SimpleNamespace(max_model_len=8)
        kv_params = self._reuse_kv_params([42])
        request = SimpleNamespace(kv_transfer_params=kv_params, max_tokens=1)
        engine_prompt = {"prompt_token_ids": list(range(5))}
        with mock.patch.object(
            patch_mod, "_original_preprocess_chat", self._fake_original(engine_prompt)
        ):
            _conversation, [result] = _run_coro(
                patch_mod.OpenAIServingChatPreprocessPatch._preprocess_chat(
                    serving, request, self._fake_tokenizer(), [], None, "auto"))

        self.assertEqual(result["prefilled_token_ids"], [42])
        self.assertIn("prefilled_token", kv_params)

    def test_chat_preprocess_rejects_overlong_prompt(self) -> None:
        serving = SimpleNamespace(max_model_len=8)
        request = SimpleNamespace(kv_transfer_params=None, max_tokens=1)
        engine_prompt = {"prompt_token_ids": list(range(8))}
        with mock.patch.object(
            patch_mod, "_original_preprocess_chat", self._fake_original(engine_prompt)
        ):
            with self.assertRaises(ValueError):
                _run_coro(
                    patch_mod.OpenAIServingChatPreprocessPatch._preprocess_chat(
                        serving, request, None, [], None, "auto"))

    def test_chat_preprocess_allows_prompt_within_budget(self) -> None:
        serving = SimpleNamespace(max_model_len=8)
        request = SimpleNamespace(kv_transfer_params=None, max_tokens=1)
        engine_prompt = {"prompt_token_ids": list(range(7))}
        with mock.patch.object(
            patch_mod, "_original_preprocess_chat", self._fake_original(engine_prompt)
        ):
            _conversation, [result] = _run_coro(
                patch_mod.OpenAIServingChatPreprocessPatch._preprocess_chat(
                    serving, request, None, [], None, "auto"))
        self.assertEqual(result["prompt_token_ids"], list(range(7)))

    def test_serving_preprocess_rejects_overlong_kv_transfer_prompt(self) -> None:
        serving = SimpleNamespace(max_model_len=8)
        request = SimpleNamespace(
            kv_transfer_params={"prompt_token_ids": list(range(8))},
            max_tokens=1,
        )
        with mock.patch.object(
            patch_mod, "_original_preprocess_chat",
            self._fake_original({"prompt_token_ids": []}),
        ):
            with self.assertRaises(ValueError):
                _run_coro(
                    patch_mod.OpenAIServingPatch._preprocess_chat(
                        serving, request, None, [], None, "auto"))

    def test_serving_preprocess_allows_kv_transfer_prompt_within_budget(self) -> None:
        serving = SimpleNamespace(max_model_len=8)
        request = SimpleNamespace(
            kv_transfer_params={"prompt_token_ids": list(range(7))},
            max_tokens=1,
        )
        with mock.patch.object(
            patch_mod, "_original_preprocess_chat",
            self._fake_original({"prompt_token_ids": []}),
        ):
            _conversation, [result] = _run_coro(
                patch_mod.OpenAIServingPatch._preprocess_chat(
                    serving, request, None, [], None, "auto"))
        self.assertEqual(result["prompt_token_ids"], list(range(7)))


class TestEnforceSpeculativeGenerationBudget(unittest.TestCase):
    # max_model_len=20, num_spec=3 -> margin=9, effective=11.
    def _cfg(self, num_spec=3):
        return SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=20),
            speculative_config=SimpleNamespace(num_speculative_tokens=num_spec),
        )

    def _req(self, prompt_len, max_tokens):
        return SimpleNamespace(
            request_id="r",
            prompt_token_ids=list(range(prompt_len)),
            sampling_params=SimpleNamespace(max_tokens=max_tokens),
        )

    def test_over_budget_max_tokens_is_capped(self) -> None:
        # Crash case: serving fills max_tokens to 16 (> effective 11); cap to 7.
        req = self._req(prompt_len=4, max_tokens=16)
        patch_mod.enforce_speculative_generation_budget(self._cfg(), req)
        self.assertEqual(req.sampling_params.max_tokens, 7)  # effective 11 - 4

    def test_within_budget_is_noop(self) -> None:
        req = self._req(prompt_len=4, max_tokens=5)  # 9 <= 11
        patch_mod.enforce_speculative_generation_budget(self._cfg(), req)
        self.assertEqual(req.sampling_params.max_tokens, 5)

    def test_no_room_raises(self) -> None:
        req = self._req(prompt_len=13, max_tokens=7)  # prompt already > effective 11
        with self.assertRaises(ValueError) as ctx:
            patch_mod.enforce_speculative_generation_budget(self._cfg(), req)
        self.assertEqual(ctx.exception.parameter, "input_tokens")

    def test_no_speculation_is_noop(self) -> None:
        req = self._req(prompt_len=4, max_tokens=16)
        patch_mod.enforce_speculative_generation_budget(self._cfg(num_spec=0), req)
        self.assertEqual(req.sampling_params.max_tokens, 16)

    def test_none_max_tokens_is_noop(self) -> None:
        req = self._req(prompt_len=4, max_tokens=None)
        patch_mod.enforce_speculative_generation_budget(self._cfg(), req)
        self.assertIsNone(req.sampling_params.max_tokens)


class _FakeAttentionSpec:
    pass


class _FakeKVCacheManagerForLoadRecovery:
    def __init__(self, block_id_groups: dict[str, tuple[list[int], ...]]) -> None:
        self.block_id_groups = block_id_groups
        self.evicted_blocks: list[set[int]] = []

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        return self.block_id_groups[request_id]

    def evict_blocks(self, block_ids: set[int]) -> None:
        self.evicted_blocks.append(block_ids)


class _FakeSchedulerForLoadRecovery:
    def __init__(
        self,
        requests: list[SimpleNamespace],
        block_id_groups: dict[str, tuple[list[int], ...]],
        *,
        recompute_kv_load_failures: bool,
    ) -> None:
        self.waiting = requests
        self.running: list[SimpleNamespace] = []
        self.block_size = 4
        self.kv_cache_manager = _FakeKVCacheManagerForLoadRecovery(block_id_groups)
        self.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=object()),
                SimpleNamespace(kv_cache_spec=_FakeAttentionSpec()),
            ]
        )
        self.failed_recving_kv_req_ids: set[str] = set()
        self.recompute_kv_load_failures = recompute_kv_load_failures

    def _update_requests_with_invalid_blocks(
        self, requests, invalid_block_ids, evict_blocks=True
    ):
        return patch_mod.HybridKVLoadFailureSchedulerPatch._update_requests_with_invalid_blocks(
            self, requests, invalid_block_ids, evict_blocks
        )


def _make_load_recovery_request(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        status=RequestStatus.WAITING_FOR_REMOTE_KVS,
        num_computed_tokens=0,
        num_external_computed_tokens=12,
        num_cached_tokens=0,
    )


class TestHybridKVLoadFailureRecovery(unittest.TestCase):
    def _scheduler(self, *, recompute: bool) -> _FakeSchedulerForLoadRecovery:
        request = _make_load_recovery_request("req-1")
        return _FakeSchedulerForLoadRecovery(
            [request],
            {"req-1": ([100, 101, 102], [200, 201, 202])},
            recompute_kv_load_failures=recompute,
        )

    def test_selects_attention_group_for_hybrid_invalid_blocks(self) -> None:
        scheduler = self._scheduler(recompute=True)

        with mock.patch.object(patch_mod, "AttentionSpec", _FakeAttentionSpec):
            affected, affected_tokens, blocks_to_evict = (
                patch_mod.HybridKVLoadFailureSchedulerPatch
                ._update_requests_with_invalid_blocks(
                    scheduler, scheduler.waiting, {201}, evict_blocks=False
                )
            )

        request = scheduler.waiting[0]
        self.assertEqual(affected, {"req-1"})
        self.assertEqual(affected_tokens, 8)
        self.assertEqual(blocks_to_evict, set())
        self.assertEqual(request.num_computed_tokens, 4)
        self.assertEqual(request.num_external_computed_tokens, 4)

    def test_single_group_preserves_upstream_behavior(self) -> None:
        request = _make_load_recovery_request("req-1")
        scheduler = _FakeSchedulerForLoadRecovery(
            [request], {"req-1": ([100, 101, 102],)}, recompute_kv_load_failures=True
        )

        affected, affected_tokens, blocks_to_evict = (
            patch_mod.HybridKVLoadFailureSchedulerPatch
            ._update_requests_with_invalid_blocks(
                scheduler, scheduler.waiting, {101}, evict_blocks=True
            )
        )

        self.assertEqual(affected, {"req-1"})
        self.assertEqual(affected_tokens, 8)
        self.assertEqual(blocks_to_evict, {101, 102})
        self.assertEqual(request.num_computed_tokens, 4)

    def test_recompute_policy_marks_async_request_for_retry(self) -> None:
        scheduler = self._scheduler(recompute=True)

        with mock.patch.object(patch_mod, "AttentionSpec", _FakeAttentionSpec):
            failed_req_ids = Scheduler._handle_invalid_blocks(scheduler, {201})

        self.assertEqual(failed_req_ids, set())
        self.assertEqual(scheduler.failed_recving_kv_req_ids, {"req-1"})
        self.assertEqual(scheduler.waiting[0].num_computed_tokens, 4)

    def test_fail_policy_returns_only_the_affected_request(self) -> None:
        scheduler = self._scheduler(recompute=False)

        with mock.patch.object(patch_mod, "AttentionSpec", _FakeAttentionSpec):
            failed_req_ids = Scheduler._handle_invalid_blocks(scheduler, {201})

        self.assertEqual(failed_req_ids, {"req-1"})
        self.assertEqual(scheduler.failed_recving_kv_req_ids, set())


if __name__ == "__main__":
    unittest.main()
