# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-with-batch-queue DP scheduling patch.

This patch keeps ``DPEngineCoreProc.step_with_batch_queue`` as the central
scheduling path: schedule/submit stays non-blocking, model outputs are drained in
batch-queue order, and 0-token control batches go through the same drain path
instead of returning early to the outer loop.

When a queued step schedules no real tokens, the idle rank submits its dummy
batch asynchronously before draining the queued output. The dummy future is
waited only after ``Scheduler.update_from_output()``, and before the global
unfinished-request sync, so dummy execution can overlap host-side output drain
without crossing into the next engine step.
"""

import time
from concurrent.futures import Future
from typing import Any, cast

from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import DPEngineCoreProc
from vllm.v1.outputs import ModelRunnerOutput

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


def _start_dummy_batch_async(engine: DPEngineCoreProc) -> Any | None:
    if engine.model_executor.is_sleeping:
        return None

    executor = engine.model_executor
    kwargs: dict[str, Any] = {"non_block": True}
    output_rank = getattr(executor, "output_rank", None)
    if output_rank is not None:
        kwargs["unique_reply_rank"] = output_rank
    return executor.collective_rpc("execute_dummy_batch", **kwargs)


def _wait_future(future: Any | None) -> None:
    if future is not None and hasattr(future, "result"):
        future.result()


def _mark_dummy_submitted(engine: DPEngineCoreProc, future: Any | None) -> None:
    engine._omni_dummy_submitted_in_step = future is not None


def _clear_dummy_submitted(engine: DPEngineCoreProc) -> bool:
    submitted = bool(getattr(engine, "_omni_dummy_submitted_in_step", False))
    engine._omni_dummy_submitted_in_step = False
    return submitted


def _yield_after_dummy_submit(
    engine: DPEngineCoreProc, dummy_future: Any | None
) -> None:
    engine._omni_no_execute_yield_handled = False
    if dummy_future is not None and engine.scheduler.has_requests():
        # Let KV-transfer/background threads run while the worker/device is
        # executing dummy, instead of adding another 1 ms after the whole step.
        time.sleep(0.001)
        engine._omni_no_execute_yield_handled = True


@register_patch("StepWithBatchQueueAsyncDummyDPEnginePatch", DPEngineCoreProc)
class StepWithBatchQueueAsyncDummyDPEnginePatch(VLLMPatch):
    _attr_names_to_apply = [
        "_process_engine_step",
        "step_with_batch_queue",
    ]

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        self._omni_no_execute_yield_handled = False
        batch_queue = self.batch_queue
        if batch_queue is None:
            raise RuntimeError("batch_queue is not initialized")
        if len(batch_queue) >= self.batch_queue_size:
            raise RuntimeError("batch_queue is full before scheduling")

        model_executed = False
        deferred_scheduler_output = None
        exec_future = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(
                    scheduler_output, non_block=True
                )
            if self.is_ec_consumer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                if not scheduler_output.pending_structured_output_tokens:
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                batch_queue.appendleft((future, scheduler_output, exec_future))
                if model_executed and len(batch_queue) < self.batch_queue_size and not batch_queue[-1][0].done():
                    _mark_dummy_submitted(self, None)
                    return None, model_executed

        elif not batch_queue:
            _mark_dummy_submitted(self, None)
            return None, False

        dummy_future = None
        if not model_executed:
            with self.log_iteration_details(None):
                dummy_future = _start_dummy_batch_async(self)
            _yield_after_dummy_submit(self, dummy_future)

        future, scheduler_output, exec_model_fut = batch_queue.pop()
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                exec_model_fut.result()
                raise RuntimeError("unexpected error")

        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        if deferred_scheduler_output:
            if self.check_for_draft_tokens:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                if draft_token_ids is not None:
                    self.scheduler.update_draft_token_ids_in_output(
                        draft_token_ids, deferred_scheduler_output
                    )
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))

        _wait_future(dummy_future)
        _mark_dummy_submitted(self, dummy_future)

        return engine_core_outputs, model_executed

    def _process_engine_step(self) -> bool:
        outputs, model_executed = self.step_fn()
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        self.post_step(model_executed)

        dummy_submitted = _clear_dummy_submitted(self)
        yield_handled = bool(
            getattr(self, "_omni_no_execute_yield_handled", False)
        )
        self._omni_no_execute_yield_handled = False
        if (
            not yield_handled
            and not model_executed
            and self.scheduler.has_requests()
        ):
            time.sleep(0.001)

        return model_executed or dummy_submitted
