import os

def patch_dp_engine_core_proc():
    if os.getenv("OMNI_PD_HYBRID", "0") == "0":
        return

    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.engine.core import DPEngineCoreProc

    def custom_run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""
        from vllm.logger import logger

        while True:
            self._process_input_queue()

            phase = self._get_global_phase_hint()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()

            if phase == "prefill":
                has_prefill_local = self.scheduler.has_prefill_requests()
                if has_prefill_local:
                    self._process_engine_step()
                else:
                    self.execute_dummy_batch("prefill")
            else:
                if local_unfinished_reqs:
                    self._process_engine_step()
                else:
                    if self.scheduler.has_finished_requests():
                        self._process_engine_step()
                    else:
                        self.execute_dummy_batch("decode")

            local_unfinished_reqs = self.scheduler.has_unfinished_requests() or self.scheduler.has_finished_requests()

            if not self.enable_sleep_mode:
                # disable all-reduce operation, a workaround for manual api-server scale-out
                continue

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs)

            if not self.engines_running:
                if self.dp_rank == 0:
                    # Notify client that we are pausing the loop.
                    logger.debug("Wave %d finished, pausing engine loop.",
                                 self.current_wave)
                    self.output_queue.put_nowait(
                        EngineCoreOutputs(wave_complete=self.current_wave))
                self.current_wave += 1
    DPEngineCoreProc.run_busy_loop = custom_run_busy_loop

patch_dp_engine_core_proc()