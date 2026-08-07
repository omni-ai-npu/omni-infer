# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Fix: https://github.com/vllm-project/vllm/pull/37452
#
# Eliminates the TOCTOU port race condition in DPCoordinator ZMQ socket binding.
# The coordinator was calling get_open_port() to pre-select a port, then passing
# it as a string to the child process, which would bind to it later. Between port
# selection and binding, another process could take the port.
#
# Fix: bind with port=0 (OS atomically assigns) + zmq.LAST_ENDPOINT to read back
# the actual address, communicated back to the parent via multiprocessing.Pipe.

import copy
import multiprocessing
import multiprocessing.connection
import time
import weakref

import msgspec.msgpack
import zmq

import vllm.utils.network_utils as network_utils
import vllm.v1.engine.coordinator as coordinator
from vllm.logger import init_logger
from vllm.utils.network_utils import get_tcp_uri, make_zmq_socket
from vllm.utils.system_utils import get_mp_context
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)


def _patch_enabled_in_env(patch_name: str) -> bool:
    patches = envs.OMNI_NPU_VLLM_PATCHES.strip()
    if not patches or patches == "ALL":
        return True
    return patch_name in {item.strip() for item in patches.split(",") if item.strip()}


def _ensure_split_zmq_path_port_fix() -> None:
    if not _patch_enabled_in_env("SplitZMQPathPortFix"):
        return
    applied = getattr(network_utils, "_omni_npu_applied_patches", {})
    applied_by = applied.get("split_zmq_path")
    if applied_by == "SplitZMQPathPortFix":
        return
    if applied_by is not None:
        raise RuntimeError(
            "vllm.utils.network_utils.split_zmq_path already patched by "
            f"{applied_by}, cannot apply SplitZMQPathPortFix"
        )
    SplitZMQPathPortFix.apply()


@register_patch("SplitZMQPathPortFix", network_utils)
class SplitZMQPathPortFix(VLLMPatch):
    _attr_names_to_apply = ["split_zmq_path"]

    @staticmethod
    def split_zmq_path(path: str) -> tuple[str, str, str]:
        from urllib.parse import urlparse

        parsed = urlparse(path)
        if not parsed.scheme:
            raise ValueError(f"Invalid zmq path: {path}")

        scheme = parsed.scheme
        host = parsed.hostname or ""
        # Fix: use explicit None-check instead of "or" so port=0 is preserved
        port = "" if parsed.port is None else str(parsed.port)

        if scheme == "tcp" and not all((host, port)):
            raise ValueError(f"Invalid zmq path: {path}")

        if scheme != "tcp" and port:
            raise ValueError(f"Invalid zmq path: {path}")

        return scheme, host, port


@register_patch("DPCoordinatorPortRaceFix", coordinator.DPCoordinator)
class DPCoordinatorPortRaceFix(VLLMPatch):
    _attr_names_to_apply = ["_wait_for_zmq_addrs", "__init__"]

    def _wait_for_zmq_addrs(self, zmq_addr_pipe):
        try:
            ready = multiprocessing.connection.wait(
                [zmq_addr_pipe, self.proc.sentinel], timeout=30
            )
            if not ready:
                raise RuntimeError(
                    "DP Coordinator process failed to report ZMQ addresses "
                    "during startup."
                )
            try:
                return zmq_addr_pipe.recv()
            except EOFError:
                raise RuntimeError(
                    "DP Coordinator process failed during startup."
                ) from None
        finally:
            zmq_addr_pipe.close()

    def __init__(  # noqa: F811
        self, parallel_config, enable_wave_coordination: bool = True
    ):
        dp_size = parallel_config.data_parallel_size
        assert dp_size > 1, "Coordinator only used for data parallel"

        host = parallel_config.data_parallel_master_ip
        external_lb = parallel_config.data_parallel_external_lb
        hybrid_lb = parallel_config.data_parallel_hybrid_lb

        local_only = not (external_lb or hybrid_lb)
        local_only_eng = dp_size == parallel_config.data_parallel_size_local

        # Core fix: use port=0 for TCP so the OS assigns atomically at bind time
        def bind_address(local: bool) -> str:
            return (
                get_engine_client_zmq_addr(local_only=True, host=host)
                if local
                else get_tcp_uri(host, 0)
            )

        front_publish_address = bind_address(local_only)
        back_publish_address = bind_address(local_only_eng)
        back_output_address = bind_address(local_only_eng)

        context = get_mp_context()
        parent_zmq_addr_pipe, child_zmq_addr_pipe = context.Pipe(duplex=False)
        self.proc: multiprocessing.Process = context.Process(
            target=coordinator.DPCoordinatorProc.run_coordinator,
            name="VLLM_DP_Coordinator",
            kwargs={
                "engine_count": parallel_config.data_parallel_size,
                "front_publish_address": front_publish_address,
                "back_output_address": back_output_address,
                "back_publish_address": back_publish_address,
                "zmq_addr_pipe": child_zmq_addr_pipe,
                "enable_wave_coordination": enable_wave_coordination,
            },
            daemon=True,
        )
        self.proc.start()
        child_zmq_addr_pipe.close()

        # Block until child reports the kernel-assigned addresses
        (
            front_publish_address,
            back_output_address,
            back_publish_address,
        ) = self._wait_for_zmq_addrs(parent_zmq_addr_pipe)

        self.stats_publish_address = front_publish_address
        self.coord_in_address = back_publish_address
        self.coord_out_address = back_output_address
        self._finalizer = weakref.finalize(self, shutdown, [self.proc])


@register_patch("DPCoordinatorProcPortRaceFix", coordinator.DPCoordinatorProc)
class DPCoordinatorProcPortRaceFix(VLLMPatch):
    _attr_names_to_apply = ["run_coordinator", "process_input_socket"]

    @staticmethod
    def run_coordinator(
        engine_count: int,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
        min_stats_update_interval_ms: int = 100,
        enable_wave_coordination: bool = True,
    ):
        from vllm.utils.system_utils import set_process_title

        _ensure_split_zmq_path_port_fix()

        coordinator_proc = coordinator.DPCoordinatorProc(
            engine_count=engine_count,
            min_stats_update_interval_ms=min_stats_update_interval_ms,
            enable_wave_coordination=enable_wave_coordination,
        )
        try:
            DPCoordinatorProcPortRaceFix.process_input_socket(
                coordinator_proc,
                front_publish_address,
                back_output_address,
                back_publish_address,
                zmq_addr_pipe,
            )
        except KeyboardInterrupt:
            logger.info("DP Coordinator process exiting")
        finally:
            if zmq_addr_pipe is not None:
                zmq_addr_pipe.close()

    def process_input_socket(
        self,
        front_publish_address: str,
        back_output_address: str,
        back_publish_address: str,
        zmq_addr_pipe=None,
    ):
        decoder = MsgpackDecoder(EngineCoreOutputs)

        # For tracking request wave progression.
        current_wave = 0
        engines_running = False

        # For tracking request counts for internal load-balancing.
        stats_changed = False
        last_stats_step = -1
        last_stats_wave = -1
        last_step_counts: list[list[int]] | None = None

        with (
            make_zmq_socket(
                path=front_publish_address,
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_front,
            make_zmq_socket(
                path=back_output_address,
                ctx=self.ctx,
                socket_type=zmq.PULL,
                bind=True,
            ) as output_back,
            make_zmq_socket(
                path=back_publish_address,
                ctx=self.ctx,
                socket_type=zmq.XPUB,
                bind=True,
            ) as publish_back,
        ):
            # PR #37452: report kernel-assigned addresses back to parent
            if zmq_addr_pipe is not None:
                try:
                    zmq_addr_pipe.send(
                        (
                            publish_front.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            output_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                            publish_back.getsockopt(zmq.LAST_ENDPOINT).decode(),
                        )
                    )
                finally:
                    zmq_addr_pipe.close()

            # Wait until all engines subscribe.
            for _ in self.engines:
                if publish_back.recv() != b"\x01":
                    logger.error(
                        "DP Coordinator received unexpected message while "
                        "waiting for engines to subscribe"
                    )
                    return
            # Send ready message to engines.
            publish_back.send(b"READY")

            logger.info("All engine subscriptions received by DP coordinator")

            poller = zmq.Poller()
            poller.register(publish_front, zmq.POLLIN)
            poller.register(output_back, zmq.POLLIN)
            last_publish_time = 0
            while True:
                elapsed = int(time.time() * 1000) - last_publish_time
                # Send at stats_update_interval_ms interval if the stats have
                # changed, or otherwise every 5 seconds.
                wait_for = self.stats_update_interval_ms if stats_changed else 5000

                # Wait at least 50ms to ensure we've received all stats for
                # the current step.
                min_timeout = 50 if last_step_counts is None else 0

                events = poller.poll(timeout=max(min_timeout, wait_for - elapsed))
                if not events:
                    # Poller timeout - publish current stats to front-ends.
                    if last_step_counts is not None:
                        engine_req_counts_list = last_step_counts
                        last_step_counts = None
                    else:
                        engine_req_counts_list = self._get_engine_counts()
                        stats_changed = False

                    to_publish = (
                        engine_req_counts_list,
                        current_wave,
                        engines_running,
                    )
                    publish_front.send(msgspec.msgpack.encode(to_publish))
                    last_publish_time = int(time.time() * 1000)
                    continue

                events = dict(events)
                wave_state_changed = False

                if publish_front in events:
                    buffer = publish_front.recv()
                    if buffer in (b"\x01", b"\x00"):
                        # Ignore subscription messages.
                        continue

                    decoded = msgspec.msgpack.decode(buffer)
                    if (
                        isinstance(decoded, (list, tuple))
                        and len(decoded) == 2
                        and decoded[0] == "SCALE_ELASTIC_EP"
                    ):
                        # Handle scale up notification
                        new_engine_count = decoded[1]
                        current_count = len(self.engines)
                        if new_engine_count > current_count:
                            for _ in range(new_engine_count - current_count):
                                self.engines.append(coordinator.EngineState())
                            engines_running = False
                            logger.info(
                                "DPCoordinator scaled up from %s to %s engines",
                                current_count,
                                new_engine_count,
                            )
                        else:
                            self.engines = self.engines[:new_engine_count]
                            logger.info(
                                "DPCoordinator scaled down from %s to %s engines",
                                current_count,
                                new_engine_count,
                            )
                        continue

                    # Wave coordination: handle new-request messages from
                    # front-end.
                    if self.enable_wave_coordination:
                        engine_to_exclude, wave = decoded
                        if not engines_running:
                            if wave < current_wave:
                                engine_to_exclude = None

                            engines_running = True
                            wave_state_changed = True
                            self._send_start_wave(
                                publish_back, current_wave, engine_to_exclude
                            )

                if output_back in events:
                    # We received a message from one of the engines.
                    buffer = output_back.recv()
                    outputs: EngineCoreOutputs = decoder.decode(buffer)

                    assert not outputs.outputs, "decode-only branch should not emit token outputs."
                    assert outputs.utility_output is None, "decode-only branch expects utility_output to be None."

                    eng_index = outputs.engine_index
                    scheduler_stats = outputs.scheduler_stats
                    if scheduler_stats:
                        stats = self.engines[eng_index].request_counts
                        stats_step = scheduler_stats.step_counter
                        stats_wave = scheduler_stats.current_wave
                        if (
                            stats_wave > last_stats_wave
                            or stats_wave == last_stats_wave
                            and stats_step > last_stats_step
                        ):
                            if stats_changed:
                                last_step_counts = self._get_engine_counts(do_copy=True)
                            last_stats_step = stats_step
                            last_stats_wave = stats_wave
                        elif stats_wave != last_stats_wave or (
                            stats_step != last_stats_step
                        ):
                            logger.warning(
                                "Received stats for out-of-order "
                                "step (%d, %d) from engine %d (expected "
                                "> (%d, %d))",
                                stats_wave,
                                stats_step,
                                eng_index,
                                last_stats_wave,
                                last_stats_step,
                            )
                        stats[0] = scheduler_stats.num_waiting_reqs
                        stats[1] = scheduler_stats.num_running_reqs
                        stats_changed = True

                    # Wave coordination: handle wave completion and start
                    # notifications.
                    if self.enable_wave_coordination:
                        if (wave := outputs.wave_complete) is not None:
                            if current_wave <= wave:
                                new_wave = wave + 1
                                logger.debug(
                                    "Moving DP wave from %d to %d.",
                                    current_wave,
                                    new_wave,
                                )
                                current_wave = new_wave
                                engines_running = False
                                wave_state_changed = True
                        elif (wave := outputs.start_wave) is not None and (
                            wave > current_wave
                            or (wave == current_wave and not engines_running)
                        ):
                            logger.debug(
                                "Starting wave %d after notification of "
                                "stale wave request from engine.",
                                wave,
                            )
                            current_wave = wave
                            engines_running = True
                            wave_state_changed = True
                            self._send_start_wave(publish_back, wave, eng_index)

                if wave_state_changed:
                    message = (None, current_wave, engines_running)
                    publish_front.send(msgspec.msgpack.encode(message))
