# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from omni_npu.v1.utils import switch_torch_device
from omni_npu.profiler.wrapper import _to_bool
from omni_npu.worker.npu_worker import (
    NPUWorker,
    NPUMemorySnapshot,
    _patch_npu_triton_capabilities,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import ModelRunnerOutput
from tests.unit.platform.utils import create_vllm_config, DeviceConfig


@pytest.mark.parametrize(
    ("has_triton_package", "available_device", "expected"),
    [
        (False, "npu", False),
        (True, None, False),
        (True, "cuda", True),
        (True, "npu", True),
    ],
)
def test_patch_npu_has_triton_updates_dynamo_alias(
    monkeypatch, has_triton_package, available_device, expected
):
    from torch._dynamo import device_interface
    from torch._dynamo import utils as dynamo_utils
    from torch.utils import _triton as torch_triton

    old_has_triton = MagicMock(
        side_effect=TypeError("NoneType cannot be compared with int")
    )
    stale_dynamo_has_triton = MagicMock(
        side_effect=TypeError("NoneType cannot be compared with int")
    )
    old_device_supports_tma = MagicMock(
        side_effect=TypeError("NoneType cannot be compared with tuple")
    )

    monkeypatch.setattr(torch_triton, "has_triton", old_has_triton)
    monkeypatch.setattr(dynamo_utils, "has_triton", stale_dynamo_has_triton)
    monkeypatch.setattr(
        torch_triton, "_device_supports_tma", old_device_supports_tma
    )
    monkeypatch.setattr(
        torch_triton, "has_triton_package", lambda: has_triton_package
    )

    get_interface_for_device = MagicMock(
        side_effect=lambda device: SimpleNamespace(
            is_available=lambda: device == available_device
        )
    )
    monkeypatch.setattr(
        device_interface,
        "get_interface_for_device",
        get_interface_for_device,
    )

    _patch_npu_triton_capabilities()

    assert torch_triton.has_triton() is expected
    assert dynamo_utils.has_triton is torch_triton.has_triton
    old_has_triton.assert_not_called()
    stale_dynamo_has_triton.assert_not_called()
    if not has_triton_package:
        get_interface_for_device.assert_not_called()


@pytest.mark.parametrize(
    ("npu_available", "hip", "expected"),
    [
        (True, None, True),
        (False, None, False),
        (True, "hip", False),
    ],
)
def test_patch_npu_device_supports_tma(
    monkeypatch, npu_available, hip, expected
):
    from torch._dynamo import utils as dynamo_utils
    from torch.utils import _triton as torch_triton

    monkeypatch.setattr(
        torch_triton, "has_triton", MagicMock(return_value=True)
    )
    monkeypatch.setattr(
        dynamo_utils, "has_triton", MagicMock(return_value=True)
    )
    old_device_supports_tma = MagicMock(
        side_effect=TypeError("NoneType cannot be compared with tuple")
    )
    monkeypatch.setattr(
        torch_triton, "_device_supports_tma", old_device_supports_tma
    )
    monkeypatch.setattr("torch.npu.is_available", lambda: npu_available)
    monkeypatch.setattr(torch.version, "hip", hip)

    _patch_npu_triton_capabilities()

    assert torch_triton._device_supports_tma() is expected
    old_device_supports_tma.assert_not_called()


def test_patch_npu_tma_probe_avoids_cuda_device_capability(monkeypatch):
    from torch._dynamo import utils as dynamo_utils
    from torch.utils import _triton as torch_triton

    host_tma_probes = (
        torch_triton.has_triton_experimental_host_tma,
        torch_triton.has_triton_tensor_descriptor_host_tma,
    )
    monkeypatch.setattr(
        torch_triton, "has_triton", MagicMock(return_value=True)
    )
    monkeypatch.setattr(
        dynamo_utils, "has_triton", MagicMock(return_value=True)
    )
    monkeypatch.setattr(
        torch_triton,
        "_device_supports_tma",
        MagicMock(side_effect=TypeError("NoneType cannot be compared with tuple")),
    )
    monkeypatch.setattr(torch_triton, "has_triton_package", lambda: True)
    monkeypatch.setattr("torch.npu.is_available", lambda: False)
    monkeypatch.setattr(torch.version, "hip", None)

    _patch_npu_triton_capabilities()

    for host_tma_probe in host_tma_probes:
        assert host_tma_probe.__wrapped__() is False


def test_init_device_patches_triton_capabilities_after_setting_device(monkeypatch):
    class DtypeCheckReached(Exception):
        pass

    events = []

    def logical_to_visible(local_rank):
        events.append(("resolve_device", local_rank))
        return 5

    def make_device(spec):
        events.append(("make_device", spec))
        return spec

    def set_device(device):
        events.append(("set_device", device))

    def patch_triton_capabilities():
        events.append(("patch_triton_capabilities", None))

    def check_dtype(dtype):
        events.append(("check_dtype", dtype))
        raise DtypeCheckReached

    worker = SimpleNamespace(
        device_config=SimpleNamespace(device=SimpleNamespace(type="npu")),
        parallel_config=SimpleNamespace(distributed_executor_backend="ray"),
        local_rank=3,
        model_config=SimpleNamespace(dtype="float16"),
    )
    platform = SimpleNamespace(
        device_type="npu",
        logical_device_id_to_visible_device_id=logical_to_visible,
        check_if_supports_dtype=check_dtype,
    )
    monkeypatch.setattr("omni_npu.worker.npu_worker.current_platform", platform)
    monkeypatch.setattr("omni_npu.worker.npu_worker.torch.device", make_device)
    monkeypatch.setattr("omni_npu.worker.npu_worker.torch.npu.set_device", set_device)
    monkeypatch.setattr(
        "omni_npu.worker.npu_worker._patch_npu_triton_capabilities",
        patch_triton_capabilities,
    )

    init_device = getattr(NPUWorker.init_device, "__wrapped__", NPUWorker.init_device)
    with pytest.raises(DtypeCheckReached):
        init_device(worker)

    assert events == [
        ("resolve_device", 3),
        ("make_device", "npu:5"),
        ("set_device", "npu:5"),
        ("patch_triton_capabilities", None),
        ("check_dtype", "float16"),
    ]


class TestNpuWorker:

    def setup_method(self):
        self.vllm_cfg = create_vllm_config()
        self.vllm_cfg.device_config = DeviceConfig("npu")
        self.worker = None

    def _create_worker(self, monkeypatch):
        """Create an NPUWorker instance with necessary dependencies mocked.
        
        This helper method sets up all required mocks to create an NPUWorker
        instance without requiring actual NPU hardware or full vLLM dependencies.
        """
        # Mock current_platform
        mock_platform = SimpleNamespace(
            device_type="npu",
            pre_register_and_update=lambda: None,
            set_device=lambda device: None,
            dist_backend="hccl",
            is_sleep_mode_available=lambda: True,
        )
        monkeypatch.setattr("omni_npu.worker.npu_worker.current_platform",
                            mock_platform)

        # Mock torch.npu related operations
        monkeypatch.setattr("torch.npu.empty_cache", lambda: None)
        monkeypatch.setattr("torch.npu.mem_get_info", lambda device=None: (1000, 2000))
        monkeypatch.setattr("torch.npu.reset_peak_memory_stats", lambda device=None: None)
        monkeypatch.setattr("torch.npu.max_memory_allocated", lambda device=None: 500)
        monkeypatch.setattr("torch.npu.memory_allocated", lambda device=None: 400)
        monkeypatch.setattr("torch.npu.memory_reserved", lambda device=None: 600)

        # Mock distributed initialization
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.init_worker_distributed_environment",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr("omni_npu.worker.npu_worker.set_random_seed",
                            lambda seed: None)

        # Mock NPUModelRunner
        mock_model_runner = MagicMock()
        monkeypatch.setattr(
            "omni_npu.worker.npu_model_runner.NPUModelRunner",
            lambda *args, **kwargs: mock_model_runner,
        )

        # Mock report_usage_stats to avoid importing vllm code
        monkeypatch.setattr(
            "vllm.v1.utils.report_usage_stats",
            lambda vllm_config: None,
        )

        worker = NPUWorker(
            vllm_config=self.vllm_cfg,
            local_rank=0,
            rank=0,
            distributed_init_method="tcp://localhost:12345",
            is_driver_worker=True,
        )
        worker.model_runner = mock_model_runner
        return worker

    def test_get_kv_connector_handshake_metadata(self, monkeypatch):
        """Test get_kv_connector_handshake_metadata method.
        
        Verifies the method handles different scenarios:
        - When kv_transfer_group is not available
        - When kv_transfer_group exists but has no metadata
        - When metadata is available and properly formatted
        """
        worker = self._create_worker(monkeypatch)

        # Test case: no kv_transfer_group available
        monkeypatch.setattr(
            "vllm.v1.worker.gpu_worker.has_kv_transfer_group",
            lambda: False,
        )
        assert worker.get_kv_connector_handshake_metadata() is None

        # Test case: kv_transfer_group exists but has no metadata
        monkeypatch.setattr(
            "vllm.v1.worker.gpu_worker.has_kv_transfer_group",
            lambda: True,
        )
        mock_connector = MagicMock()
        mock_connector.get_handshake_metadata.return_value = None
        monkeypatch.setattr(
            "vllm.v1.worker.gpu_worker.get_kv_transfer_group",
            lambda: mock_connector,
        )
        assert worker.get_kv_connector_handshake_metadata() is None

        # Test case: metadata is available
        mock_metadata = {"key": "value"}
        mock_connector.get_handshake_metadata.return_value = mock_metadata
        mock_tp_group = SimpleNamespace(rank_in_group=0)
        monkeypatch.setattr(
            "vllm.v1.worker.gpu_worker.get_tp_group",
            lambda: mock_tp_group,
        )
        monkeypatch.setattr(
            "vllm.v1.worker.gpu_worker.get_pp_group",
            lambda: SimpleNamespace(rank_in_group=0),
        )
        result = worker.get_kv_connector_handshake_metadata()
        assert result == {(0, 0): mock_metadata}

    @pytest.mark.skip(reason="Skipping test_init_device due to mock conflicts @sunhaochen")
    def test_init_device(self, monkeypatch):
        """Test init_device method.
        
        Verifies that device initialization sets up the worker correctly,
        including device assignment, memory snapshots, and model runner creation.
        """
        worker = self._create_worker(monkeypatch)

        # Mock device_config
        worker.local_rank = 0
        # Create mock model_config with required attributes
        mock_hf_config = SimpleNamespace(
            model_type="test_model",
            quantization_config=None,
        )
        mock_registry = MagicMock()

        worker.model_config = SimpleNamespace(
            seed=42,
            hf_config=mock_hf_config,
            registry=mock_registry,
        )
        worker.cache_config = SimpleNamespace(gpu_memory_utilization=0.9)
        worker.rank = 0
        worker.scheduler_config = SimpleNamespace(enable_chunked_prefill=True)

        # Mock _init_profiler
        mock_profiler = MagicMock()
        worker._init_profiler = MagicMock(return_value=mock_profiler)

        # Mock load_model_extra_config to avoid dependencies
        mock_load_config = MagicMock()
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.load_model_extra_config",
            mock_load_config,
        )

        # Mock torch_npu.npu.get_device_name to avoid device dependency
        monkeypatch.setattr(
            "torch_npu.npu.get_device_name",
            lambda device: "Ascend910B",
        )

        # Mock NPUModelRunner initialization
        mock_model_runner = MagicMock()
        monkeypatch.setattr(
            "omni_npu.worker.npu_model_runner.NPUModelRunner",
            MagicMock(return_value=mock_model_runner),
        )

        worker.init_device()

        assert worker.device == torch.device("npu:0")
        assert hasattr(worker, "init_snapshot")
        assert hasattr(worker, "requested_memory")
        assert worker.model_runner is mock_model_runner
        assert worker.profiler is mock_profiler
        mock_load_config.assert_called_once_with(worker.model_config,
                                                 worker.vllm_config,
                                                 worker.scheduler_config)

    @pytest.mark.skip(reason="Skipping test_init_device_with_custom_model_enable due to mock conflicts @sunhaochen")
    def test_init_device_with_custom_model_enable(self, monkeypatch):
        """Test init_device with VLLM_CUSTOM_MODEL_ENABLE environment variable (covers lines 91-95).
        
        Verifies that when VLLM_CUSTOM_MODEL_ENABLE is set, layer parallel
        initialization is triggered with the correct backend.
        """
        worker = self._create_worker(monkeypatch)

        worker.local_rank = 0
        # Create mock hf_config with required attributes
        mock_hf_config = SimpleNamespace(
            model_type="test_model",
            quantization_config=None,
        )
        worker.model_config = SimpleNamespace(seed=42,
                                              hf_config=mock_hf_config)
        worker.cache_config = SimpleNamespace(gpu_memory_utilization=0.9)
        worker.rank = 0
        worker.scheduler_config = SimpleNamespace(enable_chunked_prefill=True)

        # Mock _init_profiler
        mock_profiler = MagicMock()
        worker._init_profiler = MagicMock(return_value=mock_profiler)

        # Set environment variable
        monkeypatch.setenv("VLLM_PLUGINS", "omni_custom_models")

        # Mock ensure_layer_parallel_initialized
        ensure_called = {"called": False}

        def mock_ensure_layer_parallel_initialized(backend):
            ensure_called["called"] = True
            ensure_called["backend"] = backend

        monkeypatch.setattr(
            "omni_npu.v1.distributed.parallel_state_ext.ensure_layer_parallel_initialized",
            mock_ensure_layer_parallel_initialized,
        )

        # Mock load_model_extra_config to avoid file system dependencies
        mock_load_config = MagicMock()
        # Mock both potential import paths
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.load_model_extra_config",
            mock_load_config,
        )
        monkeypatch.setattr(
            "omni_npu.model_config.config_loader.loader.load_model_extra_config",
            mock_load_config,
        )

        # Mock torch_npu.npu.get_device_name to avoid device dependency
        monkeypatch.setattr(
            "torch_npu.npu.get_device_name",
            lambda device: "Ascend910B",
        )

        # Mock NPUModelRunner initialization
        mock_model_runner = MagicMock()
        monkeypatch.setattr(
            "omni_npu.worker.npu_model_runner.NPUModelRunner",
            MagicMock(return_value=mock_model_runner),
        )

        worker.init_device()

        assert ensure_called["called"] is True
        assert ensure_called["backend"] == "hccl"
        # Verify load_model_extra_config was called exactly once
        mock_load_config.assert_called_once()

    def test_init_device_unsupported_device_type(self, monkeypatch):
        """Test init_device with unsupported device type (covers line 103).
        
        Verifies that a RuntimeError is raised when an unsupported device type
        is provided.
        """
        worker = self._create_worker(monkeypatch)

        worker.local_rank = 0
        worker.model_config = SimpleNamespace(seed=42)
        worker.cache_config = SimpleNamespace(gpu_memory_utilization=0.9)
        worker.rank = 0
        worker._init_profiler = lambda: None

        # Mock device to an unsupported device type
        mock_device = SimpleNamespace(type="cpu")
        worker.device_config.device = mock_device

        with pytest.raises(RuntimeError, match="Not support device type"):
            worker.init_device()

    @pytest.mark.skip(
        reason=
        "Skipping test_npu_runner_init_with_rejection_sampler due to mock conflicts @sunhaochen"
    )
    def test_init_profiler_full(self, monkeypatch):
        """Test _init_profiler full implementation (covers lines 236-267).
        
        Verifies that profiler initialization works correctly when all
        required environment variables are set, including token threshold
        and stop step configuration.
        """
        worker = self._create_worker(monkeypatch)

        # Mock environment variables
        monkeypatch.setenv("PROFILER_TOKEN_THRESHOLD", "10")
        monkeypatch.setenv("PROFILER_STOP_STEP", "5")
        monkeypatch.setenv("VLLM_TORCH_PROFILER_DIR", "/tmp/profiler")

        # Mock torch_npu.profiler
        mock_profiler = MagicMock()
        mock_experimental_config = MagicMock()
        mock_tensorboard_handler = MagicMock()

        monkeypatch.setattr(
            "torch_npu.profiler._ExperimentalConfig",
            lambda **kwargs: mock_experimental_config,
        )
        monkeypatch.setattr(
            "torch_npu.profiler.tensorboard_trace_handler",
            lambda path: mock_tensorboard_handler,
        )
        monkeypatch.setattr(
            "torch_npu.profiler.profile",
            lambda **kwargs: mock_profiler,
        )
        monkeypatch.setattr("vllm.envs.VLLM_TORCH_PROFILER_DIR",
                            "/tmp/profiler")
        monkeypatch.setattr("vllm.envs.VLLM_TORCH_PROFILER_RECORD_SHAPES",
                            True)
        monkeypatch.setattr(
            "vllm.envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY", True)
        monkeypatch.setattr("vllm.envs.VLLM_TORCH_PROFILER_WITH_STACK", True)
        monkeypatch.setattr("vllm.envs.VLLM_TORCH_PROFILER_WITH_FLOPS", True)

        result = worker._init_profiler()

        assert result is mock_profiler
        assert worker.profiler_token_threshold == 10
        assert worker.profiler_stop_step == 5
        assert worker._use_token_for_profile is True
        assert worker.profile_already_start is False
        assert worker.profile_finished is False

    def test_determine_available_memory(self, monkeypatch):
        """Test determine_available_memory with the GPU-aligned calculation.

        Verifies memory calculation in different scenarios:
        - When kv_cache_memory_bytes is explicitly specified
        - When memory needs to be calculated from available memory
        """
        worker = self._create_worker(monkeypatch)

        # Case 1: kv_cache_memory_bytes is explicitly specified
        worker.cache_config = SimpleNamespace(
            kv_cache_memory_bytes=1000,
            gpu_memory_utilization=0.9,
        )
        worker.model_runner.profile_run = MagicMock()

        result = worker.determine_available_memory()
        assert result == 1000
        worker.model_runner.profile_run.assert_called_once()

        # Case 2: Normal calculation with new formula
        worker.cache_config.kv_cache_memory_bytes = None
        worker.model_runner.profile_run.reset_mock()
        worker.model_runner.model_memory_usage = 300
        worker.init_snapshot = SimpleNamespace(non_torch_memory=100)

        call_count = [0]

        def mock_snapshot_measure(self):
            call_count[0] += 1
            self.torch_peak = 500 if call_count[0] == 2 else 400
            self.torch_memory = 600 if call_count[0] == 1 else 700
            self.free_memory = 1000 if call_count[0] == 1 else 800
            self.total_memory = 2000
            self.cuda_memory = 1000 if call_count[0] == 1 else 1200
            self.non_torch_memory = 100 if call_count[0] == 1 else 250
            self.timestamp = 0.0

        monkeypatch.setattr(NPUMemorySnapshot, "measure", mock_snapshot_measure)

        result = worker.determine_available_memory()
        # available = 2000 * 0.9 - 300 - (500-400) - (250-100) = 1800 - 300 - 100 - 150 = 1250
        assert result == 1250

    def test_determine_available_memory_reset_peak_exception(self, monkeypatch):
        """Test determine_available_memory with reset_peak_memory_stats exception handling.

        Verifies that exceptions from reset_peak_memory_stats are properly
        handled and the method continues execution.
        """
        worker = self._create_worker(monkeypatch)

        worker.cache_config = SimpleNamespace(
            kv_cache_memory_bytes=None,
            gpu_memory_utilization=0.9,
        )
        worker.model_runner.profile_run = MagicMock()
        worker.model_runner.model_memory_usage = 300
        worker.init_snapshot = SimpleNamespace(non_torch_memory=100)

        # Mock reset_peak_memory_stats to raise an exception
        monkeypatch.setattr(
            "torch.npu.reset_peak_memory_stats",
            lambda device=None: (_ for _ in ()).throw(Exception("Reset failed"))
        )

        call_count = [0]

        def mock_snapshot_measure(self):
            call_count[0] += 1
            self.torch_peak = 500 if call_count[0] == 2 else 400
            self.torch_memory = 600
            self.free_memory = 1000
            self.total_memory = 2000
            self.cuda_memory = 1000
            self.non_torch_memory = 250
            self.timestamp = 0.0

        monkeypatch.setattr(NPUMemorySnapshot, "measure", mock_snapshot_measure)

        result = worker.determine_available_memory()
        assert result == 1250

    def test_model_runner_proxy_methods(self, monkeypatch):
        """Test all methods that directly proxy to model_runner (consolidated test).
        
        Verifies that all proxy methods correctly delegate to the model_runner
        and return the expected values.
        """
        worker = self._create_worker(monkeypatch)

        # Prepare mock return values
        mock_kv_spec = {"layer_0": MagicMock()}
        mock_model = MagicMock()
        mock_tasks = (MagicMock(), )
        mock_draft_tokens = MagicMock()
        mock_sample_result = MagicMock()
        mock_lora_result = True
        mock_lora_set = {1, 2, 3}

        # Set model_runner return values
        worker.model_runner.get_kv_cache_spec.return_value = mock_kv_spec
        worker.model_runner.get_model.return_value = mock_model
        worker.model_runner.get_supported_tasks.return_value = mock_tasks
        worker.model_runner.take_draft_token_ids.return_value = mock_draft_tokens
        worker.model_runner.sample_tokens.return_value = mock_sample_result
        worker.model_runner.add_lora.return_value = mock_lora_result
        worker.model_runner.remove_lora.return_value = mock_lora_result
        worker.model_runner.list_loras.return_value = mock_lora_set
        worker.model_runner.pin_lora.return_value = mock_lora_result

        # Test get_kv_cache_spec
        assert worker.get_kv_cache_spec() is mock_kv_spec
        worker.model_runner.get_kv_cache_spec.assert_called_once()

        # Test get_model
        assert worker.get_model() is mock_model
        worker.model_runner.get_model.assert_called_once()

        # Test get_supported_tasks
        assert worker.get_supported_tasks() is mock_tasks
        worker.model_runner.get_supported_tasks.assert_called_once()

        # Test take_draft_token_ids
        assert worker.take_draft_token_ids() is mock_draft_tokens
        worker.model_runner.take_draft_token_ids.assert_called_once()

        # Test sample_tokens
        mock_grammar_output = MagicMock()
        assert worker.sample_tokens(mock_grammar_output) is mock_sample_result
        worker.model_runner.sample_tokens.assert_called_once_with(
            mock_grammar_output)

        # Test add_lora
        mock_lora_request = MagicMock()
        assert worker.add_lora(mock_lora_request) is mock_lora_result
        worker.model_runner.add_lora.assert_called_once_with(mock_lora_request)

        # Test remove_lora
        assert worker.remove_lora(1) is mock_lora_result
        worker.model_runner.remove_lora.assert_called_once_with(1)

        # Test list_loras
        assert worker.list_loras() is mock_lora_set
        worker.model_runner.list_loras.assert_called_once()

        # Test pin_lora
        assert worker.pin_lora(1) is mock_lora_result
        worker.model_runner.pin_lora.assert_called_once_with(1)

    def test_initialize_from_config(self, monkeypatch):
        """Test initialize_from_config method.
        
        Verifies that KV transfer initialization and cache initialization
        are properly called with the correct arguments.
        """
        worker = self._create_worker(monkeypatch)
        worker.model_config = SimpleNamespace(enable_return_routed_experts=False)
        worker.cache_config = SimpleNamespace(num_gpu_blocks=None)
        mock_kv_cache_config = MagicMock()
        mock_kv_cache_config.num_blocks = 8
        mock_kv_cache_config.needs_kv_cache_zeroing = False

        # Mock ensure_kv_transfer_initialized
        ensure_kv_initialized_called = {"called": False}

        def mock_ensure_kv_initialized(vllm_config, kv_cache_config):
            ensure_kv_initialized_called["called"] = True
            ensure_kv_initialized_called["args"] = (vllm_config,
                                                    kv_cache_config)

        monkeypatch.setattr(
            "vllm.v1.worker.gpu_worker.ensure_kv_transfer_initialized",
            mock_ensure_kv_initialized,
        )

        # Mock NpuMemAllocator
        worker._maybe_get_memory_pool_context = MagicMock(
            return_value=nullcontext()
        )

        # Test case: ENABLE_OMNI_CACHE=0 (原 use_omni_cache=False)
        worker.initialize_from_config(mock_kv_cache_config)
        worker.model_runner.initialize_kv_cache.assert_called_once_with(
            mock_kv_cache_config)

        assert ensure_kv_initialized_called["args"] == (
            worker.vllm_config,
            mock_kv_cache_config,
        )

    def test_profile(self, monkeypatch):
        """Test profile method.
        
        Verifies profiler behavior in different scenarios:
        - When profiler is None (should raise RuntimeError)
        - Normal profiler start/stop operations
        - When token threshold is enabled
        """
        worker = self._create_worker(monkeypatch)

        # Test case: profiler is None
        worker.profiler = None
        worker.profiler_config = None
        with pytest.raises(RuntimeError, match="Profiling is not enabled"):
            worker.profile()

        # Test case: normal profiler operation
        mock_profiler = MagicMock()
        worker.profiler = mock_profiler
        worker.profiler_config = SimpleNamespace(profiler="torch")
        worker._use_token_for_profile = False

        worker.profile(is_start=True)
        mock_profiler.start.assert_called_once()

        worker.profile(is_start=False)
        mock_profiler.stop.assert_called_once()

        # Explicit profile requests start regardless of auto-profile state.
        worker._use_token_for_profile = True
        mock_profiler.reset_mock()
        worker.profile(is_start=True)
        mock_profiler.start.assert_called_once()

    def test_compile_or_warm_up_model(self, monkeypatch):
        """Test compile_or_warm_up_model method.
        
        Verifies that model capture is called and random seed is set.
        """
        worker = self._create_worker(monkeypatch)
        worker.model_config = SimpleNamespace(enforce_eager=False, seed=42)

        worker.compile_or_warm_up_model()

        worker.model_runner.capture_model.assert_called_once()
        # Verify set_random_seed is called (verified through monkeypatch)

    def test_load_model(self, monkeypatch):
        """Test load_model method.
        
        Verifies that model loading delegates to model_runner.load_model
        and handles memory pool context correctly.
        """
        worker = self._create_worker(monkeypatch)
        worker.model_config = SimpleNamespace(enable_cumem_allocator=False)
        worker.vllm_config.weight_transfer_config = None
        worker._maybe_get_memory_pool_context = MagicMock(
            return_value=nullcontext()
        )
        worker._scoped_allocator_max_split = MagicMock(
            return_value=nullcontext()
        )

        worker.load_model()

        worker.model_runner.load_model.assert_called_once_with(
            load_dummy_weights=False
        )

    def test_execute_dummy_batch(self, monkeypatch):
        """Test execute_dummy_batch method.
        
        Verifies that dummy batch execution calls model_runner._dummy_run
        with the correct parameters.
        """
        worker = self._create_worker(monkeypatch)
        worker.model_runner.uniform_decode_query_len = 1

        worker.execute_dummy_batch()

        worker.model_runner._dummy_run.assert_called_once_with(
            1, uniform_decode=True, force_attention=True)

    def test_execute_model(self, monkeypatch):
        """Test execute_model method.
        
        Verifies model execution in different scenarios:
        - Normal execution without profiler
        - Execution with profiler but token threshold disabled
        - Execution with token threshold enabled (profiler start/stop logic)
        """
        worker = self._create_worker(monkeypatch)

        # Ensure _use_token_for_profile attribute exists (avoid AttributeError)
        # This attribute is usually initialized in _init_profiler(), but may not be called in tests
        if not hasattr(worker, "_use_token_for_profile"):
            worker._use_token_for_profile = False
        if not hasattr(worker, "profile_already_start"):
            worker.profile_already_start = False
        if not hasattr(worker, "profile_finished"):
            worker.profile_finished = False
        if not hasattr(worker, "profile_step"):
            worker.profile_step = 0
        if not hasattr(worker, "profiler_token_threshold"):
            worker.profiler_token_threshold = 0
        if not hasattr(worker, "profiler_stop_step"):
            worker.profiler_stop_step = 0

        # Mock scheduler_output
        mock_scheduler_output = SimpleNamespace(
            total_num_scheduled_tokens=3,
            num_scheduled_tokens=[1, 2, 3],
        )
        mock_output = IntermediateTensors(MagicMock())
        worker.model_runner.execute_model.return_value = mock_output
        
        # Test case: normal execution (no profiler)
        worker.profiler = None
        mock_tp_group = SimpleNamespace(rank_in_group=0)
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.get_tp_group",
            lambda: mock_tp_group,
        )
        
        mock_send_recv = lambda *args, **kwargs: MagicMock()
        mock_pp_group = MagicMock()
        mock_pp_group.is_first_rank = False
        mock_pp_group.recv_tensor_dict = mock_send_recv
        mock_pp_group.send_tensor_dict = mock_send_recv
        monkeypatch.setattr("omni_npu.worker.npu_worker.get_pp_group",
                            mock_pp_group)
        result = worker.execute_model(mock_scheduler_output)
        assert result is None
        worker.model_runner.execute_model.assert_called_once_with(
            mock_scheduler_output, None)

        # Test case: pp logic
        mock_output = ModelRunnerOutput(req_ids=MagicMock(), req_id_to_index=MagicMock())
        worker.model_runner.execute_model.return_value = mock_output
        
        mock_pp_group.is_first_rank = True
        monkeypatch.setattr("omni_npu.worker.npu_worker.get_pp_group",
                            mock_pp_group)
        result = worker.execute_model(mock_scheduler_output)

        # Test case: profiler exists but token threshold is not enabled
        mock_profiler = MagicMock()
        worker.profiler = mock_profiler
        worker._use_token_for_profile = False
        worker.profile_already_start = False
        worker.profile_finished = False

        worker.model_runner.execute_model.reset_mock()
        result = worker.execute_model(mock_scheduler_output)
        assert result is mock_output
        mock_profiler.start.assert_not_called()
        mock_profiler.stop.assert_not_called()

        # Test case: token threshold is enabled
        monkeypatch.setattr(worker, "_is_auto_profiler_mode", lambda: True)

        worker._use_token_for_profile = True
        worker.profile_already_start = False
        worker.profile_finished = False
        worker.profiler_token_threshold = 3
        worker.profiler_stop_step = 5
        worker.profile_step = 0
        worker.profiler_skip_requests = 0
        worker.enable_prefill_profiler = False
        worker._requests_seen = 0

        mock_scheduler_output.scheduled_new_reqs = [MagicMock()]

        # Prefill step (has new reqs): arms the profiler but does not start it
        worker.model_runner.execute_model.reset_mock()
        mock_profiler.reset_mock()
        result = worker.execute_model(mock_scheduler_output)
        assert worker.profile_already_start is False
        assert worker._requests_seen == 1
        mock_profiler.start.assert_not_called()

        # Pure decode step at the threshold: starts the profiler
        mock_scheduler_output.scheduled_new_reqs = []
        result = worker.execute_model(mock_scheduler_output)
        assert worker.profile_already_start is True
        mock_profiler.start.assert_called_once()

        # After multiple calls, profiler should be stopped
        worker.profile_step = 6
        result = worker.execute_model(mock_scheduler_output)
        assert worker.profile_finished is True
        mock_profiler.stop.assert_called_once()

    def test_execute_model_profiler_phase_conditions(self, monkeypatch):
        """Test prefill/decode phase gating in execute_model (token-based)."""
        worker = self._create_worker(monkeypatch)

        monkeypatch.setattr(worker, "_is_auto_profiler_mode", lambda: True)
        mock_pp_group = MagicMock()
        monkeypatch.setattr("omni_npu.worker.npu_worker.get_pp_group",
                            lambda: mock_pp_group)
        worker.model_runner.execute_model.return_value = ModelRunnerOutput(
            req_ids=MagicMock(), req_id_to_index=MagicMock())

        worker.profiler = MagicMock()
        worker._use_token_for_profile = True
        worker.profile_already_start = False
        worker.profile_finished = False
        worker.profile_step = 0
        worker.profiler_token_threshold = 4
        worker.profiler_stop_step = 100
        worker.profiler_skip_requests = 0
        worker.enable_prefill_profiler = False
        worker._requests_seen = 1

        def _step(num_tokens, new_reqs):
            return worker.execute_model(SimpleNamespace(
                total_num_scheduled_tokens=num_tokens,
                scheduled_new_reqs=new_reqs,
            ))

        # Decode step at the threshold but with a new request: len(new_reqs)
        # != 0, so it is (partly) prefill and must not start the profiler.
        _step(4, [MagicMock()])
        worker.profiler.start.assert_not_called()

        # Token count above the threshold without prefill profiling: no start
        _step(10, [])
        worker.profiler.start.assert_not_called()

        # Pure decode step at the threshold: starts the profiler
        _step(4, [])
        worker.profiler.start.assert_called_once()

        # ENABLE_PREFILL_PROFILER on: a prefill step (tokens > threshold)
        # starts the profiler
        worker.profiler = MagicMock()
        worker.profile_already_start = False
        worker.profile_step = 0
        worker.enable_prefill_profiler = True
        _step(10, [MagicMock()])
        worker.profiler.start.assert_called_once()

        # profiler_skip_requests defers profiling until enough requests seen
        worker.profiler = MagicMock()
        worker.profile_already_start = False
        worker.profile_step = 0
        worker.enable_prefill_profiler = False
        worker.profiler_skip_requests = 2
        worker._requests_seen = 0

        _step(4, [MagicMock()])   # seen=1, not > 2
        _step(4, [])              # decode at threshold but still skipping
        worker.profiler.start.assert_not_called()
        _step(4, [MagicMock(), MagicMock()])  # seen=3 > 2 but has new reqs
        worker.profiler.start.assert_not_called()
        _step(4, [])              # armed pure decode step
        worker.profiler.start.assert_called_once()

    def test_init_profiler(self, monkeypatch):
        """The worker delegates profiler construction to NpuProfilerWrapper."""
        worker = self._create_worker(monkeypatch)
        mock_profiler = MagicMock()
        mock_wrapper = MagicMock(return_value=mock_profiler)
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.NpuProfilerWrapper", mock_wrapper
        )
        monkeypatch.setattr(
            "vllm.distributed.utils.get_worker_rank_suffix",
            lambda global_rank: "rank_0",
        )
        worker.profiler = None
        worker.profiler_config = SimpleNamespace(profiler="torch")

        worker._init_profiler("ut")

        assert worker.profiler is mock_profiler
        mock_wrapper.assert_called_once_with(
            worker.profiler_config,
            worker_name="ut_rank_0",
            local_rank=worker.local_rank,
        )

    @pytest.mark.parametrize("value,expected", [
        ("1", True),
        ("true", True),
        ("True", True),
        (" TRUE ", True),
        (True, True),
        ("0", False),
        ("false", False),
        ("False", False),
        (False, False),
        (None, False),
        ("", False),
        ("yes", False),
    ])
    def test_to_bool(self, value, expected):
        assert _to_bool(value) is expected

    def test_sleep(self, monkeypatch):
        """Test the sleep method."""
        worker = self._create_worker(monkeypatch)
        monkeypatch.setattr("torch.npu.synchronize", lambda: None)
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.on_ascend950", lambda: False)

        mock_model = MagicMock()
        worker.model_runner.get_model.return_value = mock_model
        worker.model_runner.drafter = None

        stor = MagicMock()
        stor.nbytes.return_value = 128
        kv_tensor = MagicMock()
        kv_tensor.untyped_storage.return_value = stor
        worker.model_runner.kv_caches = [[kv_tensor]]

        monkeypatch.setattr("torch.npu.mem_get_info", lambda:
                            (15 * (1 << 30), 20 * (1 << 30)))

        worker.sleep(level=1)

        worker.model_runner.unregister_kv_caches.assert_called_once_with()
        mock_model.to.assert_called_once_with("cpu")
        stor.resize_.assert_called_once_with(0)

    def _make_tensor_like(self, device_type: str):
        """Helper: build a mock param/buffer with mutable .data and a device.type."""
        tensor = MagicMock()
        tensor.device = SimpleNamespace(type=device_type)
        tensor.data = MagicMock(name=f"{device_type}_data")
        return tensor

    def _setup_sleep_common(self, monkeypatch, cast_calls):
        """Common monkeypatching for sleep cast tests."""
        monkeypatch.setattr("torch.npu.synchronize", lambda: None)
        monkeypatch.setattr("torch.npu.mem_get_info", lambda:
                            (15 * (1 << 30), 20 * (1 << 30)))

        # Cast-related unit tests trigger the explicit NZ → ND cast only when the A5 path is taken by default.
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.on_ascend950", lambda: True)

        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.torch_npu.get_npu_format",
            lambda t: 29)

        def fake_cast(tensor, fmt):
            cast_calls.append((tensor, fmt))
            return tensor

        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.torch_npu.npu_format_cast",
            fake_cast)

    def test_sleep_casts_npu_params_and_buffers_to_nd(self, monkeypatch):
        """sleep should cast NPU parameters/buffers to ND before .to('cpu')."""
        worker = self._create_worker(monkeypatch)
        cast_calls = []
        self._setup_sleep_common(monkeypatch, cast_calls)

        npu_param = self._make_tensor_like("npu")
        npu_buffer = self._make_tensor_like("npu")

        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("p0", npu_param)]
        mock_model.named_buffers.return_value = [("b0", npu_buffer)]
        worker.model_runner.get_model.return_value = mock_model
        worker.model_runner.drafter = None
        worker.model_runner.kv_caches = []

        worker.sleep(level=1)

        # Both param and buffer get format-cast.
        assert len(cast_calls) == 2
        # The target format for all should be ND (value 2, equal to torch_npu.Format.ND)
        assert all(int(fmt) == 2 for _, fmt in cast_calls)
        # And the model is moved to CPU afterwards.
        mock_model.to.assert_called_once_with("cpu")

    def test_sleep_skips_non_npu_tensors(self, monkeypatch):
        """sleep should not call npu_format_cast on tensors whose device is not NPU."""
        worker = self._create_worker(monkeypatch)
        cast_calls = []
        self._setup_sleep_common(monkeypatch, cast_calls)

        cpu_param = self._make_tensor_like("cpu")
        cpu_buffer = self._make_tensor_like("cpu")

        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("p0", cpu_param)]
        mock_model.named_buffers.return_value = [("b0", cpu_buffer)]
        worker.model_runner.get_model.return_value = mock_model
        worker.model_runner.drafter = None
        worker.model_runner.kv_caches = []

        worker.sleep(level=1)

        assert cast_calls == []
        mock_model.to.assert_called_once_with("cpu")

    def test_sleep_casts_drafter_model_when_present(self, monkeypatch):
        """sleep should also cast and move drafter model to CPU when drafter is set."""
        worker = self._create_worker(monkeypatch)
        cast_calls = []
        self._setup_sleep_common(monkeypatch, cast_calls)

        main_param = self._make_tensor_like("npu")
        main_model = MagicMock()
        main_model.named_parameters.return_value = [("p_main", main_param)]
        main_model.named_buffers.return_value = []

        drafter_param = self._make_tensor_like("npu")
        drafter_model = MagicMock()
        drafter_model.named_parameters.return_value = [("p_draft", drafter_param)]
        drafter_model.named_buffers.return_value = []

        worker.model_runner.get_model.return_value = main_model
        worker.model_runner.get_drafter_model.return_value = drafter_model
        worker.model_runner.drafter = object()  # truthy
        worker.model_runner.kv_caches = []

        worker.sleep(level=1)

        cast_targets = [t for t, _ in cast_calls]
        assert main_param.data in cast_targets
        assert drafter_param.data in cast_targets
        main_model.to.assert_called_once_with("cpu")
        drafter_model.to.assert_called_once_with("cpu")

    def test_sleep_cast_happens_before_model_to_cpu(self, monkeypatch):
        """The NZ→ND cast must happen BEFORE model.to('cpu') is invoked."""
        worker = self._create_worker(monkeypatch)
        monkeypatch.setattr("torch.npu.synchronize", lambda: None)
        monkeypatch.setattr("torch.npu.mem_get_info", lambda:
                            (15 * (1 << 30), 20 * (1 << 30)))
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.on_ascend950", lambda: True)
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.torch_npu.get_npu_format",
            lambda t: 29)

        call_order = []
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.torch_npu.npu_format_cast",
            lambda t, fmt: (call_order.append("cast"), t)[1])

        npu_param = self._make_tensor_like("npu")
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("p0", npu_param)]
        mock_model.named_buffers.return_value = []
        mock_model.to.side_effect = lambda *a, **kw: call_order.append("to_cpu")
        worker.model_runner.get_model.return_value = mock_model
        worker.model_runner.drafter = None
        worker.model_runner.kv_caches = []

        worker.sleep(level=1)

        assert call_order == ["cast", "to_cpu"]

    def test_sleep_skips_format_cast_on_non_ascend950(self, monkeypatch):
        """On non-A5 platforms (where on_ascend950() == False), 
        explicit NZ → ND cast should not be performed, and the model 
        should simply be moved directly to CPU via .to('cpu')."""
        worker = self._create_worker(monkeypatch)
        monkeypatch.setattr("torch.npu.synchronize", lambda: None)
        monkeypatch.setattr("torch.npu.mem_get_info", lambda:
                            (15 * (1 << 30), 20 * (1 << 30)))
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.on_ascend950", lambda: False)

        cast_calls = []
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.torch_npu.npu_format_cast",
            lambda t, fmt: (cast_calls.append((t, fmt)), t)[1])

        main_param = self._make_tensor_like("npu")
        main_model = MagicMock()
        main_model.named_parameters.return_value = [("p_main", main_param)]
        main_model.named_buffers.return_value = []

        drafter_param = self._make_tensor_like("npu")
        drafter_model = MagicMock()
        drafter_model.named_parameters.return_value = [("p_draft", drafter_param)]
        drafter_model.named_buffers.return_value = []

        worker.model_runner.get_model.return_value = main_model
        worker.model_runner.get_drafter_model.return_value = drafter_model
        worker.model_runner.drafter = object()
        worker.model_runner.kv_caches = []

        worker.sleep(level=1)

        assert cast_calls == []
        main_model.to.assert_called_once_with("cpu")
        drafter_model.to.assert_called_once_with("cpu")

    def test_sleep_replaces_param_data_with_cast_result(self, monkeypatch):
        """sleep should write the cast result back into param.data (so .to('cpu') sees ND)."""
        worker = self._create_worker(monkeypatch)
        cast_calls = []

        monkeypatch.setattr("torch.npu.synchronize", lambda: None)
        monkeypatch.setattr("torch.npu.mem_get_info", lambda:
                            (15 * (1 << 30), 20 * (1 << 30)))
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.on_ascend950", lambda: True)
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.torch_npu.get_npu_format",
            lambda t: 29)

        cast_result = MagicMock(name="nd_tensor")

        def fake_cast(tensor, fmt):
            cast_calls.append((tensor, fmt))
            return cast_result

        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.torch_npu.npu_format_cast",
            fake_cast)

        npu_param = self._make_tensor_like("npu")
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = [("p0", npu_param)]
        mock_model.named_buffers.return_value = []
        worker.model_runner.get_model.return_value = mock_model
        worker.model_runner.drafter = None
        worker.model_runner.kv_caches = []

        worker.sleep(level=1)

        # param.data should now point to the cast result (the ND tensor).
        assert npu_param.data is cast_result

    def test_wake_up_weights(self, monkeypatch):
        """wake_up(tags=['weights']) moves main (and drafter) model to NPU."""
        worker = self._create_worker(monkeypatch)
        monkeypatch.setattr("torch.npu.synchronize", lambda: None)

        mock_main = MagicMock()
        mock_drafter_model = MagicMock()
        worker.model_runner.get_model.return_value = mock_main
        worker.model_runner.get_drafter_model.return_value = mock_drafter_model
        worker.model_runner.drafter = object()

        worker.wake_up(tags=["weights"])

        mock_main.to.assert_called_once_with("npu")
        mock_drafter_model.to.assert_called_once_with("npu")

    def test_wake_up_kv_cache_enforce_eager(self, monkeypatch):
        """wake_up(tags=['kv_cache']) restores KV storages and reregister; no capture when eager."""
        worker = self._create_worker(monkeypatch)
        monkeypatch.setattr("torch.npu.synchronize", lambda: None)
        mock_recapture = MagicMock()
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.set_aclgraph_recapture",
            mock_recapture)

        worker.kv_nbytes = [[256]]
        stor = MagicMock()
        kv_tensor = MagicMock()
        kv_tensor.untyped_storage.return_value = stor
        worker.model_runner.kv_caches = [[kv_tensor]]
        worker.model_config = SimpleNamespace(enforce_eager=True)

        worker.wake_up(tags=["kv_cache"])

        stor.resize_.assert_called_once_with(256)
        worker.model_runner.reregister_kv_caches.assert_called_once_with()
        mock_recapture.assert_not_called()
        worker.model_runner.capture_model.assert_not_called()

    def test_wake_up_kv_cache_triggers_recapture(self, monkeypatch):
        """wake_up(kv_cache) triggers acl graph recapture when not enforce_eager."""
        worker = self._create_worker(monkeypatch)
        monkeypatch.setattr("torch.npu.synchronize", lambda: None)
        mock_recapture = MagicMock()
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.set_aclgraph_recapture",
            mock_recapture)

        worker.kv_nbytes = [[128]]
        stor = MagicMock()
        kv_tensor = MagicMock()
        kv_tensor.untyped_storage.return_value = stor
        worker.model_runner.kv_caches = [[kv_tensor]]
        worker.model_config = SimpleNamespace(enforce_eager=False)

        worker.wake_up(tags=["kv_cache"])

        mock_recapture.assert_called_once_with(True)
        worker.model_runner.capture_model.assert_called_once_with()

    def test_wake_up_forbidden_under_full_async_rl(self, monkeypatch):
        """In fully-asynchronous RL scenarios, wake_up should fail fast."""
        worker = self._create_worker(monkeypatch)
        worker.is_full_async_rl = True

        mock_set_flag = MagicMock()
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.set_aclgraph_recapture",
            mock_set_flag,
        )

        sentinel = MagicMock()
        worker.model_runner.kv_caches = sentinel

        with pytest.raises(RuntimeError, match="wake_up is forbidden"):
            worker.wake_up(tags=["kv_cache"])

        mock_set_flag.assert_not_called()
        worker.model_runner.reregister_kv_caches.assert_not_called()
        worker.model_runner.capture_model.assert_not_called()
        sentinel.__iter__.assert_not_called()

    def test_sleep_forbidden_under_full_async_rl(self, monkeypatch):
        """In fully-asynchronous RL scenarios, sleep should fail fast."""
        worker = self._create_worker(monkeypatch)
        worker.is_full_async_rl = True
        with pytest.raises(RuntimeError, match="sleep is forbidden"):
            worker.sleep(level=1)

    def test_recapture_model_short_circuits_when_not_full_async_rl(
            self, monkeypatch):
        """In non-fully-asynchronous RL scenarios, recapture_model should directly no-op."""
        worker = self._create_worker(monkeypatch)
        worker.is_full_async_rl = False
        worker.model_config = SimpleNamespace(enforce_eager=False)

        mock_set_flag = MagicMock()
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.set_aclgraph_recapture",
            mock_set_flag,
        )

        worker.recapture_model()

        mock_set_flag.assert_not_called()
        worker.model_runner.capture_model.assert_not_called()

    def test_recapture_model_short_circuits_when_enforce_eager(
            self, monkeypatch):
        """When enforce_eager is True (no aclgraph), recapture_model is a no-op."""
        worker = self._create_worker(monkeypatch)
        worker.is_full_async_rl = True
        worker.model_config = SimpleNamespace(enforce_eager=True)

        mock_set_flag = MagicMock()
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.set_aclgraph_recapture",
            mock_set_flag,
        )

        worker.recapture_model()

        mock_set_flag.assert_not_called()
        worker.model_runner.capture_model.assert_not_called()

    def test_recapture_model_triggers_capture(self, monkeypatch):
        """Normal aclgraph path: must set the global recapture flag *and then*
        invoke model_runner.capture_model.

        The call order matters: capture_model internally calls
        consume_aclgraph_recapture(), which only marks wrappers for recapture
        if the flag is already True. So set_aclgraph_recapture MUST come
        strictly before capture_model.
        """
        worker = self._create_worker(monkeypatch)
        worker.is_full_async_rl = True
        worker.model_config = SimpleNamespace(enforce_eager=False)

        call_order = []
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.set_aclgraph_recapture",
            lambda enable: call_order.append(("set", enable)),
        )
        worker.model_runner.capture_model.side_effect = \
            lambda: call_order.append(("capture",))

        worker.recapture_model()

        worker.model_runner.capture_model.assert_called_once_with()
        assert call_order == [("set", True), ("capture",)], \
            f"set_aclgraph_recapture must run before capture_model, got {call_order}"

    def test_recapture_model_propagates_capture_error(self, monkeypatch):
        """If capture_model fails, wrap and re-raise, and clear the flag."""
        worker = self._create_worker(monkeypatch)
        worker.is_full_async_rl = True
        worker.model_config = SimpleNamespace(enforce_eager=False)

        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.set_aclgraph_recapture",
            lambda enable: None,
        )
        mock_consume = MagicMock(return_value=True)
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.consume_aclgraph_recapture",
            mock_consume,
        )
        worker.model_runner.capture_model.side_effect = RuntimeError(
            "aclgraph capture failed")

        with pytest.raises(RuntimeError, match="ACLGraph recapture failed"):
            worker.recapture_model()

        worker.model_runner.capture_model.assert_called_once_with()
        mock_consume.assert_called_once_with()

    def test_recapture_model_clears_flag_on_success(self, monkeypatch):
        """Even on success, finally must call consume (no-op if already
        consumed inside capture_model)."""
        worker = self._create_worker(monkeypatch)
        worker.is_full_async_rl = True
        worker.model_config = SimpleNamespace(enforce_eager=False)

        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.set_aclgraph_recapture",
            lambda enable: None,
        )
        mock_consume = MagicMock(return_value=False)
        monkeypatch.setattr(
            "omni_npu.worker.npu_worker.consume_aclgraph_recapture",
            mock_consume,
        )

        worker.recapture_model()

        worker.model_runner.capture_model.assert_called_once_with()
        mock_consume.assert_called_once_with()


class TestMemorySnapshot:
    """Tests for the NPU-specific memory snapshot implementation."""

    def test_default_values(self):
        """Test MemorySnapshot default initialization."""
        with switch_torch_device():
            snapshot = NPUMemorySnapshot(device="npu:0", auto_measure=False)
        assert snapshot.torch_peak == 0
        assert snapshot.free_memory == 0
        assert snapshot.total_memory == 0
        assert snapshot.cuda_memory == 0
        assert snapshot.torch_memory == 0
        assert snapshot.non_torch_memory == 0
        assert snapshot.timestamp == 0.0

    def test_measure_calculates_non_torch_memory(self, monkeypatch):
        """Test that measure correctly calculates non_torch_memory."""
        monkeypatch.setattr(
            "torch.npu.memory_stats",
            lambda device: {"allocated_bytes.all.peak": 100},
        )
        monkeypatch.setattr("torch.npu.mem_get_info", lambda device: (900, 1000))
        monkeypatch.setattr("torch.npu.memory_reserved", lambda device: 50)
        monkeypatch.setattr("vllm.platforms.current_platform.is_cuda",
                            lambda: False)

        with switch_torch_device():
            snapshot = NPUMemorySnapshot(device="npu:0", auto_measure=True)

        assert snapshot.torch_peak == 100
        assert snapshot.free_memory == 900
        assert snapshot.total_memory == 1000
        assert snapshot.cuda_memory == 100  # total - free
        assert snapshot.torch_memory == 50
        assert snapshot.non_torch_memory == 50  # cuda_memory - torch_memory

    def test_measure_with_none_device(self, monkeypatch):
        """Test MemorySnapshot with device=None."""
        monkeypatch.setattr(
            "torch.npu.memory_stats",
            lambda device: {"allocated_bytes.all.peak": 200},
        )
        monkeypatch.setattr("torch.npu.mem_get_info", lambda device: (800, 1000))
        monkeypatch.setattr("torch.npu.memory_reserved", lambda device: 100)
        monkeypatch.setattr("vllm.platforms.current_platform.current_device",
                            lambda: "npu:0")
        monkeypatch.setattr("vllm.platforms.current_platform.is_cuda",
                            lambda: False)

        with switch_torch_device():
            snapshot = NPUMemorySnapshot(device=None, auto_measure=True)

        assert snapshot.torch_peak == 200
        assert snapshot.cuda_memory == 200  # 1000 - 800
        assert snapshot.non_torch_memory == 100  # 200 - 100


class TestInitWorkerDistributedEnvironment:
    """Tests for init_worker_distributed_environment function."""

    @pytest.fixture
    def mock_vllm_config(self):
        """Create a mock VllmConfig for testing."""
        vllm_config = MagicMock()
        vllm_config.attention_config = MagicMock()
        vllm_config.attention_config.backend = "attention_backend"
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.disable_custom_all_reduce = False
        vllm_config.parallel_config.world_size = 4
        vllm_config.parallel_config.tensor_parallel_size = 2
        vllm_config.parallel_config.pipeline_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.decode_context_parallel_size = 1
        return vllm_config

    def test_init_with_default_init_method(self, mock_vllm_config, monkeypatch):
        """Test initialization with default init_method when world_ranks is None."""
        from omni_npu.worker.npu_worker import init_worker_distributed_environment

        # Track calls
        calls = []

        def mock_init_batch_invariance():
            calls.append(('init_batch_invariance',))

        def mock_set_custom_all_reduce(enable):
            calls.append(('set_custom_all_reduce', enable))

        def mock_init_distributed_environment(world_size, rank, init_method, local_rank, backend):
            calls.append(('init_distributed_environment', world_size, rank, init_method, local_rank, backend))

        def mock_ensure_model_parallel_initialized(tp_size, pp_size, pcp_size, dcp_size):
            calls.append(('ensure_model_parallel_initialized', tp_size, pp_size, pcp_size, dcp_size))

        def mock_ensure_ec_transfer_initialized(config):
            calls.append(('ensure_ec_transfer_initialized', config))

        monkeypatch.setattr("omni_npu.worker.npu_worker.init_batch_invariance", mock_init_batch_invariance)
        monkeypatch.setattr("omni_npu.worker.npu_worker.set_custom_all_reduce", mock_set_custom_all_reduce)
        monkeypatch.setattr("omni_npu.worker.npu_worker.init_distributed_environment", mock_init_distributed_environment)
        monkeypatch.setattr("omni_npu.worker.npu_worker.ensure_model_parallel_initialized", mock_ensure_model_parallel_initialized)
        monkeypatch.setattr("omni_npu.worker.npu_worker.ensure_ec_transfer_initialized", mock_ensure_ec_transfer_initialized)

        # Execute
        init_worker_distributed_environment(
            vllm_config=mock_vllm_config,
            rank=0,
            distributed_init_method="tcp://localhost:12345",
            local_rank=0,
            backend="hccl",
            world_ranks=None,
        )

        # Verify init_distributed_environment was called (not init_world_group)
        assert any(call[0] == 'init_distributed_environment' for call in calls)
        assert any(call[0] == 'ensure_ec_transfer_initialized' for call in calls)

    def test_init_with_env_method_when_no_init_method(self, mock_vllm_config, monkeypatch):
        """Test initialization uses 'env://' when distributed_init_method is None."""
        from omni_npu.worker.npu_worker import init_worker_distributed_environment

        captured_init_method = []

        def mock_init_distributed_environment(world_size, rank, init_method, local_rank, backend):
            captured_init_method.append(init_method)

        monkeypatch.setattr("omni_npu.worker.npu_worker.init_batch_invariance", lambda: None)
        monkeypatch.setattr("omni_npu.worker.npu_worker.set_custom_all_reduce", lambda x: None)
        monkeypatch.setattr("omni_npu.worker.npu_worker.init_distributed_environment", mock_init_distributed_environment)
        monkeypatch.setattr("omni_npu.worker.npu_worker.ensure_model_parallel_initialized", lambda *args: None)
        monkeypatch.setattr("omni_npu.worker.npu_worker.ensure_ec_transfer_initialized", lambda x: None)

        # Execute with None init_method
        init_worker_distributed_environment(
            vllm_config=mock_vllm_config,
            rank=0,
            distributed_init_method=None,
            local_rank=0,
            backend="hccl",
            world_ranks=None,
        )

        # Verify env:// was used as default
        assert captured_init_method[0] == "env://"

    def test_init_with_world_ranks(self, mock_vllm_config, monkeypatch):
        """Test initialization with world_ranks provided (RL scenario)."""
        from omni_npu.worker.npu_worker import init_worker_distributed_environment

        # Track calls
        calls = []

        def mock_init_world_group(ranks, local_rank, backend):
            calls.append(('init_world_group', ranks, local_rank, backend))

        monkeypatch.setattr("omni_npu.worker.npu_worker.init_batch_invariance", lambda: None)
        monkeypatch.setattr("omni_npu.worker.npu_worker.set_custom_all_reduce", lambda x: None)
        monkeypatch.setattr("omni_npu.worker.npu_worker.init_world_group", mock_init_world_group)
        monkeypatch.setattr("omni_npu.worker.npu_worker.ensure_model_parallel_initialized", lambda *args: None)
        monkeypatch.setattr("omni_npu.worker.npu_worker.ensure_ec_transfer_initialized", lambda x: None)

        world_ranks = [0, 1, 2, 3]

        # Execute
        init_worker_distributed_environment(
            vllm_config=mock_vllm_config,
            rank=0,
            distributed_init_method="tcp://localhost:12345",
            local_rank=0,
            backend="hccl",
            world_ranks=world_ranks,
        )

        # Verify init_world_group was called with correct arguments
        assert len(calls) == 1
        assert calls[0][0] == 'init_world_group'
        assert calls[0][1] == world_ranks
        assert calls[0][2] == 0
        assert calls[0][3] == "hccl"

    def test_ensure_model_parallel_initialized_called(self, mock_vllm_config, monkeypatch):
        """Test that ensure_model_parallel_initialized is called with correct parameters."""
        from omni_npu.worker.npu_worker import init_worker_distributed_environment

        captured_args = []

        def mock_ensure_model_parallel_initialized(tp_size, pp_size, pcp_size, dcp_size):
            captured_args.append((tp_size, pp_size, pcp_size, dcp_size))

        monkeypatch.setattr("omni_npu.worker.npu_worker.init_batch_invariance", lambda: None)
        monkeypatch.setattr("omni_npu.worker.npu_worker.set_custom_all_reduce", lambda x: None)
        monkeypatch.setattr("omni_npu.worker.npu_worker.init_distributed_environment", lambda *args: None)
        monkeypatch.setattr("omni_npu.worker.npu_worker.ensure_model_parallel_initialized", mock_ensure_model_parallel_initialized)
        monkeypatch.setattr("omni_npu.worker.npu_worker.ensure_ec_transfer_initialized", lambda x: None)

        # Execute
        init_worker_distributed_environment(
            vllm_config=mock_vllm_config,
            rank=0,
            local_rank=0,
            backend="hccl",
            world_ranks=None,
        )

        # Verify
        assert len(captured_args) == 1
        assert captured_args[0] == (2, 1, 1, 1)  # tp_size, pp_size, pcp_size, dcp_size


class TestInitWorldGroup:
    """Tests for init_world_group function."""

    def test_init_world_group_success(self, monkeypatch):
        """Test successful initialization of world group."""
        from omni_npu.worker.npu_worker import init_world_group

        # Mock torch.distributed
        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 4
        monkeypatch.setattr("torch.distributed", mock_dist)

        # Mock parallel_state
        mock_parallel_state = MagicMock()
        mock_parallel_state._WORLD = None
        mock_world_group = MagicMock()
        mock_parallel_state.init_world_group.return_value = mock_world_group
        monkeypatch.setattr("omni_npu.worker.npu_worker.parallel_state", mock_parallel_state)

        # Mock GroupCoordinator
        mock_group_coordinator = MagicMock()
        monkeypatch.setattr("omni_npu.worker.npu_worker.GroupCoordinator", mock_group_coordinator)

        ranks = [0, 1, 2, 3]

        # Execute
        init_world_group(ranks=ranks, local_rank=0, backend="hccl")

        # Verify
        mock_parallel_state.init_world_group.assert_called_once_with(ranks, 0, "hccl")
        assert mock_parallel_state._WORLD == mock_world_group

    def test_init_world_group_raises_when_dist_not_initialized(self, monkeypatch):
        """Test that RuntimeError is raised when torch.distributed is not initialized."""
        from omni_npu.worker.npu_worker import init_world_group

        # Mock torch.distributed as not initialized
        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = False
        monkeypatch.setattr("torch.distributed", mock_dist)

        with pytest.raises(RuntimeError, match="torch.distributed must be initialized"):
            init_world_group(ranks=[0, 1, 2, 3], local_rank=0, backend="hccl")

    def test_init_world_group_raises_when_world_already_initialized(self, monkeypatch):
        """Test that RuntimeError is raised when _WORLD is already initialized."""
        from omni_npu.worker.npu_worker import init_world_group

        # Mock torch.distributed as initialized
        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = True
        monkeypatch.setattr("torch.distributed", mock_dist)

        # Mock parallel_state with _WORLD already set
        mock_parallel_state = MagicMock()
        mock_parallel_state._WORLD = MagicMock()  # Already initialized
        monkeypatch.setattr("omni_npu.worker.npu_worker.parallel_state", mock_parallel_state)

        with pytest.raises(RuntimeError, match="_WORLD must not be initialized"):
            init_world_group(ranks=[0, 1, 2, 3], local_rank=0, backend="hccl")

    def test_init_world_group_sets_local_synchronization_when_ranks_mismatch(self, monkeypatch):
        """Test that use_local_synchronization is set when ranks length != world_size."""
        from omni_npu.worker.npu_worker import init_world_group

        # Mock torch.distributed
        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 8  # world_size is 8
        monkeypatch.setattr("torch.distributed", mock_dist)

        # Mock parallel_state
        mock_parallel_state = MagicMock()
        mock_parallel_state._WORLD = None
        mock_world_group = MagicMock()
        mock_parallel_state.init_world_group.return_value = mock_world_group
        monkeypatch.setattr("omni_npu.worker.npu_worker.parallel_state", mock_parallel_state)

        # Mock GroupCoordinator
        mock_group_coordinator = MagicMock()
        monkeypatch.setattr("omni_npu.worker.npu_worker.GroupCoordinator", mock_group_coordinator)

        # Use fewer ranks than world_size
        ranks = [0, 1, 2, 3]  # len(ranks) = 4, world_size = 8

        # Execute
        init_world_group(ranks=ranks, local_rank=0, backend="hccl")

        # Verify use_local_synchronization was set to True
        assert mock_group_coordinator.use_local_synchronization is True

    def test_init_world_group_no_local_synchronization_when_ranks_match(self, monkeypatch):
        """Test that use_local_synchronization is not set when ranks length == world_size."""
        from omni_npu.worker.npu_worker import init_world_group

        # Mock torch.distributed
        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 4  # world_size matches len(ranks)
        monkeypatch.setattr("torch.distributed", mock_dist)

        # Mock parallel_state
        mock_parallel_state = MagicMock()
        mock_parallel_state._WORLD = None
        mock_world_group = MagicMock()
        mock_parallel_state.init_world_group.return_value = mock_world_group
        monkeypatch.setattr("omni_npu.worker.npu_worker.parallel_state", mock_parallel_state)

        # Mock GroupCoordinator
        mock_group_coordinator = MagicMock()
        monkeypatch.setattr("omni_npu.worker.npu_worker.GroupCoordinator", mock_group_coordinator)

        ranks = [0, 1, 2, 3]  # len(ranks) = 4, world_size = 4

        # Execute
        init_world_group(ranks=ranks, local_rank=0, backend="hccl")

        # Verify use_local_synchronization was NOT set to True
        # (it should remain at its default value)
        assert not hasattr(mock_group_coordinator, 'use_local_synchronization') or \
               mock_group_coordinator.use_local_synchronization != True
