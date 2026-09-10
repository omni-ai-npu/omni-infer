# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from itertools import count
import inspect
import multiprocessing as mp
import functools
from unittest.mock import MagicMock

import torch
from vllm.config import VllmConfig, SchedulerConfig, ModelConfig, CacheConfig, KVTransferConfig, DeviceConfig
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, FullAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput, KVConnectorOutput
from vllm.v1.request import Request
from vllm import SamplingParams


def _mock_with_dafault_value(cls):
    mock_obj = MagicMock()

    for k, v in cls.__dict__.items():
        if k.startswith("__"):
            continue
        setattr(mock_obj, k, v)
    
    return mock_obj


def create_vllm_config(
    kv_role: str="kv_producer",
    kv_connector: str = "LLMDataDistConnector",
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 64,
    block_size: int = 16,
    max_model_len: int = 10000,
    enable_chunked_prefill: bool = True,
    enable_permute_local_kv: bool = False,
) -> VllmConfig:
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        enable_chunked_prefill=enable_chunked_prefill,
        is_encoder_decoder=False,
    )
    model_config = _mock_with_dafault_value(ModelConfig)
    model_config.max_model_len = max_model_len
    model_config.is_encoder_decoder = False
    model_config.is_multimodal_model = False
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector=kv_connector,
        kv_role=kv_role,
        enable_permute_local_kv=enable_permute_local_kv,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
    )
    vllm_config.model_config = model_config
    return vllm_config

def create_scheduler(
    vllm_config: VllmConfig,
    num_blocks: int = 10000,
) -> Scheduler:
    """Initialize Scheduler For Testing."""
    block_size = vllm_config.cache_config.block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,  # A large number of blocks to hold all requests
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32
                )
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    structured_output_manager = MagicMock()
    structured_output_manager.should_advance.return_value = False
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=structured_output_manager,
        block_size=block_size,
    )

_request_count = count(1)
def create_request(
    request_id: int | None = None,
    num_tokens: int = 10,
    common_prefix_len=0,
    max_tokens: int = 16,
    do_remote_decode: bool = False,
    do_remote_prefill: bool = False,
    num_remote_blocks: int = 3,
    block_size: int = 16,
) -> Request:
    """Make dummy request for testing."""
    assert num_tokens >= common_prefix_len >= 0

    if request_id is None:
        request_id = next(_request_count)

    kv_transfer_params: dict[str, Any] | None = None

    if do_remote_decode:
        assert not do_remote_prefill
        kv_transfer_params = dict(do_remote_prefill=False, do_remote_decode=True)
    elif do_remote_prefill:
        kv_transfer_params = dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_engine_id="my-engine-id",
            remote_block_ids=list(range(num_remote_blocks)),
            remote_host="my-host",
            remote_port=1234,
        )

    max_tokens = 1 if do_remote_decode else max_tokens
    sampling_params = SamplingParams(max_tokens=max_tokens)

    common_prefix = [1] * common_prefix_len if common_prefix_len > 0 else []
    suffix = [i * request_id for i in range(num_tokens - common_prefix_len)]
    prompt_token_ids = common_prefix + suffix

    req = Request(
        request_id=f"id-{request_id}",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=None,
        eos_token_id=None
    )
    req.kv_transfer_params = kv_transfer_params
    return req


def create_model_runner_output(
    reqs: list[Request] = None,
    finished_sending: set[str] | None = None,
    finished_recving: set[str] | None = None,
    invalid_block_ids: set[int] | None = None,
    token_id: int = 1,
) -> ModelRunnerOutput:
    """Make dummy model runner output for testing."""

    # Make request data.
    req_ids = [req.request_id for req in reqs] if reqs else []
    req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

    # Make sampled tokens.
    sampled_token = token_id
    sampled_token_ids = [[sampled_token] for _ in req_ids]

    kv_connector_output = (
        None
        if (
            finished_sending is None
            and finished_recving is None
            and invalid_block_ids is None
        )
        else KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            invalid_block_ids=invalid_block_ids or set(),
        )
    )

    # Make output data structure.
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        kv_connector_output=kv_connector_output,
    )

def _catch_exception_to_queue(fun, queue, args, kwargs):
    try:
        fun(*args, **kwargs)
        queue.put(None)
    except Exception as e:
        queue.put(str(e))

def run_in_process(fun):
    # Python does not offer a mechanism to forcibly terminate threads.
    # To avoid thread leakage in tests that start background threads, this decorator executes the test in a subprocess.
    # The subprocess exits when the test completes, guaranteeing that all threads started by the test are properly cleaned up.
    @functools.wraps(fun)
    def new_fun(*args, **kwargs):
        q = mp.Queue()
        p = mp.Process(target=_catch_exception_to_queue, args=(fun, q, args, kwargs))
        p.start()
        ret = q.get()
        p.join()
        if ret is not None:
            raise RuntimeError(ret)
    return new_fun
