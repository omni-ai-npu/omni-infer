from vllm.v1.core import kv_cache_utils
from vllm.v1.engine import core as engine_core
from vllm.v1.worker import block_table
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker import gpu_input_batch
from vllm.v1.core import kv_cache_manager as orig_manager
from vllm.v1.core.sched import scheduler


ENABLED = False


from . import kv_cache_interface as itfc
from .kv_cache_interface import (
    get_kv_cache_config_omni_type,
    get_omni_hybrid_kv_cache_spec,
    OmniMultiGroupBlockTable,
    OmniAttentionSpec,
)
from .kv_cache_manager import (
    OmniKVCacheBlocks,
    OmniKVCacheManager,
)
from .pd import OmniBiGroupDataDistManager
from .utils import compute_omni_attn_metadata


def apply_omni_attn_patch(enable=False, is_kv_consumer=True, config=None):
    if not enable:
        return

    global ENABLED
    ENABLED = True

    if config is not None:
        # update hyperparameters from command line args
        if "sink" in config:
            sink_val = config["sink"]
            if not isinstance(sink_val, int):
                raise ValueError(f"sink should be int, but is given {sink_val}")
            itfc.SINK = sink_val
        if "recent" in config:
            recent_val = config["recent"]
            if not isinstance(recent_val, int):
                raise ValueError(f"recent should be int, but is given {recent_val}")
            itfc.RECENT = recent_val
        if "beta" in config:
            beta_val = config["beta"]
            if not isinstance(beta_val, float) or beta_val <= 0 or beta_val >= 1:
                raise ValueError(f"beta should be float in (0,1), but is given {beta_val}")
            itfc.BETA = beta_val
        if "pattern" in config:
            pattern = config["pattern"]
            if not isinstance(pattern, list) or any(pi not in [0,1] for pi in pattern):
                raise ValueError(f"pattern should be a list of 0s and 1s, but is given {pattern}")
            itfc.PATTERN = pattern

    if is_kv_consumer:
        # use Omni-related classes and methods only for KV consumers
        kv_cache_utils.get_kv_cache_config = get_kv_cache_config_omni_type
        engine_core.get_kv_cache_config = get_kv_cache_config_omni_type
        GPUModelRunner.get_kv_cache_spec = get_omni_hybrid_kv_cache_spec
        block_table.MultiGroupBlockTable = OmniMultiGroupBlockTable
        gpu_input_batch.MultiGroupBlockTable = OmniMultiGroupBlockTable

        orig_manager.KVCacheBlocks = OmniKVCacheBlocks
        orig_manager.KVCacheManager = OmniKVCacheManager
        scheduler.KVCacheBlocks = OmniKVCacheBlocks
        scheduler.KVCacheManager = OmniKVCacheManager


__all__ = [
    "apply_omni_attn_patch",
]
