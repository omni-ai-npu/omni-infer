from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec
from omni_npu.vllm_patches.core import VLLMPatch, register_patch


logger = init_logger(__name__)


@register_patch("MambaSpecPatch", MambaSpec)
class MambaSpecPatch(VLLMPatch):
    _attr_names_to_apply = ['max_memory_usage_bytes']

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        return self.page_size_bytes * (2 + self.num_speculative_blocks)


@register_patch("MambaBasePatch", MambaBase)
class MambaBasePatch(VLLMPatch):
    _attr_names_to_apply = ['get_kv_cache_spec']

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        mamba_block_size = vllm_config.cache_config.mamba_block_size
        page_size_padded = vllm_config.cache_config.mamba_page_size_padded
        return MambaSpec(
            shapes=self.get_state_shape(),
            dtypes=self.get_state_dtype(),
            block_size=mamba_block_size,
            page_size_padded=page_size_padded,
            mamba_type=self.mamba_type,
            num_speculative_blocks=(
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            ),
        )