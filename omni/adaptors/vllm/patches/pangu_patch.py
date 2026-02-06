from omni.adaptors.vllm.utils import get_attr_by_names, _round_up
from vllm.logger import init_logger

logger = init_logger(__name__)
# The following patches are corresponding to vllm-0.9.0


def patch_pangu():
    from typing import Optional
    from vllm.config import ModelConfig
    from vllm.v1.core.kv_cache_manager import KVCacheManager, KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock, hash_request_tokens
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SingleTypeKVCacheManager
    from vllm.v1.request import Request

    @property
    def is_deepseek_mla(self) -> bool:
        kv_lora_dim_names = ['attention_kv_lora_dim', 'kv_lora_rank']
        kv_lora_dim = get_attr_by_names(
            self.hf_text_config, kv_lora_dim_names, None)
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in \
                ('deepseek_v2', 'deepseek_v3', 'deepseek_v32', 'deepseek_mtp', 'pangu_ultra_moe', 'longcat_flash', 'kimi_k2', 'pangu_moe_v2_mtp'):
            return kv_lora_dim is not None
        elif self.hf_text_config.model_type == 'eagle':
            # if the model is an EAGLE module, check for the
            # underlying architecture
            return self.hf_text_config.model.model_type in \
                ('deepseek_v2', 'deepseek_v3', 'deepseek_v32', 'pangu_ultra_moe', 'longcat_flash', 'kimi_k2') \
                and kv_lora_dim is not None
        return False

    def _verify_with_expert_parallelism(self) -> None:
        num_expert_names = [
            "moe_num_experts",  # Dbrx
            "num_experts",  # Jamba
            "n_routed_experts",  # DeepSeek
            "num_local_experts",  # Mixtral
            "num_routed_experts",  # Pangu
        ]
        num_experts = 0
        for name in num_expert_names:
            num_experts = getattr(self.hf_text_config, name, 0)
            if num_experts > 0:
                break
        if num_experts < 1:
            raise ValueError(
                "Number of experts in the model must be greater than 0 "
                "when expert parallelism is enabled.")

    def get_head_size(self) -> int:
        if self.is_deepseek_mla:
            qk_rope_dim_names = ['attention_qk_rope_dim', 'qk_rope_head_dim']
            kv_lora_dim_names = ['attention_kv_lora_dim', 'kv_lora_rank']
            qk_rope_dim = get_attr_by_names(
                self.hf_text_config, qk_rope_dim_names, 0)
            kv_lora_dim = get_attr_by_names(
                self.hf_text_config, kv_lora_dim_names, 0)
            if self.use_mla:
                return kv_lora_dim + qk_rope_dim
            else:
                qk_dim_names = ['attention_qk_dim', 'qk_nope_head_dim']
                qk_dim = get_attr_by_names(
                    self.hf_text_config, qk_dim_names, 0)
                if qk_rope_dim and qk_dim:
                    return qk_rope_dim + qk_dim

    # for mtp
    from vllm.config import SpeculativeConfig
    from transformers import PretrainedConfig
    import vllm.envs as envs

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        if hf_config.model_type in ["deepseek_v3", "deepseek_v32", "kimi_k2"]:
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update({
                "n_predict": n_predict,
                "architectures": ["DeepSeekMTPModel"]
            })

        if hf_config.model_type == "pangu_ultra_moe":
            hf_config.model_type = "pangu_ultra_moe_mtp"
        if hf_config.model_type == "pangu_ultra_moe_mtp":
            n_predict = getattr(hf_config, "num_mtp_layers", None)
            hf_config.update({
                "n_predict": n_predict,
                "architectures": ["PanguUltraMoEMTPModel"]
            })
        if hf_config.model_type == "PanguProMoE":
            hf_config.model_type = "pangu_moe_v2_mtp"
        if hf_config.model_type == "pangu_moe_v2_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update({
                "n_predict": n_predict,
                "architectures": ["PanguProMoEMTPModel"]
            })

        if hf_config.architectures[0] == "MiMoForCausalLM":
            hf_config.model_type = "mimo_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update({
                "num_hidden_layers": 0,
                "n_predict": n_predict,
                "architectures": ["MiMoMTPModel"]
            })
            return hf_config

        return hf_config

    def __post_init__(self):

        # Note: "method" is a new parameter that helps to extend the
        # configuration of non-model-based proposers, and the "model" parameter
        # will be used to set the draft model, eagle head, or additional weight
        # when needed. If users do not specify "method", the speculative method
        # will be detected automatically if possible. If the speculative method
        # can not be detected, it will be considered as the "draft_model" by
        # default.

        if self.model is None and self.num_speculative_tokens is not None:
            # TODO(Shangming): Refactor mtp configuration logic when supporting
            # mtp acceleration for more models besides deepseek_v3
            if self.target_model_config and \
                (self.target_model_config.hf_text_config.model_type \
                        in ["deepseek_v3", "deepseek_v32", "kimi_k2"] or
                    self.target_model_config.hf_text_config.model_type \
                        == "mimo" or
                    self.target_model_config.hf_text_config.model_type \
                        == "pangu_ultra_moe" or
                    self.target_model_config.hf_text_config.model_type \
                        == "PanguProMoE" or
                    self.target_model_config.hf_text_config.model_type \
                        == "qwen3_moe" or
                    self.target_model_config.hf_text_config.model_type \
                        == "glm4_moe"):
                # use the draft model from the same model:
                self.model = self.target_model_config.model
            elif self.method in ("ngram", "[ngram]"):
                self.model = "ngram"
            else:
                raise ValueError("num_speculative_tokens was provided without "
                                 "speculative model.")

        # Automatically configure the method for ngram when "model" is used
        # instead of "method"
        if self.method is None and (self.model is not None
                                    and self.model in ("ngram", "[ngram]")):
            self.method = "ngram"

        if self.method in ("ngram", "[ngram]"):
            # Unified to "ngram" internally
            self.method = "ngram"
            # Set default values if not provided
            if (self.prompt_lookup_min is None
                    and self.prompt_lookup_max is None):
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                assert self.prompt_lookup_max is not None
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                assert self.prompt_lookup_min is not None
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min < 1:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must be > 0")
            if self.prompt_lookup_max < 1:
                raise ValueError(
                    f"prompt_lookup_max={self.prompt_lookup_max} must be > 0")
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}")

            # TODO: current we still need extract vocab_size from target model
            # config, in future, we may try refactor it out, and set
            # draft related config as None here.
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        else:
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if self.model is not None:
                self.draft_model_config = ModelConfig(
                    model=self.model,
                    task="draft",
                    tokenizer=self.target_model_config.tokenizer,
                    tokenizer_mode=self.target_model_config.tokenizer_mode,
                    trust_remote_code=self.target_model_config.
                    trust_remote_code,
                    allowed_local_media_path=self.target_model_config.
                    allowed_local_media_path,
                    dtype=self.target_model_config.dtype,
                    seed=self.target_model_config.seed,
                    revision=self.revision,
                    code_revision=self.code_revision,
                    tokenizer_revision=self.target_model_config.
                    tokenizer_revision,
                    spec_target_max_model_len=self.target_model_config.
                    max_model_len,
                    quantization=self.quantization,
                    enforce_eager=self.target_model_config.enforce_eager,
                    max_seq_len_to_capture=self.target_model_config.
                    max_seq_len_to_capture,
                    max_logprobs=self.target_model_config.max_logprobs,
                    hf_overrides=SpeculativeConfig.hf_config_override,
                )

                # Automatically detect the method
                if self.method in ('eagle', 'eagle3'):
                    pass
                elif "eagle-" in self.draft_model_config.model.lower() or \
                        "eagle3-" in self.draft_model_config.model.lower():
                    self.method = "eagle"
                elif self.draft_model_config.hf_config.model_type == "medusa":
                    self.method = "medusa"
                elif (self.draft_model_config.hf_config.model_type ==
                      "mlp_speculator"):
                    self.method = "mlp_speculator"
                elif (self.draft_model_config.hf_config.model_type ==
                      "deepseek_mtp"):
                    self.method = "deepseek_mtp"
                    if self.num_speculative_tokens > 1:
                        logger.info(
                            "All Deepseek MTP models only have "
                            "one layer. Might need some code changes "
                            "to support multiple layers."
                        )
                elif (self.draft_model_config.hf_config.model_type ==
                      "qwen3_moe" and self.method == "deepseek_mtp"):
                    self.method = "qwen3_mtp"
                    n_predict = getattr(
                        self.draft_model_config.hf_config, "num_nextn_predict_layers", None)
                    self.draft_model_config.hf_config.model_type = "qwen3_mtp"
                    self.draft_model_config.hf_config.n_predict = n_predict
                    self.draft_model_config.hf_config.architectures = [
                        "Qwen3MTPModel"]
                    if self.num_speculative_tokens > 1:
                        logger.info(
                            "All Qwen3 MTP models only have "
                            "one layer. Might need some code changes "
                            "to support multiple layers."
                        )
                elif (self.draft_model_config.hf_config.model_type ==
                        "pangu_moe_v2_mtp"):
                    self.method = "pangu_moe_v2_mtp"
                    if self.num_speculative_tokens > 1:
                        print(
                            "All Pangu MTP models only have "
                            "one layer. Might need some code changes "
                            "to support multiple layers."
                        )
                elif (self.draft_model_config.hf_config.model_type ==
                      "pangu_ultra_moe_mtp"):
                    self.method = "pangu_ultra_moe_mtp"
                    if self.num_speculative_tokens > 1:
                        logger.info(
                            "All Pangu Ultra MoE MTP models only have " \
                            "one layer. Might need some code changes " \
                            "to support multiple layers."
                        )
                elif (self.draft_model_config.hf_config.model_type ==
                      "glm4_moe"):
                    self.method = "glm4_moe_mtp"
                    n_predict = getattr(self.draft_model_config.hf_config, "num_nextn_predict_layers", None)
                    self.draft_model_config.hf_config.model_type = "glm4_moe_mtp"
                    self.draft_model_config.hf_config.n_predict = n_predict
                    self.draft_model_config.hf_config.architectures = ["Glm4MoeMTPModel"]
                    if self.num_speculative_tokens > 1: 
                        logger.info(
                            "All GLM MTP models only have " \
                            "one layer. Might need some code changes " \
                            "to support multiple layers."
                        )
                else:
                    self.method = "draft_model"

                # Replace hf_config for EAGLE draft_model
                if self.method in ("eagle", "eagle3"):
                    if self.enable_chunked_prefill and not envs.VLLM_USE_V1:
                        raise ValueError(
                            "Chunked prefill and EAGLE are not compatible "
                            "when using V0.")

                    from vllm.transformers_utils.configs.eagle import (
                        EAGLEConfig)
                    if isinstance(self.draft_model_config.hf_config,
                                  EAGLEConfig):
                        pass
                    else:
                        eagle_config = EAGLEConfig(
                            self.draft_model_config.hf_config,
                            method=self.method)
                        self.draft_model_config.hf_config = eagle_config

                if (self.num_speculative_tokens is not None
                        and hasattr(self.draft_model_config.hf_config,
                                    "num_lookahead_tokens")):
                    self.draft_model_config.hf_config.num_lookahead_tokens = \
                        self.num_speculative_tokens

                n_predict = getattr(self.draft_model_config.hf_config,
                                    "n_predict", None)
                if n_predict is not None:
                    if self.num_speculative_tokens is None:
                        # Default to max value defined in draft model config.
                        self.num_speculative_tokens = n_predict
                    elif self.num_speculative_tokens > n_predict and \
                            self.num_speculative_tokens % n_predict != 0:
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
                            f"num_speculative_tokens:{self.num_speculative_tokens}"
                            f" must be divisible by {n_predict=}")

                self.draft_tensor_parallel_size = \
                    SpeculativeConfig._verify_and_get_draft_tp(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size,
                        self.draft_model_config.hf_config
                    )

                self.draft_model_config.max_model_len = (
                    SpeculativeConfig._maybe_override_draft_max_model_len(
                        self.max_model_len,
                        self.draft_model_config.max_model_len,
                        self.target_model_config.max_model_len,
                    ))

                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size))

        if self.acceptance_method == "typical_acceptance_sampler":
            if self.posterior_threshold is None:
                self.posterior_threshold = 0.09
            if self.posterior_alpha is None:
                self.posterior_alpha = 0.3

        self._verify_args()

    def use_eagle(self) -> bool:
        return self.method in ("eagle", "eagle3", "deepseek_mtp", "ernie_mtp", "pangu_ultra_moe_mtp", "qwen3_mtp", "pangu_moe_v2_mtp","glm4_moe_mtp")

    def patch_chat_utils():
        from vllm.entrypoints.chat_utils import BaseMultiModalItemTracker, AsyncMultiModalContentParser
        from omni.adaptors.vllm.entrypoints.chat_utils import _placeholder_str_add_pangu, parse_video
        BaseMultiModalItemTracker._placeholder_str = _placeholder_str_add_pangu
        AsyncMultiModalContentParser.parse_video = parse_video

    def patch_multimodal_utils():
        from vllm.multimodal.utils import MediaConnector
        from omni.adaptors.vllm.multimodal.utils import fetch_video_async
        MediaConnector.fetch_video_async = fetch_video_async

    def patch_multimodal_video():
        from vllm.multimodal.video import VideoMediaIO
        from omni.adaptors.vllm.multimodal.video import __init__, load_bytes
        VideoMediaIO.__init__ = __init__
        VideoMediaIO.load_bytes = load_bytes

    def patch_rotary_embedding():
        from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
        from omni.layers.rotary_embedding import get_input_positions_tensor, _pangu_omni_get_input_positions_tensor
        MRotaryEmbedding.get_input_positions_tensor = get_input_positions_tensor
        MRotaryEmbedding._pangu_omni_get_input_positions_tensor = _pangu_omni_get_input_positions_tensor
        print("+++++++++++++++++++++++patch_rotary_embedding+++++++++++++++++++++++++++")

    from vllm.utils import cdiv
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_draft_tokens: int = 0,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        dcp_size: Optional[int] = 1,
        is_swap: bool = False
    ) -> Optional[KVCacheBlocks]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_blocks).
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed 
                tokens.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such 
                as eagle.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.

        Blocks layout:
        ```
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        ```
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_new_tokens == 0 and not is_swap:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = []

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.single_type_manager.remove_skipped_blocks(
            request.request_id, request.num_computed_tokens)

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (request.num_computed_tokens +
                               num_new_computed_tokens) 
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len)
        if dcp_size > 1:
            num_tokens_need_slot = cdiv(num_tokens_need_slot, dcp_size)
        
        num_blocks_to_allocate = (
            self.single_type_manager.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=num_tokens_need_slot,
                new_computed_blocks=new_computed_block_list,
            ))

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        else:
            assert not new_computed_block_list, (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        self.single_type_manager.save_new_computed_blocks(
            request.request_id, new_computed_block_list)

        new_blocks = self.single_type_manager.allocate_new_blocks(
            request.request_id, num_tokens_need_slot)

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return KVCacheBlocks(new_blocks)

        # Speculated tokens might be rejected in the future, so we does
        # not cache any speculated tokens. We only cache blocks with
        # generated (accepted) tokens.
        num_tokens_to_cache = min(num_computed_tokens + num_new_tokens, request.num_tokens)
        self.single_type_manager.cache_blocks(
            request, self.req_to_block_hashes[request.request_id],
            num_tokens_to_cache, dcp_size)

        return KVCacheBlocks(new_blocks)
    
    def find_longest_cache_hit(self, block_hashes: list[BlockHashType],
                               max_length: int,
                               dcp_size: Optional[int] = 1) -> list[KVCacheBlock]:
        computed_blocks: list[KVCacheBlock] = []

        dcp_block_size = dcp_size * self.block_size
        max_num_blocks = max_length // dcp_block_size

        for i in range(max_num_blocks):
            block_hash = block_hashes[i]
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := self.block_pool.get_cached_block(block_hash):
                computed_blocks.append(cached_block)
            else:
                break
        if self.use_eagle and len(computed_blocks) > 0:
            computed_blocks.pop()
        return computed_blocks
    
    def get_computed_blocks(self,
                            request: Request,
                            dcp_size: Optional[int] = 1) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        # Prefix caching is disabled or
        # When the request requires prompt logprobs, we skip prefix caching.
        if (not self.enable_caching
                or request.sampling_params.prompt_logprobs is not None):
            return KVCacheBlocks.create_empty(), 0

        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes[request.request_id]
        dcp_block_size = dcp_size * self.block_size
        if not block_hashes:
            block_hashes = hash_request_tokens(self.caching_hash_fn,
                                               dcp_block_size, request)
            self.req_to_block_hashes[request.request_id] = block_hashes

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.requests += 1

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1

        computed_blocks = self.single_type_manager.find_longest_cache_hit(
            block_hashes, max_cache_hit_length, dcp_size)
        # NOTE(woosuk): Since incomplete blocks are not eligible for
        # sharing, `num_computed_tokens` is always a multiple of
        # `block_size`.
        num_computed_tokens = len(computed_blocks) * dcp_block_size

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.queries += request.num_tokens
            self.prefix_cache_stats.hits += num_computed_tokens

        return KVCacheBlocks(computed_blocks), num_computed_tokens
    
    def cache_blocks(self, request: Request, block_hashes: list[BlockHashType],
                     num_tokens: int, dcp_size: Optional[int] = 1) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            block_hashes: The block hashes of the request.
            num_tokens: The total number of tokens that need to be cached 
                (including tokens that are already cached).
        """
        num_cached_blocks = self.num_cached_block[request.request_id]
        num_full_blocks = num_tokens // (self.block_size * dcp_size)

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            block_hashes=block_hashes,
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size * dcp_size,
            hash_fn=self.caching_hash_fn,
        )

        self.num_cached_block[request.request_id] = num_full_blocks

    KVCacheManager.allocate_slots = allocate_slots
    KVCacheManager.get_computed_blocks = get_computed_blocks
    FullAttentionManager.find_longest_cache_hit = find_longest_cache_hit
    SingleTypeKVCacheManager.cache_blocks = cache_blocks

    ModelConfig.is_deepseek_mla = is_deepseek_mla
    ModelConfig._verify_with_expert_parallelism = _verify_with_expert_parallelism
    ModelConfig.get_head_size = get_head_size

    SpeculativeConfig.__post_init__ = __post_init__
    SpeculativeConfig.hf_config_override = hf_config_override
    SpeculativeConfig.use_eagle = use_eagle

    patch_chat_utils()
    patch_multimodal_utils()
    patch_multimodal_video()
    patch_rotary_embedding()

    from omni.adaptors.vllm.reasoning import register_reasoning
    from omni.adaptors.vllm.entrypoints.openai.tool_parsers import register_tool
    register_reasoning()
    register_tool()
    logger.info("++++++++++++++++++++++patch_pangu++++++++++++++++++++++++++++")
