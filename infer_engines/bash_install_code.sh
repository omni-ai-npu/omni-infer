#!/bin/bash
set -e

PATCH_ROOT=${1:-../../omni/adaptors/vllm/patches/}
VLLM_PATH=${2:-./vllm}

cd ${VLLM_PATH}
git reset --hard
git clean -fd
git checkout v0.9.0
git apply --whitespace=nowarn $PATCH_ROOT/manual_apiserver_scaleout.patch
git apply --whitespace=nowarn $PATCH_ROOT/scheduler_kv_cache_manager_partial_kv_transfer.patch
git apply --whitespace=nowarn $PATCH_ROOT/tokenizer_proc_pool.patch
git apply --whitespace=nowarn $PATCH_ROOT/mtp.patch
git apply --whitespace=nowarn $PATCH_ROOT/api_server_keepalive_timeout.patch
git apply --whitespace=nowarn $PATCH_ROOT/async_schedule_multi_step.patch
git apply --whitespace=nowarn $PATCH_ROOT/patch_support_fast_path_pull_kv.patch
git apply --whitespace=nowarn $PATCH_ROOT/patch_support_prefilled_token_skip_tokenize.patch
git apply --whitespace=nowarn $PATCH_ROOT/common_dependency.patch
git apply --whitespace=nowarn $PATCH_ROOT/omni_attn.patch
git apply --whitespace=nowarn $PATCH_ROOT/chunk_prefill_enable.patch
git apply --whitespace=nowarn $PATCH_ROOT/scheduler_abort_kv_loading_failure_request.patch
git apply --whitespace=nowarn $PATCH_ROOT/tfas_patch_request.patch
git apply --whitespace=nowarn $PATCH_ROOT/prometheus_dp_logging.patch
git apply --whitespace=nowarn $PATCH_ROOT/swap_kv_cache.patch
git apply --whitespace=nowarn $PATCH_ROOT/vllm_commit_3fc9644.patch
git apply --whitespace=nowarn $PATCH_ROOT/combine_block.patch
git apply --whitespace=nowarn $PATCH_ROOT/health_check.patch
git apply --whitespace=nowarn $PATCH_ROOT/function_calling_and_reasoning_content.patch
git apply --whitespace=nowarn $PATCH_ROOT/adaptive_speculative_decode.patch
git apply --whitespace=nowarn $PATCH_ROOT/pd_num_cached_tokens.patch
git apply --whitespace=nowarn $PATCH_ROOT/guided_decoding_adaption.patch
git apply --whitespace=nowarn $PATCH_ROOT/patch_qwen_moe_eagle3.patch
git apply --whitespace=nowarn $PATCH_ROOT/extract_layer_index.patch
git apply --whitespace=nowarn $PATCH_ROOT/overwrite_request_id_chat.patch
git apply --whitespace=nowarn $PATCH_ROOT/adapt_dsv32_prefix_continuation_feature.patch
git apply --whitespace=nowarn $PATCH_ROOT/overload_control.patch
git apply --whitespace=nowarn $PATCH_ROOT/enable_max_tokens_exclude_reasoning.patch
git apply --whitespace=nowarn $PATCH_ROOT/guided_decoding_v3.2_adaption.patch
git apply --whitespace=nowarn $PATCH_ROOT/adapt_chat_template_kwargs.patch
git apply --whitespace=nowarn $PATCH_ROOT/qwen3_next.patch
