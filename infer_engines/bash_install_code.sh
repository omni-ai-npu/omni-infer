#!/bin/bash
set -e

PATCH_ROOT=${1:-../../omni/adaptors/vllm/patches/}
VLLM_PATH=${2:-./vllm}

cd ${VLLM_PATH}

require_patch_marker() {
    local file="$1"
    local marker="$2"
    if ! git grep -q -F "$marker" -- "$file"; then
        echo "Required vLLM patch marker not found in ${file}: ${marker}" >&2
        exit 1
    fi
}

verify_parser_patches() {
    require_patch_marker "vllm/entrypoints/openai/serving_chat.py" "self.tool_parser_name = tool_parser"
    require_patch_marker "vllm/entrypoints/openai/serving_chat.py" "enable_thinking = self.should_enable_thinking(request)"
    require_patch_marker "vllm/entrypoints/openai/serving_chat.py" "def should_enable_thinking(self, request: ChatCompletionRequest):"
    require_patch_marker "vllm/entrypoints/openai/serving_chat.py" "self._finish_reasoning_streaming(reasoning_parser)"
    require_patch_marker "vllm/entrypoints/openai/serving_chat.py" "self._finish_tool_streaming(tool_parser, request)"
    require_patch_marker "vllm/entrypoints/openai/serving_chat.py" "def _merge_delta_message(cls, delta_message, extra_delta_message):"
    require_patch_marker "vllm/reasoning/deepseek_r1_reasoning_parser.py" "request: ChatCompletionRequest = None,"
}

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
git apply --whitespace=nowarn $PATCH_ROOT/guided_decoding_v3.2_adaption.patch
git apply --whitespace=nowarn $PATCH_ROOT/adapt_chat_template_kwargs.patch
git apply --whitespace=nowarn $PATCH_ROOT/fix_bugs_first_token.patch
git apply --whitespace=nowarn $PATCH_ROOT/guided_decoding_adapt_prefilled_token.patch
git apply --whitespace=nowarn $PATCH_ROOT/tracing.patch
git apply --whitespace=nowarn $PATCH_ROOT/support_v1_priority_schedule_for_xiaoyi.patch
git apply --whitespace=nowarn $PATCH_ROOT/gpt_oss_model_init.patch
git apply --whitespace=nowarn $PATCH_ROOT/openai_harmony_parser.patch
git apply --whitespace=nowarn $PATCH_ROOT/apply_tool_parser_content.patch
git apply --whitespace=nowarn $PATCH_ROOT/fix_ems.patch
git apply --whitespace=nowarn $PATCH_ROOT/glm4moe_model_init.patch
git apply --whitespace=nowarn $PATCH_ROOT/fix_fc_stream_content_type_error.patch
git apply --whitespace=nowarn $PATCH_ROOT/enable_max_tokens_exclude_reasoning.patch
git apply --whitespace=nowarn $PATCH_ROOT/remove_hard_limit_check_for_max_tokens.patch
git apply --whitespace=nowarn $PATCH_ROOT/openai_harmony_validation_error.patch
git apply --whitespace=nowarn $PATCH_ROOT/detokenizer.patch
git apply --whitespace=nowarn $PATCH_ROOT/adapt_deepseek_v32_tokenizer.patch
git apply --whitespace=nowarn $PATCH_ROOT/fix_deepseekv32_gd.patch
git apply --whitespace=nowarn $PATCH_ROOT/gpt_oss_make_request_by_chat_template.patch
git apply --whitespace=nowarn $PATCH_ROOT/openai_harmony_max_tokens_exclude_reasoning.patch
git apply --whitespace=nowarn $PATCH_ROOT/fix-stream-no-id-and-type.patch
git apply --whitespace=nowarn $PATCH_ROOT/api_server_print_uvicorn_kwargs.patch
git apply --whitespace=nowarn $PATCH_ROOT/fix_internal_server_error.patch
git apply --whitespace=nowarn $PATCH_ROOT/fix_reasoning_max_token_bug.patch
git apply --whitespace=nowarn $PATCH_ROOT/fix_prefill_profiling_bug.patch
git apply --whitespace=nowarn $PATCH_ROOT/fix_content_to_reasoning_content_for_nonstream.patch
git apply --whitespace=nowarn $PATCH_ROOT/fix_apc_mtp_conflict.patch
git apply --whitespace=nowarn $PATCH_ROOT/fix_multimtp.patch
git apply --whitespace=nowarn $PATCH_ROOT/fix_reasoning_content_to_content.patch
git apply --whitespace=nowarn $PATCH_ROOT/patch_support_omni_cache.patch
git apply --whitespace=nowarn $PATCH_ROOT/preserve_reasoning_content_input_passthrough.patch
git apply --whitespace=nowarn $PATCH_ROOT/glm_parser_finish_streaming.patch
verify_parser_patches
