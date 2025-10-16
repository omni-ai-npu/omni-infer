#!/bin/bash
set -e

PATCH_ROOT="../../omni/adaptors/sglang/patches/"

cd ./sglang
git reset --hard
git clean -fd
git checkout v0.5.0rc0
git apply --whitespace=nowarn $PATCH_ROOT/npu_support.patch
git apply --whitespace=nowarn $PATCH_ROOT/triton_placeholder.patch
git apply --whitespace=nowarn $PATCH_ROOT/add_stream_usage.patch
git apply --whitespace=nowarn $PATCH_ROOT/add_multi_api_server_deployment.patch
git apply --whitespace=nowarn $PATCH_ROOT/operator_change_to_improve_profiler.patch
git apply --whitespace=nowarn $PATCH_ROOT/log_model_forward_time_cost.patch
git apply --whitespace=nowarn $PATCH_ROOT/add_parallel.patch
git apply --whitespace=nowarn $PATCH_ROOT/mtp.patch
git apply --whitespace=nowarn $PATCH_ROOT/npu_profiling.patch
git apply --whitespace=nowarn $PATCH_ROOT/first_token.patch
git apply --whitespace=nowarn $PATCH_ROOT/rope_nope_cache_split.patch
git apply --whitespace=nowarn $PATCH_ROOT/omni_placement.patch