#!/bin/bash
set -e

PATCH_ROOT=../../omni/adaptors/vllm/patches/

cd ./vllm
git checkout v0.9.0
git apply --whitespace=nowarn $PATCH_ROOT/manual_apiserver_scaleout.patch
git apply --whitespace=nowarn $PATCH_ROOT/scheduler_kv_cache_manager_partial_kv_transfer_v0.9.0.patch
git apply --whitespace=nowarn $PATCH_ROOT/tokenizer_proc_pool_v0.9.0.patch
git apply --whitespace=nowarn $PATCH_ROOT/mtp.patch
git apply --whitespace=nowarn $PATCH_ROOT/api_server_keepalive_timeout.patch
