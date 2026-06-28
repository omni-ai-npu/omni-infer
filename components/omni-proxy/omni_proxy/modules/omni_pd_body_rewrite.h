// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <omni_proxy.h>

ngx_int_t omni_proxy_prepare_encoder_body(
    ngx_http_request_t *r,
    omni_req_context_t *ctx);

ngx_int_t omni_proxy_prepare_encoder_subrequest(
    ngx_http_request_t *r,
    ngx_http_request_t *sr,
    omni_req_context_t *ctx);

ngx_int_t omni_proxy_save_origin_body(
    ngx_http_request_t *r,
    omni_req_context_t *ctx);

ngx_int_t omni_proxy_prepare_prefill_subrequest(
    ngx_http_request_t *r,
    ngx_http_request_t *sr,
    omni_req_context_t *ctx);

void omni_proxy_prepare_decode_request_body(
    ngx_http_request_t *r,
    omni_req_context_t *ctx);

// Return values for omni_check_kv_transfer_params_null
typedef enum {
    KV_TRANSFER_PRESENT = 0,         // kv_transfer_params present with all required fields
    KV_TRANSFER_NULL = 1,            // kv_transfer_params is null
    KV_TRANSFER_ABSENT = 2,          // kv_transfer_params field not present
    KV_TRANSFER_MISSING_FIELDS = 3,  // kv_transfer_params present but required fields missing
    KV_TRANSFER_ERROR = -1,          // Parse error
} omni_kv_transfer_status_t;

ngx_int_t omni_check_kv_transfer_params_null(
    ngx_http_request_t *r,
    char *body,
    size_t body_size,
    omni_req_context_t *ctx);

ngx_int_t omni_remove_kv_transfer_params_null(
    ngx_http_request_t *r,
    char *body,
    size_t body_size,
    char **new_body_out,
    omni_req_context_t *ctx);
