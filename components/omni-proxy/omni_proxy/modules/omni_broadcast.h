// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
typedef struct {
    ngx_str_t type;
    ngx_uint_t index;
    ngx_str_t address;
    ngx_uint_t status;
    ngx_flag_t ok;
    ngx_str_t body;
    ngx_str_t content_type;
} omni_broadcast_result_t;

typedef struct {
    ngx_http_request_t *request;
    ngx_atomic_t pending;
    ngx_uint_t total;
    ngx_uint_t success;
    ngx_array_t *results;
} omni_broadcast_ctx_t;

typedef struct {
    omni_broadcast_ctx_t *ctx;
    ngx_uint_t array_index;
} omni_broadcast_subreq_ctx_t;