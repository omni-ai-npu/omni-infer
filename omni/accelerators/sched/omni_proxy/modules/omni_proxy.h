// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <stdint.h>
#include <assert.h>
#include <ngx_atomic.h>
#include <omni_shared_state.h>
#include <omni_tokenizer_worker.h>

#define OMNI_PROXY_SCHEDULE_ALGO_DEFAULT 0
#define OMNI_PROXY_SCHEDULE_ALGO_EARLIEST_BATCH 1

typedef enum {
    NGX_PREFILL_STREAM_OFF = 0,
    NGX_PREFILL_STREAM_ADD,
    NGX_PREFILL_STREAM_SET_OPT
} ngx_prefill_stream_op_e;

typedef struct
{
    ngx_str_t upstream_name;
    ngx_int_t pd_policy;
    ngx_str_t model_path;
    ngx_int_t vllm_kv_port_offset;
    ngx_int_t metrics_enabled;
    ngx_int_t kv_block_size;
    ngx_http_upstream_srv_conf_t *upstream;
    ngx_uint_t max_batch_num_token;
    ngx_uint_t prefill_max_num_seqs;
    ngx_uint_t decode_max_num_seqs;
    ngx_uint_t prefill_starvation_timeout;
    ngx_flag_t health_status_enabled;
    ngx_uint_t schedule_algo;
    ngx_prefill_stream_op_e stream_ops;
} ngx_http_omni_loc_conf_t;

typedef struct omni_req_context_s
{
    ngx_array_t *backends;
    ngx_http_upstream_conf_t upstream;

    omni_req_t *req;
    u_char *origin_body_data;
    ngx_uint_t origin_body_data_size;
    void *origin_body_tokens;
    int origin_body_tokens_size;
    u_char *prefill_response_body;
    ngx_uint_t prefill_response_body_size;
} omni_req_context_t;

typedef struct omni_worker_local_state_s
{
    pid_t pid;
    uint32_t worker;
    bool is_master_worker;
    uint32_t num_prefill_endpoints;
    uint32_t num_decode_endpoints;

    ngx_omni_tokenize_worker_t tokenize_worker;
    omni_zmq_handler_t *kv_handler_list;
    ngx_event_t omni_proxy_timer_event;
    ngx_http_output_body_filter_pt ngx_http_next_body_filter;
    int32_t req_in_groups;
    omni_req_group_t groups[PHASE_MAX];
    ngx_http_omni_loc_conf_t *loc_conf;
} omni_worker_local_state_t;

omni_global_state_t *omni_get_global_state();
omni_worker_local_state_t *omni_get_local_state();
