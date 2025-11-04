// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <stdint.h>
#include <omni_tokenizer.h>
#include <omni_zmq_handler.h>
#include <omni_radix_tree.h>
#include <ngx_atomic.h>  

#define NUM_PREFILL_BATCH_METRICS_HIS 32
#define NUM_DECODE_BATCH_METRICS_HIS 256
#define OMNI_PREFILL_BATCH_STATS_MAX 64
#define OMNI_PREFILL_BATCH_STATS_WINDOW 128
#define MAX_PREFILL_UPSTREAMS 128
#define MAX_DECODE_UPSTREAMS 1024
#define MAX_REQUEST_SLOTS 16384
#define MAX_WORKERS 320
#define OMNI_TTFT_BUCKETS_COUNT 23
#define OMNI_TPOT_BUCKETS_COUNT 20
#define OMNI_E2E_BUCKETS_COUNT 22

typedef enum omni_proxy_request_phase
{
    PHASE_TOKENIZING,
    PHASE_APC_MATCHING,
    PHASE_PREFILL_WAITING_SCHEDULE,
    PHASE_PREFILL_SCHEDULED,
    PHASE_PREFILLING,
    PHASE_DECODE_WAITING_SCHEDULE,
    PHASE_DECODE_SCHEDULED,
    PHASE_DECODING,
    PHASE_MAX
} omni_proxy_request_phase_t;

typedef enum omni_upstream_status
{
    STATUS_UNUSED,
    STATUS_ENABLE,
    STATUS_DISABLE,
    STATUS_ABANDON,
} omni_upstream_status_t;

typedef struct omni_request_metrics_s
{
    uint32_t prompt_num_tokens;
    uint32_t decoded_tokens;
    uint32_t max_tokens;

    ngx_msec_t time_received;
    ngx_msec_t time_contents_received;
    ngx_msec_t time_tokenized;
    ngx_msec_t time_apc_updated;

    ngx_msec_t time_enter_wait_prefill;
    ngx_msec_t time_prefill_scheduled;
    ngx_msec_t time_to_prefill;
    ngx_msec_t time_prefilled;

    ngx_msec_t time_enter_wait_decode;
    ngx_msec_t time_decode_scheduled;
    ngx_msec_t time_to_decode;

    ngx_msec_t time_last_reponse;
    ngx_msec_t time_first_token;
    ngx_msec_t tpot;
    ngx_msec_t ttft;
    ngx_uint_t http_status;  
} omni_request_metrics_t;

typedef struct omni_request_s
{
    uint16_t in_use;
    uint16_t slot_index;
    uint32_t phase_state;
    int worker_pid;
    void *backend;
    uint32_t last_retry;

    uint16_t prefill_upstream_endpoint_idx;
    uint16_t decode_upstream_endpoint_idx;
    omni_request_metrics_t metrics;
    omni_tokenizer_request tokenizer_req;
    uint32_t match_depths[MAX_PREFILL_UPSTREAMS];
    uint32_t max_match_depth;

    char request_id[37];
} omni_req_t;

typedef struct omni_request_pool_s
{
    // Each bit represent a request, 1 for in use.
    uint64_t in_use_bitmap[MAX_REQUEST_SLOTS / 64];
    // The number of request current in use
    uint32_t num_requests;
    uint32_t head;
    omni_req_t slots[MAX_REQUEST_SLOTS];
} omni_request_pool_t;

typedef struct omni_req_info_s
{
    uint32_t in_use;
    uint32_t slot_index;
    double weight;
} omni_req_info_t;

typedef struct omni_req_group_s
{
    uint32_t num_requests;
    uint32_t watermark;
    omni_proxy_request_phase_t phase;
    omni_req_info_t requests[MAX_REQUEST_SLOTS];
} omni_req_group_t;

typedef struct omni_batch_metrics_s
{
    uint32_t num_requests;
    uint32_t num_tokens;
    ngx_msec_t time_taken; // Since the oldest request responded in this batch
    ngx_msec_t first_response_receive_time;
    ngx_msec_t last_response_receive_time;
    ngx_msec_t first_schedule_sent_time;
    ngx_msec_t last_schedule_sent_time;
    double average_delta;
} omni_batch_metrics_t;

typedef struct omni_batch_metrics_his_s
{
    uint32_t head;
    uint32_t count;
    omni_batch_metrics_t his[NUM_PREFILL_BATCH_METRICS_HIS];
} omni_batch_metrics_his_t;

typedef struct omni_prefill_batch_stats_bucket_s
{
    ngx_msec_t durations[OMNI_PREFILL_BATCH_STATS_WINDOW];
    uint32_t count;
    uint32_t cursor;
    uint64_t total;
    ngx_msec_t cached_average;
} omni_prefill_batch_stats_bucket_t;

typedef struct omni_prefill_batch_stats_s
{
    omni_prefill_batch_stats_bucket_t buckets[OMNI_PREFILL_BATCH_STATS_MAX + 1];
} omni_prefill_batch_stats_t;

#define UPSTREAM_NAME_MAX 64
#define UPSTREAM_IP_MAX 16
#define UPSTREAM_ADDR_NAME_MAX 128

typedef struct omni_upstream_address_s
{
    struct sockaddr sockaddr;
    socklen_t socklen;
    char ip[UPSTREAM_IP_MAX];
    int port;
    char text[UPSTREAM_ADDR_NAME_MAX];
    int text_len;
    ngx_str_t name_str;
} omni_upstream_address_t;

typedef struct omni_upstream_common_s
{
    ngx_atomic_t ref;
    omni_upstream_status_t status;
    omni_upstream_address_t address;
} omni_upstream_common_t;

typedef struct omni_upstream_prefill_s
{
    omni_upstream_common_t comm;

    uint32_t num_running;
    uint32_t num_queue;
    uint32_t num_batch_exec;
    uint32_t idle_batch;
    uint32_t num_tokens;
    ngx_msec_t last_scheduled_time;
    ngx_msec_t expected_next_schedule_time;
    omni_batch_metrics_his_t his;
    omni_zmq_handler_t kv_handler;
    omni_radix_tree_t *radix_tree;
    ngx_atomic_t          healthy; // 1 for healthy, 0 for unhealthy
    ngx_msec_t            last_health_check_time;
} omni_upstream_prefill_t;

typedef struct omni_upstream_decode_s
{
    omni_upstream_common_t comm;

    uint32_t num_running;
    uint32_t num_tokens;
    uint32_t generated_tokens;
    ngx_msec_t last_scheduled_time;
    ngx_msec_t expected_next_schedule_time;
    omni_batch_metrics_his_t his;
    omni_zmq_handler_t kv_handler;
    ngx_slab_pool_t *radix_pool;
    omni_radix_tree_t *radix_tree;
    ngx_atomic_t          healthy; // 1 for healthy, 0 for unhealthy
    ngx_msec_t            last_health_check_time;
} omni_upstream_decode_t;

typedef enum omni_proxy_pd_policy_s
{
    PD_SEQUENTIAL,
    PD_PARALLEL
} omni_proxy_pd_policy_t;

typedef struct omni_global_state_s
{
    int magic;
    ngx_shmtx_t shmtx;
    ngx_shmtx_sh_t lock;
    ngx_slab_pool_t *shm;

    int has_tokenizer;
    omni_proxy_pd_policy_t pd_policy;

    omni_request_pool_t request_pool;
    omni_req_group_t groups[PHASE_MAX];
    uint16_t num_prefill_endpoints;
    uint16_t last_selected_prefill;
    uint16_t num_decode_endpoints;
    uint16_t last_selected_decode;
    uint32_t last_summary;
    uint32_t upstream_initialized;
    bool master_worker_selected;
    omni_upstream_prefill_t prefill_states[MAX_PREFILL_UPSTREAMS];
    omni_upstream_decode_t decode_states[MAX_DECODE_UPSTREAMS];

    ngx_atomic_t ttft_buckets[OMNI_TTFT_BUCKETS_COUNT];
    ngx_atomic_t ttft_sum_ms;
    ngx_atomic_t ttft_count;
    ngx_atomic_t tpot_buckets[OMNI_TPOT_BUCKETS_COUNT];
    ngx_atomic_t tpot_sum_ms;
    ngx_atomic_t tpot_count;
    ngx_atomic_t e2e_buckets[OMNI_E2E_BUCKETS_COUNT];
    ngx_atomic_t e2e_sum_ms;
    ngx_atomic_t e2e_count;
    ngx_atomic_t success_count;
    ngx_atomic_t failure_count;
    char      model_name[128];
    ngx_uint_t model_name_len;
    omni_prefill_batch_stats_t prefill_batch_stats;
} omni_global_state_t;

#define GLOBAL_STATE_SIZE sizeof(omni_global_state_t)
