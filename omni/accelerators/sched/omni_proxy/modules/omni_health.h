// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <omni_proxy.h> 


#define HEALTH_CHECK_INTERVAL 10000 // health_check_interval (10 seconds)
#define HEALTH_CHECK_URI "/health"  // vLLM health_check URI
#define HEALTH_CHECK_TIMEOUT 1000   // health_check overtime 1 second

/**
 * @brief health_check to separate different tyes of upstreams
 */
typedef struct {
    ngx_uint_t upstream_type; // 0 for prefill, 1 for decode
    ngx_uint_t upstream_index;
} omni_health_check_ctx_t;

typedef struct {
    ngx_http_request_t *request;
    ngx_atomic_t        pending_checks;
} omni_health_check_job_t;

/**
 * @brief timer funtion of health_check handler
 */
void omni_proxy_health_check_handler(ngx_event_t *ev);

void omni_run_single_health_check(omni_health_check_job_t *job, ngx_log_t *log, ngx_uint_t upstream_type, ngx_uint_t index);

void omni_health_send_response(omni_health_check_job_t *job);