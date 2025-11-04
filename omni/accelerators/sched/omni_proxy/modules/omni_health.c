// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#define _GNU_SOURCE
#include <errno.h>
#include <string.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <omni_proxy.h>
#include <omni_health.h>
#include <ngx_core.h>
#include <ngx_string.h>
#include <netinet/in.h>
#include <ngx_inet.h>
#include <omni_metrics.h>

static int udp_fd = -1;
static int report_udp_fd = -1;

// overtime settings
#define PREFILL_LIMIT 60000
#define DECODE_LIMIT 60000

void omni_health_send_response(omni_health_check_job_t *job);

// Health check handler function
static void omni_health_check_handler_internal(ngx_event_t *ev);
static void omni_health_check_read_handler(ngx_event_t *rev);
static void omni_health_check_write_handler(ngx_event_t *wev);
// static void omni_report_health_status(ngx_log_t *log);

typedef struct {
    ngx_peer_connection_t   peer;
    ngx_buf_t              *request;
    ngx_buf_t               response;
    ngx_uint_t              upstream_type;
    ngx_uint_t              upstream_index;
    u_char                  response_buf[512];
    omni_health_check_job_t *job;
} omni_health_check_conn_ctx_t;

void omni_health_send_response(omni_health_check_job_t *job) {
    ngx_http_request_t *r = job->request;

    ngx_str_t health_json = omni_health_status_export_json(omni_get_global_state(), r->pool);

    r->headers_out.status = NGX_HTTP_OK;
    r->headers_out.content_length_n = health_json.len;
    r->headers_out.content_type.len = sizeof("application/json") - 1;
    r->headers_out.content_type.data = (u_char *)"application/json";
    r->headers_out.content_type_len = r->headers_out.content_type.len;


    ngx_int_t rc = ngx_http_send_header(r);
    if (rc == NGX_ERROR || rc > NGX_OK || r->header_only) {
        ngx_http_finalize_request(r, rc);
        return;
    }

    ngx_buf_t *b = ngx_calloc_buf(r->pool);
    if (b == NULL) {
        ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return;
    }
    b->pos = health_json.data;
    b->last = health_json.data + health_json.len;
    b->memory = 1;

    b->last_buf = (r->main->count == 1);
    b->last_in_chain = 1;

    ngx_chain_t out = {b, NULL};

    ngx_http_output_filter(r, &out);
    ngx_http_finalize_request(r, NGX_OK);
}

static void omni_health_check_finalize(ngx_connection_t *c, ngx_int_t is_healthy) {
    omni_health_check_conn_ctx_t *ctx = c->data;
    omni_global_state_t *gs = omni_get_global_state();
    ngx_atomic_t *healthy_flag;
    const char* type_str = (ctx->upstream_type == 0) ? "Prefill" : "Decode";

    if (ctx->upstream_type == 0) {
        healthy_flag = &gs->prefill_states[ctx->upstream_index].healthy;
    } else {
        healthy_flag = &gs->decode_states[ctx->upstream_index].healthy;
    }

    if (is_healthy) {
        if (ngx_atomic_cmp_set(healthy_flag, 0, 1)) {
            ngx_log_error(NGX_LOG_WARN, c->log, 0, "Health check: %s upstream %ui is now HEALTHY", type_str, ctx->upstream_index);
        }
    } else {
        if (ngx_atomic_cmp_set(healthy_flag, 1, 0)) {
            ngx_log_error(NGX_LOG_WARN, c->log, 0, "Health check: %s upstream %ui is now UNHEALTHY", type_str, ctx->upstream_index);
        }
    }

    if (ngx_atomic_fetch_add(&ctx->job->pending_checks, -1) == 1) {
        omni_health_send_response(ctx->job);
    }

    ngx_close_connection(c);
}


static void omni_health_check_handler_internal(ngx_event_t *ev) {
    ngx_connection_t *c = ev->data;
    omni_health_check_finalize(c, 0);
}


static void omni_health_check_read_handler(ngx_event_t *rev) {
    ngx_connection_t *c = rev->data;
    omni_health_check_conn_ctx_t *ctx = c->data;
    ssize_t n;
    u_char *p;

    if (rev->timedout) {
        omni_health_check_finalize(c, 0);
        return;
    }

    while (1) {
        n = c->recv(c, ctx->response.last, ctx->response.end - ctx->response.last);

        if (n > 0) {
            ctx->response.last += n;
            // Find "200 OK" string
            p = ngx_strnstr(ctx->response.pos, (u_char*)"200 OK", ctx->response.last - ctx->response.pos);
            if (p != NULL) {
                omni_health_check_finalize(c, 1); // Find 200 OK，set it healthy
                return;
            }
            // if oversize, unhealthy
            if (ctx->response.last == ctx->response.end) {
                omni_health_check_finalize(c, 0);
                return;
            }
            // otherwise continue loop
            continue;
        }

        if (n == NGX_AGAIN) {
            // no more data, wait next event
            return;
        }

        // Error when (n <= 0)
        omni_health_check_finalize(c, 0);
        return;
    }
}


static void omni_health_check_write_handler(ngx_event_t *wev) {
    ngx_connection_t *c = wev->data;
    ssize_t n;

    if (wev->timedout) {
        omni_health_check_finalize(c, 0);
        return;
    }

    n = c->send(c, ((omni_health_check_conn_ctx_t*)c->data)->request->pos, ((omni_health_check_conn_ctx_t*)c->data)->request->last - ((omni_health_check_conn_ctx_t*)c->data)->request->pos);

    if (n > 0) {
        ((omni_health_check_conn_ctx_t*)c->data)->request->pos += n;
        if (((omni_health_check_conn_ctx_t*)c->data)->request->pos == ((omni_health_check_conn_ctx_t*)c->data)->request->last) {
            c->write->handler = omni_health_check_handler_internal;
            if(ngx_handle_read_event(c->read, 0) != NGX_OK) {
                omni_health_check_finalize(c, 0);
                return;
            }
            ngx_add_timer(c->read, HEALTH_CHECK_TIMEOUT);
            return;
        }
        return;
    }

    if (n == NGX_AGAIN) {
        return;
    }

    omni_health_check_finalize(c, 0);
}


void omni_run_single_health_check(omni_health_check_job_t *job, ngx_log_t *log, ngx_uint_t upstream_type, ngx_uint_t index)
{
    omni_global_state_t *gs = omni_get_global_state();
    ngx_int_t rc;
    ngx_pool_t *pool;
    ngx_connection_t *c;
    omni_health_check_conn_ctx_t *ctx;
    omni_upstream_address_t *addr;

    if (upstream_type == 0) {
        addr = &gs->prefill_states[index].comm.address;
    } else {
        addr = &gs->decode_states[index].comm.address;
    }

    pool = ngx_create_pool(1024, log);
    if (pool == NULL) return;

    ctx = ngx_pcalloc(pool, sizeof(omni_health_check_conn_ctx_t));
    if (ctx == NULL) {
        ngx_destroy_pool(pool);
        return;
    }
    ctx->job = job;
    ctx->upstream_type = upstream_type;
    ctx->upstream_index = index;

    ctx->peer.sockaddr = &addr->sockaddr;
    ctx->peer.socklen = addr->socklen;
    ctx->peer.name = &addr->name_str;
    ctx->peer.get = ngx_event_get_peer;
    ctx->peer.log = log;
    ctx->peer.log_error = NGX_ERROR_ERR;

    rc = ngx_event_connect_peer(&ctx->peer);
    if (rc == NGX_ERROR || rc == NGX_BUSY || rc == NGX_DECLINED) {
        ngx_destroy_pool(pool);
        return;
    }

    c = ctx->peer.connection;
    c->data = ctx;
    c->pool = pool;

    c->read->handler = omni_health_check_read_handler;
    c->write->handler = omni_health_check_write_handler;

    u_char *req_str = ngx_palloc(pool, 256);
    ctx->request = ngx_pcalloc(pool, sizeof(ngx_buf_t));
    ctx->request->pos = req_str;
    ctx->request->last = ngx_snprintf(req_str, 256, "GET /health HTTP/1.1\r\nHost: health-check\r\nConnection: close\r\n\r\n");
    ctx->request->memory = 1;

    ctx->response.pos = ctx->response_buf;
    ctx->response.last = ctx->response_buf;
    ctx->response.start = ctx->response_buf;
    ctx->response.end = ctx->response_buf + sizeof(ctx->response_buf);
    ctx->response.memory = 1;

    ngx_add_timer(c->write, HEALTH_CHECK_TIMEOUT);
}