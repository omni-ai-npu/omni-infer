// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <omni_proxy.h>
#include <omni_pd_body_rewrite.h>
#include <omni_scheduler.h>
#include <omni_utils.h>
#include <omni_tokenizer.h>
#include <stdbool.h>
#include <omni_apc.h>
#include <omni_metrics.h>
#include <errno.h>
#include <string.h>
#include <arpa/inet.h>
#include <inttypes.h>
#include <omni_health.h>

ngx_module_t ngx_http_omni_proxy_module;
#define PREFILL_ENDPOINTS "prefill_endpoints"
#define DECODE_ENDPOINTS "decode_endpoints"
#define ENCODE_ENDPOINTS "encode_endpoints"

#define TIMER_INTERVAL 1
#define UUID_STR_LEN 37
#define TRACE_ID_STR_LEN 56 /* 55 for trace_id, and 1 for '\0' */
#define TIME_STR_LEN 33 /* 32 for start_time_ns, and 1 for '\0' */
static u_char x_trace_headers[] = "Traceparent";
static u_char x_start_time_ns[] = "Start_time_ns";

enum req_metrics_tokens_var {
    VAR_PROMPT_NUM_TOKENS,
    VAR_DECODED_TOKENS,
    VAR_MAX_TOKENS,
    VAR_MAX_PREFILL_MATCH_DEPTH,
    VAR_MAX_DECODE_MATCH_DEPTH,
    VAR_ENCODE_UPSTREAM_IDX,
    VAR_PREFILL_UPSTREAM_IDX,
    VAR_DECODE_UPSTREAM_IDX,
};

enum req_metrics_time_var {
    VAR_TIME_RECEIVED,
    VAR_TIME_CONTENTS_RECEIVED,
    VAR_TIME_TOKENIZED,
    VAR_TIME_APC_UPDATED,

    VAR_TIME_ENTER_WAIT_PREFILL,
    VAR_TIME_PREFILL_SCHEDULED,
    VAR_TIME_TO_PREFILL,
    VAR_TIME_PREFILLED,

    VAR_TIME_ENTER_WAIT_DECODE,
    VAR_TIME_DECODE_SCHEDULED,
    VAR_TIME_TO_DECODE,

    VAR_TIME_ENCODE_START,
    VAR_TIME_ENCODE_SCHEDULED,
    VAR_TIME_ENCODE_END,

    VAR_TIME_LATENCY,
    VAR_TIME_FIRST_TOKEN,
    VAR_TPOT,
    VAR_TTFT,
};

static const char *PREFILL_URI = "/prefill_sub";
static const size_t PREFILL_URI_LEN = sizeof("/prefill_sub") - 1;
static const char *ENCODE_URI = "/encode_sub";
static const size_t ENCODE_URI_LEN = sizeof("/encode_sub") - 1;

static ngx_int_t ngx_http_omni_proxy_health_status_handler(ngx_http_request_t *r);
static ngx_int_t omni_proxy_broadcast_handler(ngx_http_request_t *r);
static void omni_proxy_broadcast_body_handler(ngx_http_request_t *r);

static char *ngx_conf_set_omni_stream_ops(ngx_conf_t *cf, ngx_command_t *cmd, void *conf);

static ngx_conf_enum_t ngx_http_omni_schedule_algos[] = {
    {ngx_string("default"), OMNI_PROXY_SCHEDULE_ALGO_DEFAULT},
    {ngx_string("earliest_batch"), OMNI_PROXY_SCHEDULE_ALGO_EARLIEST_BATCH},
    {ngx_null_string, 0}};

static omni_global_state_t *g_state;          // In share memory
static omni_worker_local_state_t local_state; // In local process memory space

static ngx_int_t ngx_http_omni_init_upstreams(ngx_cycle_t *cycle);

static ngx_int_t update_traceparent_header(ngx_http_request_t *r, const u_char *trace_headers, const u_char *trace_value, size_t trace_value_len)
{
    ngx_table_elt_t *target = NULL;
    ngx_list_part_t *part = &r->headers_in.headers.part;
    ngx_table_elt_t *h = part->elts;

    for (ngx_uint_t i = 0; /*void*/; i++) {
        if (i >= part->nelts) {
            if (part->next == NULL) break;
            part = part->next;
            h = part->elts;
            i = 0;
        }
        if (h[i].key.len == ngx_strlen(trace_headers) &&
            ngx_strncasecmp(h[i].key.data, (u_char *)trace_headers, ngx_strlen(trace_headers)) == 0) {
            target = &h[i];
            break;
        }
    }

    if (target) {
        // update trace header
        if (target->value.len == trace_value_len) {
            ngx_memcpy(target->value.data, trace_value, trace_value_len);
        } else {
            u_char *p = ngx_pnalloc(r->pool, trace_value_len);
            if (p == NULL) return NGX_ERROR;
            ngx_memcpy(p, trace_value, trace_value_len);
            target->value.data = p;
            target->value.len = trace_value_len;
        }
    }
    return NGX_OK;
}

omni_global_state_t *omni_get_global_state()
{
    return g_state;
}

omni_worker_local_state_t *omni_get_local_state()
{
    return &local_state;
}



static omni_req_t *omni_req_init(ngx_http_request_t *r)
{
    ngx_shmtx_lock(&g_state->shmtx);
    omni_req_t *req = omni_allocate_request(&g_state->request_pool, r);
    if (req == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "omni_proxy: allocate request slot failed, num_requests:%uD, head:%uD, max:%d",
                      g_state->request_pool.num_requests,
                      g_state->request_pool.head,
                      MAX_REQUEST_SLOTS);
        ngx_shmtx_unlock(&g_state->shmtx);
        return NULL;
    }
    ngx_shmtx_unlock(&g_state->shmtx);

    omni_req_context_t *ctx = ngx_pcalloc(r->pool, sizeof(omni_req_context_t));
    ctx->req = req;
    ngx_http_set_ctx(r, ctx, ngx_http_omni_proxy_module);

    omni_req_enter_phase(req, 0);
    omni_add_req_to_group(req->slot_index, &local_state.groups[0]);
    local_state.req_in_groups++;

    ngx_shmtx_lock(&g_state->shmtx);
    omni_add_req_to_group(req->slot_index, &g_state->groups[0]);
    ngx_shmtx_unlock(&g_state->shmtx);

    req->worker_pid = ngx_pid;

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "Allocate Req:%d",
                  req->slot_index);

    ngx_list_part_t *part = &r->headers_in.headers.part;
    ngx_table_elt_t *header = part->elts;
    ngx_uint_t i;
    for (i = 0; /* void */; i++) {
        if (i >= part->nelts) {
            if (part->next == NULL) break;
            part = part->next;
            header = part->elts;
            i = 0;
        }
        if (header[i].key.len == sizeof("X-Request-Id") - 1 &&
            ngx_strncasecmp(header[i].key.data, (u_char*)"X-Request-Id", sizeof("X-Request-Id") - 1) == 0) {
            size_t len = header[i].value.len;
            if (len >= UUID_STR_LEN) len = UUID_STR_LEN - 1;
            ngx_memcpy(req->request_id, header[i].value.data, len);
            req->request_id[len] = '\0';
            break;
        }
    }

    return req;
}

static void omni_req_free(omni_req_t *req)
{
    ngx_http_request_t *r = omni_get_http_request(req);
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "Free Req:%d, at:%u",
                  req->slot_index, req->phase_state);
    ngx_shmtx_lock(&g_state->shmtx);
    omni_free_request(&g_state->request_pool, req);
    ngx_shmtx_unlock(&g_state->shmtx);
}

void omni_phase_transition_all(omni_req_t *req, omni_proxy_request_phase_t from, omni_proxy_request_phase_t to)
{
    omni_local_phase_change_to(req, from, to);

    ngx_shmtx_lock(&g_state->shmtx);
    omni_global_phase_change_to(req, from, to);
    omni_req_leave_phase(req, from);
    omni_req_enter_phase(req, to);
    ngx_shmtx_unlock(&g_state->shmtx);
}

static inline omni_req_context_t *omni_get_req_ctx(ngx_http_request_t *r)
{
    return ngx_http_get_module_ctx(r, ngx_http_omni_proxy_module);
}


static inline omni_req_t *omni_get_req(ngx_http_request_t *r)
{
    omni_req_context_t *ctx = omni_get_req_ctx(r);
    if (ctx == NULL){
        ngx_log_error(NGX_LOG_NOTICE, r->connection->log, 0,
            "omni_get_req: Failed to get request context for a request that should have one, not a omni req.");
        return NULL;
    }
    return ctx->req;
}

static void omni_proxy_post_tokenized(omni_req_t *req)
{
    /* 1) snapshot pointers under lock */
    ngx_shmtx_lock(&g_state->shmtx);
    /* snapshot radix_tree pointers */
    uint16_t num_prefill = g_state->num_prefill_endpoints;
    omni_radix_tree_t *local_prefill_trees[MAX_PREFILL_UPSTREAMS];
    ngx_memzero(local_prefill_trees, sizeof(omni_radix_tree_t *) * MAX_PREFILL_UPSTREAMS);
    uint16_t prefill_cnt = 0;
    for (int i = 0; i < MAX_PREFILL_UPSTREAMS && prefill_cnt < num_prefill; i++) {
        omni_upstream_prefill_t *prefill = &g_state->prefill_states[i];
        if (prefill->comm.status != STATUS_ENABLE) {
            continue;
        }
        prefill_cnt++;
        local_prefill_trees[i] = prefill->radix_tree;
    }

    uint16_t num_decode = g_state->num_decode_endpoints;
    omni_radix_tree_t *local_decode_trees[MAX_DECODE_UPSTREAMS];
    ngx_memzero(local_decode_trees, sizeof(omni_radix_tree_t *) * MAX_DECODE_UPSTREAMS);
    uint16_t decode_cnt = 0;
    for (int i = 0; i < MAX_DECODE_UPSTREAMS && decode_cnt < num_decode; i++) {
        omni_upstream_decode_t *decode = &g_state->decode_states[i];
        if (decode->comm.status != STATUS_ENABLE) {
            continue;
        }
        decode_cnt++;
        local_decode_trees[i] = decode->radix_tree;
    }
    ngx_shmtx_unlock(&g_state->shmtx);

    /* 2) Compute matches without holding g_state lock */
    // Prefill Matching
    uint32_t local_prefill_match_depths[MAX_PREFILL_UPSTREAMS];
    ngx_uint_t computed_max_prefill = 0;

    for (uint16_t i = 0, cnt = 0; i < MAX_PREFILL_UPSTREAMS && cnt < num_prefill; i++) {
        omni_radix_tree_t *tree = local_prefill_trees[i];
        ngx_uint_t match_depth = 0;
        if (tree != NULL) {
            cnt++;
            match_depth = omni_radix_tree_match_optimistic(tree,
                                                            (uint64_t *)req->tokenizer_req.block_hashes,
                                                            req->tokenizer_req.block_hashes_len);
        }
        local_prefill_match_depths[i] = (uint32_t)match_depth;
        if (match_depth > 0) {
            ngx_log_error(NGX_LOG_INFO,
                omni_get_http_request(req)->connection->log,
                0,
                "[APC_MATCHING-%d] prefill[%ui] match_depth=%ui",
                req->slot_index,
                i,
                local_prefill_match_depths[i]);
        }

        if ((ngx_uint_t)match_depth > computed_max_prefill)
            computed_max_prefill = match_depth;
    }

    ngx_log_error(NGX_LOG_INFO,
                  omni_get_http_request(req)->connection->log,
                  0,
                  "[APC_MATCHING-%d] computed num_prefill=%ui, max_prefill_match_depth=%ui",
                  req->slot_index,
                  num_prefill,
                  computed_max_prefill);
    
    // Decode Matching
    uint32_t local_decode_match_depths[MAX_DECODE_UPSTREAMS];
    ngx_uint_t computed_max_decode = 0;
    for (uint16_t i = 0, cnt = 0; i < MAX_DECODE_UPSTREAMS && cnt < num_decode; i++) {
        omni_radix_tree_t *tree = local_decode_trees[i];
        ngx_uint_t match_depth = 0;
        if (tree != NULL) {
            cnt++;
            match_depth = omni_radix_tree_match_optimistic(tree,
                                                            (uint64_t *)req->tokenizer_req.block_hashes,
                                                            req->tokenizer_req.block_hashes_len);
        }
        local_decode_match_depths[i] = (uint32_t)match_depth;
        if (match_depth > 0) {
            ngx_log_error(NGX_LOG_INFO,
                omni_get_http_request(req)->connection->log,
                0,
                "[APC_MATCHING-%d] decode[%ui] match_depth=%ui",
                req->slot_index,
                i,
                local_decode_match_depths[i]);
        }        

        if ((ngx_uint_t)match_depth > computed_max_decode)
            computed_max_decode = match_depth;
    }

    ngx_log_error(NGX_LOG_INFO,
                  omni_get_http_request(req)->connection->log,
                  0,
                  "[APC_MATCHING-%d] computed num_decode=%ui, max_decode_match_depth=%ui",
                  req->slot_index,
                  num_decode,
                  computed_max_decode);

    /* 3) write results to shared request */
    ngx_memcpy(req->prefill_match_depths, local_prefill_match_depths, sizeof(local_prefill_match_depths));
    req->max_prefill_match_depth = (uint32_t)computed_max_prefill;
    ngx_memcpy(req->decode_match_depths, local_decode_match_depths, sizeof(local_decode_match_depths));
    req->max_decode_match_depth = (uint32_t)computed_max_decode;

    /* Finally perform phase transition and timestamp updates */
    struct timeval tv;
    omni_get_current_time(&tv);
    ngx_http_request_t *r = omni_get_http_request(req);

    if (g_state->pd_policy == PD_AGGREGATION) {
        omni_phase_transition_all(req, 0, PHASE_DECODE_WAITING_SCHEDULE);
        req->metrics.time_enter_wait_decode = tv;
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                    "<<<Action: Enter state D waiting; Timestamp:%d.%06d; RequestID:%s", tv.tv_sec, tv.tv_usec, req->request_id);
    } else {
        omni_phase_transition_all(req, 0, PHASE_PREFILL_WAITING_SCHEDULE);
        req->metrics.time_enter_wait_prefill = tv;

        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                    "<<<Action: Enter state P waiting; Timestamp:%d.%06d; RequestID:%s", tv.tv_sec, tv.tv_usec, req->request_id);
    }
}

static void omni_proxy_begin_tokenization(ngx_http_request_t *r, omni_req_t *req, omni_req_context_t *ctx)
{
    req->metrics.prompt_num_tokens = r->request_length / 8; // will be updated after tokenization

    if (!g_state->has_tokenizer)
    {
        omni_proxy_post_tokenized(req);
        return;
    }

    req->tokenizer_req.input_data = ctx->origin_body_data;
    req->tokenizer_req.input_len = ctx->origin_body_data_size;

    // Chat template expansion buffer
    req->tokenizer_req.prompt_buf_size = req->tokenizer_req.input_len + 512;
    req->tokenizer_req.prompt = ngx_palloc(r->pool, req->tokenizer_req.prompt_buf_size);

    req->tokenizer_req.input_ids_buf_size = req->tokenizer_req.prompt_buf_size;
    req->tokenizer_req.input_ids = ngx_palloc(r->pool, sizeof(int64_t) * req->tokenizer_req.input_ids_buf_size);

    req->tokenizer_req.block_hashes_buf_size = req->tokenizer_req.input_ids_buf_size / local_state.loc_conf->kv_block_size;
    req->tokenizer_req.block_hashes = ngx_palloc(r->pool, sizeof(int64_t) * req->tokenizer_req.block_hashes_buf_size);
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "buf sizes: prompt %zu, input_ids %zu, block_hashes %zu, %p, %p, %p\n",
                  req->tokenizer_req.prompt_buf_size,
                  req->tokenizer_req.input_ids_buf_size,
                  req->tokenizer_req.block_hashes_buf_size,
                  req->tokenizer_req.prompt,
                  req->tokenizer_req.input_ids,
                  req->tokenizer_req.block_hashes);
    omni_tokenizer_worker_submit(&local_state.tokenize_worker, req->slot_index);
}

static void omni_encoder_release_sched(omni_req_t *req, ngx_uint_t encode_items)
{
    if (!req->has_encode_sched) {
        return;
    }

    omni_upstream_encode_t *es = &g_state->encode_states[req->encode_upstream_endpoint_idx];
    ngx_atomic_fetch_add(&es->num_running, -1);
    if (es->num_tokens >= encode_items) {
        ngx_atomic_fetch_add(&es->num_tokens, -encode_items);
    } else {
        es->num_tokens = 0;
    }
    ngx_atomic_fetch_add(&es->comm.ref, -1);
    req->has_encode_sched = false;
}

static ngx_int_t ngx_http_encoder_post_subrequest(ngx_http_request_t *subr, void *data, ngx_int_t rc)
{
    omni_req_t *req = data;
    ngx_http_request_t *r = omni_get_http_request(req);
    omni_req_context_t *ctx = omni_get_req_ctx(r);
    ngx_uint_t encode_items = (req->encode_item_count > 0) ? req->encode_item_count : ctx->encoder_item_count;
    ngx_chain_t *cl;
    size_t total = 0;
    u_char *p;

    if (rc != NGX_OK)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "[Encoder-%d] subrequest failed with code %i", req->slot_index, rc);
        ngx_http_finalize_request(r->main, NGX_HTTP_BAD_GATEWAY);
        omni_encoder_release_sched(req, encode_items);
        return rc;
    }

    if (subr->headers_out.status != NGX_HTTP_OK)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "[Encoder-%d] subrequest returned status %i", req->slot_index, subr->headers_out.status);
        ngx_http_finalize_request(r->main, NGX_HTTP_BAD_GATEWAY);
        omni_encoder_release_sched(req, encode_items);
        return NGX_ERROR;
    }

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[Encoder-%d] processed %ui multimodal items", req->slot_index, ctx->encoder_item_count);

    struct timeval tv;
    omni_get_current_time(&tv);
    req->metrics.time_encode_end = tv;

    omni_encoder_release_sched(req, encode_items);

    for (cl = subr->out; cl; cl = cl->next)
    {
        total += ngx_buf_size(cl->buf);
    }

    ctx->encoder_response_body = ngx_palloc(r->main->pool, total + 1);
    if (ctx->encoder_response_body == NULL)
    {
        ngx_http_finalize_request(r, NGX_ERROR);
        ngx_log_error(
            NGX_LOG_ERR, r->connection->log, 0,
            "done encoder subrequest, malloc encoder response body buffer failed");
        return rc;
    }

    p = ctx->encoder_response_body;
    for (cl = subr->out; cl; cl = cl->next)
    {
        size_t buf_size = ngx_buf_size(cl->buf);
        if (buf_size > 0)
        {
            p = ngx_cpymem(p, cl->buf->pos, buf_size);
        }
    }
    *p = '\0';
    ctx->encoder_response_body_size = total;

    omni_proxy_begin_tokenization(r, req, ctx);
    return NGX_OK;
}

static ngx_int_t omni_proxy_start_encoder_subrequest(ngx_http_request_t *r, omni_req_context_t *ctx)
{
    omni_req_t *req = ctx->req;

    size_t prelen = ENCODE_URI_LEN + r->uri.len;
    if (r->args.len)
    {
        prelen += 1 + r->args.len;
    }
    u_char *encoder_uri = ngx_pnalloc(r->pool, prelen + 1);
    if (encoder_uri == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "omni_proxy: allocate encoder uri failed.");
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    u_char *p = encoder_uri;
    p = ngx_cpymem(p, ENCODE_URI, ENCODE_URI_LEN);
    p = ngx_cpymem(p, r->uri.data, r->uri.len);
    if (r->args.len)
    {
        *p++ = '?';
        p = ngx_cpymem(p, r->args.data, r->args.len);
    }
    *p = '\0';

    ngx_str_t sub_uri;
    sub_uri.len = ngx_strlen(encoder_uri);
    sub_uri.data = encoder_uri;

    ngx_http_post_subrequest_t *psr = ngx_pcalloc(r->pool, sizeof(ngx_http_post_subrequest_t));
    if (psr == NULL) {
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }
    psr->handler = ngx_http_encoder_post_subrequest;
    psr->data = req;

    ngx_str_t args = ngx_null_string;
    ngx_http_request_t *sr;

    ngx_int_t rc = ngx_http_subrequest(r, &sub_uri, &args, &sr, psr,
                                       NGX_HTTP_SUBREQUEST_IN_MEMORY);
    if (rc != NGX_OK)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "omni_proxy: create encoder subrequest failed.");
        return rc;
    }

    sr->method = r->method;
    sr->method_name = r->method_name;

    ngx_http_set_ctx(sr, ctx, ngx_http_omni_proxy_module);

    if (omni_proxy_prepare_encoder_subrequest(r, sr, ctx) != NGX_DONE)
    {
        return NGX_ERROR;
    }

    ngx_http_run_posted_requests(r->connection);
    return NGX_OK;
}

static void omni_proxy_req_body_handler(ngx_http_request_t *r)
{
    omni_req_t *req = omni_get_req(r);
    omni_req_context_t *ctx = omni_get_req_ctx(r);
    omni_get_current_time(&req->metrics.time_contents_received);
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Prefill-%d]: request body received", req->slot_index);

    omni_proxy_save_origin_body(r, ctx);

    if (g_state->num_encode_endpoints > 0) {
        struct timeval tv;
        omni_get_current_time(&tv);
        req->metrics.time_encode_start = tv;

        ngx_int_t enc_rc = omni_proxy_prepare_encoder_body(r, ctx);
        if (enc_rc == NGX_OK) {
            ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                          "[Encoder-%d] preparing %ui multimodal items", req->slot_index, ctx->encoder_item_count);
            ngx_int_t schedule_rc = omni_proxy_schedule_encode(g_state, req, ctx);
            if (schedule_rc != NGX_OK) {
                ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                              "[Encoder-%d] schedule failed", req->slot_index);
                ngx_http_finalize_request(r, NGX_HTTP_SERVICE_UNAVAILABLE);
                return;
            }
            if (omni_proxy_start_encoder_subrequest(r, ctx) != NGX_OK) {
                ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
            }
            return;
        }
        if (enc_rc != NGX_DECLINED) {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                          "[Encoder-%d] failed to parse multimodal content", req->slot_index);
            ngx_http_finalize_request(r, NGX_HTTP_BAD_REQUEST);
            return;
        }
    }

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                    "<<<Action: Enter state tokenization; Timestamp:%d.%06d; RequestID:%s", req->metrics.time_contents_received.tv_sec, req->metrics.time_contents_received.tv_usec, req->request_id);

    omni_proxy_begin_tokenization(r, req, ctx);
}

static inline void omni_proxy_cleanup_req(omni_req_t *req)
{
    omni_proxy_request_phase_t phases[PHASE_MAX];
    size_t count = 0;
    omni_req_get_phases(req, phases, &count);

    for (size_t i = 0; i < count; ++i)
    {
        omni_proxy_request_phase_t phase = phases[i];
        ngx_shmtx_lock(&g_state->shmtx);

        if (req->has_encode_sched)
        {
            omni_upstream_encode_t *es =
                &g_state->encode_states[req->encode_upstream_endpoint_idx];

            ngx_atomic_fetch_add(&es->num_running, -1);
            ngx_atomic_fetch_add(&es->comm.ref, -1);

            if (es->num_tokens >= req->encode_item_count)
            {
                ngx_atomic_fetch_add(&es->num_tokens, -req->encode_item_count);
            }
            else
            {
                es->num_tokens = 0;
            }
            req->has_encode_sched = false;
            ngx_log_error(NGX_LOG_INFO, omni_get_http_request(req)->connection->log, 0,
                          "[Encode Release Stats] req=%uD, items=%uD; encode_idx=%uD, num_running=%uD, num_tokens(after)=%uD",
                          req->slot_index,
                          req->encode_item_count,
                          req->encode_upstream_endpoint_idx,
                          es->num_running,
                          es->num_tokens);
        }

        if (phase == PHASE_PREFILLING)
        {
            omni_upstream_prefill_t *ps =
                &g_state->prefill_states[req->prefill_upstream_endpoint_idx];

            ngx_atomic_fetch_add(&ps->num_running, -1);
            ngx_atomic_fetch_add(&ps->comm.ref, -1);

            if (ps->num_tokens >= req->metrics.prompt_num_tokens)
            {
                ngx_atomic_fetch_add(&ps->num_tokens, -req->metrics.prompt_num_tokens);
            }
            else
            {
                ps->num_tokens = 0;
            }
            ngx_log_error(NGX_LOG_INFO, omni_get_http_request(req)->connection->log, 0,
                          "[Prefill Release Stats] req=%uD, prompt_tokens=%uD, decoded_tokens=%uD; "
                          "prefill_idx=%uD, num_running=%uD, num_tokens(after)=%uD",
                          req->slot_index,
                          req->metrics.prompt_num_tokens,
                          req->metrics.decoded_tokens,
                          req->prefill_upstream_endpoint_idx,
                          ps->num_running,
                          ps->num_tokens);
        }

        if (phase == PHASE_DECODING)
        {
            omni_upstream_decode_t *ds =
                &g_state->decode_states[req->decode_upstream_endpoint_idx];

            ngx_atomic_fetch_add(&ds->num_running, -1);
            ngx_atomic_fetch_add(&ds->comm.ref, -1);

            if (ds->num_tokens >= req->metrics.prompt_num_tokens + req->metrics.decoded_tokens)
            {
                ngx_atomic_fetch_add(&ds->num_tokens, -(req->metrics.prompt_num_tokens + req->metrics.decoded_tokens));
            }
            else
            {
                ds->num_tokens = 0;
            }
            ngx_log_error(NGX_LOG_INFO, omni_get_http_request(req)->connection->log, 0,
                          "[Decode Release Stats] req=%uD, prompt_tokens=%uD, decoded_tokens=%uD; "
                          "decode_idx=%uD, num_running=%uD, num_tokens(after)=%uD",
                          req->slot_index,
                          req->metrics.prompt_num_tokens,
                          req->metrics.decoded_tokens,
                          req->decode_upstream_endpoint_idx,
                          ds->num_running,
                          ds->num_tokens);
        }
        if (phase == PHASE_PREFILL_WAITING_SCHEDULE) {
            if (req->has_prefill_sched) {
                omni_remove_req_from_group_by_req_index(req->slot_index, &g_state->groups[PHASE_PREFILL_SCHEDULED]);
            } else {
                omni_remove_req_from_group_by_req_index(req->slot_index, &g_state->groups[PHASE_PREFILL_WAITING_SCHEDULE]);
            }
        } else if (phase == PHASE_DECODE_WAITING_SCHEDULE) {
            if (req->has_decode_sched) {
                omni_remove_req_from_group_by_req_index(req->slot_index, &g_state->groups[PHASE_DECODE_SCHEDULED]);
            } else {
                omni_remove_req_from_group_by_req_index(req->slot_index, &g_state->groups[PHASE_DECODE_WAITING_SCHEDULE]);
            }
        } else {
            omni_remove_req_from_group_by_req_index(req->slot_index, &g_state->groups[phase]);
        }
        ngx_shmtx_unlock(&g_state->shmtx);

        omni_remove_req_from_group_by_req_index(req->slot_index, &local_state.groups[phase]);
    }
    local_state.req_in_groups--;
}

static void omni_proxy_main_req_cleanup(void *data)
{
    omni_req_t *req = data;
    omni_proxy_cleanup_req(req);

    ngx_http_request_t *r = omni_get_http_request(req);
    struct timeval tv;
    omni_get_current_time(&tv);
    if (req->metrics.http_status >= 200 && req->metrics.http_status < 300)
    {
        ngx_atomic_fetch_add(&omni_get_global_state()->success_count, 1);
    }
    else
    {
        ngx_atomic_fetch_add(&omni_get_global_state()->failure_count, 1);
    }
    if (req->metrics.decoded_tokens > 0 && (req->metrics.tpot.tv_sec > 0 || req->metrics.tpot.tv_usec > 0))
    {
        omni_metrics_record_tpot(g_state, TIMEVAL_TO_MSEC(req->metrics.tpot));
    }
    if ((req->metrics.time_last_reponse.tv_sec > 0 || req->metrics.time_last_reponse.tv_usec > 0) && (req->metrics.time_received.tv_sec > 0 || req->metrics.time_received.tv_usec > 0) &&
        TIMEVAL_TO_MSEC(req->metrics.time_last_reponse) >= TIMEVAL_TO_MSEC(req->metrics.time_received))
    {
        struct timeval time_diff;
        timersub(&req->metrics.time_last_reponse, &req->metrics.time_received, &time_diff);
        ngx_msec_t e2e_ms = TIMEVAL_TO_MSEC(time_diff);
        omni_metrics_record_e2e(omni_get_global_state(), e2e_ms);
    }
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                "<<<Action: Received all tokens; Timestamp:%d.%06d; RequestID:%s", tv.tv_sec, tv.tv_usec, req->request_id);
    ngx_log_error(NGX_LOG_INFO, omni_get_http_request(req)->connection->log, 0,
                  "[Decode-%d]: Done from %d.",
                  req->slot_index, req->decode_upstream_endpoint_idx);

    omni_req_free(req);
}

static ngx_int_t omni_proxy_handler(ngx_http_request_t *r)
{
    if (r->parent != NULL)
    {
        return NGX_DECLINED;
    }

    omni_req_t *req = omni_req_init(r);
    if (req == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "omni_proxy: allocate omni req failed.");
        return NGX_HTTP_TOO_MANY_REQUESTS;
    }

    ngx_http_omni_loc_conf_t *olcf;

    olcf = ngx_http_get_module_loc_conf(r, ngx_http_omni_proxy_module);
    if (olcf == NULL || olcf->upstream_name.len == 0)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "omni_proxy: no upstream_name in loc conf");
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    g_state->pd_policy = olcf->pd_policy;
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[Prefill-%d]: enter main request handler, policy:%d",
                  req->slot_index, g_state->pd_policy);

    omni_get_current_time(&req->metrics.time_received);

    g_state->decode_pod_size = olcf->decode_pod_size;
    g_state->prefill_pod_size = olcf->prefill_pod_size;

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                    "<<<Action: Time received; Timestamp:%d.%06d; RequestID:%s", req->metrics.time_received.tv_sec, req->metrics.time_received.tv_usec, req->request_id);

    ngx_http_cleanup_t *cleanup = ngx_http_cleanup_add(r, 0);
    cleanup->handler = omni_proxy_main_req_cleanup;
    cleanup->data = req;

    ngx_int_t rc = ngx_http_read_client_request_body(r, omni_proxy_req_body_handler);
    if (rc == NGX_AGAIN)
    {
        return NGX_DONE;
    }
    else if (rc >= NGX_HTTP_SPECIAL_RESPONSE)
    {
        return rc;
    }

    return NGX_DONE;
}

static u_char *omni_json_escape(u_char *dst, const u_char *src, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        switch (src[i]) {
            case '"':
            case '\\':
                *dst++ = '\\';
                *dst++ = src[i];
                break;
            case '\n':
                *dst++ = '\\';
                *dst++ = 'n';
                break;
            case '\r':
                *dst++ = '\\';
                *dst++ = 'r';
                break;
            case '\t':
                *dst++ = '\\';
                *dst++ = 't';
                break;
            case '\b':
                *dst++ = '\\';
                *dst++ = 'b';
                break;
            case '\f':
                *dst++ = '\\';
                *dst++ = 'f';
                break;
            default:
                *dst++ = src[i];
                break;
        }
    }
    return dst;
}

static void omni_proxy_broadcast_send_response(omni_broadcast_ctx_t *ctx)
{
    ngx_http_request_t *r = ctx->request;
    size_t base_size = 1024;
    size_t per_result_size = 256;
    size_t out_cap = base_size + ctx->total * per_result_size;
    out_cap = ngx_max(out_cap, 65536);

    u_char *out = ngx_pnalloc(r->pool, out_cap);
    if (out == NULL) {
        ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return;
    }

    u_char *p = out;
    u_char *end = out + out_cap;
    omni_broadcast_result_t *results = ctx->results ? ctx->results->elts : NULL;

    ngx_uint_t safe_total = (results == NULL) ? 0 : ngx_min(ctx->total, ctx->results->nelts);

    p = ngx_snprintf(p, end - p, "{\"uri\":\"");
    p = omni_json_escape(p, r->uri.data, r->uri.len);
    p = ngx_snprintf(p, end - p, "\",\"results\":[");
    for (ngx_uint_t i = 0; i < safe_total; i++) {
        if (i > 0) {
            p = ngx_snprintf(p, end - p, ",");
        }

        size_t body_len = results[i].body.len;
        size_t escaped_len = (body_len > 0) ? body_len * 4 : 0;
        if ((size_t)(end - p) < escaped_len + 256) {
            size_t needed = escaped_len + 256;
            size_t offset = p - out;
            out_cap += needed - (end - p);
            u_char *new_out = ngx_pnalloc(r->pool, out_cap);
            if (new_out == NULL) {
                ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
                return;
            }
            ngx_memcpy(new_out, out, offset);
            p = new_out + offset;
            out = new_out;
            end = out + out_cap;
        }
        p = ngx_snprintf(p, end - p, "{\"type\":\"%V\",\"index\":%ui,\"address\":\"%V\",\"ok\":%s,\"status\":%ui,\"body\":\"",
                         &results[i].type,
                         results[i].index,
                         &results[i].address,
                         results[i].ok ? "true" : "false",
                         results[i].status);
        if (escaped_len > 0) {
            p = omni_json_escape(p, results[i].body.data, body_len);
        }
        p = ngx_snprintf(p, end - p, "\"}");     
    }

    p = ngx_snprintf(p,
                     end - p,
                     "],\"summary\":{\"total\":%ui,\"success\":%ui,\"failed\":%ui}} \n",
                     ctx->total,
                     ctx->success,
                     ctx->total - ctx->success);

    if (p >= end) {
        ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return;
    }

    r->headers_out.status = NGX_HTTP_OK;
    r->headers_out.content_length_n = p - out;
    ngx_str_set(&r->headers_out.content_type, "application/json");
    r->headers_out.content_type_len = r->headers_out.content_type.len;

    ngx_int_t hrc = ngx_http_send_header(r);
    if (hrc == NGX_ERROR || hrc > NGX_OK || r->header_only) {
        ngx_http_finalize_request(r, hrc);
        return;
    }

    ngx_buf_t *b = ngx_calloc_buf(r->pool);
    if (b == NULL) {
        ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return;
    }

    b->pos = out;
    b->last = p;
    b->memory = 1;
    b->last_buf = 1;
    b->last_in_chain = 1;

    ngx_chain_t chain = {b, NULL};
    ngx_int_t rc = ngx_http_output_filter(r, &chain);
    ngx_http_finalize_request(r, rc == NGX_ERROR ? NGX_ERROR : NGX_OK);
}

static ngx_int_t omni_proxy_broadcast_post_subrequest(ngx_http_request_t *subr, void *data, ngx_int_t rc)
{
    omni_broadcast_subreq_ctx_t *sub_ctx = data;
    omni_broadcast_ctx_t *ctx = sub_ctx->ctx;

    omni_broadcast_result_t *results = ctx->results->elts;
    omni_broadcast_result_t *result = &results[sub_ctx->array_index];

    if (rc != NGX_OK) {
        result->status = NGX_HTTP_BAD_GATEWAY;
        result->ok = 0;
        ngx_log_error(NGX_LOG_INFO, subr->connection->log, 0,
                      "subrequest to %V failed, rc: %i", &result->address, rc);
    } else {
        result->status = subr->headers_out.status;
        if (subr->headers_out.status >= NGX_HTTP_OK && subr->headers_out.status < NGX_HTTP_SPECIAL_RESPONSE) {
            result->ok = 1;
            ctx->success++;
        } else {
            result->ok = 0;
            ngx_log_error(NGX_LOG_INFO, subr->connection->log, 0,
                          "subrequest to %V returned status: %ui", &result->address, subr->headers_out.status);
        }

        size_t body_total = 0;
        ngx_chain_t *cl;
        for (cl = subr->out; cl; cl = cl->next) {
            body_total += ngx_buf_size(cl->buf);
        }

        ngx_str_t *content_type = &subr->headers_out.content_type;
        if (content_type != NULL && content_type->len > 0) {
            result->content_type.data = ngx_pnalloc(ctx->request->pool, content_type->len + 1);
            if (result->content_type.data != NULL) {
                ngx_memcpy(result->content_type.data, content_type->data, content_type->len);
                result->content_type.data[content_type->len] = '\0';
                result->content_type.len = content_type->len;
            }
        }

        if (body_total > 0) {
            result->body.data = ngx_pnalloc(ctx->request->pool, body_total + 1);
            if (result->body.data != NULL) {
                u_char *p = result->body.data;
                for (cl = subr->out; cl; cl = cl->next) {
                    size_t buf_size = ngx_buf_size(cl->buf);
                    if (buf_size > 0) {
                        p = ngx_cpymem(p, cl->buf->pos, buf_size);
                    }
                }
                *p = '\0';
                result->body.len = body_total;
            }
        }
    }

    if (ngx_atomic_fetch_add(&ctx->pending, -1) == 1) {
        omni_proxy_broadcast_send_response(ctx);
    }
    return NGX_OK;
}

static ngx_int_t omni_proxy_broadcast_schedule_subrequest(ngx_http_request_t *r,
                                                          omni_broadcast_ctx_t *ctx,
                                                          ngx_str_t *type,
                                                          ngx_uint_t index,
                                                          omni_upstream_address_t *addr)
{
    omni_broadcast_result_t *result = ngx_array_push(ctx->results);
    if (result == NULL) {
        return NGX_ERROR;
    }
    ngx_memzero(result, sizeof(*result));
    result->type = *type;
    result->index = index;

    result->address.data = ngx_pnalloc(r->pool, addr->text_len + 1);
    if (result->address.data == NULL) {
        return NGX_ERROR;
    }
    ngx_memcpy(result->address.data, addr->text, addr->text_len);
    result->address.data[addr->text_len] = '\0';
    result->address.len = addr->text_len;

    ngx_str_t sub_uri = ngx_string("/omni_proxy_broadcast_sub");
    
    size_t args_len = sizeof("target=") - 1 + addr->text_len;

    u_char *args_buf = ngx_pnalloc(r->pool, args_len + 1);
    if (args_buf == NULL) {
        return NGX_ERROR;
    }

    u_char *ap = args_buf;
    ap = ngx_cpymem(ap, "target=", sizeof("target=") - 1);
    ap = ngx_cpymem(ap, addr->text, addr->text_len);
    *ap = '\0';
    
    ngx_str_t args;
    args.data = args_buf;
    args.len = ap - args_buf;
    
    ngx_http_post_subrequest_t *psr = ngx_pcalloc(r->pool, sizeof(ngx_http_post_subrequest_t));
    omni_broadcast_subreq_ctx_t *sub_ctx = ngx_pcalloc(r->pool, sizeof(omni_broadcast_subreq_ctx_t));
    if (psr == NULL || sub_ctx == NULL) {
        return NGX_ERROR;
    }
    
    sub_ctx->ctx = ctx;
    sub_ctx->array_index = ctx->results->nelts - 1;
    psr->handler = omni_proxy_broadcast_post_subrequest;
    psr->data = sub_ctx;
    
    ngx_http_request_t *sr;
    ngx_int_t rc = ngx_http_subrequest(r, &sub_uri, &args, &sr, psr, NGX_HTTP_SUBREQUEST_IN_MEMORY);
    if (rc != NGX_OK) {
        result->status = NGX_HTTP_INTERNAL_SERVER_ERROR;
        return NGX_ERROR;
    }
    
    sr->method = r->method;
    sr->method_name = r->method_name;
    sr->request_body = r->request_body;
    sr->headers_in.content_length_n = r->headers_in.content_length_n;
    // sr->uri = r->uri;
    sr->unparsed_uri = r->unparsed_uri;
    sr->valid_unparsed_uri = r->valid_unparsed_uri;
    
    ngx_atomic_fetch_add(&ctx->pending, 1);
    return NGX_OK;
}

static void omni_proxy_broadcast_body_handler(ngx_http_request_t *r)
{
    omni_global_state_t *gs = omni_get_global_state();
    if (gs == NULL) {
        ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return;
    }

    omni_req_context_t *req_ctx = ngx_http_get_module_ctx(r, ngx_http_omni_proxy_module);
    if (req_ctx == NULL || req_ctx->broadcast_ctx == NULL) {
        ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return;
    }

    omni_broadcast_ctx_t *ctx = req_ctx->broadcast_ctx;

    ngx_str_t encode = ngx_string("encode");
    ngx_str_t prefill = ngx_string("prefill");
    ngx_str_t decode = ngx_string("decode");

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "Broadcast handler called");

    ngx_shmtx_lock(&gs->shmtx);

    for (ngx_uint_t i = 0, active = 0; i < MAX_ENCODE_UPSTREAMS && active < gs->num_encode_endpoints; i++) {
        if (gs->encode_states[i].comm.status != STATUS_ENABLE) {
            continue;
        }
        active++;

        ctx->total++;
        if (omni_proxy_broadcast_schedule_subrequest(r, ctx, &encode, i, &gs->encode_states[i].comm.address) != NGX_OK) {
            omni_broadcast_result_t *res = ngx_array_push(ctx->results);
            if (res != NULL) {
                ngx_memzero(res, sizeof(*res));
                res->ok = 0;
                res->status = NGX_HTTP_INTERNAL_SERVER_ERROR;
            }
        }
    }

    for (ngx_uint_t i = 0, active = 0; i < MAX_PREFILL_UPSTREAMS && active < gs->num_prefill_endpoints; i++) {
        if (gs->prefill_states[i].comm.status != STATUS_ENABLE) {
            continue;
        }
        active++;

        ctx->total++;
        if (omni_proxy_broadcast_schedule_subrequest(r, ctx, &prefill, i, &gs->prefill_states[i].comm.address) != NGX_OK) {
            omni_broadcast_result_t *res = ngx_array_push(ctx->results);
            if (res != NULL) {
                ngx_memzero(res, sizeof(*res));
                res->ok = 0;
                res->status = NGX_HTTP_INTERNAL_SERVER_ERROR;
            }
        }
    }

    for (ngx_uint_t i = 0, active = 0; i < MAX_DECODE_UPSTREAMS && active < gs->num_decode_endpoints; i++) {
        if (gs->decode_states[i].comm.status != STATUS_ENABLE) {
            continue;
        }
        active++;

        ctx->total++;
        if (omni_proxy_broadcast_schedule_subrequest(r, ctx, &decode, i, &gs->decode_states[i].comm.address) != NGX_OK) {
            omni_broadcast_result_t *res = ngx_array_push(ctx->results);
            if (res != NULL) {
                ngx_memzero(res, sizeof(*res));
                res->ok = 0;
                res->status = NGX_HTTP_INTERNAL_SERVER_ERROR;
            }
        }
    }
    ngx_shmtx_unlock(&gs->shmtx);
    if (ctx->pending == 0) {
        omni_proxy_broadcast_send_response(ctx);
        return;
    }

    ngx_http_run_posted_requests(r->connection);
}

static ngx_int_t omni_proxy_broadcast_handler(ngx_http_request_t *r)
{
    if (r->method != NGX_HTTP_POST && r->method != NGX_HTTP_GET) {
        return NGX_HTTP_NOT_ALLOWED;
    }

    omni_req_context_t *req_ctx = ngx_http_get_module_ctx(r, ngx_http_omni_proxy_module);
    if (req_ctx == NULL) {
        req_ctx = ngx_pcalloc(r->pool, sizeof(omni_req_context_t));
        if (req_ctx == NULL) {
            return NGX_HTTP_INTERNAL_SERVER_ERROR;
        }
        ngx_http_set_ctx(r, req_ctx, ngx_http_omni_proxy_module);
    }

    req_ctx->broadcast_ctx = ngx_pcalloc(r->pool, sizeof(omni_broadcast_ctx_t));
    if (req_ctx->broadcast_ctx == NULL) {
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    omni_broadcast_ctx_t *ctx = req_ctx->broadcast_ctx;
    ctx->request = r;
    ctx->pending = 0;
    ctx->total = 0;
    ctx->success = 0;
    omni_global_state_t *gs = omni_get_global_state();
    if (gs == NULL) {
        ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    ngx_shmtx_lock(&gs->shmtx);

    ngx_uint_t max_results = gs->num_prefill_endpoints + gs->num_decode_endpoints;

    ngx_shmtx_unlock(&gs->shmtx);

    ctx->results = ngx_array_create(r->pool, max_results, sizeof(omni_broadcast_result_t));
    if (ctx->results == NULL) {
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    ngx_int_t rc = ngx_http_read_client_request_body(r, omni_proxy_broadcast_body_handler);
    if (rc == NGX_AGAIN) {
        return NGX_DONE;
    }
    if (rc >= NGX_HTTP_SPECIAL_RESPONSE) {
        return rc;
    }
    return NGX_DONE;
}

static ngx_int_t ngx_http_prefill_post_subrequest(ngx_http_request_t *subr, void *data, ngx_int_t rc)
{
    omni_req_t *req = data;
    ngx_chain_t *cl;
    size_t total = 0;
    u_char *p;
    ngx_http_request_t *r = omni_get_http_request(req);
    omni_req_context_t *ctx = omni_get_req_ctx(subr);

    omni_get_current_time(&req->metrics.time_prefilled);
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                    "<<<Action: Finish P running; Timestamp:%d.%06d; RequestID:%s", req->metrics.time_prefilled.tv_sec, req->metrics.time_prefilled.tv_usec, req->request_id);

    ngx_connection_t *c = r->connection;

    ctx = ngx_http_get_module_ctx(r, ngx_http_omni_proxy_module);

    if (c->read->timer_set)
    {
        ngx_del_timer(c->read);
    }
    if (c->write->timer_set)
    {
        ngx_del_timer(c->write);
    }

    r->read_event_handler = ngx_http_request_empty_handler;
    r->write_event_handler = ngx_http_request_empty_handler;

    if (subr == NULL || subr->headers_out.headers.part.elts == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "Upstream headers_in is NULL!");
    } else {
        ngx_list_part_t *part = &subr->headers_out.headers.part;
        ngx_table_elt_t *header = part->elts;
        ngx_uint_t i;
        for (i = 0; /* void */; i++) {
            if (i >= part->nelts) {
                if (part->next == NULL) break;
                part = part->next;
                header = part->elts;
                i = 0;
            }
            /*update Traceparent from P node*/
            if (header[i].key.len == sizeof(x_trace_headers)-1 &&
            ngx_strncasecmp(header[i].key.data, x_trace_headers, sizeof(x_trace_headers)-1) == 0){
                ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "****Find Traceparent return from P****; Traceparent:%s", header[i].value.data);
                u_char *trace_value = header[i].value.data;
                size_t trace_value_len = header[i].value.len;
                u_char *saved = ngx_pnalloc(r->pool, trace_value_len);
                if (saved) {ngx_memcpy(saved, trace_value, trace_value_len);}
                update_traceparent_header(r, x_trace_headers, saved, trace_value_len);
            }
            /*update Start_time_ns from P node*/
            if (header[i].key.len == sizeof(x_start_time_ns)-1 &&
            ngx_strncasecmp(header[i].key.data, x_start_time_ns, sizeof(x_start_time_ns)-1) == 0){
                ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "****Find Start_time_ns return from P****; Start_time_ns:%s", header[i].value.data);
                u_char *trace_value = header[i].value.data;
                size_t trace_value_len = header[i].value.len;
                u_char *saved = ngx_pnalloc(r->pool, trace_value_len);
                if (saved) {ngx_memcpy(saved, trace_value, trace_value_len);}
                update_traceparent_header(r, x_start_time_ns, saved, trace_value_len);
            }
        }
    }

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[Prefill-%d] Done from %d", req->slot_index, req->prefill_upstream_endpoint_idx);

    if (rc != NGX_OK)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "[Prefill-%d] subrequest failed with code %i", req->slot_index, rc);
        ngx_http_finalize_request(r->main, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return rc;
    }

    for (cl = subr->out; cl; cl = cl->next)
    {
        total += ngx_buf_size(cl->buf);
    }

    ctx->prefill_response_body = ngx_palloc(r->main->pool, total + 1);
    if (ctx->prefill_response_body == NULL)
    {
        ngx_http_finalize_request(r, NGX_ERROR);
        ngx_log_error(
            NGX_LOG_ERR, r->connection->log, 0,
            "done prefill subrequest, malloc prefill response body buffer failed");
        return rc;
    }

    p = ctx->prefill_response_body;
    for (cl = subr->out; cl; cl = cl->next)
    {
        size_t buf_size = ngx_buf_size(cl->buf);
        if (buf_size > 0)
        {
            p = ngx_cpymem(p, cl->buf->pos, buf_size);
        }
    }
    *p = '\0';
    ctx->prefill_response_body_size = total;

    omni_upstream_prefill_t *us = &g_state->prefill_states[req->prefill_upstream_endpoint_idx];

    if (subr->headers_out.status < NGX_HTTP_OK ||
        subr->headers_out.status >= NGX_HTTP_SPECIAL_RESPONSE)
    {
        ngx_log_error(NGX_LOG_INFO,
                      r->connection->log,
                      0,
                      "[Prefill-%d] subrequest returned non-success status:%i",
                      req->slot_index,
                      subr->headers_out.status);

        r->headers_out.status = subr->headers_out.status;
        r->headers_out.content_length_n = ctx->prefill_response_body_size;
        r->headers_out.content_type = subr->headers_out.content_type;
        r->headers_out.content_type_len = subr->headers_out.content_type_len;

        ngx_int_t send_rc = ngx_http_send_header(r);
        if (send_rc == NGX_ERROR || send_rc > NGX_OK)
        {
            ngx_http_finalize_request(r, send_rc);
            return NGX_DONE;
        }

        ngx_buf_t *b = ngx_calloc_buf(r->pool);
        if (b == NULL)
        {
            ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
            return NGX_DONE;
        }

        b->pos = ctx->prefill_response_body;
        b->last = ctx->prefill_response_body + ctx->prefill_response_body_size;
        b->memory = 1;
        b->last_buf = 1;
        b->last_in_chain = 1;

        ngx_chain_t out;
        out.buf = b;
        out.next = NULL;

        ngx_int_t body_rc = ngx_http_output_filter(r, &out);
        ngx_http_finalize_request(r, (body_rc == NGX_ERROR) ? NGX_ERROR : NGX_OK);
        return NGX_DONE;
    }

    // Check if kv_transfer_params is null or absent in prefill response
    // If so, return prefill response directly to client without forwarding to decode
    ngx_int_t kv_null = omni_check_kv_transfer_params_null(r,
        (char *)ctx->prefill_response_body, ctx->prefill_response_body_size, ctx);
    if (kv_null == KV_TRANSFER_NULL || kv_null == KV_TRANSFER_ABSENT || kv_null == KV_TRANSFER_MISSING_FIELDS)
    {
        ngx_log_error(NGX_LOG_WARN, r->connection->log, 0,
                      "[Prefill-%d] kv_transfer_params is %s, returning prefill response directly to client",
                      req->slot_index,
                      kv_null == KV_TRANSFER_NULL ? "null" :
                      (kv_null == KV_TRANSFER_ABSENT ? "absent" : "missing required fields"));

        // If kv_transfer_params is null or missing required fields, remove it from response body
        char *response_body = (char *)ctx->prefill_response_body;
        size_t response_body_size = ctx->prefill_response_body_size;

        char *new_body = NULL;
        ngx_int_t new_size = -1;
        if (kv_null == KV_TRANSFER_NULL || kv_null == KV_TRANSFER_MISSING_FIELDS)
        {
            new_size = omni_remove_kv_transfer_params_null(r,
                (char *)ctx->prefill_response_body, ctx->prefill_response_body_size, &new_body, ctx);
            if (new_size > 0 && new_body != NULL)
            {
                response_body = new_body;
                response_body_size = (size_t)new_size;
            }
        }

        r->headers_out.status = NGX_HTTP_OK;
        r->headers_out.content_length_n = response_body_size;
        r->headers_out.content_type = subr->headers_out.content_type;
        r->headers_out.content_type_len = subr->headers_out.content_type_len;

        ngx_int_t send_rc = ngx_http_send_header(r);
        if (send_rc == NGX_ERROR || send_rc > NGX_OK)
        {
            ngx_http_finalize_request(r, send_rc);
            return NGX_DONE;
        }

        ngx_buf_t *b = ngx_create_temp_buf(r->pool, response_body_size);
        if (b == NULL)
        {
            ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
            return NGX_DONE;
        }

        ngx_memcpy(b->pos, response_body, response_body_size);
        b->last = b->pos + response_body_size;
        b->last_buf = 1;
        b->last_in_chain = 1;

        ngx_chain_t out;
        out.buf = b;
        out.next = NULL;

        ngx_int_t body_rc = ngx_http_output_filter(r, &out);
        ngx_http_finalize_request(r, (body_rc == NGX_ERROR) ? NGX_ERROR : NGX_OK);
        return NGX_DONE;
    }

    omni_batch_metrics_t *current_batch = &us->his.his[us->his.head];
    omni_batch_metrics_t *request_batch = current_batch;
    ngx_msec_t response_time = ngx_current_msec;
    ngx_msec_t schedule_time = TIMEVAL_TO_MSEC(req->metrics.time_to_prefill);
    ngx_msec_t delta = (current_batch->last_response_receive_time > 0) ?
                         (response_time - current_batch->last_response_receive_time) :
                         (21); // If first，force delta > 20 to get a new batch

    // Need a smarter value from statistics or work out by the number of tokens scheduled
    if (delta > 20)
    {
        if (current_batch->last_response_receive_time == 0) {
            us->num_batch_exec = us->num_queue;
            us->num_queue = 0;
        }
        // 1. **calculate last batch**
        if (current_batch->num_requests > 0)
        { // not a empty batch
            if (current_batch->first_schedule_sent_time > 0 &&
                current_batch->last_response_receive_time >= current_batch->first_schedule_sent_time)
            {
                current_batch->time_taken =
                    current_batch->last_response_receive_time - current_batch->first_schedule_sent_time;
            }

            ngx_msec_t gap = 0;
            if (current_batch->last_schedule_sent_time > 0 &&
                current_batch->first_response_receive_time >= current_batch->last_schedule_sent_time)
            {
                gap = current_batch->first_response_receive_time - current_batch->last_schedule_sent_time;
            }


            ngx_log_error(NGX_LOG_INFO, ngx_cycle->log,
                          0,
                          "[Prefill-Batch-End] Batch head=%ui duration=%M first_schedule=%M last_schedule=%M first_response=%M last_response=%M gap=%M tokens=%ui batch_exec=%ui queue=%ui",
                          us->his.head,
                          current_batch->time_taken,
                          current_batch->first_schedule_sent_time,
                          current_batch->last_schedule_sent_time,
                          current_batch->first_response_receive_time,
                          current_batch->last_response_receive_time,
                          gap,
                          current_batch->num_tokens,
                          us->num_batch_exec,
                          us->num_queue);

            if (gap > 0)
            {
                omni_scheduler_record_prefill_batch_stat(g_state, current_batch->num_requests, gap);
            }

            if (current_batch->num_requests == 1 && us->idle_batch == true)
            {
                //omni_scheduler_record_prefill_batch_stat(g_state, current_batch->num_requests, gap);
                us->idle_batch = false;
            }

            us->num_batch_exec = us->num_queue;
            us->num_queue = 0;
        }

        // 2. **start a new batch**
        if (us->his.count < NUM_PREFILL_BATCH_METRICS_HIS) {
            us->his.count++;
        }
        us->his.head = (us->his.head + 1) % NUM_PREFILL_BATCH_METRICS_HIS;

        omni_batch_metrics_t *new_batch = &us->his.his[us->his.head];
        ngx_memzero(new_batch, sizeof(omni_batch_metrics_t));

        new_batch->first_response_receive_time = response_time;
        new_batch->last_response_receive_time = response_time;
        new_batch->num_requests = 1;
        new_batch->num_tokens = req->metrics.prompt_num_tokens;
        if (schedule_time > 0 && response_time >= schedule_time)
        {
            new_batch->time_taken = response_time - schedule_time;
        }
        new_batch->first_schedule_sent_time = schedule_time;
        new_batch->last_schedule_sent_time = schedule_time;
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[Prefill-Batch-Metrics] head=%ui last_response_receive_time=%M num_requests=%ui num_tokens=%ui first_schedule=%M last_schedule=%M time_taken=%M batch_exec=%ui queue=%ui",
                      us->his.head,
                      new_batch->last_response_receive_time,
                      new_batch->num_requests,
                      new_batch->num_tokens,
                      new_batch->first_schedule_sent_time,
                      new_batch->last_schedule_sent_time,
                      new_batch->time_taken,
                      (ngx_uint_t)us->num_batch_exec,
                      (ngx_uint_t)us->num_queue);
    }
    else
    {
        // **add to current batch**
        current_batch->last_response_receive_time = response_time;
        current_batch->num_requests++;
        current_batch->num_tokens += req->metrics.prompt_num_tokens;
        // average_delta
        current_batch->average_delta = current_batch->average_delta * (current_batch->num_requests - 1) +
                                       delta / (current_batch->num_requests);
        if (current_batch->first_schedule_sent_time == 0 ||
            schedule_time < current_batch->first_schedule_sent_time)
        {
            current_batch->first_schedule_sent_time = schedule_time;
        }
        if (schedule_time > current_batch->last_schedule_sent_time)
        {
            current_batch->last_schedule_sent_time = schedule_time;
        }
        if (current_batch->first_schedule_sent_time > 0 &&
            current_batch->last_response_receive_time >= current_batch->first_schedule_sent_time)
        {
            current_batch->time_taken =
                current_batch->last_response_receive_time - current_batch->first_schedule_sent_time;
        }
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[Prefill-Batch-Metrics] head=%ui last_response_receive_time=%M num_requests=%ui num_tokens=%ui first_schedule=%M last_schedule=%M time_taken=%M batch_exec=%ui queue=%ui",
                      us->his.head,
                      current_batch->last_response_receive_time,
                      current_batch->num_requests,
                      current_batch->num_tokens,
                      current_batch->first_schedule_sent_time,
                      current_batch->last_schedule_sent_time,
                      current_batch->time_taken,
                      us->num_batch_exec,
                      us->num_queue);
    }

    ngx_atomic_fetch_add(&us->num_running, -1);
    ngx_atomic_fetch_add(&us->comm.ref, -1);
    ngx_atomic_fetch_add(&us->num_tokens, -req->metrics.prompt_num_tokens);

    // check policy
    if (g_state->pd_policy == PD_SEQUENTIAL) {
        omni_phase_transition_all(req, PHASE_PREFILLING, PHASE_DECODE_WAITING_SCHEDULE);
        omni_get_current_time(&req->metrics.time_enter_wait_decode);
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                    "<<<Action: Enter state D waiting; Timestamp:%d.%06d; RequestID:%s", req->metrics.time_enter_wait_decode.tv_sec, req->metrics.time_enter_wait_decode.tv_usec, req->request_id);
    } else {
        omni_remove_req_from_group_by_req_index(req->slot_index, &omni_get_local_state()->groups[PHASE_PREFILLING]);

        ngx_shmtx_lock(&g_state->shmtx);
        omni_remove_req_from_group_by_req_index(req->slot_index, &omni_get_global_state()->groups[PHASE_PREFILLING]);
        omni_req_leave_phase(req, PHASE_PREFILLING);
        ngx_shmtx_unlock(&g_state->shmtx);
    }

    return NGX_DONE;
}

static ngx_int_t omni_proxy_prepare_bootstrap_info(ngx_http_request_t *r, omni_req_context_t *ctx)
{
    omni_req_t *req = ctx->req;
    omni_global_state_t *gs = omni_get_global_state();

    ctx->bootstrap_host.data = NULL;
    ctx->bootstrap_host.len = 0;
    ctx->bootstrap_port.data = NULL;
    ctx->bootstrap_port.len = 0;
    ctx->bootstrap_room.data = NULL;
    ctx->bootstrap_room.len = 0;

    if (req == NULL || gs == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "prefill: missing request or global state when preparing bootstrap info");
        return NGX_ERROR;
    }

    if (req->prefill_upstream_endpoint_idx >= MAX_PREFILL_UPSTREAMS) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "prefill: invalid prefill upstream index %ui when preparing bootstrap info",
                      req->prefill_upstream_endpoint_idx);
        return NGX_ERROR;
    }

    omni_upstream_prefill_t *prefill = &gs->prefill_states[req->prefill_upstream_endpoint_idx];
    const char *ip = prefill->comm.address.ip;
    if (ip != NULL && ip[0] != '\0') {
        size_t len = ngx_strlen(ip);
        ctx->bootstrap_host.data = ngx_pnalloc(r->pool, len + 1);
        if (ctx->bootstrap_host.data == NULL)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                          "prefill: failed to allocate bootstrap host buffer");
            return NGX_ERROR;
        }
        ngx_memcpy(ctx->bootstrap_host.data, ip, len);
        ctx->bootstrap_host.data[len] = '\0';
        ctx->bootstrap_host.len = len;
    } else {
        ctx->bootstrap_host.data = (u_char *)"";
        ctx->bootstrap_host.len = 0;
    }

    if (prefill->comm.address.port > 0) {
        ctx->bootstrap_port.data = ngx_pnalloc(r->pool, NGX_INT_T_LEN + 1);
        if (ctx->bootstrap_port.data == NULL) {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                          "prefill: failed to allocate bootstrap port buffer");
            return NGX_ERROR;
        }
        /* get port from setting later */
        ctx->bootstrap_port.len = ngx_sprintf(ctx->bootstrap_port.data, "%d", 8998)
                                   - ctx->bootstrap_port.data;
        ctx->bootstrap_port.data[ctx->bootstrap_port.len] = '\0';
    } else {
        ctx->bootstrap_port.data = NULL;
        ctx->bootstrap_port.len = 0;
    }

    ctx->bootstrap_room.data = ngx_pnalloc(r->pool, NGX_INT64_LEN + 1);
    if (ctx->bootstrap_room.data == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "prefill: failed to allocate bootstrap room buffer");
        return NGX_ERROR;
    }
    uint64_t room_id = (((uint64_t)ngx_random()) << 32) | (uint64_t)ngx_random();
    room_id &= 0x7FFFFFFFFFFFFFFFULL;
    ctx->bootstrap_room.len = ngx_sprintf(ctx->bootstrap_room.data, "%uL", room_id)
                              - ctx->bootstrap_room.data;
    ctx->bootstrap_room.data[ctx->bootstrap_room.len] = '\0';

    return NGX_OK;
}

static ngx_int_t ngx_http_prefill_wakeup(omni_req_t *req)
{
    ngx_http_request_t *r = omni_get_http_request(req);
    omni_req_context_t *ctx = omni_get_req_ctx(r);

    if (g_state->pd_policy == PD_PARALLEL) {
        if (omni_proxy_prepare_bootstrap_info(r, ctx) != NGX_OK) {
            return NGX_HTTP_INTERNAL_SERVER_ERROR;
        }
        omni_add_req_to_group(req->slot_index, &omni_get_local_state()->groups[PHASE_DECODE_WAITING_SCHEDULE]);

        ngx_shmtx_lock(&g_state->shmtx);
        omni_add_req_to_group(req->slot_index, &omni_get_global_state()->groups[PHASE_DECODE_WAITING_SCHEDULE]);
        omni_req_enter_phase(req, PHASE_DECODE_WAITING_SCHEDULE);
        ngx_shmtx_unlock(&g_state->shmtx);
        struct timeval tv;
        omni_get_current_time(&tv);
        req->metrics.time_enter_wait_decode = tv;

        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                    "<<<Action: Enter state D waiting; Timestamp:%d.%06d; RequestID:%s", tv.tv_sec, tv.tv_usec, req->request_id);
    }

    size_t prelen = PREFILL_URI_LEN + r->uri.len;
    if (r->args.len)
    {
        prelen += 1 + r->args.len;
    }
    u_char *prefill_uri = ngx_pnalloc(r->pool, prelen + 1);
    if (prefill_uri == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "omni_proxy: allocate prefill uri failed.");
        ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    u_char *p = prefill_uri;
    p = ngx_cpymem(p, PREFILL_URI, PREFILL_URI_LEN);
    p = ngx_cpymem(p, r->uri.data, r->uri.len);
    if (r->args.len)
    {
        *p++ = '?';
        p = ngx_cpymem(p, r->args.data, r->args.len);
    }
    *p = '\0';

    ngx_str_t sub_uri;
    sub_uri.len = ngx_strlen(prefill_uri);
    sub_uri.data = prefill_uri;

    ngx_http_post_subrequest_t *psr = ngx_pcalloc(r->pool, sizeof(ngx_http_post_subrequest_t));
    psr->handler = ngx_http_prefill_post_subrequest;
    psr->data = req;

    ngx_str_t args = ngx_null_string;
    ngx_http_request_t *sr;

    ngx_int_t rc = ngx_http_subrequest(r, &sub_uri, &args, &sr, psr,
                                       NGX_HTTP_SUBREQUEST_IN_MEMORY);
    if (rc != NGX_OK)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "omni_proxy: create subrequest failed.");
        ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
        return rc;
    }
    sr->method = r->method;
    sr->method_name = r->method_name;

    ngx_http_set_ctx(sr, ctx, ngx_http_omni_proxy_module);

    omni_proxy_prepare_prefill_subrequest(r, sr, ctx);

    omni_phase_transition_all(req, PHASE_PREFILL_SCHEDULED, PHASE_PREFILLING);

    omni_get_current_time(&req->metrics.time_to_prefill);

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                    "<<<Action: Enter state P running; Timestamp:%d.%06d; RequestID:%s", req->metrics.time_to_prefill.tv_sec, req->metrics.time_to_prefill.tv_usec, req->request_id);

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Prefill-%d] Submit to:%d",
                  req->slot_index, req->prefill_upstream_endpoint_idx);

    // ngx_http_core_run_phases(sr);
    ngx_http_run_posted_requests(r->connection);

    return NGX_OK;
}

static void omni_proxy_req_prefill_resched(omni_req_t *req, ngx_uint_t new_idx)
{
    ngx_uint_t old_idx = req->prefill_upstream_endpoint_idx;

    g_state->prefill_states[old_idx].num_running--;
    g_state->prefill_states[old_idx].num_tokens -= req->metrics.prompt_num_tokens;
    ngx_atomic_fetch_add(&g_state->prefill_states[old_idx].comm.ref, -1);

    g_state->prefill_states[new_idx].num_running++;
    g_state->prefill_states[new_idx].num_tokens += req->metrics.prompt_num_tokens;

    req->prefill_upstream_endpoint_idx = new_idx;
    req->prefill_group_id = g_state->prefill_states[new_idx].comm.group_id;
    ngx_atomic_fetch_add(&g_state->prefill_states[new_idx].comm.ref, 1);
}

static ngx_http_upstream_rr_peer_t *omni_proxy_req_prefill_resched_by_peer(ngx_http_upstream_rr_peers_t *peers,
    ngx_http_request_t *r)
{
    omni_req_t *req = omni_get_req(r);
    ngx_http_upstream_rr_peer_t *peer;
    for (peer = peers->peer; peer != NULL; peer = peer->next) {
        ngx_shmtx_lock(&g_state->shmtx);
        for (uint32_t j = 0; j < MAX_PREFILL_UPSTREAMS; j++) {
            omni_upstream_common_t *comm = &g_state->prefill_states[j].comm;
            if (comm->status == STATUS_ABANDON) {
                continue;
            }
            if (comm->status == STATUS_UNUSED) {
                break;
            }
            ngx_http_upstream_rr_peer_lock(peers, peer);
            if (ngx_cmp_sockaddr(peer->sockaddr, peer->socklen, &comm->address.sockaddr, comm->address.socklen, 1) == NGX_OK) {
                ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "Prefill resched from %ui to %ui",
                    req->prefill_upstream_endpoint_idx, j);
                omni_proxy_req_prefill_resched(req, j);
                ngx_http_upstream_rr_peer_ref(peers, peer);
                ngx_http_upstream_rr_peer_unlock(peers, peer);
                ngx_shmtx_unlock(&g_state->shmtx);
                return peer;
            }
            ngx_http_upstream_rr_peer_unlock(peers, peer);
        }
        ngx_shmtx_unlock(&g_state->shmtx);
    }
    return NULL;
}

static ngx_int_t omni_proxy_get_peer(ngx_peer_connection_t *pc,
                                     void *data)
{
    ngx_http_upstream_rr_peer_data_t *rrp = data;
    ngx_http_request_t *r = (ngx_http_request_t *)rrp->data;
    omni_req_t *req = omni_get_req(r);
    ngx_http_upstream_rr_peers_t *peers = rrp->peers;
    ngx_http_upstream_rr_peer_t *peer;

    assert(omni_req_is_in_phase(req, PHASE_PREFILLING));

    ngx_http_upstream_rr_peers_rlock(peers);

    omni_upstream_common_t *comm = &g_state->prefill_states[req->prefill_upstream_endpoint_idx].comm;
    for (peer = peers->peer; peer != NULL; peer = peer->next) {
        ngx_http_upstream_rr_peer_lock(peers, peer);
        if (ngx_cmp_sockaddr(peer->sockaddr, peer->socklen, &comm->address.sockaddr, comm->address.socklen, 1) == NGX_OK) {
            pc->sockaddr = peer->sockaddr;
            pc->socklen = peer->socklen;
            pc->name = &peer->name;
            rrp->current = peer;
            ngx_http_upstream_rr_peer_ref(peers, peer);

            ngx_http_upstream_rr_peer_unlock(peers, peer);
            break;
        }
        ngx_http_upstream_rr_peer_unlock(peers, peer);
    }

    if (peer == NULL) {
        peer = omni_proxy_req_prefill_resched_by_peer(peers, r);
        if (peer == NULL) {
            ngx_http_upstream_rr_peers_unlock(peers);
            ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "Prefill %ui req %s:%d no peer match",
                req->prefill_upstream_endpoint_idx, comm->address.ip, comm->address.port);
            return NGX_ERROR;
        }
        pc->sockaddr = peer->sockaddr;
        pc->socklen = peer->socklen;
        pc->name = &peer->name;
        rrp->current = peer;
    }

    ngx_http_upstream_rr_peers_unlock(peers);

    ngx_uint_t idx = req->prefill_upstream_endpoint_idx;
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[Prefill-%d] Upstream set: %d, running:%d",
                  req->slot_index, idx, g_state->prefill_states[idx].num_running);

    return NGX_OK;
}

static ngx_int_t omni_proxy_get_encode_peer(ngx_peer_connection_t *pc,
                                            void *data)
{
    ngx_http_upstream_rr_peer_data_t *rrp = data;
    ngx_http_request_t *r = (ngx_http_request_t *)rrp->data;
    omni_req_t *req = omni_get_req(r);
    ngx_http_upstream_rr_peers_t *peers = rrp->peers;
    ngx_http_upstream_rr_peer_t *peer;

    if (req->encode_upstream_endpoint_idx >= MAX_ENCODE_UPSTREAMS) {
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                      "Encode req %uD invalid upstream index %uD",
                      req->slot_index,
                      req->encode_upstream_endpoint_idx);
        return NGX_ERROR;
    }

    ngx_http_upstream_rr_peers_rlock(peers);

    omni_upstream_common_t *comm = &g_state->encode_states[req->encode_upstream_endpoint_idx].comm;
    for (peer = peers->peer; peer != NULL; peer = peer->next) {
        ngx_http_upstream_rr_peer_lock(peers, peer);
        if (ngx_cmp_sockaddr(peer->sockaddr, peer->socklen, &comm->address.sockaddr, comm->address.socklen, 1) == NGX_OK) {
            pc->sockaddr = peer->sockaddr;
            pc->socklen = peer->socklen;
            pc->name = &peer->name;
            rrp->current = peer;
            ngx_http_upstream_rr_peer_ref(peers, peer);

            ngx_http_upstream_rr_peer_unlock(peers, peer);
            break;
        }
        ngx_http_upstream_rr_peer_unlock(peers, peer);
    }

    ngx_http_upstream_rr_peers_unlock(peers);

    if (peer == NULL) {
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                      "Encode req %uD no peer match %s:%d",
                      req->slot_index,
                      comm->address.ip,
                      comm->address.port);
        return NGX_ERROR;
    }

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "[Encoder-%d] Upstream set: %d, running:%d",
                  req->slot_index,
                  req->encode_upstream_endpoint_idx,
                  g_state->encode_states[req->encode_upstream_endpoint_idx].num_running);

    return NGX_OK;
}

static ngx_int_t omni_proxy_upstream_init(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf)
{
    omni_req_context_t *ctx = ngx_http_get_module_ctx(r, ngx_http_omni_proxy_module);
    if (ctx == NULL || ctx->req == NULL) {
        return ngx_http_upstream_init_round_robin_peer(r, uscf);
    }
    omni_req_t *req = omni_get_req(r);
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[-%d] init upstream at phase: %u",
                  req->slot_index, req->phase_state);

    ngx_http_upstream_t *u = r->upstream;
    u->conf->send_lowat = 0;
    ngx_http_upstream_rr_peer_data_t *rrp;
    if (ngx_http_upstream_init_round_robin_peer(r, uscf) != NGX_OK)
    {
        return NGX_ERROR;
    }
    rrp = u->peer.data;

    u->peer.get = omni_proxy_get_peer;
    u->peer.data = rrp;
    rrp->data = (uintptr_t)r;

    return NGX_OK;
}

static ngx_int_t omni_proxy_upstream_init_encode(ngx_http_request_t *r, ngx_http_upstream_srv_conf_t *uscf)
{
    ngx_http_upstream_t *u = r->upstream;
    u->conf->send_lowat = 0;
    ngx_http_upstream_rr_peer_data_t *rrp;
    if (ngx_http_upstream_init_round_robin_peer(r, uscf) != NGX_OK)
    {
        return NGX_ERROR;
    }
    rrp = u->peer.data;

    u->peer.get = omni_proxy_get_encode_peer;
    u->peer.data = rrp;
    rrp->data = (uintptr_t)r;

    return NGX_OK;
}

static void omni_proxy_update_decode_stats(ngx_http_request_t *r, ngx_buf_t *buf, ssize_t bytes)
{
    omni_req_t *req = omni_get_req(r);

    // Update request level statistics

    if (req->metrics.time_last_reponse.tv_sec > 0 || req->metrics.time_last_reponse.tv_usec > 0)
    {
        req->metrics.decoded_tokens++;
        ngx_uint_t decode_idx = req->decode_upstream_endpoint_idx;
        ngx_atomic_fetch_add(&g_state->decode_states[decode_idx].num_tokens, 1);

        if (req->metrics.decoded_tokens != 0)
        {
            struct timeval tv;
            omni_get_current_time(&tv);
            int64_t total_usec = (((req->metrics.tpot.tv_sec * 1000000 + req->metrics.tpot.tv_usec) * (req->metrics.decoded_tokens - 1)) + (tv.tv_sec * 1000000 + tv.tv_usec) - (req->metrics.time_last_reponse.tv_sec * 1000000 + req->metrics.time_last_reponse.tv_usec)) /
            req->metrics.decoded_tokens;
            req->metrics.tpot.tv_sec = total_usec / 1000000;
            req->metrics.tpot.tv_usec = total_usec % 1000000;

        }
        if (req->metrics.decoded_tokens == 2){
            struct timeval tv;
            omni_get_current_time(&tv);
            ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                        "<<<Action: Proxy got second token; Timestamp:%d.%06d; RequestID:%s", tv.tv_sec, tv.tv_usec, req->request_id);
        }else if (req->metrics.decoded_tokens == 3){
            struct timeval tv;
            omni_get_current_time(&tv);
            ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                        "<<<Action: Proxy got third token; Timestamp:%d.%06d; RequestID:%s", tv.tv_sec, tv.tv_usec, req->request_id);
        }
        ngx_log_error(NGX_LOG_DEBUG, r->connection->log, 0,
                      "[Decode Update Stats]  req %d prompt_tokens= %ui decoded_tokens= %ui; decode upstream %d num_tokens=%ui",
                      req->slot_index,
                      req->metrics.prompt_num_tokens,
                      req->metrics.decoded_tokens,
                      decode_idx,
                      (unsigned long long)g_state->decode_states[decode_idx].num_tokens);
    }
    else
    {
        struct timeval tv;
        omni_get_current_time(&tv);
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                    "<<<Action: Proxy got first token; Timestamp:%d.%06d; RequestID:%s", tv.tv_sec, tv.tv_usec, req->request_id);

        timersub(&tv, &req->metrics.time_received, &req->metrics.ttft);
        timersub(&tv, &req->metrics.time_to_decode, &req->metrics.tpot);
        req->metrics.time_first_token = req->metrics.tpot;
        req->metrics.decoded_tokens++;
        omni_metrics_record_ttft(g_state, TIMEVAL_TO_MSEC(req->metrics.ttft));

        ngx_uint_t decode_idx = req->decode_upstream_endpoint_idx;
        ngx_atomic_fetch_add(&g_state->decode_states[decode_idx].num_tokens, 1);

        ngx_log_error(NGX_LOG_DEBUG, r->connection->log, 0,
                      "[Decode Update Stats]  req %d prompt_tokens= %ui decoded_tokens= %ui; decode upstream %d num_tokens=%ui",
                      req->slot_index,
                      req->metrics.prompt_num_tokens,
                      req->metrics.decoded_tokens,
                      decode_idx,
                      (unsigned long long)g_state->decode_states[decode_idx].num_tokens);
    }

    omni_get_current_time(&req->metrics.time_last_reponse);

    omni_upstream_decode_t *us = &g_state->decode_states[req->decode_upstream_endpoint_idx];
    // Update batch level statistics
    omni_batch_metrics_t *batch = &us->his.his[us->his.head];
    ngx_msec_t delta = ngx_current_msec - batch->last_response_receive_time;

    // Need a smarter value from statistics or work out by the number of tokens scheduled
    if (delta > 10)
    {
        // An new batch comes back
        if (us->his.count < NUM_PREFILL_BATCH_METRICS_HIS - 1)
        {
            us->his.count++;
        }

        us->his.head = (us->his.head + 1) % NUM_PREFILL_BATCH_METRICS_HIS;

        batch = &us->his.his[us->his.head];
        ngx_memzero(batch, sizeof(omni_batch_metrics_t));

        batch->first_response_receive_time = ngx_current_msec;
        batch->last_response_receive_time = ngx_current_msec;
        batch->num_requests = 1;
        batch->num_tokens = req->metrics.prompt_num_tokens + req->metrics.decoded_tokens;
    }
    else
    {
        batch->average_delta = batch->average_delta * (batch->num_requests - 1) +
                               delta / (batch->num_requests);
        batch->last_response_receive_time = ngx_current_msec;
        batch->num_requests++;
        batch->num_tokens += req->metrics.prompt_num_tokens + req->metrics.decoded_tokens;
    }
}

static ngx_int_t ngx_http_omni_create_request(ngx_http_request_t *r)
{
    ngx_chain_t *cl;
    size_t body_len = 0;
    omni_req_context_t *ctx = ngx_http_get_module_ctx(r, ngx_http_omni_proxy_module);

    omni_proxy_prepare_decode_request_body(r, ctx);

    if (r->request_body == NULL || r->request_body->bufs == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "ngx_http_omni_create_request: request body is NULL");
        return NGX_ERROR;
    }

    for (cl = r->request_body->bufs; cl; cl = cl->next)
    {
        body_len += ngx_buf_size(cl->buf);
    }

    size_t header_len = sizeof("POST ") - 1 + r->uri.len + sizeof(" HTTP/1.1\r\n") - 1;

    ngx_list_part_t *part = &r->headers_in.headers.part;
    ngx_table_elt_t *h = part->elts;
    ngx_uint_t i;

    for (;;)
    {
        for (i = 0; i < part->nelts; i++)
        {
            if (h[i].key.len == sizeof("Content-Length") - 1 &&
                ngx_strncasecmp(h[i].key.data,
                                (u_char *)"Content-Length",
                                h[i].key.len) == 0)
            {
                continue;
            }

            header_len += h[i].key.len + sizeof(": ") - 1 + h[i].value.len + sizeof("\r\n") - 1;
        }

        if (part->next == NULL)
        {
            break;
        }
        part = part->next;
        h = part->elts;
    }

    header_len += sizeof("Content-Length: ") - 1 + NGX_OFF_T_LEN + sizeof("\r\n\r\n") - 1;

    ngx_buf_t *hdr = ngx_create_temp_buf(r->pool, header_len);
    if (hdr == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "ngx_http_omni_create_request: failed to create temp buffer for request header");
        return NGX_ERROR;
    }

    hdr->last = ngx_snprintf(hdr->last, hdr->end - hdr->last,
                             "POST %V HTTP/1.1\r\n", &r->uri);

    part = &r->headers_in.headers.part;
    h = part->elts;

    for (;;)
    {
        for (i = 0; i < part->nelts; i++)
        {
            if (h[i].key.len == sizeof("Content-Length") - 1 &&
                ngx_strncasecmp(h[i].key.data,
                                (u_char *)"Content-Length",
                                h[i].key.len) == 0)
            {
                continue;
            }

            hdr->last = ngx_snprintf(hdr->last, hdr->end - hdr->last,
                                     "%V: %V\r\n", &h[i].key, &h[i].value);
        }

        if (part->next == NULL)
        {
            break;
        }
        part = part->next;
        h = part->elts;
    }

    hdr->last = ngx_snprintf(hdr->last, hdr->end - hdr->last,
                             "Content-Length: %O\r\n\r\n", (off_t)body_len);

    cl = ngx_alloc_chain_link(r->pool);
    if (cl == NULL)
    {
        return NGX_ERROR;
    }

    cl->buf = hdr;
    cl->next = r->request_body->bufs;
    r->upstream->request_bufs = cl;

    return NGX_OK;
}

static ngx_int_t ngx_http_omni_reinit_request(ngx_http_request_t *r)
{
    ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                  "ngx_http_omni_reinit_request(ngx_http_request_t *r): [Line %d]", __LINE__);
    return NGX_OK;
}

static ngx_int_t ngx_http_omni_process_header(ngx_http_request_t *r)
{
    ngx_int_t rc;
    ngx_http_upstream_t *u;
    ngx_table_elt_t *h;

    u = r->upstream;

    for (;;)
    {
        rc = ngx_http_parse_header_line(r, &u->buffer, 1);

        if (rc == NGX_OK)
        {
            h = ngx_list_push(&r->headers_out.headers);
            if (h == NULL)
            {
                ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                              "omni_proxy: push header failed.");
                return NGX_HTTP_INTERNAL_SERVER_ERROR;
            }

            h->key.len = r->header_name_end - r->header_name_start;
            h->key.data = ngx_pnalloc(r->pool, h->key.len);
            if (h->key.data == NULL)
            {
                return NGX_ERROR;
            }
            ngx_memcpy(h->key.data, r->header_name_start, h->key.len);

            h->value.len = r->header_end - r->header_start;
            h->value.data = ngx_pnalloc(r->pool, h->value.len);
            if (h->value.data == NULL)
            {
                return NGX_ERROR;
            }
            ngx_memcpy(h->value.data, r->header_start, h->value.len);

            h->lowcase_key = ngx_pnalloc(r->pool, h->key.len);
            if (h->lowcase_key == NULL)
            {
                return NGX_ERROR;
            }
            ngx_strlow(h->lowcase_key, h->key.data, h->key.len);

            h->hash = ngx_hash_key_lc(h->lowcase_key, h->key.len);

            continue;
        }

        if (rc == NGX_HTTP_PARSE_HEADER_DONE)
        {
            // Mark as processed to avoid forward to filter below
            // u->buffer.last = u->buffer.pos;

            ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                          "omni_process_header: finished parsing header");

            return NGX_OK;
        }

        if (rc == NGX_AGAIN)
        {
            return NGX_AGAIN;
        }

        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "omni_process_header: invalid header from upstream");
        return NGX_HTTP_UPSTREAM_INVALID_HEADER;
    }
}

static ngx_int_t ngx_http_omni_process_status_line(ngx_http_request_t *r)
{
    ngx_int_t rc;
    ngx_http_upstream_t *u = r->upstream;
    ngx_http_status_t st;
    ngx_memzero(&st, sizeof(st));

    rc = ngx_http_parse_status_line(r, &u->buffer, &st);

    if (rc == NGX_AGAIN)
    {
        return NGX_AGAIN;
    }

    if (rc == NGX_ERROR)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "omni_proxy: invalid status line from upstream");
        return NGX_HTTP_UPSTREAM_INVALID_HEADER;
    }

    u->headers_in.status_n = st.code;

    u->headers_in.status_line.len = st.end - st.start;
    u->headers_in.status_line.data = ngx_pnalloc(r->pool, u->headers_in.status_line.len);
    if (u->headers_in.status_line.data == NULL)
    {
        return NGX_ERROR;
    }
    ngx_memcpy(u->headers_in.status_line.data, st.start, u->headers_in.status_line.len);

    {
        omni_req_t *req = omni_get_req(r);
        if (req)
        {
            req->metrics.http_status = (ngx_uint_t)st.code;
        }
    }

    return ngx_http_omni_process_header(r);
}

static ngx_int_t ngx_http_omni_input_filter_init(void *data)
{
    return NGX_OK;
}

static ngx_int_t ngx_http_omni_input_filter(void *data, ssize_t bytes)
{
    ngx_http_request_t *r = data;
    ngx_http_upstream_t *u = r->upstream;

    ngx_log_error(NGX_LOG_DEBUG, r->connection->log, 0,
                  "ngx_http_omni_input_filter: bytes %lu", bytes);

    u->buffer.last = u->buffer.pos + bytes;
    if (u->buffer.pos != NULL && u->buffer.pos < u->buffer.last)
    {
        size_t len = u->buffer.last - u->buffer.pos;
        ngx_buf_t *b = ngx_create_temp_buf(r->pool, len);
        if (b == NULL)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                          "ngx_http_omni_input_filter, create buf failed");
            return NGX_ERROR;
        }
        ngx_memcpy(b->pos, u->buffer.pos, len);
        b->last = b->pos + len;
        b->memory = 1;
        b->flush = 1;

        ngx_chain_t *cl = ngx_alloc_chain_link(r->pool);
        if (cl == NULL)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                          "ngx_http_omni_input_filter, allocate chain failed");
            return NGX_ERROR;
        }
        cl->buf = b;
        cl->next = NULL;

        ngx_int_t rc = ngx_http_output_filter(r, cl);
        if (rc == NGX_ERROR)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                          "ngx_http_omni_input_filter, putput failed");
            return NGX_ERROR;
        }
    }

    omni_proxy_update_decode_stats(r, &u->buffer, bytes);

    return NGX_OK;
}

static void ngx_http_omni_finalize_request(ngx_http_request_t *r, ngx_int_t rc)
{
}

static ngx_int_t ngx_http_omni_start_decode_upstream(ngx_http_request_t *r)
{
    omni_req_t *req = omni_get_req(r);
    ngx_http_omni_loc_conf_t *olcf;

    if (ngx_http_upstream_create(r) != NGX_OK)
    {
        return NGX_ERROR;
    }
    olcf = ngx_http_get_module_loc_conf(r, ngx_http_omni_proxy_module);
    if (olcf->upstream == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "omni_proxy: no upstream in loc conf");
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    ngx_http_upstream_t *u = r->upstream;

    u->conf = olcf->upstream->srv_conf[ngx_http_upstream_module.ctx_index];
    u->conf->buffer_size = 1024;
    u->conf->send_lowat = 0;

    u->create_request = ngx_http_omni_create_request;
    u->reinit_request = ngx_http_omni_reinit_request;
    u->process_header = ngx_http_omni_process_status_line;
    u->input_filter_init = ngx_http_omni_input_filter_init;
    u->input_filter = ngx_http_omni_input_filter;
    u->input_filter_ctx = r;
    u->finalize_request = ngx_http_omni_finalize_request;

    u->output.tag = (ngx_buf_tag_t)&ngx_http_omni_proxy_module;

    if (req->decode_upstream_endpoint_idx >= MAX_DECODE_UPSTREAMS) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "omni_proxy: decode upstream endpoint index %d out of bounds for configured size %d",
                      req->decode_upstream_endpoint_idx, g_state->num_decode_endpoints);
        return NGX_ERROR;
    }
    omni_upstream_address_t *addr = &g_state->decode_states[req->decode_upstream_endpoint_idx].comm.address;

    u->resolved = ngx_pcalloc(r->pool, sizeof(ngx_http_upstream_resolved_t));
    if (u->resolved == NULL)
    {
        return NGX_ERROR;
    }

    struct sockaddr *sa = ngx_pcalloc(r->pool, addr->socklen);
    if (sa == NULL)
    {
        return NGX_ERROR;
    }
    ngx_memcpy(sa, &addr->sockaddr, addr->socklen);

    u->resolved->sockaddr = sa;
    u->resolved->port = addr->port;
    u->resolved->socklen = addr->socklen;
    u->resolved->naddrs = 1;

    u->resolved->host.len = addr->text_len;
    u->resolved->host.data = addr->text;

    ngx_http_upstream_init(r);

    return NGX_OK;
}

static ngx_int_t ngx_http_decode_wakeup(omni_req_t *req)
{
    ngx_http_request_t *r = omni_get_http_request(req);
    ngx_int_t rc = ngx_http_omni_start_decode_upstream(r);

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Decode-%d]: wakeup", req->slot_index);

    omni_phase_transition_all(req, PHASE_DECODE_SCHEDULED, PHASE_DECODING);

    omni_get_current_time(&req->metrics.time_to_decode);

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                "<<<Action: Enter state D running; Timestamp:%d.%06d; RequestID:%s", req->metrics.time_to_decode.tv_sec, req->metrics.time_to_decode.tv_usec, req->request_id);

    if (rc == NGX_OK)
    {
        /* ngx_http_upstream_init will drive request; return NGX_OK from post_subrequest */
        return NGX_OK;
    }

    ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                  "omni_proxy: start decode upstream failed.");
    ngx_http_finalize_request(r, NGX_HTTP_INTERNAL_SERVER_ERROR);
    return NGX_OK;
}

typedef ngx_int_t (*omni_run_handle_t)(omni_req_t *);

static void omni_proxy_run_group(int phase_from, omni_run_handle_t handle)
{
    omni_req_group_t *group = &local_state.groups[phase_from];

    for (uint32_t i = 0; i < group->watermark; ++i)
    {
        omni_req_info_t *info = &group->requests[i];
        if (info->in_use)
        {
            omni_req_t *req = omni_info_to_req(info);
            ngx_http_request_t *r = omni_get_http_request(req);
            ngx_int_t rc = handle(req);
            if (!(rc == NGX_OK || rc == NGX_DONE))
            {
                ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "[Wakeup-%d] Failed: %d",
                              req->slot_index, rc);
            }
        }
    }
}

void print_summary()
{
    if (ngx_current_msec - g_state->last_summary < 10000)
    {
        return;
    }
    g_state->last_summary = ngx_current_msec;

    ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                  "Active requests: %d, prefill waiting: %d, prefill running: %d "
                  "decode waiting: %d, decode running: %d, pid: %d\n",
                  g_state->request_pool.num_requests,
                  g_state->groups[PHASE_PREFILL_WAITING_SCHEDULE].num_requests,
                  g_state->groups[PHASE_PREFILLING].num_requests,
                  g_state->groups[PHASE_DECODE_WAITING_SCHEDULE].num_requests,
                  g_state->groups[PHASE_DECODING].num_requests,
                  ngx_pid);
}
static void omni_update_local_waiting(omni_worker_local_state_t *local_state,
                                      omni_req_group_t *group,
                                      omni_proxy_request_phase_t to)
{
    for (uint32_t i = 0; i < group->watermark; ++i)
    {
        omni_req_info_t *info = &group->requests[i];
        omni_req_t *req = omni_info_to_req(info);
        if (omni_req_is_in_phase(req, PHASE_PREFILL_WAITING_SCHEDULE) && req->has_prefill_sched) {
            omni_req_leave_phase(req, PHASE_PREFILL_WAITING_SCHEDULE);
            omni_req_enter_phase(req, PHASE_PREFILL_SCHEDULED);
        }
        if (omni_req_is_in_phase(req, PHASE_DECODE_WAITING_SCHEDULE) && req->has_decode_sched) {
            omni_req_leave_phase(req, PHASE_DECODE_WAITING_SCHEDULE);
            omni_req_enter_phase(req, PHASE_DECODE_SCHEDULED);
        }
        if (req->in_use && omni_req_is_in_phase(req, to))
        {
            omni_remove_from_group_by_req_info(info, group);
            omni_add_req_to_group(req->slot_index,
                                  &local_state->groups[to]);
        }
    }

    omni_sort_compact_group(group);
}

// Global scheduler has changed the req->phase to PHASE_PREFILL_SCHEDULED, here to update local state
// to make sure local state is consistent with global state
static inline void omni_update_local_prefill_waiting(omni_worker_local_state_t *local_state)
{
    omni_update_local_waiting(local_state, &local_state->groups[PHASE_PREFILL_WAITING_SCHEDULE],
                              PHASE_PREFILL_SCHEDULED);
}

static inline void omni_update_local_decode_waiting(omni_worker_local_state_t *local_state)
{
    omni_update_local_waiting(local_state, &local_state->groups[PHASE_DECODE_WAITING_SCHEDULE],
                              PHASE_DECODE_SCHEDULED);
}

static void omni_proxy_schedule(omni_global_state_t *gs)
{
    if (omni_is_master_worker(&local_state))
    {
        ngx_shmtx_lock(&gs->shmtx);
        if (local_state.loc_conf) {
            omni_proxy_schedule_prefill(gs, local_state.loc_conf);
            omni_proxy_schedule_decode(gs, local_state.loc_conf);
        }
        ngx_shmtx_unlock(&gs->shmtx);
    }
}

static void omni_proxy_timer_handler(ngx_event_t *ev)
{
    omni_proxy_schedule(g_state);

    // Global state has moved on, local state needs to be updated
    omni_update_local_prefill_waiting(&local_state);
    omni_update_local_decode_waiting(&local_state);

    omni_proxy_run_group(PHASE_PREFILL_SCHEDULED, ngx_http_prefill_wakeup);
    omni_proxy_run_group(PHASE_DECODE_SCHEDULED, ngx_http_decode_wakeup);

    if (ngx_exiting && local_state.req_in_groups == 0) {
        ngx_log_error(NGX_LOG_INFO, ev->log, 0, "Worker %P timer exit", ngx_pid);
        return;
    }
    ngx_add_timer(&local_state.omni_proxy_timer_event, TIMER_INTERVAL);

    // print_summary();
}

static ngx_int_t omni_proxy_init_req_groups_with_slots(omni_req_group_t groups[], uint32_t max_slots, ngx_slab_pool_t *shpool)
{
    for (int i = 0; i < PHASE_MAX; i++)
    {
        groups[i].phase = i;
        groups[i].num_requests = 0;
        groups[i].watermark = 0;

        size_t req_info_size = (size_t)max_slots * sizeof(omni_req_info_t);
        groups[i].requests = ngx_slab_alloc(shpool, req_info_size);
        if (groups[i].requests == NULL) {
            return NGX_ERROR;
        }
        ngx_memzero(groups[i].requests, req_info_size);
    }
    return NGX_OK;
}

static ngx_int_t omni_proxy_global_state_init(ngx_shm_zone_t *zone, void *data)
{
    ngx_uint_t i;

    ngx_cycle_t *cycle = zone->data;
    if (data != NULL) {
        // Reload: check if max_request_slots changed
        ngx_http_omni_main_conf_t *mcf = ngx_http_cycle_get_module_main_conf(cycle, ngx_http_omni_proxy_module);
        uint32_t new_max_slots = mcf->max_request_slots;

        if (new_max_slots != g_state->request_pool.max_slots) {
            ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                "omni_proxy: max_request_slots cannot be changed on reload. "
                "Old value: %ui, New value: %ui. Please restart instead of reload.",
                g_state->request_pool.max_slots, new_max_slots);
            return NGX_ERROR;
        }

        ngx_log_error(NGX_LOG_INFO, zone->shm.log, 0, "reuse global state share mem when reload");
        if (ngx_http_omni_init_upstreams(cycle) != NGX_OK) {
            ngx_log_error(NGX_LOG_EMERG, cycle->log, 0, "Init global upstream failed when reload");
            return NGX_ERROR;
        }
        g_state->master_worker_pid = 0;
        return NGX_OK;
    }

    if (zone->shm.addr == NULL)
    {
        return NGX_ERROR;
    }

    ngx_slab_pool_t *shpool = (ngx_slab_pool_t *) zone->shm.addr;
    if (zone->shm.exists) {
        zone->data = shpool->data;
        return NGX_OK;
    }

    g_state = ngx_slab_alloc(shpool, sizeof(omni_global_state_t));
    if (g_state == NULL) {
        return NGX_ERROR;
    }
    ngx_memzero(g_state, GLOBAL_STATE_SIZE);
    g_state->shm = shpool;
    g_state->magic = 47;

    g_state->num_prefill_endpoints = 0;
    g_state->num_decode_endpoints = 0;
    g_state->num_encode_endpoints = 0;

    // Get max_request_slots from main conf
    ngx_http_omni_main_conf_t *mcf = ngx_http_cycle_get_module_main_conf(cycle, ngx_http_omni_proxy_module);
    uint32_t max_slots = mcf->max_request_slots;

    // Allocate in_use_bitmap (rounded up: ceil(max_slots/64))
    size_t bitmap_size = ((max_slots + 63) / 64) * sizeof(uint64_t);
    g_state->request_pool.in_use_bitmap = ngx_slab_alloc(shpool, bitmap_size);
    if (g_state->request_pool.in_use_bitmap == NULL) {
        ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
            "omni_proxy: failed to allocate in_use_bitmap");
        return NGX_ERROR;
    }
    ngx_memzero(g_state->request_pool.in_use_bitmap, bitmap_size);

    // Allocate slots
    size_t slots_size = (size_t)max_slots * sizeof(omni_req_t);
    g_state->request_pool.slots = ngx_slab_alloc(shpool, slots_size);
    if (g_state->request_pool.slots == NULL) {
        ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
            "omni_proxy: failed to allocate slots");
        return NGX_ERROR;
    }
    ngx_memzero(g_state->request_pool.slots, slots_size);

    // Initialize request pool
    g_state->request_pool.max_slots = max_slots;
    g_state->request_pool.num_requests = 0;
    g_state->request_pool.head = 0;

    // Initialize groups with requests array
    if (omni_proxy_init_req_groups_with_slots(g_state->groups, max_slots, shpool) != NGX_OK) {
        ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
            "omni_proxy: failed to allocate requests array");
        return NGX_ERROR;
    }

    ngx_shmtx_create(&g_state->shmtx, &g_state->lock,
                     (u_char *)"omni_proxy_lock");

    if (ngx_http_omni_init_upstreams(cycle) != NGX_OK) {
        ngx_log_error(NGX_LOG_EMERG, cycle->log, 0, "Init global upstream failed");
        return NGX_ERROR;
    }

    // Calculate total memory size
    size_t groups_size = PHASE_MAX * (size_t)max_slots * sizeof(omni_req_info_t);
    size_t total_size = GLOBAL_STATE_SIZE + bitmap_size + slots_size + groups_size;
    printf("shared memory initialed: %p, total size: %zuKB (static:%ldKB, bitmap:%zuKB, slots:%zuKB, groups:%zuKB)\n",
           zone->shm.addr, total_size / 1024,
           GLOBAL_STATE_SIZE / 1024, bitmap_size / 1024, slots_size / 1024, groups_size / 1024);

    return NGX_OK;
}

static void *ngx_http_omni_create_main_conf(ngx_conf_t *cf)
{
    ngx_http_omni_main_conf_t *conf;

    conf = ngx_pcalloc(cf->pool, sizeof(ngx_http_omni_main_conf_t));
    if (conf == NULL)
    {
        return NULL;
    }

    conf->max_request_slots = NGX_CONF_UNSET_UINT;

    return conf;
}

static char *ngx_http_omni_merge_main_conf(ngx_conf_t *cf, void *conf)
{
    ngx_http_omni_main_conf_t *prev = ((ngx_http_conf_ctx_t *)cf->ctx)->main_conf[ngx_http_omni_proxy_module.ctx_index];
    ngx_http_omni_main_conf_t *conf_t = conf;

    ngx_conf_merge_uint_value(conf_t->max_request_slots, prev->max_request_slots, MAX_REQUEST_SLOTS);

    return NGX_CONF_OK;
}

ngx_int_t omni_proxy_init_global_state(ngx_conf_t *cf)
{
    ngx_str_t name = ngx_string("omni_proxy_state");
    printf("Init global state with size:%ldK\n", GLOBAL_STATE_SIZE / 1024);

    ngx_shm_zone_t *zone = ngx_shared_memory_add(
        cf,
        &name,
        NGX_MAX_INT32_VALUE,
        &ngx_http_omni_proxy_module);

    if (zone == NULL)
    {
        return NGX_ERROR;
    }

    zone->shm.log->data = cf->cycle;
    zone->data = cf->cycle;

    zone->init = omni_proxy_global_state_init;

    return NGX_OK;
}

static void *ngx_http_omni_create_loc_conf(ngx_conf_t *cf)
{
    ngx_http_omni_loc_conf_t *conf;

    conf = ngx_pcalloc(cf->pool, sizeof(ngx_http_omni_loc_conf_t));
    if (conf == NULL)
    {
        return NULL;
    }

    conf->upstream_name.data = NULL;
    conf->upstream_name.len = 0;

    conf->pd_policy = NGX_CONF_UNSET;
    conf->model_path.data = NULL;
    conf->model_path.len = 0;
    conf->kv_block_size = 128;

    conf->vllm_kv_port_offset = NGX_CONF_UNSET;
    conf->max_batch_num_token = NGX_CONF_UNSET_UINT;
    conf->prefill_max_num_seqs = NGX_CONF_UNSET_UINT;
    conf->decode_max_num_seqs = NGX_CONF_UNSET_UINT;
    conf->max_tokens_weight = NGX_CONF_UNSET_UINT;
    conf->prefill_starvation_timeout = NGX_CONF_UNSET_UINT;
    conf->schedule_algo = NGX_CONF_UNSET_UINT;
    conf->prefill_pod_size = NGX_CONF_UNSET_UINT;
    conf->decode_pod_size = NGX_CONF_UNSET_UINT;
    conf->health_status_enabled = NGX_CONF_UNSET;
    conf->stream_ops = (ngx_prefill_stream_op_e) NGX_CONF_UNSET_UINT;
    conf->prefill_groups = NULL;
    conf->decode_groups = NULL;

    return conf;
}

static char *ngx_http_omni_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child)
{
    ngx_http_omni_loc_conf_t *prev = parent;
    ngx_http_omni_loc_conf_t *conf = child;

    ngx_conf_merge_str_value(conf->upstream_name, prev->upstream_name, "");

    if (conf->pd_policy == NGX_CONF_UNSET)
    {
        conf->pd_policy = (prev->pd_policy != NGX_CONF_UNSET) ? prev->pd_policy : NGX_CONF_UNSET;
    }

    ngx_conf_merge_str_value(conf->model_path, prev->model_path, "");

    if (conf->vllm_kv_port_offset == NGX_CONF_UNSET)
    {
        conf->vllm_kv_port_offset = (prev->vllm_kv_port_offset != NGX_CONF_UNSET) ? prev->vllm_kv_port_offset : NGX_CONF_UNSET;
    }

    ngx_conf_merge_uint_value(conf->max_batch_num_token, prev->max_batch_num_token, 32000);
    ngx_conf_merge_uint_value(conf->prefill_max_num_seqs, prev->prefill_max_num_seqs, 32);
    ngx_conf_merge_uint_value(conf->decode_max_num_seqs, prev->decode_max_num_seqs, 32);
    ngx_conf_merge_uint_value(conf->max_tokens_weight, prev->max_tokens_weight, 0);
    ngx_conf_merge_uint_value(conf->prefill_starvation_timeout, prev->prefill_starvation_timeout, 400);
    ngx_conf_merge_uint_value(conf->schedule_algo, prev->schedule_algo, 0);

    ngx_conf_merge_uint_value(conf->prefill_pod_size, prev->prefill_pod_size, 1);
    ngx_conf_merge_uint_value(conf->decode_pod_size, prev->decode_pod_size, 1);    

    if (conf->metrics_enabled == NGX_CONF_UNSET)
    {
        conf->metrics_enabled = (prev->metrics_enabled != NGX_CONF_UNSET) ? prev->metrics_enabled : NGX_CONF_UNSET;
    }

    ngx_conf_merge_value(conf->health_status_enabled, prev->health_status_enabled, 0);

    if (conf->prefill_groups == NULL) {
        conf->prefill_groups = prev->prefill_groups;
    }

    if (conf->decode_groups == NULL) {
        conf->decode_groups = prev->decode_groups;
    }

    if (conf->model_path.len != 0 || conf->vllm_kv_port_offset != NGX_CONF_UNSET ||
        conf->pd_policy != NGX_CONF_UNSET || conf->prefill_groups != NULL || conf->decode_groups != NULL)
    {
        local_state.loc_conf = conf;
    }

    if ((ngx_uint_t)conf->stream_ops == NGX_CONF_UNSET_UINT)
    {
        if ((ngx_uint_t)prev->stream_ops == NGX_CONF_UNSET_UINT)
        {
            conf->stream_ops = NGX_PREFILL_STREAM_OFF;
        }
        else
        {
            conf->stream_ops = prev->stream_ops;
        }
    }

    return NGX_CONF_OK;
}

static ngx_int_t omni_proxy_parse_group_token(ngx_conf_t *cf, ngx_command_t *cmd, ngx_str_t *token,
    ngx_uint_t *group_id, ngx_uint_t *repeat)
{
    u_char *sep = ngx_strlchr(token->data, token->data + token->len, ':');
    if (sep == NULL) {
        return NGX_ERROR;
    }

    ngx_str_t left;
    ngx_str_t right;
    left.data = token->data;
    left.len = (size_t)(sep - token->data);
    right.data = sep + 1;
    right.len = (size_t)((token->data + token->len) - right.data);

    ngx_int_t gid = ngx_atoi(left.data, left.len);
    ngx_int_t cnt = ngx_atoi(right.data, right.len);
    if (gid == NGX_ERROR || gid < 0 || cnt == NGX_ERROR || cnt <= 0) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                           "invalid group segment \"%V\" in \"%V\" directive",
                           token, &cmd->name);
        return NGX_ERROR;
    }
    *group_id = (ngx_uint_t) gid;
    *repeat = (ngx_uint_t) cnt;
    return NGX_OK;
}

static char *omni_proxy_set_groups(ngx_conf_t *cf, ngx_command_t *cmd, ngx_array_t **groups)
{
    ngx_str_t *value = cf->args->elts;
    ngx_uint_t count = cf->args->nelts;

    if (*groups != NULL) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                           "\"%V\" directive is duplicate", &cmd->name);
        return NGX_CONF_ERROR;
    }

    if (count < 2) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                           "invalid number of arguments in \"%V\" directive", &cmd->name);
        return NGX_CONF_ERROR;
    }

    *groups = ngx_array_create(cf->pool, count - 1, sizeof(ngx_uint_t));
    if (*groups == NULL) {
        return NGX_CONF_ERROR;
    }

    for (ngx_uint_t i = 1; i < count; i++) {
        ngx_uint_t group_id = 0;
        ngx_uint_t repeat = 0;
        if (omni_proxy_parse_group_token(cf, cmd, &value[i], &group_id, &repeat) != NGX_OK) {
            return NGX_CONF_ERROR;
        }
        for (ngx_uint_t r = 0; r < repeat; r++) {
            ngx_uint_t *slot = ngx_array_push(*groups);
            if (slot == NULL) {
                return NGX_CONF_ERROR;
            }
            *slot = group_id;
        }
    }

    return NGX_CONF_OK;
}

static char *omni_proxy_set_prefill_groups(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_omni_loc_conf_t *olcf = conf;
    return omni_proxy_set_groups(cf, cmd, &olcf->prefill_groups);
}

static char *omni_proxy_set_decode_groups(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_omni_loc_conf_t *olcf = conf;
    return omni_proxy_set_groups(cf, cmd, &olcf->decode_groups);
}

static char *ngx_conf_set_omni_stream_ops(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_omni_loc_conf_t *olcf = conf;
    ngx_str_t *value = cf->args->elts;

    if (cf->args->nelts != 2)
    {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                           "invalid number of arguments in \"%V\" directive",
                           &cmd->name);
        return NGX_CONF_ERROR;
    }

    if (ngx_strcmp(value[1].data, "off") == 0)
    {
        olcf->stream_ops = NGX_PREFILL_STREAM_OFF;
    }
    else if (ngx_strcmp(value[1].data, "add") == 0)
    {
        olcf->stream_ops = NGX_PREFILL_STREAM_ADD;
    }
    else if (ngx_strcmp(value[1].data, "set_opt") == 0)
    {
        olcf->stream_ops = NGX_PREFILL_STREAM_SET_OPT;
    }
    else
    {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                           "invalid value \"%V\" in \"%V\" directive, it must be \"add\", \"set_opt\", or \"off\"",
                           &value[1], &cmd->name);
        return NGX_CONF_ERROR;
    }

    return NGX_CONF_OK;
}

static void omni_upstream_addr_set(omni_upstream_address_t *addr, ngx_http_upstream_rr_peer_t *peer)
{
    addr->socklen = peer->socklen;
    ngx_memcpy(&addr->sockaddr, peer->sockaddr, peer->socklen);

    addr->text_len = peer->name.len;
    if (addr->text_len < UPSTREAM_ADDR_NAME_MAX - 1) {
        ngx_memcpy(addr->text, peer->name.data, peer->name.len);
        addr->text[addr->text_len] = '\0';
    } else {
        ngx_memcpy(addr->text, peer->name.data, UPSTREAM_ADDR_NAME_MAX - 1);
        addr->text[UPSTREAM_ADDR_NAME_MAX - 1] = '\0';
        addr->text_len = UPSTREAM_ADDR_NAME_MAX - 1;
    }

    if (peer->sockaddr->sa_family == AF_INET) {
        struct sockaddr_in *ipv4 = (struct sockaddr_in *)&addr->sockaddr;
        inet_ntop(AF_INET, &(ipv4->sin_addr), addr->ip, UPSTREAM_IP_MAX);
        addr->port = ntohs(ipv4->sin_port);
    } else if (peer->sockaddr->sa_family == AF_INET6) {
        struct sockaddr_in6 *ipv6 = (struct sockaddr_in6 *)&addr->sockaddr;
        inet_ntop(AF_INET6, &(ipv6->sin6_addr), addr->ip, UPSTREAM_IP_MAX);
        addr->port = ntohs(ipv6->sin6_port);
    }
}

static int32_t omni_proxy_get_group_id(ngx_cycle_t *cycle, bool is_prefill, ngx_uint_t idx)
{
    if (local_state.loc_conf == NULL) {
        ngx_log_error(NGX_LOG_ERR, cycle->log, 0, "local_state.loc_conf is NULL");
        return -1;
    }

    ngx_array_t *groups = is_prefill ? local_state.loc_conf->prefill_groups : local_state.loc_conf->decode_groups;
    if (groups == NULL) {
        return 0;
    }

    if (idx >= groups->nelts) {
        ngx_log_error(NGX_LOG_ERR, cycle->log, 0, "idx %ui larger than %s group nelts:%ui",
            idx, is_prefill ? "Prefill" : "Decode", groups->nelts);
        return -1;
    }

    ngx_uint_t *ids = groups->elts;
    return ids[idx];
}

static void omni_upstream_disable(bool is_prefill, omni_upstream_status_t from, omni_upstream_status_t to)
{
    ngx_uint_t i;
    ngx_uint_t max_num = is_prefill ? MAX_PREFILL_UPSTREAMS : MAX_DECODE_UPSTREAMS;
    for (i = 0; i < max_num; i++) {
        omni_upstream_common_t *comm = is_prefill ? &g_state->prefill_states[i].comm : &g_state->decode_states[i].comm;
        if (comm->status == STATUS_UNUSED) {
            break;
        }
        if (comm->status == from) {
            comm->status = to;
        }
    }
}

static void omni_upstream_reuse(bool is_prefill, ngx_http_upstream_rr_peer_t *peer, ngx_uint_t idx)
{
    omni_upstream_common_t *comm = is_prefill ? &g_state->prefill_states[idx].comm : &g_state->decode_states[idx].comm;
    omni_upstream_addr_set(&comm->address, peer);
    comm->status = STATUS_ENABLE;
    if (is_prefill) {
        g_state->prefill_states[idx].num_running = 0;
        g_state->prefill_states[idx].num_tokens = 0;
        g_state->prefill_states[idx].last_scheduled_time = 0;
        g_state->prefill_states[idx].expected_next_schedule_time = 0;
        ngx_memzero(&g_state->prefill_states[idx].his, sizeof(omni_batch_metrics_his_t));
    } else {
        g_state->decode_states[idx].num_running = 0;
        g_state->decode_states[idx].num_tokens = 0;
        g_state->decode_states[idx].generated_tokens = 0;
        g_state->decode_states[idx].last_scheduled_time = 0;
        g_state->decode_states[idx].expected_next_schedule_time = 0;
        ngx_memzero(&g_state->decode_states[idx].his, sizeof(omni_batch_metrics_his_t));
    }
}

static ngx_int_t omni_upstream_add(bool is_prefill, ngx_cycle_t *cycle, ngx_http_upstream_rr_peers_t *peers)
{
    ngx_uint_t max_num = is_prefill ? MAX_PREFILL_UPSTREAMS : MAX_DECODE_UPSTREAMS;
    if (peers->number >= max_num) {
        ngx_log_error(NGX_LOG_WARN, cycle->log, 0,
            "%s endpoint num %ui exceeds maximum %ui", is_prefill ? "Prefill" : "Decode", peers->number, max_num);
        return NGX_ERROR;
    }

    ngx_uint_t cnt = 0;
    ngx_http_upstream_rr_peer_t *peer;
    for (peer = peers->peer; peer != NULL; peer = peer->next) {
        ngx_http_upstream_rr_peer_lock(peers, peer);
        int32_t group_id = omni_proxy_get_group_id(cycle, is_prefill, cnt);
        if (group_id < 0) {
            ngx_http_upstream_rr_peer_unlock(peers, peer);
            return NGX_ERROR;
        }
        omni_upstream_common_t *comm;
        ngx_int_t abandon_idx = -1;
        ngx_uint_t j;
        for (j = 0; j < max_num; j++) {
            comm = is_prefill ? &g_state->prefill_states[j].comm : &g_state->decode_states[j].comm;
            if (comm->status == STATUS_ABANDON) {
                if (comm->ref == 0 && abandon_idx == -1) {
                    abandon_idx = j;
                }
                continue;
            }
            if (comm->status != STATUS_UNUSED) {
                /* ip/port matches exist upstream, continue use it */
                if (ngx_cmp_sockaddr(peer->sockaddr, peer->socklen, &comm->address.sockaddr, comm->address.socklen, 1) != NGX_OK) {
                    continue;
                }
            } else {
                /* not match any exist upstream, add */
                omni_upstream_addr_set(&comm->address, peer);

                if (is_prefill && local_state.loc_conf->vllm_kv_port_offset != NGX_CONF_UNSET) {
                    g_state->prefill_states[j].radix_tree = omni_radix_tree_init(g_state->shm);
                    if (g_state->prefill_states[j].radix_tree == NULL) {
                        ngx_http_upstream_rr_peer_unlock(peers, peer);
                        ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                                      "Failed to create prefill radix hash_tree, idx=%ui", cnt);
                        return NGX_ERROR;
                    }
                }

                if (!is_prefill && local_state.loc_conf->vllm_kv_port_offset != NGX_CONF_UNSET) {
                    g_state->decode_states[j].radix_tree = omni_radix_tree_init(g_state->shm);
                    if (g_state->decode_states[j].radix_tree == NULL) {
                        ngx_http_upstream_rr_peer_unlock(peers, peer);
                        ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                                      "Failed to create decode radix hash_tree, idx=%ui", cnt);
                        return NGX_ERROR;
                    }
                }
            }
            comm->status = STATUS_ENABLE;
            break;
        }

        if (j >= max_num) {
            if (abandon_idx != -1) {
                omni_upstream_reuse(is_prefill, peer, abandon_idx);
                ngx_log_error(NGX_LOG_NOTICE, cycle->log, 0, "%s peer %ui reuse endpoint[%ui]",
                    is_prefill ? "Prefill" : "Decode", cnt, abandon_idx);
                j = abandon_idx;
                comm = is_prefill ? &g_state->prefill_states[j].comm : &g_state->decode_states[j].comm;
            } else {
                ngx_http_upstream_rr_peer_unlock(peers, peer);
                return NGX_ERROR;
            }
        }

        comm->group_id = (uint32_t)group_id;
        ngx_log_error(NGX_LOG_NOTICE, cycle->log, 0, "Add %s peer %ui in endpoint[%ui] -> %s:%d (group=%uD)",
            is_prefill ? "Prefill" : "Decode", cnt, j, comm->address.ip, comm->address.port, comm->group_id);
        cnt++;
        ngx_http_upstream_rr_peer_unlock(peers, peer);
    }

    return NGX_OK;
}

static void omni_upstream_disable_encode(omni_upstream_status_t from, omni_upstream_status_t to)
{
    for (ngx_uint_t i = 0; i < MAX_ENCODE_UPSTREAMS; i++) {
        omni_upstream_common_t *comm = &g_state->encode_states[i].comm;
        if (comm->status == STATUS_UNUSED) {
            break;
        }
        if (comm->status == from) {
            comm->status = to;
        }
    }
}

static void omni_upstream_reuse_encode(ngx_http_upstream_rr_peer_t *peer, ngx_uint_t idx)
{
    omni_upstream_common_t *comm = &g_state->encode_states[idx].comm;
    omni_upstream_addr_set(&comm->address, peer);
    comm->status = STATUS_ENABLE;
    g_state->encode_states[idx].num_running = 0;
    g_state->encode_states[idx].num_tokens = 0;
}

static ngx_int_t omni_upstream_add_encode(ngx_cycle_t *cycle, ngx_http_upstream_rr_peers_t *peers)
{
    if (peers->number >= MAX_ENCODE_UPSTREAMS) {
        ngx_log_error(NGX_LOG_WARN, cycle->log, 0,
                      "Encode endpoint num %ui exceeds maximum %ui", peers->number, (ngx_uint_t)MAX_ENCODE_UPSTREAMS);
        return NGX_ERROR;
    }

    ngx_uint_t cnt = 0;
    ngx_http_upstream_rr_peer_t *peer;
    for (peer = peers->peer; peer != NULL; peer = peer->next) {
        ngx_http_upstream_rr_peer_lock(peers, peer);
        omni_upstream_common_t *comm;
        ngx_int_t abandon_idx = -1;
        ngx_uint_t j;
        for (j = 0; j < MAX_ENCODE_UPSTREAMS; j++) {
            comm = &g_state->encode_states[j].comm;
            if (comm->status == STATUS_ABANDON) {
                if (comm->ref == 0 && abandon_idx == -1) {
                    abandon_idx = j;
                }
                continue;
            }
            if (comm->status != STATUS_UNUSED) {
                if (ngx_cmp_sockaddr(peer->sockaddr, peer->socklen, &comm->address.sockaddr, comm->address.socklen, 1) != NGX_OK) {
                    continue;
                }
            } else {
                omni_upstream_addr_set(&comm->address, peer);
                g_state->encode_states[j].num_running = 0;
                g_state->encode_states[j].num_tokens = 0;
            }
            comm->status = STATUS_ENABLE;
            break;
        }

        if (j >= MAX_ENCODE_UPSTREAMS) {
            if (abandon_idx != -1) {
                omni_upstream_reuse_encode(peer, abandon_idx);
                ngx_log_error(NGX_LOG_NOTICE, cycle->log, 0, "Encode peer %ui reuse endpoint[%ui]",
                              cnt, abandon_idx);
                j = abandon_idx;
                comm = &g_state->encode_states[j].comm;
            } else {
                ngx_http_upstream_rr_peer_unlock(peers, peer);
                return NGX_ERROR;
            }
        }

        ngx_log_error(NGX_LOG_NOTICE, cycle->log, 0, "Add Encode peer %ui in endpoint[%ui] -> %s:%d",
                      cnt, j, comm->address.ip, comm->address.port);
        cnt++;
        ngx_http_upstream_rr_peer_unlock(peers, peer);
    }

    return NGX_OK;
}

static ngx_int_t ngx_http_omni_init_upstreams(ngx_cycle_t *cycle)
{
    ngx_uint_t i;
    ngx_uint_t j;
    ngx_http_upstream_main_conf_t *umcf = ngx_http_cycle_get_module_main_conf(cycle, ngx_http_upstream_module);

    if (umcf == NULL) {
        return NGX_OK;
    }

    ngx_array_t *upstreams = &umcf->upstreams;
    ngx_uint_t nupstreams = upstreams->nelts;
    ngx_http_upstream_srv_conf_t **uscfp = upstreams->elts;

    ngx_shmtx_lock(&g_state->shmtx);
    for (i = 0; i < nupstreams; i++) {
        ngx_http_upstream_srv_conf_t *uscf = uscfp[i];
        if (uscf == NULL) {
            continue;
        }

        ngx_http_upstream_rr_peers_t *peers = uscf->peer.data;
        if (peers == NULL) {
            ngx_log_error(NGX_LOG_WARN, cycle->log, 0,
                         "No peers found for upstream %V", &uscf->host);
            continue;
        }

        ngx_http_upstream_rr_peers_rlock(peers);

        ngx_int_t use_custom_peer = 0;
        ngx_int_t use_encode_peer = 0;
        if (ngx_strncmp(uscf->host.data, PREFILL_ENDPOINTS, sizeof(PREFILL_ENDPOINTS) - 1) == 0) {
            /* all endpoint set disable before p/d num update */
            omni_upstream_disable(true, STATUS_ENABLE, STATUS_DISABLE);
            /* enable/add endpoint by new conf */
            if (omni_upstream_add(true, cycle, peers) != NGX_OK) {
                /* if fail here, upstream still can be used */
                ngx_http_upstream_rr_peers_unlock(peers);
                ngx_shmtx_unlock(&g_state->shmtx);
                return NGX_ERROR;
            }
            /* set abandon if not reuse in reload, and will not reuse in reload later */
            omni_upstream_disable(true, STATUS_DISABLE, STATUS_ABANDON);
            g_state->num_prefill_endpoints = uscf->servers->nelts >= MAX_PREFILL_UPSTREAMS ?
                MAX_PREFILL_UPSTREAMS : uscf->servers->nelts;
            use_custom_peer = 1;
        } else if (ngx_strncmp(uscf->host.data, DECODE_ENDPOINTS, sizeof(DECODE_ENDPOINTS) - 1) == 0) {
            omni_upstream_disable(false, STATUS_ENABLE, STATUS_DISABLE);
            if (omni_upstream_add(false, cycle, peers) != NGX_OK) {
                ngx_http_upstream_rr_peers_unlock(peers);
                ngx_shmtx_unlock(&g_state->shmtx);
                return NGX_ERROR;
            }
            omni_upstream_disable(false, STATUS_DISABLE, STATUS_ABANDON);
            g_state->num_decode_endpoints = uscf->servers->nelts >= MAX_DECODE_UPSTREAMS ?
                MAX_DECODE_UPSTREAMS : uscf->servers->nelts;
            use_custom_peer = 1;
        } else if (ngx_strncmp(uscf->host.data, ENCODE_ENDPOINTS, sizeof(ENCODE_ENDPOINTS) - 1) == 0) {
            omni_upstream_disable_encode(STATUS_ENABLE, STATUS_DISABLE);
            if (omni_upstream_add_encode(cycle, peers) != NGX_OK) {
                ngx_http_upstream_rr_peers_unlock(peers);
                ngx_shmtx_unlock(&g_state->shmtx);
                return NGX_ERROR;
            }
            omni_upstream_disable_encode(STATUS_DISABLE, STATUS_ABANDON);
            g_state->num_encode_endpoints = uscf->servers->nelts >= MAX_ENCODE_UPSTREAMS ?
                MAX_ENCODE_UPSTREAMS : uscf->servers->nelts;
            use_custom_peer = 1;
            use_encode_peer = 1;
        } else {
            ngx_http_upstream_rr_peers_unlock(peers);
            continue;
        }
        ngx_http_upstream_rr_peers_unlock(peers);

        if (use_custom_peer) {
            uscf->peer.init = use_encode_peer ? omni_proxy_upstream_init_encode : omni_proxy_upstream_init;
        }
    }

    ngx_log_error(NGX_LOG_INFO, cycle->log, 0,
                 "Upstream initialization completed - Encode: %d, Prefill: %d, Decode: %d",
                 g_state->num_encode_endpoints, g_state->num_prefill_endpoints, g_state->num_decode_endpoints);

    ngx_shmtx_unlock(&g_state->shmtx);

    return NGX_OK;
}

static ngx_int_t ngx_http_omni_proxy_metrics_handler(ngx_http_request_t *r)
{
    ngx_http_omni_loc_conf_t *plcf;
    ngx_int_t rc;
    ngx_buf_t *b;
    ngx_chain_t out;

    plcf = ngx_http_get_module_loc_conf(r, ngx_http_omni_proxy_module);

    if (!plcf || !plcf->metrics_enabled)
    {
        return NGX_DECLINED;
    }

    ngx_str_t metrics = omni_metrics_export(g_state);

    r->headers_out.status = NGX_HTTP_OK;
    r->headers_out.content_length_n = metrics.len;

    r->headers_out.content_type.len = sizeof("text/plain") - 1;
    r->headers_out.content_type.data = (u_char *)"text/plain";
    r->headers_out.content_type_len = sizeof("text/plain") - 1;

    rc = ngx_http_send_header(r);
    if (rc == NGX_ERROR || rc > NGX_OK)
    {
        return rc;
    }

    b = ngx_calloc_buf(r->pool);
    if (b == NULL)
    {
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    b->pos = metrics.data;
    b->last = metrics.data + metrics.len;
    b->memory = 1;
    b->last_buf = 1;
    b->last_in_chain = 1;

    out.buf = b;
    out.next = NULL;

    return ngx_http_output_filter(r, &out);
}

static int ngx_uint_cmp(const void *a, const void *b)
{
    ngx_uint_t x = *(ngx_uint_t *)a;
    ngx_uint_t y = *(ngx_uint_t *)b;
    if (x < y) {
        return -1;
    }
    if (x > y) {
        return 1;
    }
    return 0;
}

static ngx_uint_t dedup_sorted_uint_array(ngx_uint_t *arr, ngx_uint_t n)
{
    if (n == 0) {
        return 0;
    }

    ngx_uint_t write_idx = 1;
    for (ngx_uint_t i = 1; i < n; i++) {
        if (arr[i] != arr[i - 1]) {
            arr[write_idx++] = arr[i];
        }
    }
    return write_idx;
}

static ngx_int_t omni_groups_equal_unique(ngx_conf_t *cf, ngx_array_t *prefill_group, ngx_array_t *decode_group)
{
    ngx_uint_t *prefill_group_set = NULL;
    ngx_uint_t *decode_group_set = NULL;
    ngx_uint_t prefill_group_num = prefill_group->nelts;
    ngx_uint_t decode_group_num = decode_group->nelts;

    if (prefill_group_num == 0 || decode_group_num == 0) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
            "prefill_group_num or decode_group_num should not be 0");
        return NGX_ERROR;
    }

    prefill_group_set = ngx_palloc(cf->pool, prefill_group_num * sizeof(ngx_uint_t));
    decode_group_set = ngx_palloc(cf->pool, decode_group_num * sizeof(ngx_uint_t));
    if (prefill_group_set == NULL || decode_group_set == NULL) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
            "prefill_group_set or decode_group_set ngx_palloc failed");
        return NGX_ERROR;
    }

    ngx_memcpy(prefill_group_set, prefill_group->elts, prefill_group_num * sizeof(ngx_uint_t));
    ngx_memcpy(decode_group_set, decode_group->elts, decode_group_num * sizeof(ngx_uint_t));

    ngx_qsort(prefill_group_set, prefill_group_num, sizeof(ngx_uint_t), ngx_uint_cmp);
    ngx_qsort(decode_group_set, decode_group_num, sizeof(ngx_uint_t), ngx_uint_cmp);

    ngx_uint_t dedup_prefill_group_num = dedup_sorted_uint_array(prefill_group_set, prefill_group_num);
    ngx_uint_t dedup_decode_group_num = dedup_sorted_uint_array(decode_group_set, decode_group_num);

    if (dedup_prefill_group_num != dedup_decode_group_num) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
            "dedup_prefill_group_num %ui and dedup_prefill_group_num %ui do not match");
        return NGX_ERROR;
    }

    for (ngx_uint_t i = 0; i < dedup_prefill_group_num; i++) {
        if (prefill_group_set[i] != decode_group_set[i]) {
            ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                "elements of prefill_group_set and decode_group_set do not match");
            return NGX_ERROR;
        }
    }

    return NGX_OK;
}

static ngx_int_t omni_proxy_check_upstream_group_setting(ngx_conf_t *cf)
{
    ngx_http_omni_loc_conf_t *olcf = local_state.loc_conf;

    if (olcf->prefill_groups == NULL && olcf->decode_groups != NULL) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
            "omni_proxy_decode_groups is set but omni_proxy_prefill_groups is missing");
        return NGX_ERROR;
    }

    if (olcf->prefill_groups != NULL && olcf->decode_groups == NULL) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
            "omni_proxy_prefill_groups is set but omni_proxy_decode_groups is missing");
        return NGX_ERROR;
    }

    if (olcf->prefill_groups == NULL && olcf->decode_groups == NULL) {
        return NGX_OK;
    }

    if (omni_groups_equal_unique(cf, olcf->prefill_groups, olcf->decode_groups) != NGX_OK) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
            "omni_proxy_prefill_groups and omni_proxy_decode_groups must contain the same unique group IDs");
        return NGX_ERROR;
    }

    ngx_uint_t i;
    ngx_uint_t j;
    ngx_http_upstream_main_conf_t *umcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_upstream_module);
    if (umcf == NULL) {
        return NGX_ERROR;
    }

    ngx_array_t *upstreams = &umcf->upstreams;
    ngx_uint_t nupstreams = upstreams->nelts;
    ngx_http_upstream_srv_conf_t **uscfp = upstreams->elts;

    for (i = 0; i < nupstreams; i++) {
        ngx_http_upstream_srv_conf_t *uscf = uscfp[i];
        if (uscf == NULL) {
            continue;
        }

        ngx_http_upstream_rr_peers_t *peers = uscf->peer.data;
        if (peers == NULL) {
            ngx_log_error(NGX_LOG_WARN, cf->log, 0,
                         "No peers found for upstream %V", &uscf->host);
            continue;
        }

        ngx_http_upstream_rr_peers_rlock(peers);

        if (ngx_strncmp(uscf->host.data, PREFILL_ENDPOINTS, sizeof(PREFILL_ENDPOINTS) - 1) == 0) {
            if (peers->number != olcf->prefill_groups->nelts) {
                ngx_http_upstream_rr_peers_unlock(peers);
                ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                    "number of prefill upstream server %ui and group %ui not equal", peers->number, olcf->prefill_groups->nelts);
                return NGX_ERROR;
            }
        } else if (ngx_strncmp(uscf->host.data, DECODE_ENDPOINTS, sizeof(DECODE_ENDPOINTS) - 1) == 0) {
            if (peers->number != olcf->decode_groups->nelts) {
                ngx_http_upstream_rr_peers_unlock(peers);
                ngx_conf_log_error(NGX_LOG_EMERG, cf, 0,
                    "number of decode upstream server %ui and group %ui not equal", peers->number, olcf->decode_groups->nelts);
                return NGX_ERROR;
            }
        } else {
            ngx_http_upstream_rr_peers_unlock(peers);
            continue;
        }
        ngx_http_upstream_rr_peers_unlock(peers);
    }

    return NGX_OK;
}

static ngx_int_t omni_proxy_post_config(ngx_conf_t *cf)
{
    if (omni_proxy_check_upstream_group_setting(cf) != NGX_OK) {
        return NGX_ERROR;
    }
    
    if (omni_proxy_init_global_state(cf) != NGX_OK)
    {
        return NGX_ERROR;
    }

    // Initialize local_state.groups with allocated requests (using local memory)
    ngx_http_omni_main_conf_t *mcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_omni_proxy_module);
    uint32_t max_slots = mcf->max_request_slots;
    for (int i = 0; i < PHASE_MAX; i++)
    {
        local_state.groups[i].phase = i;
        local_state.groups[i].num_requests = 0;
        local_state.groups[i].watermark = 0;
        size_t req_info_size = (size_t)max_slots * sizeof(omni_req_info_t);
        local_state.groups[i].requests = ngx_alloc(req_info_size, cf->log);
        if (local_state.groups[i].requests == NULL) {
            ngx_log_error(NGX_LOG_EMERG, cf->log, 0,
                "omni_proxy: failed to allocate requests array for phase %d", i);
            return NGX_ERROR;
        }
        ngx_memzero(local_state.groups[i].requests, req_info_size);
    }

    ngx_http_core_main_conf_t *cmcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_core_module);

    ngx_http_handler_pt *h = ngx_array_push(&cmcf->phases[NGX_HTTP_CONTENT_PHASE].handlers);
    if (h == NULL)
    {
        return NGX_ERROR;
    }
    *h = ngx_http_omni_proxy_metrics_handler;

    h = ngx_array_push(&cmcf->phases[NGX_HTTP_CONTENT_PHASE].handlers);
    if (h == NULL) {
        return NGX_ERROR;
    }
    *h = ngx_http_omni_proxy_health_status_handler;

    ngx_conf_log_error(NGX_LOG_NOTICE, cf, 0, "omni proxy post config finished");
    return NGX_OK;
}

static void ngx_omni_tokenizer_pipe_handler(ngx_event_t *ev)
{
    ngx_omni_tokenize_worker_t *worker = ev->data;
    uint32_t slot_id;
    ssize_t nread;

    while ((nread = read(worker->resp_pipe[OMNI_PIPE_READ], &slot_id, sizeof(slot_id))) > 0)
    {
        ngx_log_error(NGX_LOG_DEBUG, ev->log, 0, "Tokenize done for%u\n", slot_id);
        omni_req_t *req = omni_id_to_req(slot_id);
        if (req != NULL)
        {
            ngx_log_error(NGX_LOG_DEBUG, ev->log, 0,
                          "Tokenizer completed for slot: %ui, prompt: %s",
                          slot_id, req->tokenizer_req.prompt);

            struct timeval tv;
            omni_get_current_time(&tv);
            req->metrics.time_tokenized = tv;
            printf("Tokenize %ld, time(us) taken:%lu\n", req->tokenizer_req.input_len, (tv.tv_sec - req->metrics.time_contents_received.tv_sec) * 1000000 + tv.tv_usec - req->metrics.time_contents_received.tv_usec);
            print_tokenize_result(&req->tokenizer_req);
            req->metrics.prompt_num_tokens = req->tokenizer_req.input_ids_len;
            ngx_log_error(NGX_LOG_INFO,
                  ev->log,
                  0,
                  "[Tokenize-%d] get prompt_tokens %d",
                  slot_id,
                  req->tokenizer_req.input_ids_len);

            ngx_http_request_t *r = omni_get_http_request(req);
            ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                        "<<<Action: Enter state APC matching; Timestamp:%d.%06d; RequestID:%s", tv.tv_sec, tv.tv_usec, req->request_id);

            omni_proxy_post_tokenized(req);
        }
        else
        {
            ngx_log_error(NGX_LOG_DEBUG, ev->log, 0,
                          "Receive illegal slot id from tokenizer thread: %u",
                          slot_id);
        }
    }

    if (nread == -1 && errno != EAGAIN && errno != EWOULDBLOCK)
    {
        ngx_log_error(NGX_LOG_ERR, ev->log, ngx_errno,
                      "Error reading from response pipe");
        exit(-1);
    }
}

static ngx_int_t omni_proxy_init_tokenizer_worker(ngx_cycle_t *cycle)
{
    ngx_connection_t *c;

    if (local_state.loc_conf->model_path.len == 0)
    {
        return NGX_OK;
    }

    local_state.tokenize_worker.model_path = local_state.loc_conf->model_path;
    local_state.tokenize_worker.kv_block_size = local_state.loc_conf->kv_block_size;

    if (omni_tokenizer_worker_init(cycle, &local_state.tokenize_worker) != NGX_OK)
    {
        ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                      "Failed to initialize tokenizer worker");
        return NGX_ERROR;
    }

    c = ngx_get_connection(local_state.tokenize_worker.resp_pipe[OMNI_PIPE_READ], cycle->log);
    if (c == NULL)
    {
        omni_tokenizer_worker_exit(&local_state.tokenize_worker);
        ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                      "Failed to get connection for response pipe");
        return NGX_ERROR;
    }

    if (ngx_nonblocking(local_state.tokenize_worker.resp_pipe[OMNI_PIPE_READ]) == -1)
    {
        ngx_free_connection(c);
        omni_tokenizer_worker_exit(&local_state.tokenize_worker);
        ngx_log_error(NGX_LOG_EMERG, cycle->log, ngx_errno,
                      "Failed to set response pipe non-blocking");
        return NGX_ERROR;
    }

    c->read->handler = ngx_omni_tokenizer_pipe_handler;
    c->read->data = &local_state.tokenize_worker;
    c->read->log = cycle->log;

    if (ngx_add_conn(c) != NGX_OK)
    {
        ngx_free_connection(c);
        omni_tokenizer_worker_exit(&local_state.tokenize_worker);
        ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                      "Failed to add response pipe connection to event loop");
        return NGX_ERROR;
    }

    local_state.tokenize_worker.resp_connection = c;

    ngx_log_error(NGX_LOG_INFO, cycle->log, 0,
                  "Omni proxy process initialized with tokenizer worker, model: %V",
                  &local_state.loc_conf->model_path);

    g_state->has_tokenizer = true;
    return NGX_OK;
}

static void omni_proxy_kv_event_handler(struct omni_zmq_handler_s *handler,
                      const char *topic,
                      const void *message,
                      size_t length)
{
    KVEventBatch *batch = parse_kv_event_batch(message, length);
    if (!batch)
    {
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0, "Failed to parse KV event batch");
        return;
    }

    ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0, "[Worker %d] Received KV event batch with %d events",
        ngx_worker, batch->events_count);

    omni_radix_tree_t *target_radix_tree = NULL;

    if (handler->index < MAX_PREFILL_UPSTREAMS){
        omni_upstream_prefill_t *prefill = &g_state->prefill_states[handler->index];
        target_radix_tree = prefill->radix_tree;
    } else {
        omni_upstream_decode_t *decode = &g_state->decode_states[handler->index - MAX_PREFILL_UPSTREAMS];
        target_radix_tree = decode->radix_tree;
    }
    // assert(target_radix_tree==null)
    if (!target_radix_tree)
    {
        if (handler->index < MAX_PREFILL_UPSTREAMS){
            ngx_log_error(NGX_LOG_EMERG, ngx_cycle->log, 0,
                "omni_proxy_kv_event_handler: Radix tree for prefill %d is not initialized; shared state corrupted. Aborting.",
                handler->index);
        } else {
            ngx_log_error(NGX_LOG_EMERG, ngx_cycle->log, 0,
                "omni_proxy_kv_event_handler: Radix tree for decode %d is not initialized; shared state corrupted. Aborting.",
                handler->index - MAX_PREFILL_UPSTREAMS);
        }

        free_kv_event_batch(batch);

        /* Best-effort unlock if global mutex held elsewhere; unlocking here reduces chance of leaving a mutex locked.
           Note: if g_state itself is corrupted this may be a no-op or unsafe, but we attempt to be tidy. */
        ngx_shmtx_unlock(&g_state->shmtx);

        /* Ensure failure is loud in both debug and release builds */
        assert(target_radix_tree != NULL && "omni_proxy_kv_event_handler: radix_tree not initialized");
        abort();
        return;
    }

    for (size_t i = 0; i < batch->events_count; i++)
    {
        KVCacheEvent *event = (KVCacheEvent *)batch->events[i];
        if (!event)
            continue;

        switch (event->type)
        {
        case KV_EVENT_BLOCK_STORED:
        {
            int64_t *block_hashes = event->data.block_stored.block_hashes;
            size_t block_hashes_count = event->data.block_stored.block_hashes_count;
            int64_t parent_block_hash = event->data.block_stored.parent_block_hash;
            if (block_hashes_count > 0 && block_hashes != NULL)
            {
                omni_radix_tree_add_chain(target_radix_tree,
                                          (uint64_t *)block_hashes,
                                          (ngx_uint_t)block_hashes_count);
            }
            break;
        }
        case KV_EVENT_BLOCK_REMOVED:
        {
            int64_t *hashes = event->data.block_removed.block_hashes;
            size_t count = event->data.block_removed.block_hashes_count;
            if (hashes && count > 0)
            {
                for (size_t k = 0; k < count; ++k)
                {
                    uint64_t h = (uint64_t)hashes[k];
                    if (omni_radix_tree_remove(target_radix_tree, h) == NGX_OK)
                    {
                        if (handler->index < MAX_PREFILL_UPSTREAMS){
                            ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                                "Removed block hash %" PRIu64 " from radix tree (prefill %d)",
                                h, handler->index);
                        } else {
                            ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                                "Removed block hash %" PRIu64 " from radix tree (decode %d)",
                                h, handler->index - MAX_PREFILL_UPSTREAMS);                           
                        }
                    }
                    else
                    {
                        if (handler->index < MAX_PREFILL_UPSTREAMS){
                            ngx_log_error(NGX_LOG_DEBUG, ngx_cycle->log, 0,
                                "Block hash %" PRIu64 " not found in radix tree (prefill %d)",
                                h, handler->index);
                        } else {
                            ngx_log_error(NGX_LOG_DEBUG, ngx_cycle->log, 0,
                                "Block hash %" PRIu64 " not found in radix tree (decode %d)",
                                h, handler->index - MAX_PREFILL_UPSTREAMS);
                        }
                    }
                }
            }
            break;
        }
        case KV_EVENT_ALL_BLOCKS_CLEARED:
        {
            // Re-initialize the radix tree for this upstream
            omni_radix_tree_destroy(target_radix_tree);
            target_radix_tree = omni_radix_tree_init(g_state->shm);
            if (target_radix_tree == NULL) {
                ngx_log_error(NGX_LOG_EMERG, ngx_cycle->log, 0,
                              "Failed to create radix hash_tree in KV_EVENT_ALL_BLOCKS_CLEARED");
            }
            if (handler->index < MAX_PREFILL_UPSTREAMS){
                g_state->prefill_states[handler->index].radix_tree = target_radix_tree;
            } else {
                g_state->decode_states[handler->index - MAX_PREFILL_UPSTREAMS].radix_tree = target_radix_tree;
            }
            break;
        }
        default:
            break;
        }
    }

    free_kv_event_batch(batch);
}

static ngx_int_t omni_proxy_init_kv_listener(ngx_cycle_t *cycle)
{
    if (local_state.loc_conf->vllm_kv_port_offset == NGX_CONF_UNSET)
    {
        return NGX_OK;
    }

    ngx_core_conf_t *ccf = (ngx_core_conf_t *)ngx_get_conf(ngx_cycle->conf_ctx, ngx_core_module);

    uint16_t prefill_cnt = 0;
    for (int i = 0; i < MAX_PREFILL_UPSTREAMS && prefill_cnt < g_state->num_prefill_endpoints; i++) {
        omni_upstream_prefill_t *prefill = &g_state->prefill_states[i];
        if (prefill->comm.status != STATUS_ENABLE) {
            continue;
        }
        prefill_cnt++;

        if (prefill_cnt % ccf->worker_processes == ngx_worker)
        {
            if (prefill->radix_tree == NULL)
            {
                ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                              "prefill radix hash_tree is NULL, idx=%d", i);
                return NGX_ERROR;
            }

            printf("Start radix tree unit test...\n");
            omni_radix_tree_test(prefill->radix_tree);
            printf("Radix tree unit test done.\n");

            u_char *buf = ngx_palloc(cycle->pool, 64);
            u_char *last = ngx_snprintf((u_char *)buf, 64, "%s:%d",
                                        prefill->comm.address.ip,
                                        prefill->comm.address.port + local_state.loc_conf->vllm_kv_port_offset);

            printf("address: %s\n", buf);
            ngx_str_t addr = {last - buf, buf};

            ngx_str_t topic = ngx_string("");
            omni_zmq_handler_t *kv_handler = ngx_pcalloc(cycle->pool, sizeof(omni_zmq_handler_t));
            if (kv_handler == NULL) {
                ngx_log_error(NGX_LOG_EMERG, cycle->log, 0, "kv_handler alloc fail, idx=%d", i);
                return NGX_ERROR;
            }
            kv_handler->index = i;
            ngx_int_t ret = omni_zmq_handler_init(cycle, kv_handler, addr, topic, omni_proxy_kv_event_handler);
            if (ret != NGX_OK) {
                return ret;
            }
            kv_handler->next = local_state.kv_handler_list;
            local_state.kv_handler_list = kv_handler;
        }
    }

    uint16_t decode_cnt = 0;
    for (int i = 0; i < MAX_DECODE_UPSTREAMS && decode_cnt < g_state->num_decode_endpoints; i++) {
        omni_upstream_decode_t *decode = &g_state->decode_states[i];
        if (decode->comm.status != STATUS_ENABLE) {
            continue;
        }
        decode_cnt++;

        if (decode_cnt % ccf->worker_processes == ngx_worker)
        {
            if (decode->radix_tree == NULL)
            {
                ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                              "decode radix hash_tree is NULL, idx=%d", i);
                return NGX_ERROR;
            }

            printf("Start radix tree unit test...\n");
            omni_radix_tree_test(decode->radix_tree);
            printf("Radix tree unit test done.\n");

            u_char *buf = ngx_palloc(cycle->pool, 64);
            u_char *last = ngx_snprintf((u_char *)buf, 64, "%s:%d",
                                        decode->comm.address.ip,
                                        decode->comm.address.port + local_state.loc_conf->vllm_kv_port_offset);

            printf("address: %s\n", buf);
            ngx_str_t addr = {last - buf, buf};

            ngx_str_t topic = ngx_string(""); 
            omni_zmq_handler_t *kv_handler = ngx_pcalloc(cycle->pool, sizeof(omni_zmq_handler_t));
            if (kv_handler == NULL) {
                ngx_log_error(NGX_LOG_EMERG, cycle->log, 0, "kv_handler alloc fail, idx=%d", MAX_PREFILL_UPSTREAMS + i);
                return NGX_ERROR;
            }
            kv_handler->index = MAX_PREFILL_UPSTREAMS + i;
            ngx_int_t ret = omni_zmq_handler_init(cycle, kv_handler, addr, topic, omni_proxy_kv_event_handler);
            if (ret != NGX_OK) {
                return ret;
            }
            kv_handler->next = local_state.kv_handler_list;
            local_state.kv_handler_list = kv_handler;
        }
    }
}

static ngx_int_t omni_proxy_init_process(ngx_cycle_t *cycle)
{
    local_state.pid = ngx_pid;
    local_state.worker = ngx_worker;

    if (local_state.loc_conf) {
        ngx_log_error(NGX_LOG_WARN, cycle->log, 0,
                      "[OMNI SCHED] Worker %P Initialized: max_batch_num_token=%ui, prefill_max_num_seqs=%ui, prefill_starvation_timeout=%ui, decode_max_num_seqs=%ui, max_tokens_weight=%ui.",
                      ngx_pid,
                      local_state.loc_conf->max_batch_num_token,
                      local_state.loc_conf->prefill_max_num_seqs,
                      local_state.loc_conf->prefill_starvation_timeout,
                      local_state.loc_conf->decode_max_num_seqs,
                      local_state.loc_conf->max_tokens_weight);
    }

    local_state.omni_proxy_timer_event.handler = omni_proxy_timer_handler;
    local_state.omni_proxy_timer_event.log = cycle->log;
    local_state.omni_proxy_timer_event.data = NULL;

    if (omni_proxy_init_tokenizer_worker(cycle) != NGX_OK) {
        ngx_log_error(NGX_LOG_EMERG, cycle->log, 0,
                      "Worker %ui: Failed to initialize tokenizer worker", ngx_worker);
        return NGX_ERROR;
    }

    omni_register_worker(g_state, &local_state);

    ngx_log_error(NGX_LOG_INFO, cycle->log, 0,
              "Worker %ui: Init timer, pid: %P, g_state: %p, is_master_worker %d",
              ngx_worker, ngx_pid, g_state, local_state.is_master_worker);

    ngx_add_timer(&local_state.omni_proxy_timer_event, TIMER_INTERVAL);

    omni_proxy_init_kv_listener(cycle);

    return NGX_OK;
}

static void omni_proxy_exit_process(ngx_cycle_t *cycle)
{
    if (local_state.omni_proxy_timer_event.timer_set)
    {
        ngx_del_timer(&local_state.omni_proxy_timer_event);
    }

    if (local_state.tokenize_worker.resp_connection != NULL)
    {
        ngx_del_conn(local_state.tokenize_worker.resp_connection, 0);
        ngx_free_connection(local_state.tokenize_worker.resp_connection);
    }

    if (local_state.loc_conf->model_path.len != 0)
    {
        omni_tokenizer_worker_exit(&local_state.tokenize_worker);
    }

    omni_zmq_handler_exit();

    ngx_log_error(NGX_LOG_INFO, cycle->log, 0,
                  "Omni proxy process exited, tokenizer worker cleaned up");
}

static char *omni_proxy_broadcast_conf(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_str_t *value = cf->args->elts;
    if (cf->args->nelts != 2) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0, "invalid number of arguments in \"%V\"", &cmd->name);
        return NGX_CONF_ERROR;
    }

    if (ngx_strcmp(value[1].data, "on") != 0) {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0, "omni_proxy_broadcast_conf only supports \"on\"");
        return NGX_CONF_ERROR;
    }

    ngx_http_core_loc_conf_t *clcf = ngx_http_conf_get_module_loc_conf(cf, ngx_http_core_module);
    clcf->handler = omni_proxy_broadcast_handler;
    return NGX_CONF_OK;
}

static char *omni_proxy_init_conf(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_core_loc_conf_t *clcf = ngx_http_conf_get_module_loc_conf(cf, ngx_http_core_module);
    clcf->handler = omni_proxy_handler;

    ngx_memzero(&local_state, sizeof(omni_worker_local_state_t));

    ngx_http_omni_loc_conf_t *olcf = conf;
    ngx_str_t *value = cf->args->elts;

    if (cf->args->nelts != 2)
    {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0, "invalid number of arguments in \"%V\"", &cmd->name);
        return NGX_CONF_ERROR;
    }

    olcf->upstream_name = value[1];

    ngx_url_t u;
    ngx_memzero(&u, sizeof(ngx_url_t));
    u.url = value[1];
    u.no_resolve = 1;

    olcf->upstream = ngx_http_upstream_add(cf, &u, 0);
    if (olcf->upstream == NULL)
    {
        return NGX_CONF_ERROR;
    }

    return NGX_CONF_OK;
}

static char *ngx_http_omni_proxy_pd_policy(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_omni_loc_conf_t *olcf = conf;
    ngx_str_t *value = cf->args->elts;

    if (cf->args->nelts != 2)
    {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0, "invalid number of arguments in \"%V\"", &cmd->name);
        return NGX_CONF_ERROR;
    }

    if (ngx_strcmp(value[1].data, "sequential") == 0)
    {
        olcf->pd_policy = PD_SEQUENTIAL;
    }
    else if (ngx_strcmp(value[1].data, "parallel") == 0)
    {
        olcf->pd_policy = PD_PARALLEL;
    }
    else if (ngx_strcmp(value[1].data, "aggregation") == 0)
    {
        olcf->pd_policy = PD_AGGREGATION;
    }
    else
    {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0, "invalid policy \"%V\"", &value[1]);
        return NGX_CONF_ERROR;
    }

    return NGX_CONF_OK;
}

static char *ngx_http_omni_proxy_model_path(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_omni_loc_conf_t *olcf = conf;
    ngx_str_t *value = cf->args->elts;

    if (cf->args->nelts != 2)
    {
        ngx_conf_log_error(NGX_LOG_EMERG, cf, 0, "invalid number of arguments in \"%V\"", &cmd->name);
        return NGX_CONF_ERROR;
    }

    olcf->model_path = value[1];
    return NGX_CONF_OK;
}

static char *omni_proxy_metrics(ngx_conf_t *cf, ngx_command_t *cmd, void *conf)
{
    ngx_http_omni_loc_conf_t *plcf = conf;

    plcf->metrics_enabled = 1;
    return NGX_CONF_OK;
}

static ngx_int_t ngx_http_omni_proxy_health_status_handler(ngx_http_request_t *r)
{
    ngx_http_omni_loc_conf_t *olcf;
    ngx_int_t rc;
    ngx_buf_t *b;
    ngx_chain_t out;

    omni_global_state_t *gs = omni_get_global_state();
    omni_health_check_job_t *job;
    ngx_uint_t i, total_checks;

    olcf = ngx_http_get_module_loc_conf(r, ngx_http_omni_proxy_module);

    if (!olcf || !olcf->health_status_enabled) {
        return NGX_DECLINED;
    }

    job = ngx_pcalloc(r->pool, sizeof(omni_health_check_job_t));
    if (job == NULL) {
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }
    job->request = r;

    total_checks = gs->num_prefill_endpoints + gs->num_decode_endpoints + gs->num_encode_endpoints;
    if (total_checks == 0) {
        omni_health_send_response(job);
        return NGX_DONE;
    }
    job->pending_checks = total_checks;

    r->main->count++;

    // --- add r->connection->log ---
    uint16_t cnt = 0;
    for (i = 0; i < MAX_PREFILL_UPSTREAMS && cnt < gs->num_prefill_endpoints; i++) {
        if (gs->prefill_states[i].comm.status != STATUS_ENABLE) {
            continue;
        }
        cnt++;
        omni_run_single_health_check(job, r->connection->log, OMNI_UPSTREAM_TYPE_PREFILL, i);
    }
    for (i = 0, cnt = 0; i < MAX_ENCODE_UPSTREAMS && cnt < gs->num_encode_endpoints; i++) {
        if (gs->encode_states[i].comm.status != STATUS_ENABLE) {
            continue;
        }
        cnt++;
        omni_run_single_health_check(job, r->connection->log, OMNI_UPSTREAM_TYPE_ENCODE, i);
    }
    for (i = 0, cnt = 0; i < MAX_DECODE_UPSTREAMS && cnt < gs->num_decode_endpoints; i++) {
        if (gs->decode_states[i].comm.status != STATUS_ENABLE) {
            continue;
        }
        cnt++;
        omni_run_single_health_check(job, r->connection->log, OMNI_UPSTREAM_TYPE_DECODE, i);
    }

    return NGX_DONE;
}

static ngx_int_t omni_req_uint_var_get(ngx_http_request_t *r,
    ngx_http_variable_value_t *v, uintptr_t data)
{
    omni_req_t *ctx = omni_get_req(r);
    if (ctx == NULL) {
        v->not_found = 1;
        return NGX_OK;
    }

    u_char *p = ngx_pnalloc(r->pool, NGX_INT_T_LEN);
    if (p == NULL) {
        v->not_found = 1;
        return NGX_ERROR;
    }
    uint32_t value;
    switch (data) {
        case VAR_PROMPT_NUM_TOKENS:
            value = ctx->metrics.prompt_num_tokens;
            break;
        case VAR_DECODED_TOKENS:
            value = ctx->metrics.decoded_tokens;
            break;
        case VAR_MAX_TOKENS:
            value = ctx->metrics.max_tokens;
            break;
        case VAR_MAX_PREFILL_MATCH_DEPTH:
            value = ctx->max_prefill_match_depth;
            break;
        case VAR_MAX_DECODE_MATCH_DEPTH:
            value = ctx->max_decode_match_depth;
            break;
        case VAR_ENCODE_UPSTREAM_IDX:
            value = ctx->encode_upstream_endpoint_idx;
            break;
        case VAR_PREFILL_UPSTREAM_IDX:
            value = ctx->prefill_upstream_endpoint_idx;
            break;
        case VAR_DECODE_UPSTREAM_IDX:
            value = ctx->decode_upstream_endpoint_idx;
            break;
        default :
            v->not_found = 1;
            return NGX_ERROR;
    }

    v->len = ngx_sprintf(p, "%ui", value) - p;
    v->data = p;
    v->valid = 1;
    v->no_cacheable = 1;
    v->not_found = 0;

    return NGX_OK;
}

static ngx_int_t omni_req_time_var_get(ngx_http_request_t *r,
    ngx_http_variable_value_t *v, uintptr_t data)
{
    omni_req_t *ctx = omni_get_req(r);
    if (ctx == NULL) {
        v->not_found = 1;
        return NGX_OK;
    }

    u_char *p = ngx_pnalloc(r->pool, NGX_INT_T_LEN);
    if (p == NULL) {
        v->not_found = 1;
        return NGX_ERROR;
    }

    struct timeval time;
    switch (data) {
        case VAR_TIME_RECEIVED:
            time = ctx->metrics.time_received;
            break;
        case VAR_TIME_CONTENTS_RECEIVED:
            timersub(&ctx->metrics.time_contents_received, &ctx->metrics.time_received, &time);
            break;
        case VAR_TIME_TOKENIZED:
            timersub(&ctx->metrics.time_tokenized, &ctx->metrics.time_received, &time);
            break;
        case VAR_TIME_APC_UPDATED:
            timersub(&ctx->metrics.time_apc_updated, &ctx->metrics.time_received, &time);
            break;
        case VAR_TIME_ENTER_WAIT_PREFILL:
            timersub(&ctx->metrics.time_enter_wait_prefill, &ctx->metrics.time_received, &time);
            break;
        case VAR_TIME_PREFILL_SCHEDULED:
            timersub(&ctx->metrics.time_prefill_scheduled, &ctx->metrics.time_received, &time);
            break;
        case VAR_TIME_TO_PREFILL:
            timersub(&ctx->metrics.time_to_prefill, &ctx->metrics.time_received, &time);
            break;
        case VAR_TIME_PREFILLED:
            timersub(&ctx->metrics.time_prefilled, &ctx->metrics.time_received, &time);
            break;
        case VAR_TIME_ENTER_WAIT_DECODE:
            timersub(&ctx->metrics.time_enter_wait_decode, &ctx->metrics.time_received, &time);
            break;
        case VAR_TIME_DECODE_SCHEDULED:
            timersub(&ctx->metrics.time_decode_scheduled, &ctx->metrics.time_received, &time);
            break;
        case VAR_TIME_TO_DECODE:
            timersub(&ctx->metrics.time_to_decode, &ctx->metrics.time_received, &time);
            break;
        case VAR_TIME_ENCODE_START:
            timersub(&ctx->metrics.time_encode_start, &ctx->metrics.time_received, &time);
            break;
        case VAR_TIME_ENCODE_SCHEDULED:
            timersub(&ctx->metrics.time_encode_scheduled, &ctx->metrics.time_received, &time);
            break;
        case VAR_TIME_ENCODE_END:
            timersub(&ctx->metrics.time_encode_end, &ctx->metrics.time_received, &time);
            break;
        case VAR_TIME_LATENCY:
            timersub(&ctx->metrics.time_last_reponse, &ctx->metrics.time_received, &time);
            break;
        case VAR_TIME_FIRST_TOKEN:
            time = ctx->metrics.time_first_token;
            break;
        case VAR_TPOT:
            time = ctx->metrics.tpot;
            break;
        case VAR_TTFT:
            time = ctx->metrics.ttft;
            break;
        default :
            v->not_found = 1;
            return NGX_ERROR;
    }

    v->len = ngx_sprintf(p, "%d.%06d", time.tv_sec,time.tv_usec) - p;
    v->data = p;
    v->valid = 1;
    v->no_cacheable = 1;
    v->not_found = 0;

    return NGX_OK;
}

static ngx_http_variable_t ngx_http_omni_variables[] = {
    {ngx_string("promt_tks"), NULL, omni_req_uint_var_get,
     VAR_PROMPT_NUM_TOKENS, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("decoded_tks"), NULL, omni_req_uint_var_get,
     VAR_DECODED_TOKENS, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("max_tks"), NULL, omni_req_uint_var_get,
     VAR_MAX_TOKENS, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("prefill_max_match"), NULL, omni_req_uint_var_get,
     VAR_MAX_PREFILL_MATCH_DEPTH, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("decode_max_match"), NULL, omni_req_uint_var_get,
     VAR_MAX_DECODE_MATCH_DEPTH, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("prefill_idx"), NULL, omni_req_uint_var_get,
     VAR_PREFILL_UPSTREAM_IDX, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("decode_idx"), NULL, omni_req_uint_var_get,
     VAR_DECODE_UPSTREAM_IDX, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("encode_idx"), NULL, omni_req_uint_var_get,
     VAR_ENCODE_UPSTREAM_IDX, NGX_HTTP_VAR_CHANGEABLE, 0},

    {ngx_string("rcved"), NULL, omni_req_time_var_get,
     VAR_TIME_RECEIVED, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("cont_rcved"), NULL, omni_req_time_var_get,
     VAR_TIME_CONTENTS_RECEIVED, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("tknized"), NULL, omni_req_time_var_get,
     VAR_TIME_TOKENIZED, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("apc"), NULL, omni_req_time_var_get,
     VAR_TIME_APC_UPDATED, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("wait_p"), NULL, omni_req_time_var_get,
     VAR_TIME_ENTER_WAIT_PREFILL, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("p_sched"), NULL, omni_req_time_var_get,
     VAR_TIME_PREFILL_SCHEDULED, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("to_p"), NULL, omni_req_time_var_get,
     VAR_TIME_TO_PREFILL, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("p_ed"), NULL, omni_req_time_var_get,
     VAR_TIME_PREFILLED, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("wait_d"), NULL, omni_req_time_var_get,
     VAR_TIME_ENTER_WAIT_DECODE, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("d_sched"), NULL, omni_req_time_var_get,
     VAR_TIME_DECODE_SCHEDULED, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("to_d"), NULL, omni_req_time_var_get,
     VAR_TIME_TO_DECODE, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("e_start"), NULL, omni_req_time_var_get,
     VAR_TIME_ENCODE_START, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("e_sched"), NULL, omni_req_time_var_get,
     VAR_TIME_ENCODE_SCHEDULED, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("e_end"), NULL, omni_req_time_var_get,
     VAR_TIME_ENCODE_END, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("latency"), NULL, omni_req_time_var_get,
     VAR_TIME_LATENCY, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("1st_tk"), NULL, omni_req_time_var_get,
     VAR_TIME_FIRST_TOKEN, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("tpot"), NULL, omni_req_time_var_get,
     VAR_TPOT, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_string("ttft"), NULL, omni_req_time_var_get,
     VAR_TTFT, NGX_HTTP_VAR_CHANGEABLE, 0},
    {ngx_null_string, NULL, NULL, 0, 0, 0}
};

static ngx_int_t ngx_http_omni_add_variables(ngx_conf_t *cf)
{
    ngx_http_variable_t *var, *v;

    for (v = ngx_http_omni_variables; v->name.len; v++) {
        var = ngx_http_add_variable(cf, &v->name, v->flags);
        if (var == NULL) {
            return NGX_ERROR;
        }

        var->get_handler = v->get_handler;
        var->data = v->data;
    }

    return NGX_OK;
}

static ngx_command_t omni_proxy_commands[] = {

    {ngx_string("omni_proxy"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     omni_proxy_init_conf,
     NGX_HTTP_LOC_CONF_OFFSET,
     0,
     NULL},

    {ngx_string("omni_proxy_broadcast"),
     NGX_HTTP_LOC_CONF | NGX_CONF_FLAG,
     omni_proxy_broadcast_conf,
     NGX_HTTP_LOC_CONF_OFFSET,
     0,
     NULL},

    {ngx_string("omni_proxy_metrics"),
     NGX_HTTP_LOC_CONF | NGX_CONF_FLAG, // Now takes an argument
     omni_proxy_metrics,
     NGX_HTTP_LOC_CONF_OFFSET,
     offsetof(ngx_http_omni_loc_conf_t, metrics_enabled),
     NULL},

    {ngx_string("omni_proxy_pd_policy"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     ngx_http_omni_proxy_pd_policy,
     NGX_HTTP_LOC_CONF_OFFSET,
     0,
     NULL},

    {ngx_string("omni_proxy_model_path"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     ngx_http_omni_proxy_model_path,
     NGX_HTTP_LOC_CONF_OFFSET,
     0,
     NULL},

    {ngx_string("omni_proxy_vllm_kv_port_offset"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     ngx_conf_set_num_slot,
     NGX_HTTP_LOC_CONF_OFFSET,
     offsetof(ngx_http_omni_loc_conf_t, vllm_kv_port_offset),
     NULL},

    {ngx_string("omni_proxy_max_batch_num_token"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     ngx_conf_set_num_slot,
     NGX_HTTP_LOC_CONF_OFFSET,
     offsetof(ngx_http_omni_loc_conf_t, max_batch_num_token),
     NULL},

    {ngx_string("omni_proxy_prefill_max_num_seqs"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     ngx_conf_set_num_slot,
     NGX_HTTP_LOC_CONF_OFFSET,
     offsetof(ngx_http_omni_loc_conf_t, prefill_max_num_seqs),
     NULL},

    {ngx_string("omni_proxy_decode_max_num_seqs"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     ngx_conf_set_num_slot,
     NGX_HTTP_LOC_CONF_OFFSET,
     offsetof(ngx_http_omni_loc_conf_t, decode_max_num_seqs),
     NULL},

    {ngx_string("omni_proxy_max_tokens_weight"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     ngx_conf_set_num_slot,
     NGX_HTTP_LOC_CONF_OFFSET,
     offsetof(ngx_http_omni_loc_conf_t, max_tokens_weight),
     NULL},

    {ngx_string("omni_proxy_prefill_starvation_timeout"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     ngx_conf_set_num_slot,
     NGX_HTTP_LOC_CONF_OFFSET,
     offsetof(ngx_http_omni_loc_conf_t, prefill_starvation_timeout),
     NULL},

    {ngx_string("omni_proxy_health_status"),
     NGX_HTTP_LOC_CONF | NGX_CONF_FLAG,
     ngx_conf_set_flag_slot,
     NGX_HTTP_LOC_CONF_OFFSET,
     offsetof(ngx_http_omni_loc_conf_t, health_status_enabled),
     NULL},

    {ngx_string("omni_proxy_schedule_algo"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     ngx_conf_set_enum_slot,
     NGX_HTTP_LOC_CONF_OFFSET,
     offsetof(ngx_http_omni_loc_conf_t, schedule_algo),
     ngx_http_omni_schedule_algos},
    
    {ngx_string("prefill_pod_size"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     ngx_conf_set_num_slot,
     NGX_HTTP_LOC_CONF_OFFSET,
     offsetof(ngx_http_omni_loc_conf_t, prefill_pod_size),
     NULL},
    
    {ngx_string("decode_pod_size"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     ngx_conf_set_num_slot,
     NGX_HTTP_LOC_CONF_OFFSET,
     offsetof(ngx_http_omni_loc_conf_t, decode_pod_size),
     NULL},

    {ngx_string("omni_proxy_max_request_slots"),
     NGX_HTTP_MAIN_CONF | NGX_CONF_TAKE1,
     ngx_conf_set_num_slot,
     NGX_HTTP_MAIN_CONF_OFFSET,
     offsetof(ngx_http_omni_main_conf_t, max_request_slots),
     NULL},

    {ngx_string("stream_ops"),
     NGX_HTTP_LOC_CONF | NGX_CONF_TAKE1,
     ngx_conf_set_omni_stream_ops,
     NGX_HTTP_LOC_CONF_OFFSET,
     offsetof(ngx_http_omni_loc_conf_t, stream_ops),
     NULL},

    {ngx_string("omni_proxy_prefill_groups"),
     NGX_HTTP_LOC_CONF | NGX_CONF_1MORE,
     omni_proxy_set_prefill_groups,
     NGX_HTTP_LOC_CONF_OFFSET,
     0,
     NULL},

    {ngx_string("omni_proxy_decode_groups"),
     NGX_HTTP_LOC_CONF | NGX_CONF_1MORE,
     omni_proxy_set_decode_groups,
     NGX_HTTP_LOC_CONF_OFFSET,
     0,
     NULL},

    ngx_null_command};

static ngx_http_module_t omni_proxy_module_ctx = {
    ngx_http_omni_add_variables, /* preconfiguration */
    omni_proxy_post_config, /* postconfiguration */
    ngx_http_omni_create_main_conf, /* create main configuration */
    ngx_http_omni_merge_main_conf,  /* init main configuration */

    NULL, /* create server configuration */
    NULL, /* merge server configuration */

    ngx_http_omni_create_loc_conf, /* create location configuration */
    ngx_http_omni_merge_loc_conf   /* merge location configuration */
};

ngx_module_t ngx_http_omni_proxy_module = {
    NGX_MODULE_V1,
    &omni_proxy_module_ctx,  /* module context */
    omni_proxy_commands,     /* module directives */
    NGX_HTTP_MODULE,         /* module type */
    NULL,                    /* init master */
    NULL,                    /* init module */
    omni_proxy_init_process, /* init process */
    NULL,                    /* init thread */
    NULL,                    /* exit thread */
    omni_proxy_exit_process, /* exit process */
    NULL,                    /* exit master */
    NGX_MODULE_V1_PADDING,
};
