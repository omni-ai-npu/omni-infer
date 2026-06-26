// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>

#include <uuid/uuid.h>

#define UUID_STR_LEN 37 /* 36 for uuid, and 1 for '\0' */
#define TRACE_ID_STR_LEN 56 /* 55 for trace_id, and 1 for '\0' */
#define TIME_STR_LEN 33 /* 32 for start_time_ns, and 1 for '\0' */

typedef struct {
    ngx_flag_t enable;
    ngx_flag_t enable_trace;
} ngx_http_set_request_id_conf_t;

/* Function declarations */
static void *ngx_http_set_request_id_create_loc_conf(ngx_conf_t *cf);
static char *ngx_http_set_request_id_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child);
static ngx_int_t ngx_http_set_request_id_post_config(ngx_conf_t *cf);
static ngx_int_t ngx_http_set_request_id_handler(ngx_http_request_t *r);


static ngx_command_t ngx_http_set_request_id_commands[] = {
    {
        ngx_string("set_request_id"),
        NGX_HTTP_MAIN_CONF | NGX_HTTP_SRV_CONF | NGX_HTTP_LOC_CONF | NGX_CONF_FLAG,
        ngx_conf_set_flag_slot,
        NGX_HTTP_LOC_CONF_OFFSET,
        offsetof(ngx_http_set_request_id_conf_t, enable),
        NULL
    },
    { ngx_string("set_trace_headers_force"),
      NGX_HTTP_MAIN_CONF | NGX_HTTP_SRV_CONF | NGX_HTTP_LOC_CONF | NGX_CONF_FLAG,
      ngx_conf_set_flag_slot,
      NGX_HTTP_LOC_CONF_OFFSET,
      offsetof(ngx_http_set_request_id_conf_t, enable_trace),
      NULL },
    ngx_null_command
};

static ngx_http_module_t ngx_http_set_request_id_module_ctx = {
    NULL,                                      /* preconfiguration */
    ngx_http_set_request_id_post_config,       /* postconfiguration */
    NULL,                                      /* create main configuration */
    NULL,                                      /* init main configuration */
    NULL,                                      /* create server configuration */
    NULL,                                      /* merge server configuration */
    ngx_http_set_request_id_create_loc_conf,   /* create location configuration */
    ngx_http_set_request_id_merge_loc_conf     /* merge location configuration */
};

ngx_module_t ngx_http_set_request_id_module = {NGX_MODULE_V1,
    &ngx_http_set_request_id_module_ctx,  // Module context
    ngx_http_set_request_id_commands,     // Module commands
    NGX_HTTP_MODULE,                      // Module type
    NULL,                                 // init master
    NULL,                                 // init module
    NULL,                                 // init process
    NULL,                                 // init thread
    NULL,                                 // exit thread
    NULL,                                 // exit process
    NULL,                                 // exit master
    NGX_MODULE_V1_PADDING};

static void gen_uuid(unsigned char out[UUID_STR_LEN])
{
    uuid_t uuid_data;
    uuid_generate(uuid_data);
    uuid_unparse_lower(uuid_data, (char *)out);
    return;
}

static void gen_traceparent(u_char out[TRACE_ID_STR_LEN])
{
    /* 55 = 2 + 1 + 32 + 1 + 16 + 1 + 2 + 1  (end \0) */
    u_char trace_id[16];
    u_char span_id[8];

    /*  16+8 bytes */
    for (size_t i = 0; i < 16; i++) trace_id[i] = ngx_random() & 0xff;
    for (size_t i = 0; i < 8; i++)  span_id[i]  = ngx_random() & 0xff;

    uint64_t trace_hi, trace_lo;
    memcpy(&trace_hi, trace_id,      8);
    memcpy(&trace_lo, trace_id + 8,  8);

    uint64_t span;
    memcpy(&span, span_id, 8);
    ngx_snprintf(out, TRACE_ID_STR_LEN, "00-%016xL%016xL-%016xL-01",
                 trace_hi, trace_lo, span);

}

static void gen_start_time_ns(unsigned char out[TIME_STR_LEN])
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned long long nsec = (unsigned long long)tv.tv_sec * 1000000000ULL + (unsigned long long)tv.tv_usec * 1000ULL;
    snprintf((char *)out, TIME_STR_LEN, "%llu", nsec);
    // out end with '\0'
}
static u_char x_request_id[] = "X-Request-Id";
static u_char x_trace_headers[] = "Traceparent";
static u_char x_start_time_ns[] = "Start_time_ns";
static u_char x_ttft_time_ns[] = "TTFT_Start_time";
static u_char x_ttft_traceparent[] = "TTFT_traceparent";

static ngx_int_t add_header(ngx_http_request_t *r, u_char *key, size_t key_len, u_char *value, size_t value_len, size_t content_len)
{
    ngx_list_part_t *part = &r->headers_in.headers.part;
    ngx_table_elt_t *header = part->elts;
    ngx_uint_t i;
    u_char *p;

    ngx_table_elt_t *h = ngx_list_push(&r->headers_in.headers);
    if (h == NULL) {
        return NGX_ERROR;
    }
    // key
    p = ngx_palloc(r->pool, key_len);
    if (p == NULL) return NGX_ERROR;
    ngx_memcpy(p, key, key_len);
    h->key.len = key_len -1;
    h->key.data = p;
    h->hash = ngx_hash_key_lc(h->key.data, h->key.len);

    // value
    if (value_len != 0){
        p = ngx_palloc(r->pool, value_len);
        if (p == NULL) return NGX_ERROR;
        ngx_memcpy(p, value, value_len);
        h->value.len = value_len -1;
        h->value.data = p;
    }
    if (content_len != 0) {
        p = ngx_palloc(r->pool, content_len+1);
        if (p == NULL) return NGX_ERROR;
        ngx_memcpy(p, value, content_len);
        p[content_len] = '\0';
        h->value.len = content_len;
        h->value.data = p;
    }

    h->lowcase_key = ngx_pnalloc(r->pool, h->key.len);
    if (h->lowcase_key == NULL) return NGX_ERROR;
    ngx_strlow(h->lowcase_key, h->key.data, h->key.len);

#if defined(nginx_version) && nginx_version >= 1023000
    h->next = NULL;
#endif

    return NGX_OK;
}

static ngx_int_t ngx_http_set_request_id_handler(ngx_http_request_t *r)
{
    u_char *p;
    ngx_table_elt_t *h;
    ngx_list_part_t *part;
    ngx_table_elt_t *header;
    ngx_uint_t i;
    ngx_http_set_request_id_conf_t *conf;
    struct timeval tv;
    ngx_int_t rc;

    conf = ngx_http_get_module_loc_conf(r, ngx_http_set_request_id_module);
    
    if (r != r->main) {
        // Skip adding a new request id for subrequests
        return NGX_DECLINED;
    }

    if (conf->enable){
        // First, check if X-Request-Id already exists
        part = &r->headers_in.headers.part;
        header = part->elts;
        int found = 0;
        for (i = 0; /* void */; i++) {
            if (i >= part->nelts) {
                if (part->next == NULL) {
                    break;
                }
                part = part->next;
                header = part->elts;
                i = 0;
            }
            if (header[i].key.len == sizeof(x_request_id) - 1 &&
                ngx_strncasecmp(header[i].key.data, x_request_id, sizeof(x_request_id) - 1) == 0) {
                // X-Request-Id already exists, skip adding a new one
                found = 1;
                break;
            }
        }
        // If not found X-Request-Id, create a new Header structure
        if (!found) {
            unsigned char uuid[UUID_STR_LEN];
            gen_uuid(uuid);
            gettimeofday(&tv, NULL);
            ngx_log_error(
                NGX_LOG_INFO, r->connection->log, 0, "<<<Action: Start to schedule; Timestamp:%d.%06d; RequestID:%s", tv.tv_sec, tv.tv_usec, uuid);
            rc = add_header(r, x_request_id, sizeof(x_request_id), uuid, UUID_STR_LEN, 0);
            if (rc != NGX_OK){
                ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "add_header X-Request-Id failed: %i", rc);
                return rc;
            }

        }
    }

    /* ====== Add: trace_headers（W3C Traceparent and Start_time_ns) ====== */
    int found =0;
    part = &r->headers_in.headers.part;
    header = part->elts;
    // If Traceparent already exists, skip adding a new Traceparent, and add TTFT_traceparent, Start_time_ns, TTFT_Start_time
    for (i = 0; /* void */; i++) {
        if (i >= part->nelts) {
            if (part->next == NULL) break;
            part = part->next;
            header = part->elts;
            i = 0;
        }
        if (header[i].key.len == sizeof(x_trace_headers) - 1 &&
            ngx_strncasecmp(header[i].key.data, x_trace_headers, sizeof(x_trace_headers) - 1) == 0) {
            ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "****Traceparent already exists****; trace_headers:%s", header[i].value.data);
            found = 1;
            /* ====== Add: TTFT_traceparent ====== */
            rc = add_header(r, x_ttft_traceparent, sizeof(x_ttft_traceparent), header[i].value.data, TRACE_ID_STR_LEN, 0);
            if (rc != NGX_OK){
                ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "add_header TTFT_traceparent failed: %i", rc);
                return rc;
            }
            break;
        }
    }

    // If traceparent does not exists, and "set_trace_headers_force" is off, then do nothing
    /* Add "set_trace_headers_force on/off" in global_proxy.sh or omni_proxy.sh */
    if (!conf->enable_trace && found ==0 ) {
        return NGX_DECLINED;
    }
    // If traceparent does not exists, and set_trace_headers_force=on, then force to add trace_headers: Traceparent, TTFT_traceparent, Start_time_ns, TTFT_Start_time
    if (!found) {
        u_char tp[TRACE_ID_STR_LEN];
        gen_traceparent(tp);
        gettimeofday(&tv, NULL);
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "****Action: Inject Traceparent****; Timestamp:%d.%06d; trace_headers:%s", tv.tv_sec, tv.tv_usec, tp);

        rc = add_header(r, x_trace_headers, sizeof(x_trace_headers), tp, TRACE_ID_STR_LEN, 0);
        if (rc != NGX_OK){
                ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "add_header Traceparent failed: %i", rc);
                return rc;
        }
        rc = add_header(r, x_ttft_traceparent, sizeof(x_ttft_traceparent), tp, TRACE_ID_STR_LEN, 0);
        if (rc != NGX_OK){
                ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "add_header TTFT_traceparent failed: %i", rc);
                return rc;
        }
    }

    /* ====== Add: Start_time_ns ====== */
    u_char start_time_ns[TIME_STR_LEN];
    gen_start_time_ns(start_time_ns);
    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0, "****Action: Inject start_time****; start_time:%s", start_time_ns);
    rc = add_header(r, x_start_time_ns, sizeof(x_start_time_ns), start_time_ns, 0 ,ngx_strlen(start_time_ns));
    if (rc != NGX_OK){
                ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "add_header Start_time_ns failed: %i", rc);
                return rc;
    }
    /* ====== Add: TTFT_Start_time ====== */
    rc = add_header(r, x_ttft_time_ns, sizeof(x_ttft_time_ns), start_time_ns, 0 ,ngx_strlen(start_time_ns));
    if (rc != NGX_OK){
                ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "add_header TTFT_Start_time failed: %i", rc);
                return rc;
    }
    return NGX_DECLINED;
}

static void *ngx_http_set_request_id_create_loc_conf(ngx_conf_t *cf)
{
    ngx_http_set_request_id_conf_t *conf;

    conf = ngx_pcalloc(cf->pool, sizeof(ngx_http_set_request_id_conf_t));
    if (conf == NULL) {
        return NULL;
    }

    conf->enable = NGX_CONF_UNSET;
    conf->enable_trace = NGX_CONF_UNSET;

    return conf;
}

static char *ngx_http_set_request_id_merge_loc_conf(ngx_conf_t *cf, void *parent, void *child)
{
    ngx_http_set_request_id_conf_t *prev = parent;
    ngx_http_set_request_id_conf_t *conf = child;

    ngx_conf_merge_value(conf->enable, prev->enable, 0);
    ngx_conf_merge_value(conf->enable_trace, prev->enable_trace, 0);

    return NGX_CONF_OK;
}

static ngx_int_t ngx_http_set_request_id_post_config(ngx_conf_t *cf)
{
    ngx_http_handler_pt *h;
    ngx_http_core_main_conf_t *cmcf;

    cmcf = ngx_http_conf_get_module_main_conf(cf, ngx_http_core_module);
    h = ngx_array_push(&cmcf->phases[NGX_HTTP_REWRITE_PHASE].handlers);
    if (h == NULL) {
        return NGX_ERROR;
    }
    *h = ngx_http_set_request_id_handler;

    return NGX_OK;
}
