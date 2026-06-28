// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <omni_pd_body_rewrite.h>
#include <jsmn.h>

#define KV_HOST_STR "\"bootstrap_host\":"
#define KV_PORT_STR ",\"bootstrap_port\":"
#define KV_ROOM_STR ",\"bootstrap_room\":"

/* 7 contains one comma, 4 double quotation mark, one '}' and '\0' */
#define SGLANG_EXTRA_BODY_LEN   (strlen(KV_HOST_STR) + strlen(KV_PORT_STR) + NGX_SOCKADDR_STRLEN \
    + strlen(KV_ROOM_STR) + strlen("\"7FFFFFFFFFFFFFFF\"") + 7)

static char *prefill_response_json_keys[] = {
    "kv_transfer_params",
};
static char *encoder_response_json_keys = "ec_transfer_params";


static const ngx_str_t stream_options_content_str = ngx_string("\"stream_options\":{\"include_usage\":true,\"continuous_usage_stats\":true}");
static const ngx_str_t stream_all_content_str = ngx_string("\"stream\":true,\"stream_options\":{\"include_usage\":true,\"continuous_usage_stats\":true}");

static unsigned int prefill_response_json_keys_len = sizeof(prefill_response_json_keys) / sizeof(char *);

typedef struct omni_json_slice_s {
    int start;
    int end;
} omni_json_slice_t;

static const char *omni_mm_types[] = {"image_url", "audio_url", "input_audio"};
static const size_t omni_mm_types_len = sizeof(omni_mm_types) / sizeof(omni_mm_types[0]);

static ngx_int_t omni_json_token_eq(const char *json, const jsmntok_t *tok, const char *key)
{
    size_t key_len = ngx_strlen(key);
    if ((size_t)(tok->end - tok->start) != key_len) {
        return 0;
    }
    return ngx_strncmp((const u_char *)json + tok->start, (const u_char *)key, key_len) == 0;
}

static int omni_json_skip(const jsmntok_t *tokens, int count, int i)
{
    if (i >= count) {
        return i;
    }

    int j = i;
    int remaining = 1;

    while (remaining > 0 && j < count) {
        if (tokens[j].type == JSMN_OBJECT) {
            remaining += tokens[j].size * 2;
        } else if (tokens[j].type == JSMN_ARRAY) {
            remaining += tokens[j].size;
        }
        remaining--;
        j++;
    }

    return j;
}

/**
 * JSMN 解析函数，支持自动扩容（通过 ngx_palloc），不再需要手动预分配
 * 返回解析得到的 token 数量，或负数表示错误
 */
static int omni_origin_body_jsmn_cached(ngx_http_request_t *r,
    omni_req_context_t *ctx,
    const char *json,
    size_t len,
    jsmntok_t **tokens_out)
{
    jsmntok_t *tokens = NULL;
    size_t tokens_size = 0;
    jsmn_parser parser;
    int ret;

    if (ctx == NULL || json == NULL || tokens_out == NULL) {
        return JSMN_ERROR_INVAL;
    }

    if (ctx->body_cache.is_parsed &&
        ctx->body_cache.json_data == (u_char *)json &&  // pointer check
        ctx->body_cache.json_len == len)
    {
        *tokens_out = ctx->body_cache.tokens;
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                      "omni_origin_body_jsmn_cached: reuse cached tokens, count=%d, len=%uz",
                      ctx->body_cache.tokens_count, len);
        return ctx->body_cache.tokens_count;
    }

    jsmn_init(&parser, r->pool);
    ret = jsmn_parse(&parser, json, len, &tokens, &tokens_size);
    if (ret < 0) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "omni_origin_body_jsmn_cached: jsmn_parse failed, ret=%d", ret);
        ctx->origin_body_tokens = NULL;
        ctx->origin_body_tokens_size = ret;
        *tokens_out = NULL;
        return ret;
    }

    ctx->body_cache.json_data = (u_char *)json;
    ctx->body_cache.json_len = len;
    ctx->body_cache.tokens = tokens;
    ctx->body_cache.tokens_count = ret;
    ctx->body_cache.tokens_capacity = (int)tokens_size;
    ctx->body_cache.is_parsed = 1;

    ctx->origin_body_tokens = tokens;
    ctx->origin_body_tokens_size = ret;
    *tokens_out = tokens;
    return ret;
}

static ngx_int_t omni_json_is_mm_type(const char *json, const jsmntok_t *tok)
{
    if (tok->type != JSMN_STRING) {
        return 0;
    }
    for (size_t i = 0; i < omni_mm_types_len; i++) {
        if (omni_json_token_eq(json, tok, omni_mm_types[i])) {
            return 1;
        }
    }
    return 0;
}

static int omni_prefill_resp_jsmn_cached(ngx_http_request_t *r,
    omni_req_context_t *ctx,
    const char *json,
    size_t len,
    jsmntok_t **tokens_out)
{
    if (ctx == NULL || json == NULL || tokens_out == NULL) {
        if (tokens_out != NULL) {
            *tokens_out = NULL;
        }
        return JSMN_ERROR_INVAL;
    }

    if (ctx->prefill_resp_cache.is_parsed &&
        ctx->prefill_resp_cache.json_data == (const u_char *)json &&
        ctx->prefill_resp_cache.json_len == len)
    {
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                      "omni_prefill_resp_jsmn_cached: reuse cached tokens, "
                      "count=%d, len=%uz",
                      ctx->prefill_resp_cache.tokens_count, len);
        *tokens_out = ctx->prefill_resp_cache.tokens;
        return ctx->prefill_resp_cache.tokens_count;
    }

    jsmn_parser parser;
    jsmntok_t *tokens = NULL;
    size_t tokens_size = 0;
    jsmn_init(&parser, r->pool);
    int ret = jsmn_parse(&parser, json, len, &tokens, &tokens_size);
    if (ret < 0) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "omni_prefill_resp_jsmn_cached: jsmn_parse failed, ret=%d", ret);
        *tokens_out = NULL;
        return ret;
    }

    ctx->prefill_resp_cache.json_data = (const u_char *)json;
    ctx->prefill_resp_cache.json_len = len;
    ctx->prefill_resp_cache.tokens = tokens;
    ctx->prefill_resp_cache.tokens_count = ret;
    ctx->prefill_resp_cache.tokens_capacity = (int)tokens_size;
    ctx->prefill_resp_cache.is_parsed = 1;

    *tokens_out = tokens;
    return ret;
}

static void omni_try_set_model_name_from_json(ngx_http_request_t *r, omni_req_context_t *ctx, const u_char *json_data, size_t json_len)
{
    if (json_data == NULL || json_len == 0) {
        return;
    }

    omni_global_state_t *gs = omni_get_global_state();
    if (gs == NULL) {
        return;
    }

    if (gs->model_name_len != 0) {
        // already set
        return;
    }

    jsmntok_t *tokens = NULL;
    int ntok = omni_origin_body_jsmn_cached(r, ctx, (const char *)json_data, json_len, &tokens);
    if (ntok <= 0) {
        return;
    }

    // find top "model": "<value>"
    for (int i = 1; i < ntok - 1; i++) {
        if (tokens[i].type == JSMN_STRING) {
            int key_len = tokens[i].end - tokens[i].start;
            if (key_len == 5 && ngx_strncmp((u_char *)json_data + tokens[i].start, "model", 5) == 0) {
                jsmntok_t *val = &tokens[i + 1];
                if (val->type == JSMN_STRING) {
                    const u_char *v = (const u_char *)json_data + val->start;
                    size_t vlen = (size_t)(val->end - val->start);
                    if (vlen > sizeof(gs->model_name) - 1) {
                        vlen = sizeof(gs->model_name) - 1;
                    }

                    ngx_shmtx_lock(&gs->shmtx);
                    if (gs->model_name_len == 0) {
                        ngx_memcpy(gs->model_name, v, vlen);
                        gs->model_name[vlen] = '\0';
                        gs->model_name_len = (ngx_uint_t)vlen;
                        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                                      "Set global model_name to \"%s\" (len=%ui)",
                                      gs->model_name, gs->model_name_len);
                    }
                    ngx_shmtx_unlock(&gs->shmtx);
                }
                break;
            }
        }
    }
}
// Helper: Find the index of a key in a JSMN token array within a given scope
// scope_start: the index to start searching from (inclusive)
// scope_end: the index to stop searching at (exclusive)
// Returns -1 if not found
int find_jsmn_key_in_scope(ngx_http_request_t *r, const char *json, jsmntok_t *tokens, int tokens_size, const char *key, int scope_start, int scope_end)
{
    size_t key_len = strlen(key);
    for (int i = scope_start; i < scope_end; i++)
    {
        if (tokens[i].type == JSMN_STRING && (int)key_len == tokens[i].end - tokens[i].start &&
            strncmp(json + tokens[i].start, key, tokens[i].end - tokens[i].start) == 0)
        {
            return i;
        }
    }
    return -1;
}

// Helper: Find the index of a key in a JSMN token array
int find_jsmn_key(ngx_http_request_t *r, const char *json, jsmntok_t *tokens, int tokens_size, const char *key)
{
    ngx_log_debug1(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, "gen decode request: token size %d", tokens_size);

    size_t key_len = strlen(key);
    for (int i = 1; i < tokens_size; i++)
    {
        if (tokens[i].type == JSMN_STRING && (int)key_len == tokens[i].end - tokens[i].start &&
            strncmp(json + tokens[i].start, key, tokens[i].end - tokens[i].start) == 0)
        {
            if (i + 1 >= tokens_size)
            {
                continue;
            }
            return i;
        }
    }
    return -1;
}

// Helper: Calculate the end token index of a nested object
// Returns the index of the first token AFTER this object
int get_jsmn_nested_object_end(jsmntok_t *tokens, int tokens_size, int obj_idx)
{
    int end = obj_idx + 1;
    for (int k = 0; k < tokens[obj_idx].size && end < tokens_size; k++) {
        end = omni_json_skip(tokens, tokens_size, end);  // skip key
        end = omni_json_skip(tokens, tokens_size, end);  // skip value
    }
    return end;
}

// Check if kv_transfer_params is null or absent in prefill response
// Also validates that required fields (remote_cluster_id, remote_host_ip, remote_block_ids) are present
// Returns: 0 if present with all required fields, 1 if null, 2 if absent,
//          3 if required fields missing, -1 on parse error
ngx_int_t omni_check_kv_transfer_params_null(ngx_http_request_t *r, char *body, size_t body_size, omni_req_context_t *ctx)
{
    jsmntok_t *tokens = NULL;
    int ret = omni_prefill_resp_jsmn_cached(r, ctx, body, body_size, &tokens);
    if (ret < 0)
    {
        return KV_TRANSFER_ERROR;
    }

    int key_idx = find_jsmn_key(r, body, tokens, ret, "kv_transfer_params");
    if (key_idx == -1)
    {
        return KV_TRANSFER_ABSENT;
    }

    int val_idx = key_idx + 1;
    if (val_idx >= ret)
    {
        return KV_TRANSFER_ABSENT;
    }
    if (tokens[val_idx].type == JSMN_PRIMITIVE && tokens[val_idx].end - tokens[val_idx].start == 4 &&
        strncmp(body + tokens[val_idx].start, "null", 4) == 0)
    {
        return KV_TRANSFER_NULL;
    }

    // kv_transfer_params is present and not null, check for required fields
    if (tokens[val_idx].type != JSMN_OBJECT)
    {
        return KV_TRANSFER_ERROR;
    }

    // Calculate the scope: search only within the kv_transfer_params object
    // kv_scope_start = first token inside the object (after the OBJECT token)
    // kv_scope_end = first token AFTER the kv_transfer_params object
    int kv_scope_start = val_idx + 1;
    int kv_scope_end = get_jsmn_nested_object_end(tokens, ret, val_idx);

    // Check for remote_cluster_id within kv_transfer_params
    int remote_cluster_id_idx = find_jsmn_key_in_scope(r, body, tokens, ret, "remote_cluster_id", kv_scope_start, kv_scope_end);
    if (remote_cluster_id_idx == -1)
    {
        return KV_TRANSFER_MISSING_FIELDS;
    }

    // Check for remote_host_ip within kv_transfer_params
    int remote_host_ip_idx = find_jsmn_key_in_scope(r, body, tokens, ret, "remote_host_ip", kv_scope_start, kv_scope_end);
    if (remote_host_ip_idx == -1)
    {
        return KV_TRANSFER_MISSING_FIELDS;
    }

    // Check for remote_block_ids within kv_transfer_params
    int remote_block_ids_idx = find_jsmn_key_in_scope(r, body, tokens, ret, "remote_block_ids", kv_scope_start, kv_scope_end);
    if (remote_block_ids_idx == -1)
    {
        return KV_TRANSFER_MISSING_FIELDS;
    }

    return KV_TRANSFER_PRESENT;
}

// Remove "kv_transfer_params":<value> from JSON body, handling any position.
// Works by finding the preceding comma (or being the first key), then removing
// the entire key-value pair including surrounding comma/whitespace.
// Handles both null values and object values.
// Returns new body size, or -1 on error.
ngx_int_t omni_remove_kv_transfer_params_null(
    ngx_http_request_t *r,
    char *body,
    size_t body_size,
    char **new_body_out,
    omni_req_context_t *ctx)
{
    jsmntok_t *tokens = NULL;
    int ret = omni_prefill_resp_jsmn_cached(r, ctx, body, body_size, &tokens);
    if (ret < 0)
    {
        return -1;
    }

    int key_idx = find_jsmn_key(r, body, tokens, ret, "kv_transfer_params");
    if (key_idx == -1)
    {
        return -1;
    }

    int val_idx = key_idx + 1;
    if (val_idx >= ret)
    {
        return -1;
    }
    jsmntok_t *val_tok = &tokens[val_idx];

    // Calculate end position of the value
    size_t val_end;
    if (val_tok->type == JSMN_PRIMITIVE)
    {
        // null value
        val_end = (size_t)val_tok->end;
    }
    else if (val_tok->type == JSMN_OBJECT || val_tok->type == JSMN_ARRAY)
    {
        // Object or Array - val_tok->end contains the position after the closing } or ]
        val_end = (size_t)val_tok->end;
    }
    else
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "DEBUG remove_kv: unexpected val_type=%d", val_tok->type);
        return -1;
    }

    size_t remove_start;
    size_t remove_end;

    // jsmn token.start for strings points to the first char inside quotes,
    // so the opening quote is at token.start - 1.
    // We search backwards from the opening quote to find comma or '{'.
    size_t key_quote = (size_t)tokens[key_idx].start - 1;
    size_t p = key_quote;
    while (p > 0 && (body[p - 1] == ' ' || body[p - 1] == '\t' ||
                     body[p - 1] == '\n' || body[p - 1] == '\r'))
    {
        p--;
    }

    if (p > 0 && body[p - 1] == ',')
    {
        // kv_transfer_params follows another field: remove ',"kv_transfer_params":null'
        remove_start = p - 1;
        remove_end = val_end;
        while (remove_end < body_size && (body[remove_end] == ' ' || body[remove_end] == '\t'))
        {
            remove_end++;
        }
    }
    else
    {
        // kv_transfer_params is the first key: remove '"kv_transfer_params":<value>'
        remove_start = key_quote;
        remove_end = val_end;
        while (remove_end < body_size && (body[remove_end] == ' ' || body[remove_end] == '\t'))
        {
            remove_end++;
        }
        if (remove_end < body_size && body[remove_end] == ',')
        {
            remove_end++;
        }
        while (remove_end < body_size && (body[remove_end] == ' ' || body[remove_end] == '\t'))
        {
            remove_end++;
        }
    }

    size_t new_size = body_size - (remove_end - remove_start);
    if (new_size <= 0 || new_size >= body_size)
    {
        return -1;
    }

    char *new_body = ngx_palloc(r->pool, new_size + 1);
    if (new_body == NULL)
    {
        return -1;
    }

    // Copy before remove_start
    if (remove_start > 0)
    {
        ngx_memcpy(new_body, body, remove_start);
    }
    // Copy after remove_end
    if (remove_end < body_size)
    {
        ngx_memcpy(new_body + remove_start, body + remove_end, body_size - remove_end);
    }
    new_body[new_size] = '\0';

    *new_body_out = new_body;
    return new_size;
}

// Helper to copy a substring from JSON based on token
void json_token_tostr(const char *json, const jsmntok_t *t, char *buf, size_t buflen)
{
    size_t len = t->end - t->start;
    if (len >= buflen)
        len = buflen - 1;
    strncpy(buf, json + t->start, len);
    buf[len] = '\0';
}

void omni_proxy_prepare_decode_request_body(ngx_http_request_t *r, omni_req_context_t *ctx)
{
    // Parse prefill_response_body
    jsmntok_t *tokens = NULL;
    ngx_chain_t *chain_1st = NULL;
    ngx_chain_t *chain = NULL;
    ngx_buf_t *b = NULL;
    ngx_chain_t *chain_new = NULL;
    ngx_buf_t *b_new = NULL;
    int total_len = 0;

    ngx_log_debug2(NGX_LOG_DEBUG_HTTP,
                   r->connection->log,
                   0,
                   "gen decode request: prefill response body: %d %s",
                   ctx->prefill_response_body_size,
                   ctx->prefill_response_body);

    // PD_PARALLEL 模式下 prefill_response_body 可能为 NULL，直接跳过 jsmn 解析
    int prefill_ret = 0;
    if (ctx->prefill_response_body != NULL && ctx->prefill_response_body_size > 0) {
        prefill_ret = omni_prefill_resp_jsmn_cached(r, ctx, (const char *)ctx->prefill_response_body, ctx->prefill_response_body_size, &tokens);
        if (prefill_ret < 0)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen decode json: jsmn_parse failed %d", prefill_ret);
            ngx_http_finalize_request(r, NGX_ERROR);
            return;
        }
    }

    // Create chain for decode request
    chain = ngx_alloc_chain_link(r->pool);
    if (chain == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen decode request: failed to allocate chain link");
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }

    // 1st buf in chain is origin body
    b = ngx_pcalloc(r->pool, sizeof(ngx_buf_t));
    if (b == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen decode request: failed to ngx_pcalloc");
        ngx_http_finalize_request(r, NGX_ERROR);
        return;
    }
    b->pos = ctx->origin_body_data;
    b->last = b->pos + ctx->origin_body_data_size;
    b->memory = 1; /* content is in read-only memory */
    chain->buf = b;
    chain_1st = chain;

    // bufs then are from prefill response, one for each key
    for (int i = 0; i < (int)prefill_response_json_keys_len; i++)
    {
        int key_idx = find_jsmn_key(
            r, (char *)(ctx->prefill_response_body), tokens, prefill_ret, prefill_response_json_keys[i]);
        if (key_idx == -1)
        {
            ngx_log_error(NGX_LOG_DEBUG_HTTP,
                          r->connection->log,
                          0,
                          "gen decode request: key not found %s",
                          prefill_response_json_keys[i]);
            continue;
        }

        while (b->last > b->pos)
        {
            if (b->last[0] == '}')
            {
                break;
            }
            b->last -= 1;
        }

        // Key found, we will palloc a new buf for this key value pair string
        int val_idx = key_idx + 1;
        if (val_idx >= prefill_ret)
        {
            continue;
        }
        int val_len = tokens[val_idx].end - tokens[val_idx].start;
        int key_len = ngx_strlen(prefill_response_json_keys[i]);
        int len = key_len + val_len + 16;
        b_new = ngx_create_temp_buf(r->pool, len);
        if (b_new == NULL)
        {
            ngx_log_error(
                NGX_LOG_ERR, r->connection->log, 0, "gen decode request: failed to ngx_create_temp_buf %d", len);
            ngx_http_finalize_request(r, NGX_ERROR);
            return;
        }

        size_t pos = 0;
        b_new->pos[pos++] = ',';
        b_new->pos[pos++] = '\"';
        ngx_memcpy(b_new->pos + pos, prefill_response_json_keys[i], key_len);
        pos += ngx_strlen(prefill_response_json_keys[i]);
        b_new->pos[pos++] = '\"';
        b_new->pos[pos++] = ':';
        if (tokens[val_idx].type == JSMN_STRING)
        {
            b_new->pos[pos++] = '\"';
        }
        ngx_memcpy(b_new->pos + pos, ctx->prefill_response_body + tokens[val_idx].start, val_len);
        pos += (size_t)val_len;
        if (tokens[val_idx].type == JSMN_STRING)
        {
            b_new->pos[pos++] = '\"';
        }
        b_new->pos[pos++] = '}';
        b_new->pos[pos] = '\0';
        b_new->last = b_new->pos + pos;
        b_new->memory = 1; /* content is in read-only memory */
        b_new->last_buf = 1;
        b_new->last_in_chain = 1;

        chain_new = ngx_alloc_chain_link(r->pool);
        if (chain == NULL)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen decode request: failed to allocate new chain link");
            ngx_http_finalize_request(r, NGX_ERROR);
            return;
        }
        chain_new->buf = b_new;
        chain->next = chain_new;

        chain = chain_new;
        b = b_new;
    }

    jsmntok_t *origin_tokens = ctx->origin_body_tokens;
    size_t origin_token_size = (size_t)ctx->origin_body_tokens_size;

    if (origin_tokens == NULL || origin_token_size <= 0)
    {
        origin_tokens = NULL;
        origin_token_size = 0;
        jsmn_parser parser;
        jsmn_init(&parser, r->pool);
        int origin_ret = jsmn_parse(
            &parser, (char *)(ctx->origin_body_data), ctx->origin_body_data_size, &origin_tokens, &origin_token_size);
        if (origin_ret < 0)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen decode json: parse origin body failed %d", origin_ret);
            ngx_http_finalize_request(r, NGX_ERROR);
            return;
        }
        origin_token_size = (size_t)origin_ret;
        ctx->origin_body_tokens = origin_tokens;
        ctx->origin_body_tokens_size = origin_ret;
    }

    int need_add_stream_option = 0;
    char *stream_str = "stream";
    char *stream_opt_str = "stream_options";
    int stream_idx = -1;
    int stream_is_true = 0;

    if (origin_tokens != NULL && origin_token_size > 0)
    {
        stream_idx = find_jsmn_key(
            r, (char *)(ctx->origin_body_data), origin_tokens, origin_token_size, stream_str);
        if (stream_idx != -1)
        {
            int stream_val_idx = stream_idx + 1;
            if (stream_val_idx < (int) origin_token_size &&
                origin_tokens[stream_val_idx].type == JSMN_PRIMITIVE &&
                strncmp(ctx->origin_body_data + origin_tokens[stream_val_idx].start, "true", 4) == 0)
            {
                stream_is_true = 1;
            }
        }
    }

    int include_usage_true = 0;
    int continuous_usage_true = 0;

    if (stream_is_true)
    {
        need_add_stream_option = 0;
        int stream_option_idx = find_jsmn_key(
            r, (char *)(ctx->origin_body_data), origin_tokens, origin_token_size, stream_opt_str);
        include_usage_true = 0;
        continuous_usage_true = 0;

        if (stream_option_idx != -1)
        {
            int stream_option_val_idx = stream_option_idx + 1;
            if (stream_option_val_idx < (int) origin_token_size &&
                origin_tokens[stream_option_val_idx].type == JSMN_OBJECT)
            {
                for (int i = 0; i < origin_tokens[stream_option_val_idx].size; ++i)
                {
                    int key_token_idx = stream_option_val_idx + 1 + 2 * i;
                    if (key_token_idx >= (int) origin_token_size)
                    {
                        break;
                    }
                    char keybuf[32];
                    json_token_tostr((char *)(ctx->origin_body_data), &origin_tokens[key_token_idx], keybuf, sizeof(keybuf));
                    int val_token_idx = key_token_idx + 1;
                    if (val_token_idx >= (int) origin_token_size)
                    {
                        break;
                    }
                    if (strcmp(keybuf, "include_usage") == 0)
                    {
                        if (origin_tokens[val_token_idx].type == JSMN_PRIMITIVE &&
                            strncmp(ctx->origin_body_data + origin_tokens[val_token_idx].start, "true", 4) == 0)
                        {
                            include_usage_true = 1;
                        }
                    }
                    if (strcmp(keybuf, "continuous_usage_stats") == 0)
                    {
                        if (origin_tokens[val_token_idx].type == JSMN_PRIMITIVE &&
                            strncmp(ctx->origin_body_data + origin_tokens[val_token_idx].start, "true", 4) == 0)
                        {
                            continuous_usage_true = 1;
                        }
                    }
                }
            }
        }
        if (!(include_usage_true && continuous_usage_true))
        {
            need_add_stream_option = 1;
        }
    }
    else
    {
        need_add_stream_option = 0;
    }

    ngx_http_omni_loc_conf_t *plcf = ngx_http_get_module_loc_conf(r, ngx_http_omni_proxy_module);
    if (plcf != NULL)
    {
        switch (plcf->stream_ops)
        {
        case NGX_PREFILL_STREAM_ADD:
            need_add_stream_option = 1;
            break;
        case NGX_PREFILL_STREAM_SET_OPT:
            break;
        case NGX_PREFILL_STREAM_OFF:
        default:
            need_add_stream_option = 0;
            break;
        }
    }

    if (need_add_stream_option)
    {
        size_t content_len;
        u_char *content_data_to_copy;

        while (b->last > b->pos)
        {
            if (b->last[-1] == '}')
            {
                b->last -= 1;
                break;
            }
            b->last -= 1;
        }

        if (stream_is_true)
        {
            content_len = stream_options_content_str.len;
            content_data_to_copy = stream_options_content_str.data;
        }
        else
        {
            content_len = stream_all_content_str.len;
            content_data_to_copy = stream_all_content_str.data;
        }

        b_new = ngx_create_temp_buf(r->pool, content_len + 3);
        if (b_new == NULL)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "ngx_create_temp_buf: failed! ");
            ngx_http_finalize_request(r, NGX_ERROR);
            return;
        }

        size_t pos = 0;
        b_new->pos[pos++] = ',';
        ngx_memcpy(b_new->pos + pos, content_data_to_copy, content_len);
        pos += content_len;
        b_new->pos[pos++] = '}';
        b_new->pos[pos] = '\0';
        b_new->last = b_new->pos + pos;
        b_new->memory = 1;

        chain_new = ngx_alloc_chain_link(r->pool);
        if (chain_new == NULL)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "ngx_alloc_chain_link: failed! ");
            ngx_http_finalize_request(r, NGX_ERROR);
            return;
        }

        chain->buf->last_buf = 0;
        chain->buf->last_in_chain = 0;
        b_new->last_buf = 1;
        b_new->last_in_chain = 1;

        chain_new->buf = b_new;
        chain_new->next = NULL;
        chain->next = chain_new;
        chain = chain_new;
        b = b_new;
    }


    if (omni_get_global_state()->pd_policy == PD_PARALLEL) {
        size_t content_len;
        u_char *content_data_to_copy;

        while (b->last > b->pos)
        {
            if (b->last[-1] == '}')
            {
                b->last -= 1;
                break;
            }
            b->last -= 1;
        }

        u_int pos = 0;
        size_t buf_len = SGLANG_EXTRA_BODY_LEN;
        b_new = ngx_create_temp_buf(r->pool, buf_len);
        if (b_new == NULL)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "bootstrap_host ngx_create_temp_buf: failed! ");
            ngx_http_finalize_request(r, NGX_ERROR);
            return;
        }
        if (ctx->bootstrap_host.len == 0) {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "bootstrap_host is 0");
            ngx_http_finalize_request(r, NGX_ERROR);
            return;
        }
        pos += ngx_snprintf(b_new->pos + pos, buf_len - pos, ","KV_HOST_STR"\"%V\"", &ctx->bootstrap_host)
               - (b_new->pos + pos);

        if (ctx->bootstrap_port.len > 0) {
            pos += ngx_snprintf(b_new->pos + pos, buf_len - pos, KV_PORT_STR"\"%V\"", &ctx->bootstrap_port)
                   - (b_new->pos + pos);
        } else {
            pos += ngx_snprintf(b_new->pos + pos, buf_len - pos, KV_PORT_STR"null") - (b_new->pos + pos);
        }

        pos += ngx_snprintf(b_new->pos + pos, buf_len - pos, KV_ROOM_STR"\"%V\"", &ctx->bootstrap_room)
               - (b_new->pos + pos);

        b_new->pos[pos++] = '}';
        b_new->pos[pos] = '\0';
        b_new->last = b_new->pos + pos;
        b_new->memory = 1;

        chain_new = ngx_alloc_chain_link(r->pool);
        if (chain_new == NULL)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "ngx_alloc_chain_link: failed! ");
            ngx_http_finalize_request(r, NGX_ERROR);
            return;
        }

        chain->buf->last_buf = 0;
        chain->buf->last_in_chain = 0;
        b_new->last_buf = 1;
        b_new->last_in_chain = 1;

        chain_new->buf = b_new;
        chain_new->next = NULL;
        chain->next = chain_new;
        chain = chain_new;
    }

    if (chain->buf != NULL)
    {
        chain->buf->last_buf = 1;
        chain->buf->last_in_chain = 1;
    }

    chain->next = NULL;

    // Set up the subrequest's body structure
    if (r->request_body == NULL)
    {
        r->request_body = ngx_pcalloc(r->pool, sizeof(ngx_http_request_body_t));
        if (r->request_body == NULL)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen decode request: failed to allocate request_body_t");
            ngx_http_finalize_request(r, NGX_ERROR);
            return;
        }
    }

    // Update subrequest's body properties
    r->request_body->bufs = chain_1st;
    r->request_body->buf = NULL;

    // Clear any existing temp file in subrequest
    if (r->request_body->temp_file)
    {
        r->request_body->temp_file = NULL;
    }

    for (ngx_chain_t *cl = r->request_body->bufs; cl; cl = cl->next)
    {
        total_len += ngx_buf_size(cl->buf);
    }

    // #if (NGX_DEBUG)
    char *json_str = ngx_pcalloc(r->pool, total_len + 1);
    int cur_pos = 0;
    for (ngx_chain_t *cl = r->request_body->bufs; cl; cl = cl->next)
    {
        ngx_memcpy(json_str + cur_pos, cl->buf->pos, ngx_buf_size(cl->buf));
        cur_pos += ngx_buf_size(cl->buf);
    }

    // ngx_log_error(NGX_LOG_INFO,
    //               r->connection->log,
    //               0,
    //               "gen decode request: body for subrequest: %d %s",
    //               total_len,
    //               json_str);
    // #endif

    // Set content length in header
    r->headers_in.content_length_n = total_len;
    if (r->headers_in.content_length)
    {
        r->headers_in.content_length->value.len =
            ngx_sprintf(r->headers_in.content_length->value.data, "%uz", total_len) -
            r->headers_in.content_length->value.data;
    }

    return;
}

// Info for all modifications, in the order they appear in the JSON
typedef enum
{
    R_MAX_TOKENS,
    R_MAX_COMPLETION_TOKENS,
    R_STREAM,
    R_STREAM_OPTIONS
} region_type_t;
typedef struct
{
    region_type_t type;
    size_t idx;   // For value keys: value token index; For stream_options: key token index
    size_t start; // Start position in JSON for head (for stream_options it's the key start)
    size_t end;   // End position in JSON for tail (for stream_options it's the value end)
} region_info_t;

static int get_encode_transfer_params(ngx_http_request_t *r, omni_req_context_t *ctx, ngx_str_t *encode_str)
{
    jsmn_parser parser;
    size_t tokens_size = 0;
    jsmntok_t *tokens = NULL;
    int total_len = 0;

    jsmn_init(&parser, r->pool);
    int encoder_tokens_size =
        jsmn_parse(&parser, (char *)(ctx->encoder_response_body), ctx->encoder_response_body_size, &tokens, &tokens_size);
    if (encoder_tokens_size < 0)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen encode json: jsmn_parse failed %d", encoder_tokens_size);
        ngx_http_finalize_request(r, NGX_ERROR);
        return NGX_ERROR;
    }

    // bufs then are from encode response, one for each key
    int key_idx = find_jsmn_key(
        r, (char *)(ctx->encoder_response_body), tokens, encoder_tokens_size, encoder_response_json_keys);
    if (key_idx == -1)
    {
        ngx_log_error(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, "gen encode request: key not found, continue without encode transfer params");
        encode_str->data = NULL;
        encode_str->len = 0;
        return NGX_OK;
    }

    int val_idx = key_idx + 1;
    if (val_idx >= encoder_tokens_size)
    {
        encode_str->data = NULL;
        encode_str->len = 0;
        return NGX_OK;
    }
    encode_str->data = ctx->encoder_response_body + tokens[val_idx].start;
    encode_str->len = (size_t)(tokens[val_idx].end - tokens[val_idx].start);
    ngx_log_error(NGX_LOG_INFO,
                  r->connection->log,
                  0,
                  "encode transfer params %V", encode_str);
    return NGX_OK;
}

void gen_prefill_json_str_jsmn(
    ngx_http_request_t *r, omni_req_context_t *ctx, const char *json, size_t len, u_char **out, size_t *out_len)
{
    jsmntok_t *tokens = NULL;
    int ret = -1;
    int reuse_tokens = 0;
    region_info_t region_infos[4];
    int region_infos_count = 0;
    int max_tokens_val_idx = -1;
    int max_completion_tokens_val_idx = -1;
    int stream_val_idx = -1;
    int stream_options_key_idx = -1, stream_options_val_idx = -1;
    char keybuf[64];

    // Early return for invalid/empty input
    if (json == NULL || len == 0) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, 
                      "gen prefill json: empty json input");
        return;
    }

    ret = omni_origin_body_jsmn_cached(r, ctx, json, len, &tokens);
    if (ret < 0)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen prefill json: failed to parse json using jsmn %d", ret);
        return;
    }

    for (int i = 1; i < ret; ++i)
    {
        if (tokens[i].type == JSMN_STRING)
        {
            json_token_tostr(json, &tokens[i], keybuf, sizeof(keybuf));
            if (max_tokens_val_idx == -1 && strcmp(keybuf, "max_tokens") == 0)
            {
                max_tokens_val_idx = i + 1;
                region_infos[region_infos_count++] = (region_info_t){
                    R_MAX_TOKENS, max_tokens_val_idx, tokens[max_tokens_val_idx].start, tokens[max_tokens_val_idx].end};
            }
            if (max_completion_tokens_val_idx == -1 && strcmp(keybuf, "max_completion_tokens") == 0)
            {
                max_completion_tokens_val_idx = i + 1;
                region_infos[region_infos_count++] = (region_info_t){
                    R_MAX_COMPLETION_TOKENS, max_completion_tokens_val_idx, tokens[max_completion_tokens_val_idx].start, tokens[max_completion_tokens_val_idx].end};
            }
            if (stream_val_idx == -1 && strcmp(keybuf, "stream") == 0)
            {
                stream_val_idx = i + 1;
                region_infos[region_infos_count++] =
                    (region_info_t){R_STREAM, stream_val_idx, tokens[stream_val_idx].start, tokens[stream_val_idx].end};
            }
            if (stream_options_key_idx == -1 && strcmp(keybuf, "stream_options") == 0)
            {
                stream_options_key_idx = i;
                stream_options_val_idx = i + 1;
                // Locate key start and value end for removal, including preceding comma if any
                if (tokens[stream_options_key_idx].start < 0 ||
                    tokens[stream_options_val_idx].end < 0)
                {
                    ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                                  "gen prefill json: invalid negative stream_options token boundary");
                    return;
                }
                size_t so_key_start = (size_t) tokens[stream_options_key_idx].start;
                size_t so_val_end = (size_t) tokens[stream_options_val_idx].end;
                size_t j = so_key_start;
                while (j > 0 && json[j - 1] != ',')
                    j--;
                if (j > 0 && json[j - 1] == ',')
                    so_key_start = j - 1;
                region_infos[region_infos_count++] =
                    (region_info_t){R_STREAM_OPTIONS, stream_options_key_idx, so_key_start, so_val_end};
            }
        }
    }

    // Sort region_infos by start position (should be in order already due to scan, but robust)
    for (int i = 0; i < region_infos_count - 1; ++i)
    {
        for (int j = i + 1; j < region_infos_count; ++j)
        {
            if (region_infos[j].start < region_infos[i].start)
            {
                region_info_t tmp = region_infos[i];
                region_infos[i] = region_infos[j];
                region_infos[j] = tmp;
            }
        }
    }

    ngx_str_t encode_str = ngx_null_string;
    if (ctx->encoder_response_body != NULL && ctx->encoder_response_body_size != 0) {
        if (get_encode_transfer_params(r, ctx, &encode_str) != NGX_OK) {
            return;
        }
    }
    size_t cap = len + encode_str.len + 256; // ensure enough room for additional metadata
    u_char *newjson = ngx_palloc(r->pool, cap);
    if (!newjson)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "gen prefill json: palooc %d", cap);
        return;
    }

    size_t pos = 0;
    size_t src = 0; // source position in json to copy from

    for (int i = 0; i < region_infos_count; ++i)
    {
        region_info_t *ri = &region_infos[i];
        // Copy up to region
        if (ri->start > src)
        {
            ngx_memcpy(newjson + pos, json + src, ri->start - src);
            pos += ri->start - src;
        }
        switch (ri->type)
        {
        case R_MAX_TOKENS:
            ngx_memcpy(newjson + pos, "1", 1);
            pos += 1;
            src = ri->end;
            break;
        case R_MAX_COMPLETION_TOKENS:
            ngx_memcpy(newjson + pos, "1", 1);
            pos += 1;
            src = ri->end;
            break;
        case R_STREAM:
            ngx_memcpy(newjson + pos, "false", 5);
            pos += 5;
            src = ri->end;
            break;
        case R_STREAM_OPTIONS:
            src = ri->end;
            break;
        }
    }
    // Copy the remainder before closing }
    if (len > 1 && src < len - 1)
    {
        ngx_memcpy(newjson + pos, json + src, len - 1 - src);
        pos += len - 1 - src;
    }

    // If both "max_tokens" and "max_completion_tokens" were missing, insert "max_tokens":1 before '}'
    if (max_tokens_val_idx == -1 && max_completion_tokens_val_idx == -1)
    {
        if (pos > 0 && newjson[pos - 1] != '{')
        {
            newjson[pos++] = ',';
        }
        const char *insertion = "\"max_tokens\":1";
        size_t ins_len = ngx_strlen(insertion);
        ngx_memcpy(newjson + pos, insertion, ins_len);
        pos += ins_len;
    }

    if (omni_get_global_state()->pd_policy == PD_PARALLEL) {
        if (pos > 0 && newjson[pos - 1] != '{') {
            newjson[pos++] = ',';
        }

        pos += ngx_snprintf(newjson + pos, cap - pos, KV_HOST_STR"\"%V\"", &ctx->bootstrap_host)
               - (newjson + pos);

        if (ctx->bootstrap_port.len > 0)
        {
            pos += ngx_snprintf(newjson + pos, cap - pos, KV_PORT_STR"\"%V\"", &ctx->bootstrap_port)
                   - (newjson + pos);
        }
        else
        {
            pos += ngx_snprintf(newjson + pos, cap - pos, KV_PORT_STR":null") - (newjson + pos);
        }

        pos += ngx_snprintf(newjson + pos, cap - pos, KV_ROOM_STR"\"%V\"", &ctx->bootstrap_room)
               - (newjson + pos);
    }

    if (encode_str.len > 0) {
        if (pos > 0 && newjson[pos - 1] != '{') {
            newjson[pos++] = ',';
        }
        pos += ngx_snprintf(newjson + pos, cap - pos, "\"%s\":%V", encoder_response_json_keys, &encode_str)
               - (newjson + pos);
    }

    // Add closing '}'
    newjson[pos++] = '}';
    newjson[pos] = '\0';
    *out = newjson;
    *out_len = pos;

    ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                  "gen prefill json: output len=%uz, reused_tokens=%d", *out_len, reuse_tokens);

    return;
}

ngx_int_t omni_proxy_save_origin_body(
    ngx_http_request_t *r, omni_req_context_t *ctx)
{
    ngx_chain_t *cl;
    size_t len = 0;
    u_char *body_data = NULL;
    u_char *p;

    if (r->request_body == NULL || r->request_body->bufs == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill: request body is empty");
        return NGX_ERROR;
    }

    // Calculate total body size from the main request
    for (cl = r->request_body->bufs; cl != NULL; cl = cl->next)
    {
        ngx_buf_t *buf = cl->buf;
        if (buf->in_file)
        {
            len += (size_t)(buf->file_last - buf->file_pos);
        }
        else
        {
            off_t buf_size = ngx_buf_size(buf);
            if (buf_size > 0)
            {
                len += (size_t)buf_size;
            }
        }
    }

    if (len == 0)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill: request body length is zero");
        return NGX_ERROR;
    }

    // Allocate memory for the temporary body copy + null terminator
    body_data = ngx_palloc(r->pool, len + 1);
    if (body_data == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill: failed to allocate memory for body");
        return NGX_ERROR;
    }

    ctx->origin_body_data = body_data; // record body data for decode request
    ctx->origin_body_data_size = len;

    // Copy main request's buffer chain to a temporary contiguous block
    p = ctx->origin_body_data;
    len = ctx->origin_body_data_size;
    for (cl = r->request_body->bufs; cl != NULL; cl = cl->next)
    {
        ngx_buf_t *buf = cl->buf;
        size_t buf_size;
        if (buf->in_file)
        {
            buf_size = (size_t)(buf->file_last - buf->file_pos);
            if (buf_size > 0)
            {
                ssize_t n = ngx_read_file(buf->file, p, buf_size, buf->file_pos);
                if (n != (ssize_t)buf_size)
                {
                    ngx_log_error(NGX_LOG_ERR,
                                  r->connection->log,
                                  0,
                                  "prefill: failed to read body from file, expected %uz, got %z",
                                  buf_size,
                                  n);
                    return NGX_ERROR;
                }
                p += buf_size;
            }
        }
        else
        {
            buf_size = (size_t)ngx_buf_size(buf);
            if (buf_size > 0)
            {
                p = ngx_cpymem(p, buf->pos, buf_size);
            }
        }
    }
    *p = '\0';

    ngx_log_debug1(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, "prefill: original body for subrequest: %s", body_data);

    omni_try_set_model_name_from_json(r, ctx, ctx->origin_body_data, ctx->origin_body_data_size);

    return NGX_DONE;
}

ngx_int_t omni_proxy_prepare_prefill_subrequest(
    ngx_http_request_t *r, ngx_http_request_t *sr, omni_req_context_t *ctx)
{
    u_char *modified_json_str = NULL;
    size_t len = ctx->origin_body_data_size;
    u_char *body_data = ctx->origin_body_data;
    ngx_buf_t *b;

    // Parse original body to extract max_tokens if present
    jsmntok_t  *tokens = NULL;
    int         ntok = 0;
    omni_req_t *req;

    req = ctx->req;
    if (req == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "prefill: ctx->req is NULL, can not get max_tokens");
        return NGX_ERROR;
    }

    ntok = omni_origin_body_jsmn_cached(r, ctx, (char *)body_data, len, &tokens);
    if (ntok < 0) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
                      "prefill: jsmn_parse fail, error=%d", ntok);
    } else {
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                      "prefill: jsmn_parse return %d tokens", ntok);

        int key_idx = find_jsmn_key(r, (char *)body_data, tokens, ntok, "max_tokens");
        int completion_key_idx = find_jsmn_key(r, (char *)body_data, tokens, ntok, "max_completion_tokens");
        ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                      "prefill: find_jsmn_key returned %d for max_tokens, %d for max_completion_tokens", key_idx, completion_key_idx);

        // Prefer max_tokens, fall back to max_completion_tokens
        int effective_key_idx = (key_idx != -1) ? key_idx : completion_key_idx;

        if (effective_key_idx != -1 && effective_key_idx + 1 < ntok) {
            jsmntok_t *val_t = &tokens[effective_key_idx + 1];
            char buf[32];
            json_token_tostr((char *)body_data, val_t, buf, sizeof(buf));
            ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                          "prefill: max_tokens token string = '%s'", buf);

            req->metrics.max_tokens = (uint32_t)strtoul(buf, NULL, 10);
            ngx_log_error(NGX_LOG_INFO, r->connection->log, 0,
                          "prefill: req->metrics.max_tokens = %ui",
                          (ngx_uint_t)req->metrics.max_tokens);
        }
        else {
            req->metrics.max_tokens = 1;
            ngx_log_error(NGX_LOG_WARN, r->connection->log, 0,
                          "prefill: can not find max_tokens or max_completion_tokens, set a default value %ui",
                          (ngx_uint_t)req->metrics.max_tokens);
        }
    }

    // Parse JSON from the temporary copy
    size_t str_len = 0;
    gen_prefill_json_str_jsmn(sr, ctx, (char *)body_data, len, &modified_json_str, &str_len);
    if (modified_json_str == NULL || str_len == 0)
    {
        return NGX_ERROR;
    }

    ngx_log_debug2(NGX_LOG_DEBUG_HTTP,
                   r->connection->log,
                   0,
                   "prefill: modified body for subrequest: %d, %s",
                   str_len,
                   modified_json_str);

    b = ngx_pcalloc(r->pool, sizeof(ngx_buf_t));
    if (b == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill: failed to ngx_pcalloc buf");
        return NGX_ERROR;
    }
    b->pos = modified_json_str;
    b->last = b->pos + str_len;
    b->memory = 1;
    b->last_buf = (sr->request_body_no_buffering) ? 0 : 1;
    b->last_in_chain = 1;

    // Create new chain for subrequest
    ngx_chain_t *new_chain = ngx_alloc_chain_link(sr->pool);
    if (new_chain == NULL)
    {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill: failed to allocate chain link");
        return NGX_ERROR;
    }
    new_chain->buf = b;
    new_chain->next = NULL;

    // Set up the subrequest's body structure
    if (sr->request_body == NULL)
    {
        sr->request_body = ngx_pcalloc(sr->pool, sizeof(ngx_http_request_body_t));
        if (sr->request_body == NULL)
        {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "prefill: failed to allocate request_body_t");
            return NGX_ERROR;
        }
    }

    // Copy request method and headers
    sr->method = r->method;
    sr->method_name = r->method_name;

    // Copy request headers
    ngx_http_headers_in_t *headers_in = &sr->headers_in;
    headers_in->content_length_n = r->headers_in.content_length_n;
    headers_in->content_type = r->headers_in.content_type;

    // Update subrequest's body properties
    sr->request_body->bufs = new_chain;
    sr->request_body->buf = b;
    sr->request_body->rest = 0;
    sr->request_body_in_file_only = 0;
    sr->request_body_in_persistent_file = 0;
    sr->request_body_in_clean_file = 0;

    // Clear any existing temp file in subrequest
    if (sr->request_body->temp_file)
    {
        sr->request_body->temp_file = NULL;
    }

    // Update subrequest's Content-Length header
    sr->headers_in.content_length_n = (off_t) str_len;
    if (sr->headers_in.content_length)
    {
        sr->headers_in.content_length->value.len =
            ngx_sprintf(sr->headers_in.content_length->value.data, "%uz", str_len) -
            sr->headers_in.content_length->value.data;
    }
    sr->request_length = (off_t) str_len;

    return NGX_DONE;
}

ngx_int_t omni_proxy_prepare_encoder_body(ngx_http_request_t *r, omni_req_context_t *ctx)
{
    if (ctx->origin_body_data == NULL || ctx->origin_body_data_size == 0) {
        return NGX_ERROR;
    }

    const char *json = (const char *)ctx->origin_body_data;
    size_t len = ctx->origin_body_data_size;

    jsmntok_t *tokens = NULL;
    int ntok = omni_origin_body_jsmn_cached(r, ctx, json, len, &tokens);
    if (ntok <= 0) {
        return NGX_ERROR;
    }

    int messages_idx = -1;
    int model_val_idx = -1;
    if (tokens[0].type == JSMN_OBJECT) {
        int idx = 1;
        for (int i = 0; i < tokens[0].size && idx < ntok; i++) {
            jsmntok_t *key = &tokens[idx];
            jsmntok_t *val = &tokens[idx + 1];
            if (key->type == JSMN_STRING && omni_json_token_eq(json, key, "messages")) {
                messages_idx = idx + 1;
            } else if (key->type == JSMN_STRING && omni_json_token_eq(json, key, "model")) {
                model_val_idx = idx + 1;
            }
            idx = omni_json_skip(tokens, ntok, idx + 1);
        }
    }

    if (messages_idx == -1 || tokens[messages_idx].type != JSMN_ARRAY) {
        return NGX_DECLINED;
    }

    ngx_array_t *items = ngx_array_create(r->pool, 4, sizeof(omni_json_slice_t));
    if (items == NULL) {
        return NGX_ERROR;
    }

    int msg_idx = messages_idx + 1;
    for (int i = 0; i < tokens[messages_idx].size && msg_idx < ntok; i++) {
        if (tokens[msg_idx].type != JSMN_OBJECT) {
            msg_idx = omni_json_skip(tokens, ntok, msg_idx);
            continue;
        }

        int msg_obj_idx = msg_idx;
        int msg_field_idx = msg_idx + 1;
        for (int j = 0; j < tokens[msg_obj_idx].size && msg_field_idx < ntok; j++) {
            jsmntok_t *key = &tokens[msg_field_idx];
            jsmntok_t *val = &tokens[msg_field_idx + 1];
            if (key->type == JSMN_STRING && omni_json_token_eq(json, key, "content") && val->type == JSMN_ARRAY) {
                int content_idx = msg_field_idx + 1;
                int item_idx = content_idx + 1;
                for (int k = 0; k < tokens[content_idx].size && item_idx < ntok; k++) {
                    if (tokens[item_idx].type == JSMN_OBJECT) {
                        int item_obj_idx = item_idx;
                        int item_field_idx = item_idx + 1;
                        ngx_int_t is_mm = 0;
                        for (int m = 0; m < tokens[item_obj_idx].size && item_field_idx < ntok; m++) {
                            jsmntok_t *item_key = &tokens[item_field_idx];
                            jsmntok_t *item_val = &tokens[item_field_idx + 1];
                            if (item_key->type == JSMN_STRING && omni_json_token_eq(json, item_key, "type") &&
                                omni_json_is_mm_type(json, item_val)) {
                                is_mm = 1;
                                break;
                            }
                            item_field_idx = omni_json_skip(tokens, ntok, item_field_idx + 1);
                        }
                        if (is_mm) {
                            omni_json_slice_t *slice = ngx_array_push(items);
                            if (slice == NULL) {
                                return NGX_ERROR;
                            }
                            slice->start = tokens[item_obj_idx].start;
                            slice->end = tokens[item_obj_idx].end;
                        }
                    }
                    item_idx = omni_json_skip(tokens, ntok, item_idx);
                }
            }
            msg_field_idx = omni_json_skip(tokens, ntok, msg_field_idx + 1);
        }

        msg_idx = omni_json_skip(tokens, ntok, msg_idx);
    }

    if (items->nelts == 0) {
        return NGX_DECLINED;
    }

    size_t cap = len + 512;
    u_char *buf = ngx_palloc(r->pool, cap);
    if (buf == NULL) {
        return NGX_ERROR;
    }

    u_char *p = buf;
    u_char *end = buf + cap;
    *p++ = '{';

    ngx_uint_t needs_comma = 0;
    if (model_val_idx != -1 && model_val_idx < ntok) {
        if (p >= end) {
            return NGX_ERROR;
        }
        p = ngx_cpymem(p, "\"model\":", sizeof("\"model\":") - 1);
        if (tokens[model_val_idx].type == JSMN_STRING) {
            *p++ = '"';
            size_t vlen = (size_t)(tokens[model_val_idx].end - tokens[model_val_idx].start);
            if (p + vlen + 1 >= end) {
                return NGX_ERROR;
            }
            ngx_memcpy(p, json + tokens[model_val_idx].start, vlen);
            p += vlen;
            *p++ = '"';
        } else {
            size_t vlen = (size_t)(tokens[model_val_idx].end - tokens[model_val_idx].start);
            if (p + vlen >= end) {
                return NGX_ERROR;
            }
            ngx_memcpy(p, json + tokens[model_val_idx].start, vlen);
            p += vlen;
        }
        needs_comma = 1;
    }

    if (needs_comma) {
        *p++ = ',';
    }

    p = ngx_cpymem(p, "\"messages\":[{\"role\":\"user\",\"content\":[",
                   sizeof("\"messages\":[{\"role\":\"user\",\"content\":[") - 1);

    omni_json_slice_t *slice = items->elts;
    for (ngx_uint_t i = 0; i < items->nelts; i++) {
        size_t slice_len = (size_t)(slice[i].end - slice[i].start);
        if (p + slice_len + 2 >= end) {
            return NGX_ERROR;
        }
        ngx_memcpy(p, json + slice[i].start, slice_len);
        p += slice_len;
        if (i + 1 < items->nelts) {
            *p++ = ',';
        }
    }

    p = ngx_cpymem(p, "]}],\"max_tokens\":1,\"stream\":false}", sizeof("]}],\"max_tokens\":1,\"stream\":false}") - 1);

    ctx->encoder_body_data = buf;
    ctx->encoder_body_data_size = (ngx_uint_t)(p - buf);
    ctx->encoder_item_count = items->nelts;
    if (ctx->req != NULL) {
        ctx->req->encode_item_count = ctx->encoder_item_count;
    }

    return NGX_OK;
}

ngx_int_t omni_proxy_prepare_encoder_subrequest(
    ngx_http_request_t *r, ngx_http_request_t *sr, omni_req_context_t *ctx)
{
    if (ctx->encoder_body_data == NULL || ctx->encoder_body_data_size == 0) {
        return NGX_ERROR;
    }

    ngx_buf_t *b = ngx_pcalloc(r->pool, sizeof(ngx_buf_t));
    if (b == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "encoder: failed to ngx_pcalloc buf");
        return NGX_ERROR;
    }
    b->pos = ctx->encoder_body_data;
    b->last = b->pos + ctx->encoder_body_data_size;
    b->memory = 1;
    b->last_buf = (sr->request_body_no_buffering) ? 0 : 1;
    b->last_in_chain = 1;

    ngx_chain_t *new_chain = ngx_alloc_chain_link(sr->pool);
    if (new_chain == NULL) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "encoder: failed to allocate chain link");
        return NGX_ERROR;
    }
    new_chain->buf = b;
    new_chain->next = NULL;

    if (sr->request_body == NULL) {
        sr->request_body = ngx_pcalloc(sr->pool, sizeof(ngx_http_request_body_t));
        if (sr->request_body == NULL) {
            ngx_log_error(NGX_LOG_ERR, r->connection->log, 0, "encoder: failed to allocate request_body_t");
            return NGX_ERROR;
        }
    }

    sr->method = r->method;
    sr->method_name = r->method_name;

    ngx_http_headers_in_t *headers_in = &sr->headers_in;
    headers_in->content_length_n = r->headers_in.content_length_n;
    headers_in->content_type = r->headers_in.content_type;

    sr->request_body->bufs = new_chain;
    sr->request_body->buf = b;
    sr->request_body->rest = 0;
    sr->request_body_in_file_only = 0;
    sr->request_body_in_persistent_file = 0;
    sr->request_body_in_clean_file = 0;

    if (sr->request_body->temp_file) {
        sr->request_body->temp_file = NULL;
    }

    sr->headers_in.content_length_n = (off_t)ctx->encoder_body_data_size;
    if (sr->headers_in.content_length) {
        sr->headers_in.content_length->value.len =
            ngx_sprintf(sr->headers_in.content_length->value.data, "%uz", ctx->encoder_body_data_size) -
            sr->headers_in.content_length->value.data;
    }
    sr->request_length = (off_t)ctx->encoder_body_data_size;

    return NGX_DONE;
}
