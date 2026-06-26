// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <msgpack.h>
#include <ngx_core.h>

#include <omni_apc.h>

#define BLOCK_STORED_TAG "BlockStored"
#define BLOCK_REMOVED_TAG "BlockRemoved"
#define ALL_BLOCKS_CLEARED_TAG "AllBlocksCleared"
#define BLOCK_STORED_FIELDS 9
#define BLOCK_REMOVED_FIELDS 4
#define BLOCK_HASHES_INDEX 1
#define PARENT_BLOCK_HASH_INDEX 2
#define TOKEN_IDS_INDEX 3
#define BLOCK_SIZE_INDEX 4
#define LORA_ID_INDEX 5
#define MEDIUM_INDEX 6
#define LORA_NAME_INDEX 7
#define GROUP_IDX_INDEX 8
#define BLOCK_REMOVED_MEDIUM_INDEX 2
#define BLOCK_REMOVED_GROUP_IDX_INDEX 3

static int64_t *parse_int64_array(const msgpack_object *obj, size_t *out_count)
{
    if (obj->type != MSGPACK_OBJECT_ARRAY || obj->via.array.size == 0)
    {
        *out_count = 0;
        return NULL;
    }
    size_t n = obj->via.array.size;
    int64_t *arr = malloc(n * sizeof(int64_t));
    if (!arr)
    {
        *out_count = 0;
        return NULL;
    }
    for (size_t j = 0; j < n; ++j)
    {
        if (obj->via.array.ptr[j].type == MSGPACK_OBJECT_POSITIVE_INTEGER)
        {
            arr[j] = obj->via.array.ptr[j].via.u64;
        }
        else if (obj->via.array.ptr[j].type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
        {
            arr[j] = obj->via.array.ptr[j].via.i64;
        }
        else
        {
            arr[j] = 0;
        }
    }
    *out_count = n;
    return arr;
}

static char *parse_string(const msgpack_object *obj)
{
    const char *src;
    size_t len;

    if (obj->type == MSGPACK_OBJECT_NIL){
        return NULL;
    }

    if (obj->type != MSGPACK_OBJECT_STR){
        ngx_log_error(NGX_LOG_ERR, ngx_cycle->log, 0, "[KV EVENT] OBJECT TYPE: %d", obj->type);
        return NULL;
    }
    else {
        src = obj->via.str.ptr;
        len = obj->via.str.size;
    }

    char *dst = malloc(len + 1);

    if (!dst){
        ngx_log_error(NGX_LOG_ERR, ngx_cycle->log, 0, "[KV EVENT] NO MEMORY");
        return NULL;
    }

    memcpy(dst, src, len);
    dst[len] = '\0';
    return dst;
}

static int parse_block_stored(const msgpack_object_array *array, KVCacheEvent *event)
{
    event->type = KV_EVENT_BLOCK_STORED;

    if (array->size != BLOCK_STORED_FIELDS)
        return -1;

    size_t hashes_count = 0;
    event->data.block_stored.block_hashes = parse_int64_array(&array->ptr[BLOCK_HASHES_INDEX], &hashes_count);
    event->data.block_stored.block_hashes_count = hashes_count;

    if (array->ptr[PARENT_BLOCK_HASH_INDEX].type == MSGPACK_OBJECT_POSITIVE_INTEGER)
        event->data.block_stored.parent_block_hash = array->ptr[PARENT_BLOCK_HASH_INDEX].via.u64;
    else if (array->ptr[PARENT_BLOCK_HASH_INDEX].type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
        event->data.block_stored.parent_block_hash = array->ptr[PARENT_BLOCK_HASH_INDEX].via.i64;
    else
        event->data.block_stored.parent_block_hash = -1;

    size_t token_ids_count = 0;
    event->data.block_stored.token_ids = parse_int64_array(&array->ptr[TOKEN_IDS_INDEX], &token_ids_count);
    event->data.block_stored.token_ids_count = token_ids_count;

    if (array->ptr[BLOCK_SIZE_INDEX].type == MSGPACK_OBJECT_POSITIVE_INTEGER)
        event->data.block_stored.block_size = (int)array->ptr[BLOCK_SIZE_INDEX].via.u64;
    else if (array->ptr[BLOCK_SIZE_INDEX].type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
        event->data.block_stored.block_size = (int)array->ptr[BLOCK_SIZE_INDEX].via.i64;
    else
        event->data.block_stored.block_size = 128;

    if (array->ptr[LORA_ID_INDEX].type == MSGPACK_OBJECT_POSITIVE_INTEGER)
        event->data.block_stored.lora_id = array->ptr[LORA_ID_INDEX].via.u64;
    else if (array->ptr[LORA_ID_INDEX].type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
        event->data.block_stored.lora_id = array->ptr[LORA_ID_INDEX].via.i64;
    else if (array->ptr[LORA_ID_INDEX].type == MSGPACK_OBJECT_NIL)
        event->data.block_stored.lora_id = -1;
    else
        event->data.block_stored.lora_id = -1;

    char *medium_str = parse_string(&array->ptr[MEDIUM_INDEX]);

    if (!medium_str) {
        event->data.block_stored.medium = NULL;
    } else {
        event->data.block_stored.medium = medium_str;
    }

    char *lora_name_str = parse_string(&array->ptr[LORA_NAME_INDEX]);

    if (!lora_name_str) {
        event->data.block_stored.lora_name = NULL;
    } else {
        event->data.block_stored.lora_name = lora_name_str;
    }

    event->data.block_stored.group_idx = -1;
    if (array->ptr[GROUP_IDX_INDEX].type == MSGPACK_OBJECT_POSITIVE_INTEGER)
        event->data.block_stored.group_idx = array->ptr[GROUP_IDX_INDEX].via.u64;
    else if (array->ptr[GROUP_IDX_INDEX].type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
        event->data.block_stored.group_idx = array->ptr[GROUP_IDX_INDEX].via.i64;

    return 0;
}

static int parse_block_removed(const msgpack_object_array *array, KVCacheEvent *event)
{
    event->type = KV_EVENT_BLOCK_REMOVED;

    if (array->size != BLOCK_REMOVED_FIELDS)
        return -1;

    size_t hashes_count = 0;
    event->data.block_removed.block_hashes = parse_int64_array(&array->ptr[BLOCK_HASHES_INDEX], &hashes_count);
    event->data.block_removed.block_hashes_count = hashes_count;

    char *medium_str = parse_string(&array->ptr[BLOCK_REMOVED_MEDIUM_INDEX]);

    if (!medium_str) {
        event->data.block_removed.medium = NULL;
    } else {
        event->data.block_removed.medium = medium_str;
    }

    event->data.block_removed.group_idx = -1;
    if (array->ptr[BLOCK_REMOVED_GROUP_IDX_INDEX].type == MSGPACK_OBJECT_POSITIVE_INTEGER)
        event->data.block_removed.group_idx = array->ptr[BLOCK_REMOVED_GROUP_IDX_INDEX].via.u64;
    else if (array->ptr[BLOCK_REMOVED_GROUP_IDX_INDEX].type == MSGPACK_OBJECT_NEGATIVE_INTEGER)
        event->data.block_removed.group_idx = array->ptr[BLOCK_REMOVED_GROUP_IDX_INDEX].via.i64;
       
    return 0;
}

static int parse_all_blocks_cleared(const msgpack_object_array *array, KVCacheEvent *event)
{
    (void)array;
    event->type = KV_EVENT_ALL_BLOCKS_CLEARED;
    return 0;
}

static int parse_kv_event(const msgpack_object *obj, KVCacheEvent *event)
{
    if (obj->type != MSGPACK_OBJECT_ARRAY)
    {
        return -1;
    }

    const msgpack_object_array *array = &obj->via.array;
    if (array->size < 1 || array->ptr[0].type != MSGPACK_OBJECT_STR)
    {
        return -1;
    }

    const char *tag = array->ptr[0].via.str.ptr;
    size_t tag_len = array->ptr[0].via.str.size;

    if (tag_len == strlen(BLOCK_STORED_TAG) && memcmp(tag, BLOCK_STORED_TAG, tag_len) == 0)
    {
        if (array->size == BLOCK_STORED_FIELDS)
        {
            return parse_block_stored(array, event);
        }
    }
    else if (tag_len == strlen(BLOCK_REMOVED_TAG) && memcmp(tag, BLOCK_REMOVED_TAG, tag_len) == 0)
    {
        if (array->size == BLOCK_REMOVED_FIELDS)
        {
            return parse_block_removed(array, event);
        }
    }
    else if (tag_len == strlen(ALL_BLOCKS_CLEARED_TAG) && memcmp(tag, ALL_BLOCKS_CLEARED_TAG, tag_len) == 0)
    {
        return parse_all_blocks_cleared(array, event);
    }

    event->type = KV_EVENT_UNKNOWN;
    return -1;
}

KVEventBatch *parse_kv_event_batch(const void *payload, size_t length)
{
    msgpack_unpacked result;
    msgpack_unpacked_init(&result);

    if (msgpack_unpack_next(&result, (const char *)payload, length, NULL) != MSGPACK_UNPACK_SUCCESS)
    {
        msgpack_unpacked_destroy(&result);
        return NULL;
    }

    const msgpack_object *obj = &result.data;
    if (obj->type != MSGPACK_OBJECT_ARRAY || obj->via.array.size < 2)
    {
        msgpack_unpacked_destroy(&result);
        return NULL;
    }

    KVEventBatch *batch = malloc(sizeof(KVEventBatch));
    memset(batch, 0, sizeof(KVEventBatch));
    const msgpack_object_array *array = &obj->via.array;

    // Timestamp
    if (array->ptr[0].type == MSGPACK_OBJECT_FLOAT)
        batch->ts = array->ptr[0].via.f64;
    else if (array->ptr[0].type == MSGPACK_OBJECT_POSITIVE_INTEGER)
        batch->ts = array->ptr[0].via.u64;
    else
        batch->ts = 0;

    // Events
    if (array->ptr[1].type == MSGPACK_OBJECT_ARRAY)
    {
        size_t n = array->ptr[1].via.array.size;
        batch->events_count = n;
        batch->events = malloc(n * sizeof(KVCacheEvent *));
        for (size_t j = 0; j < n; ++j)
        {
            KVCacheEvent *event = malloc(sizeof(KVCacheEvent));
            memset(event, 0, sizeof(KVCacheEvent));
            if (parse_kv_event(&array->ptr[1].via.array.ptr[j], event) == 0)
            {
                batch->events[j] = event;
            }
            else
            {
                free(event);
                batch->events[j] = NULL;
            }
        }
    }

    msgpack_unpacked_destroy(&result);
    return batch;
}

void free_kv_event_batch(KVEventBatch *batch)
{
    if (!batch)
        return;
    for (size_t i = 0; i < batch->events_count; i++)
    {
        if (batch->events[i])
        {
            KVCacheEvent *event = batch->events[i];
            if (event->type == KV_EVENT_BLOCK_STORED)
            {
                free(event->data.block_stored.block_hashes);
                free(event->data.block_stored.token_ids);
                free(event->data.block_stored.medium);
                free(event->data.block_stored.lora_name);
            }
            else if (event->type == KV_EVENT_BLOCK_REMOVED)
            {
                free(event->data.block_removed.block_hashes);
                free(event->data.block_removed.medium);
            }
            free(event);
        }
    }
    free(batch->events);
    free(batch);
}
