// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>
#include <stdint.h>
#include <ngx_atomic.h>
#include <omni_utils.h>

omni_req_t *omni_allocate_request(omni_request_pool_t *pool, void *data)
{
    if (pool->num_requests >= pool->max_slots)
    {
        /* pool is full */
        return NULL;
    }

    size_t start = pool->head;
    size_t idx = start;

    size_t total_slots = pool->max_slots;
    size_t checked = 0;

    while (checked < total_slots)
    {
        size_t word = idx / 64;
        size_t offset = idx % 64;

        uint64_t bits = pool->in_use_bitmap[word];

        // If this word is full, skip the whole word
        if (bits == UINT64_MAX)
        {
            // Calculate how many slots remain in this word from current offset
            size_t slots_left = 64 - offset;
            // But don't overshoot the end of the bitmap
            size_t max_skip = total_slots - checked;
            size_t skip = (slots_left < max_skip) ? slots_left : max_skip;

            idx = (idx + skip) % total_slots;
            checked += skip;
            continue;
        }

        // Otherwise, check this slot
        if (!(bits & (1ULL << offset)))
        {
            // This slot is free, allocate it
            pool->in_use_bitmap[word] |= (1ULL << offset);
            pool->num_requests++;

            // Zero-initialize the slot before returning
            omni_req_t *r = &pool->slots[idx];
            memset(r, 0, sizeof(*r));

            r->in_use = 1;
            r->backend = data;
            r->slot_index = idx;

            // head always points to the next slot to try
            pool->head = (idx + 1) % total_slots;

            return r;
        }

        idx = (idx + 1) % total_slots;
        checked++;
    }

    // Should never get here if num_requests < max_slots
    return NULL;
}

/* Free a previously‐allocated request slot */
void omni_free_request(omni_request_pool_t *pool, omni_req_t *req)
{
    /* Compute index of req in the pool */
    ptrdiff_t idx = req - pool->slots;
    if (idx < 0 || idx >= (ptrdiff_t)pool->max_slots)
    {
        ngx_log_error(NGX_LOG_ERR, ngx_cycle->log, 0,
                      "omni_free_request: invalid idx=%td (max_slots=%uz)",
                      idx, pool->max_slots);
        return;
    }

    size_t word = (size_t)idx / 64;
    unsigned bit = (unsigned)idx % 64;
    uint64_t mask = (1ULL << bit);

    /* Ensure it was in use */
    if (!(pool->in_use_bitmap[word] & mask))
    {
        ngx_log_error(NGX_LOG_ERR, ngx_cycle->log, 0,
                      "omni_free_request: slot %td not in use", idx);
        return;
    }

    /* Clear the bit */
    pool->in_use_bitmap[word] &= ~mask;
    pool->num_requests--;

    req->in_use = 0;
}