// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.



#include <omni_proxy.h>
#include <omni_scheduler.h>
#include <omni_utils.h>
void omni_scheduler_record_prefill_batch_stat(omni_global_state_t *gs,
                                              uint32_t batch_size,
                                              ngx_msec_t duration)
{
    if (gs == NULL || batch_size == 0 || duration == 0)
    {
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[PrefillStat] skip invalid input: gs=%p batch_size=%ui duration=%M",
                      gs, batch_size, duration);
        return;
    }

    if (duration < 10 || duration > 600000)
    {
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[PrefillStat] skip abnormal duration=%M for batch_size=%ui",
                      duration, batch_size);
        return;
    }

    if (batch_size > OMNI_PREFILL_BATCH_STATS_MAX)
    {
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[PrefillStat] batch_size %ui > MAX %ui, clamp to MAX",
                      batch_size, OMNI_PREFILL_BATCH_STATS_MAX);
        batch_size = OMNI_PREFILL_BATCH_STATS_MAX;
    }

    omni_prefill_batch_stats_bucket_t *bucket = &gs->prefill_batch_stats.buckets[batch_size];

    if (bucket->count < OMNI_PREFILL_BATCH_STATS_WINDOW)
    {
        bucket->durations[bucket->count] = duration;
        bucket->total += (uint64_t)duration;
        bucket->count++;

        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[PrefillStat] batch=%ui append duration=%M total=%uL count=%ui avg=%M",
                      batch_size,
                      duration,
                      bucket->total,
                      bucket->count,
                      (ngx_msec_t)((bucket->total + (bucket->count / 2)) / bucket->count));

        if (bucket->count == OMNI_PREFILL_BATCH_STATS_WINDOW)
        {
            bucket->cursor = 0;
            ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                          "[PrefillStat] batch=%ui window full, reset cursor=0", batch_size);
        }
    }
    else
    {
        uint32_t pos = bucket->cursor;
        ngx_msec_t old = bucket->durations[pos];

        bucket->total -= (uint64_t)old;
        bucket->durations[pos] = duration;
        bucket->total += (uint64_t)duration;
        bucket->cursor = (pos + 1) % OMNI_PREFILL_BATCH_STATS_WINDOW;

        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[PrefillStat] batch=%ui replace pos=%ui old=%M new=%M cursor=%ui total=%uL",
                      batch_size, pos, old, duration, bucket->cursor, bucket->total);
    }

    if (bucket->count != 0)
    {
        bucket->cached_average =
            (ngx_msec_t)((bucket->total + (uint64_t)(bucket->count / 2)) / bucket->count);

        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[PrefillStat] batch=%ui new avg=%M (count=%ui window=%ui)",
                      batch_size, bucket->cached_average,
                      bucket->count, OMNI_PREFILL_BATCH_STATS_WINDOW);
    }
}


#define OMNI_SCHED_EXEC_TIME_INCREMENT ((ngx_msec_t)50)
#define OMNI_SCHED_EXEC_MAX_TIME       ((ngx_msec_t)30000)
#define OMNI_SCHED_INFLITE_TIME       ((ngx_msec_t)30)

static ngx_msec_t
omni_scheduler_recent_bucket_average(omni_global_state_t *gs, uint32_t batch_size)
{
    if (gs == NULL || batch_size == 0)
    {
        return 0;
    }

    if (batch_size > OMNI_PREFILL_BATCH_STATS_MAX)
    {
        batch_size = OMNI_PREFILL_BATCH_STATS_MAX;
    }

    omni_prefill_batch_stats_bucket_t *bucket = &gs->prefill_batch_stats.buckets[batch_size];
    if (bucket->count == 0)
    {
        return 0;
    }

    if (bucket->cached_average > OMNI_SCHED_INFLITE_TIME*batch_size)
    {
        return bucket->cached_average - OMNI_SCHED_INFLITE_TIME*batch_size;
    }

    return bucket->cached_average;
}
static ngx_msec_t
omni_scheduler_estimate_batch_exec_time(omni_global_state_t *gs, uint32_t batch_size)
{
    // Static baseline execution time mapping for small batch sizes.
    // These values represent empirical estimates for batch_size = 0, 1, 2, 3.
    static const ngx_msec_t exec_time_map[] = {0, 220, 280, 330, 380};
    static const size_t exec_time_map_len = sizeof(exec_time_map) / sizeof(exec_time_map[0]);
    if (batch_size == 0)
    {
        return 0;
    }

    if (gs != NULL)
    {
        uint32_t capped_size = batch_size;

        // Cap the batch size to prevent exceeding prefill statistics window.
        if (capped_size > OMNI_PREFILL_BATCH_STATS_MAX)
        {
            capped_size = OMNI_PREFILL_BATCH_STATS_MAX;
        }

        // Try to get the recent average execution time for this batch size.
        ngx_msec_t recent = omni_scheduler_recent_bucket_average(gs, capped_size);

        // ---------------------------------------------------------
        // Case 1.1: We have a valid recent value for this batch size.
        // ---------------------------------------------------------
        if (recent != 0)
        {
            // If actual batch size > capped size extrapolate upward.
            if (capped_size < batch_size)
            {
                uint32_t diff = batch_size - capped_size;
                uint64_t extended = (uint64_t)recent +
                                    (uint64_t)diff * (uint64_t)OMNI_SCHED_EXEC_TIME_INCREMENT;

                // Clamp to max integer range.
                if (extended > OMNI_SCHED_EXEC_MAX_TIME)
                {
                    extended = OMNI_SCHED_EXEC_MAX_TIME;
                }

                return (ngx_msec_t)extended;
            }

            // Otherwise, return the observed average directly.
            return recent;
        }

        // ---------------------------------------------------------
        // Case 1.2: No recent record for current batch size search smaller buckets.
        // ---------------------------------------------------------
        if (capped_size > 1)
        {
            for (uint32_t probe = capped_size - 1; probe > 0; probe--)
            {
                ngx_msec_t probe_recent = omni_scheduler_recent_bucket_average(gs, probe);
                if (probe_recent == 0)
                {
                    continue;
                }

                // If the found probe size < current batch size extrapolate upward.
                if (probe < batch_size)
                {
                    uint32_t diff = batch_size - probe;
                    uint64_t extended = (uint64_t)probe_recent +
                                        (uint64_t)diff * (uint64_t)OMNI_SCHED_EXEC_TIME_INCREMENT;

                    if (extended > OMNI_SCHED_EXEC_MAX_TIME)
                    {
                        extended = OMNI_SCHED_EXEC_MAX_TIME;
                    }

                    return (ngx_msec_t)extended;
                }

                // Otherwise, directly return the smaller bucket value.
                return probe_recent;
            }
        }

        // ---------------------------------------------------------
        // Case 1.3: Still no valid smaller buckets search larger buckets.
        // ---------------------------------------------------------
        if (capped_size < OMNI_PREFILL_BATCH_STATS_MAX)
        {
            for (uint32_t probe = capped_size + 1; probe <= OMNI_PREFILL_BATCH_STATS_MAX; probe++)
            {
                ngx_msec_t probe_recent = omni_scheduler_recent_bucket_average(gs, probe);
                if (probe_recent == 0)
                {
                    continue;
                }

                // If probe size < current extrapolate upward.
                if (probe < batch_size)
                {
                    uint32_t diff = batch_size - probe;
                    uint64_t extended = (uint64_t)probe_recent +
                                        (uint64_t)diff * (uint64_t)OMNI_SCHED_EXEC_TIME_INCREMENT;

                    if (extended > OMNI_SCHED_EXEC_MAX_TIME)
                    {
                        extended = OMNI_SCHED_EXEC_MAX_TIME;
                    }

                    return (ngx_msec_t)extended;
                }

                // If probe size > current extrapolate downward.
                if (probe > batch_size)
                {
                    uint32_t diff = probe - batch_size;
                    ngx_msec_t reduced = probe_recent;

                    // Prevent underflow: ensure reduced value stays positive.
                    if ((uint64_t)diff * (uint64_t)OMNI_SCHED_EXEC_TIME_INCREMENT >= (uint64_t)reduced)
                    {
                        reduced = OMNI_SCHED_EXEC_TIME_INCREMENT;
                    }
                    else
                    {
                        reduced -= diff * OMNI_SCHED_EXEC_TIME_INCREMENT;
                    }

                    return reduced;
                }

                // Exact match return it.
                return probe_recent;
            }
        }
    }

    // =========================================================
    // Case 2: No valid statistics fallback to static baseline table.
    // =========================================================
    if (batch_size < exec_time_map_len)
    {
        // Return direct static mapping if within predefined table range.
        return exec_time_map[batch_size];
    }

    // ---------------------------------------------------------
    // Case 3: Batch size exceeds static map range extrapolate linearly.
    // ---------------------------------------------------------
    uint32_t capped = (uint32_t)(exec_time_map_len - 1);
    uint32_t additional = batch_size - capped;
    uint64_t total = (uint64_t)exec_time_map[exec_time_map_len - 1] +
                     (uint64_t)additional * (uint64_t)OMNI_SCHED_EXEC_TIME_INCREMENT;

    // Cap to prevent overflow.
    if (total > OMNI_SCHED_EXEC_MAX_TIME)
    {
        total = OMNI_SCHED_EXEC_MAX_TIME;
    }

    // Final fallback estimated execution time.
    return (ngx_msec_t)total;
}

// ============================================================================
// Estimate the expected finish time of a new task.
// This is a simple linear model: base_time + exec_time.
// The base time is either the last response timestamp or the current time.
// ============================================================================
static ngx_msec_t
omni_scheduler_estimate_finish_time(ngx_msec_t last_response_time, ngx_msec_t exec_time)
{
    return ngx_current_msec + exec_time;
}


// ============================================================================
// Select the prefill node expected to finish earliest.
// This function performs a heuristic scoring across all prefill endpoints,
// estimating their wait time, current load, and batch growth impact.
// ============================================================================
static ngx_int_t
omni_scheduler_select_earliest_prefill(omni_global_state_t *gs,
                                       ngx_http_omni_loc_conf_t *olcf,
                                       ngx_msec_t *estimated_finish)
{
    ngx_int_t selected = NGX_ERROR;
    ngx_msec_t best_finish = (ngx_msec_t)-1;

    // Number of top candidates to consider for randomized selection.
    // Increasing top_k helps distribute load more evenly across prefill nodes.
    // Example:
    //   top_k = 1 deterministic best node (may cause load concentration)
    //   top_k = 3 pick randomly from top-3 candidates within threshold window
    ngx_uint_t top_k = 1;
    ngx_msec_t threshold_ms = OMNI_SCHED_EXEC_TIME_INCREMENT/2;

    ngx_uint_t cand_idx[MAX_PREFILL_UPSTREAMS];
    ngx_msec_t cand_finish[MAX_PREFILL_UPSTREAMS];
    ngx_uint_t cand_count = 0;

    uint16_t cnt = 0;
    for (uint32_t idx = 0; idx < MAX_PREFILL_UPSTREAMS && cnt < gs->num_prefill_endpoints; idx++)
    {
        omni_upstream_prefill_t *us = &gs->prefill_states[idx];
        if (us->comm.status != STATUS_ENABLE) {
            continue;
        }
        cnt++;

        if (us->num_tokens > olcf->max_batch_num_token ||
            us->num_running > olcf->prefill_max_num_seqs)
        {
            continue;
        }

        // Retrieve current batch execution metrics.
        omni_batch_metrics_t *current_batch = &us->his.his[us->his.head];
        uint32_t queued = us->num_queue;
        uint32_t executing = us->num_batch_exec;

        // Projected batch size if this new request is added.
        uint32_t projected_batch_size = queued < UINT32_MAX ? (queued + 1) : UINT32_MAX;

        // Estimate execution times for different states.
        ngx_msec_t current_exec_time = omni_scheduler_estimate_batch_exec_time(gs, executing);
        ngx_msec_t new_exec_time = omni_scheduler_estimate_batch_exec_time(gs, projected_batch_size);
        ngx_msec_t old_exec_time = omni_scheduler_estimate_batch_exec_time(gs, queued);

        // Incremental delay added by joining current queue.
        ngx_msec_t incremental_exec = 0;
        if (new_exec_time > old_exec_time && old_exec_time !=0)
        {
            incremental_exec = new_exec_time - old_exec_time;
        }

        if (current_batch->first_response_receive_time != 0
            && current_exec_time != 0
            && ngx_current_msec > current_batch->first_response_receive_time)
        {
            ngx_msec_t time_pass = ngx_current_msec - current_batch->first_response_receive_time + OMNI_SCHED_INFLITE_TIME;
            if (current_exec_time > time_pass)
            {
                current_exec_time -= time_pass;
            }
        }

        // Compute total waiting time before this batch can finish.
        // uint64_t total_wait = (uint64_t)incremental_exec * (uint64_t)queued;
        uint64_t total_wait = (uint64_t)incremental_exec * (uint64_t)(queued * queued);
        total_wait += (uint64_t)new_exec_time;
        total_wait += (uint64_t)current_exec_time;

        if (total_wait > NGX_MAX_UINT32_VALUE)
        {
            total_wait = NGX_MAX_UINT32_VALUE;
        }

        ngx_msec_t wait_time = (ngx_msec_t)total_wait;

        // Predict finish time using last response as temporal anchor.
        ngx_msec_t finish_time = omni_scheduler_estimate_finish_time(
            current_batch->first_response_receive_time,
            wait_time);

        // Log each candidates evaluation details.
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
            "[Prefill-Earliest] candidate=%ui executing=%ui first_response_receive_time=%M "
            "num_running=%ui queued=%ui total_wait=%M "
            "current_exec=%M new_exec=%M old_exec=%M wait_time=%M "
            "fin_time=%i estimated_finish=%M",
            idx,
            (ngx_uint_t)executing,
            current_batch->first_response_receive_time,
            (ngx_uint_t)us->num_running,
            (ngx_uint_t)queued,
            total_wait,
            current_exec_time,
            new_exec_time,
            old_exec_time,
            wait_time,
            (ngx_int_t)(finish_time - ngx_current_msec),
            finish_time);


        // Record candidate if within limits.
        if (cand_count < MAX_PREFILL_UPSTREAMS)
        {
            cand_idx[cand_count] = idx;
            cand_finish[cand_count] = finish_time;
            cand_count++;
        }

        // Track the globally earliest finish.
        if (selected == NGX_ERROR || finish_time < best_finish)
        {
            selected = (ngx_int_t)idx;
            best_finish = finish_time;
        }
    }

    // ------------------------------------------------------------------------
    // Step 2: No candidates available return error.
    // ------------------------------------------------------------------------
    if (cand_count == 0)
    {
        return NGX_ERROR;
    }

    // ------------------------------------------------------------------------
    // Step 3: Filter out endpoints whose finish time exceeds threshold.
    // ------------------------------------------------------------------------
    ngx_uint_t filtered_idx[MAX_PREFILL_UPSTREAMS];
    ngx_msec_t filtered_finish[MAX_PREFILL_UPSTREAMS];
    ngx_uint_t filtered_count = 0;

    for (ngx_uint_t i = 0; i < cand_count; i++)
    {
        if (cand_finish[i] <= best_finish + threshold_ms)
        {
            filtered_idx[filtered_count] = cand_idx[i];
            filtered_finish[filtered_count] = cand_finish[i];
            filtered_count++;
        }
    }

    // ------------------------------------------------------------------------
    // Step 4: Sort filtered candidates by finish time ascending.
    // ------------------------------------------------------------------------
    for (ngx_uint_t i = 0; i < filtered_count && i < top_k; i++)
    {
        ngx_uint_t best_i = i;
        for (ngx_uint_t j = i + 1; j < filtered_count; j++)
        {
            if (filtered_finish[j] < filtered_finish[best_i])
            {
                best_i = j;
            }
        }

        // Swap if necessary.
        if (best_i != i)
        {
            ngx_msec_t tmp_finish = filtered_finish[i];
            filtered_finish[i] = filtered_finish[best_i];
            filtered_finish[best_i] = tmp_finish;

            ngx_uint_t tmp_idx = filtered_idx[i];
            filtered_idx[i] = filtered_idx[best_i];
            filtered_idx[best_i] = tmp_idx;
        }
    }

    // Limit count to top_k candidates.
    if (filtered_count > top_k)
    {
        filtered_count = top_k;
    }

    // ------------------------------------------------------------------------
    // Step 5: Randomized tie-breaking among top_k candidates within threshold.
    // ------------------------------------------------------------------------
    if (filtered_count > 0)
    {
        ngx_uint_t r = (ngx_uint_t)(rand() % filtered_count);
        selected = (ngx_int_t)filtered_idx[r];
        best_finish = filtered_finish[r];

        ngx_log_error(NGX_LOG_DEBUG,
                      ngx_cycle->log,
                      0,
                      "[Prefill-Earliest] randomly selected idx=%ui among %ui candidates "
                      "(best=%M, threshold=%M, topk=%ui)",
                      filtered_idx[r],
                      filtered_count,
                      best_finish,
                      threshold_ms,
                      top_k);
    }
    else
    {
        // No candidates within acceptable threshold fallback to best only.
        ngx_log_error(NGX_LOG_INFO,
                      ngx_cycle->log,
                      0,
                      "[Prefill-Earliest] only best selected idx=%i (no candidates within threshold=%M)",
                      selected,
                      threshold_ms);
    }

    // ------------------------------------------------------------------------
    // Step 6: Return the final selection and estimated finish time.
    // ------------------------------------------------------------------------
    if (selected != NGX_ERROR && estimated_finish != NULL)
    {
        *estimated_finish = best_finish;
    }

    return selected;
}


static void update_prefill_weights(omni_req_group_t *group)
{
    uint32_t max_prompt_tokens = 0;
    ngx_msec_t max_wait_time = 0;
    for (uint32_t i = 0; i < group->watermark; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);

        if (max_prompt_tokens < req->metrics.prompt_num_tokens)
        {
            max_prompt_tokens = req->metrics.prompt_num_tokens;
        }

        ngx_msec_t waited = ngx_current_msec - req->metrics.time_received;

        if (max_wait_time < waited)
        {
            max_wait_time = waited;
        }
    }

    if (max_wait_time < 50)
    {
        max_wait_time = 50;
    }

    for (uint32_t i = 0; i < group->watermark; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);
        ngx_msec_t waited = ngx_current_msec - req->metrics.time_received;

        double token_weight = (double)(max_prompt_tokens - req->metrics.prompt_num_tokens) / max_prompt_tokens;
        double time_weight = (double)waited / max_wait_time;

        info->weight = token_weight * 0.8 + time_weight * 0.2;
    }

    omni_sort_compact_group(group);

    for (uint32_t idx = 0; idx < group->num_requests; idx++)
    {
        omni_req_info_t *info = &group->requests[idx];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[Prefill-Sort] Order %uD: slot=%uD tokens=%uD weight=%.2f",
                      idx,
                      info->slot_index,
                      req->metrics.prompt_num_tokens,
                      info->weight);
    }
}

static void update_decode_weights(omni_req_group_t *group, uint32_t max_tokens_weight)
{
    uint32_t max_total_tokens = 1;

    for (uint32_t i = 0; i < group->watermark; i++) {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use) continue;
        omni_req_t *req = omni_info_to_req(info);
        if ((req->metrics.prompt_num_tokens + max_tokens_weight * req->metrics.max_tokens) > max_total_tokens) {
            max_total_tokens = req->metrics.prompt_num_tokens + max_tokens_weight * req->metrics.max_tokens;
        }
    }

    for (uint32_t i = 0; i < group->watermark; i++) {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use) continue;
        omni_req_t *req = omni_info_to_req(info);
        info->weight = ((double)req->metrics.prompt_num_tokens + (double)max_tokens_weight * (double)req->metrics.max_tokens)/ max_total_tokens;
    }

    omni_sort_compact_group(group);

    for (uint32_t idx = 0; idx < group->num_requests; idx++) {
        omni_req_info_t *info = &group->requests[idx];
        if (!info->in_use) {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[Decode-Sort] Order %uD: slot=%uD total_tokens=%uD prompt_num_tokens=%uD max_tokens=%uD max_tokens_weight=%uD weight=%.2f",
                      idx,
                      info->slot_index,
                      req->metrics.prompt_num_tokens + max_tokens_weight * req->metrics.max_tokens,
                      req->metrics.prompt_num_tokens,
                      req->metrics.max_tokens,
                      max_tokens_weight,
                      info->weight);
    }
}

void omni_proxy_schedule_prefill(omni_global_state_t *gs, ngx_http_omni_loc_conf_t *olcf)
{
    omni_req_group_t *group = &gs->groups[PHASE_PREFILL_WAITING_SCHEDULE];

    // TODO: Check should schedule or wait based on upstream expected come back time

    update_prefill_weights(group);

    for (uint32_t i = 0; i < group->num_requests; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        omni_req_t *req = omni_info_to_req(info);

        assert(omni_req_is_in_phase(req, PHASE_PREFILL_WAITING_SCHEDULE));

        uint32_t selected = rand() % gs->num_prefill_endpoints;
        uint32_t best_match = 0;
        uint32_t best_load_tokens = UINT32_MAX;
        uint32_t best_running = UINT32_MAX;
        uint32_t best_idx = UINT32_MAX;
        ngx_flag_t used_earliest_algo = 0;
        ngx_msec_t algo_estimated_time = 0;
        uint16_t cnt = 0;

        if (olcf->schedule_algo == OMNI_PROXY_SCHEDULE_ALGO_EARLIEST_BATCH)
        {
            ngx_int_t algo_selected = omni_scheduler_select_earliest_prefill(gs, olcf, &algo_estimated_time);
            if (algo_selected != NGX_ERROR)
            {
                selected = (uint32_t)algo_selected;
                used_earliest_algo = 1;

                ngx_msec_t now_ms = ngx_current_msec;
                ngx_msec_t delta_ms = algo_estimated_time - now_ms;

                struct timeval est_tv;
                gettimeofday(&est_tv, NULL);
                est_tv.tv_sec += delta_ms / 1000;
                est_tv.tv_usec += (delta_ms % 1000) * 1000;
                if (est_tv.tv_usec >= 1000000)
                {
                    est_tv.tv_sec += est_tv.tv_usec / 1000000;
                    est_tv.tv_usec = est_tv.tv_usec % 1000000;
                }

                ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                            "[Prefill-%d] earliest-batch scheduler picked endpoint %d "
                            "(estimated_finish=%M => %d.%06d), RequestID:%s",
                            req->slot_index,
                            selected,
                            algo_estimated_time,
                            (long)est_tv.tv_sec,
                            (long)est_tv.tv_usec,
                            req->request_id);
            }
            else
            {
                ngx_log_error(NGX_LOG_WARN, ngx_cycle->log, 0,
                              "[Prefill-%d] earliest-batch scheduler had no candidate, falling back to default",
                              req->slot_index);
            }
        }

        if (!used_earliest_algo)
        {
            uint32_t best_match = 0;
            uint32_t best_load_tokens = UINT32_MAX;
            uint32_t best_running = UINT32_MAX;
            uint32_t best_idx = UINT32_MAX;

            for (uint32_t j = 0; j < MAX_PREFILL_UPSTREAMS && cnt < gs->num_prefill_endpoints; j++)
            {
                if (gs->prefill_states[j].comm.status != STATUS_ENABLE) {
                    continue;
                }
                cnt++;
                uint32_t m = req->match_depths[j];

                uint32_t load_tokens = gs->prefill_states[j].num_tokens;
                uint32_t running = gs->prefill_states[j].num_running;
                if (load_tokens > olcf->max_batch_num_token || running > olcf->prefill_max_num_seqs)
                {
                    continue;
                }
                if (m > best_match ||
                    (m == best_match && load_tokens < best_load_tokens) ||
                    (m == best_match && load_tokens == best_load_tokens && running < best_running))
                {
                    best_match = m;
                    best_load_tokens = load_tokens;
                    best_running = running;
                    best_idx = j;
                }
            }

            if (best_match > 0 && best_idx != UINT32_MAX)
            {
                selected = best_idx;
                ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0, "[Prefill-%d] Prefix cache hit on: %d with match_depth %d",
                              req->slot_index, selected, req->match_depths[selected]);
            } else {
                uint32_t least_load = UINT32_MAX;
                for (uint32_t m = gs->last_selected_prefill, cnt = 0;
                    m < MAX_PREFILL_UPSTREAMS + gs->last_selected_prefill && cnt < gs->num_prefill_endpoints; m++) {
                    uint32_t j = m % MAX_PREFILL_UPSTREAMS;
                    omni_upstream_prefill_t *prefill = &gs->prefill_states[j];
                    if (prefill->comm.status != STATUS_ENABLE) {
                        if (prefill->comm.status == STATUS_UNUSED) {
                            m = MAX_PREFILL_UPSTREAMS - 1; /* skip unuse upstream and start from 0 */
                        }
                        continue;
                    }
                    cnt++;
                    if (gs->prefill_states[j].num_tokens < least_load)
                    {
                        least_load = gs->prefill_states[j].num_tokens;
                        selected = j;
                        if (least_load == 0)
                        {
                            break;
                        }
                    }
                }
                ngx_log_error(NGX_LOG_INFO,
                              ngx_cycle->log,
                              0,
                              "[Prefill-%d] No Prefix cache hit, choose least workload Prefill %d with load %d",
                              req->slot_index,
                              selected,
                              least_load);
            }
        }

        omni_upstream_prefill_t *selected_prefill = &gs->prefill_states[selected];
        req->prefill_upstream_endpoint_idx = selected;
        gs->last_selected_prefill = selected + 1;

        if (selected_prefill->num_running == 0)
        {
            selected_prefill->num_batch_exec = 1;
            selected_prefill->num_queue = 0;
            selected_prefill->idle_batch = true;
        }
        else
        {
            selected_prefill->num_queue++;
        }

        ngx_atomic_fetch_add(&selected_prefill->num_running, 1);
        ngx_atomic_fetch_add(&selected_prefill->num_tokens, req->metrics.prompt_num_tokens);
        ngx_atomic_fetch_add(&selected_prefill->comm.ref, 1);

        omni_global_phase_change_to(req, PHASE_PREFILL_WAITING_SCHEDULE, PHASE_PREFILL_SCHEDULED);
        omni_req_leave_phase(req, PHASE_PREFILL_WAITING_SCHEDULE);
        omni_req_enter_phase(req, PHASE_PREFILL_SCHEDULED);

        // If policy is parallel, we can change to DECODE_SCHEDULED directly
        if (gs->pd_policy == PD_PARALLEL)
        {
            req->decode_upstream_endpoint_idx = 0;
            gs->decode_states[selected].num_running++;
            ngx_atomic_fetch_add(&gs->decode_states[selected].comm.ref, 1);

            omni_add_req_to_group(req->slot_index, &gs->groups[PHASE_DECODE_SCHEDULED]);
            omni_req_enter_phase(req, PHASE_DECODE_SCHEDULED);
        }

        req->metrics.time_prefill_scheduled = ngx_current_msec;

        struct timeval tv;
        gettimeofday(&tv, NULL);
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                            "<<<Action: Enter state P scheduled; Timestamp:%d.%06d; RequestID:%s", tv.tv_sec, tv.tv_usec, req->request_id);

        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0, "[Prefill-%d] Schedule to: %d",
                      req->slot_index, req->prefill_upstream_endpoint_idx);
    }

    // TODO: estimated expected next schedule time
}

void omni_proxy_schedule_decode(omni_global_state_t *gs, ngx_http_omni_loc_conf_t *olcf)
{
    omni_req_group_t *group = &gs->groups[PHASE_DECODE_WAITING_SCHEDULE];
    // TODO: Check should schedule or wait based on upstream expected come back time
    // TODO: Here we can do some estimation of pull kv finish time to make sure pull kv
    // workloads are balanced

    update_decode_weights(group, olcf->max_tokens_weight);

    for (size_t i = 0; i < group->watermark; i++)
    {
        omni_req_info_t *info = &group->requests[i];
        if (!info->in_use)
        {
            continue;
        }
        omni_req_t *req = omni_info_to_req(info);
        assert(omni_req_is_in_phase(req, PHASE_DECODE_WAITING_SCHEDULE));

        uint32_t least_load = UINT32_MAX;
        uint32_t selected = rand() % gs->num_decode_endpoints;
        uint32_t cnt = 0;
        for (int m = gs->last_selected_decode;
            m < MAX_DECODE_UPSTREAMS + gs->last_selected_decode && cnt < gs->num_decode_endpoints; m++) {
            int j = m % MAX_DECODE_UPSTREAMS;
            omni_upstream_decode_t *decode = &gs->decode_states[j];
            if (decode->comm.status != STATUS_ENABLE) {
                if (decode->comm.status == STATUS_UNUSED) {
                    m = MAX_DECODE_UPSTREAMS - 1; /* skip unuse upstream and start from 0 */
                }
                continue;
            }
            cnt++;
            if (gs->decode_states[j].num_tokens < least_load && gs->decode_states[j].num_running < olcf->decode_max_num_seqs)
            {
                least_load = gs->decode_states[j].num_tokens;
                selected = j;
                if (least_load == 0)
                {
                    break;
                }
            }
        }

        req->decode_upstream_endpoint_idx = selected;
        gs->last_selected_decode = selected + 1;
        ngx_atomic_fetch_add(&gs->decode_states[selected].num_running, 1);
        ngx_atomic_fetch_add(&gs->decode_states[selected].num_tokens, req->metrics.prompt_num_tokens);
        ngx_atomic_fetch_add(&gs->decode_states[selected].comm.ref, 1);

        omni_global_phase_change_to(req, PHASE_DECODE_WAITING_SCHEDULE, PHASE_DECODE_SCHEDULED);
        omni_req_leave_phase(req, PHASE_DECODE_WAITING_SCHEDULE);
        omni_req_enter_phase(req, PHASE_DECODE_SCHEDULED);

        req->metrics.time_decode_scheduled = ngx_current_msec;

        struct timeval tv;
        gettimeofday(&tv, NULL);
        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                            "<<<Action: Enter state D scheduled; Timestamp:%d.%06d; RequestID:%s", tv.tv_sec, tv.tv_usec, req->request_id);

        ngx_log_error(NGX_LOG_INFO, ngx_cycle->log, 0,
                      "[Decode-%d] Schedule to: %d (load=%ui)",
                      req->slot_index,
                      req->decode_upstream_endpoint_idx,
                      gs->decode_states[selected].num_tokens);
    }
}
