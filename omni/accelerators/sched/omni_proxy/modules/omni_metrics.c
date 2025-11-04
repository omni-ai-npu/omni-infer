// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <ngx_core.h>
#include <omni_metrics.h>
#include <omni_proxy.h>
#include <time.h>
#define DECODE_INSTANCE_GROUP_SIZE 16

// Label definitions
static const char *LABEL_ENDPOINT[] = {"endpoint"};
static const char *LABEL_PHASE[] = {"phase"};

// Singleton metrics registry
static omni_metric_desc_t *metrics_registry = NULL;
static size_t metrics_count = 0;

//Collector bucket 
static const ngx_uint_t OMNI_TTFT_BUCKET_BOUNDS_MS[OMNI_TTFT_BUCKETS_COUNT - 1] =
    {1, 5, 10, 20, 40, 60, 80, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 
    20000, 40000, 80000, 160000, 640000, 2560000};
static const ngx_uint_t OMNI_TPOT_BUCKET_BOUNDS_MS[OMNI_TPOT_BUCKETS_COUNT - 1] =
    {10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 2500, 5000, 7500, 10000, 
    20000, 40000, 80000};
static const ngx_uint_t OMNI_E2E_BUCKET_BOUNDS_MS[OMNI_E2E_BUCKETS_COUNT - 1] =
    {300, 500, 800, 1000, 1500, 2000, 2500, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 
    60000, 120000, 240000, 480000, 960000, 1920000, 7680000};

static ngx_uint_t
omni_hist_bucket_index_ms(ngx_msec_t v, const ngx_uint_t *bounds, ngx_uint_t n_bounds)
{
    for (ngx_uint_t i = 0; i < n_bounds; i++)
    {
        if ((ngx_uint_t)v <= bounds[i])
        {
            return i; // in i th bucket
        }
    }
    return n_bounds; // +Inf bucket
}

void omni_metrics_record_ttft(omni_global_state_t *gs, ngx_msec_t ttft_ms)
{
    if (gs == NULL || ttft_ms <= 0)
        return;
    ngx_uint_t idx = omni_hist_bucket_index_ms(ttft_ms, OMNI_TTFT_BUCKET_BOUNDS_MS,
                                               (ngx_uint_t)(OMNI_TTFT_BUCKETS_COUNT - 1));
    ngx_atomic_fetch_add(&gs->ttft_buckets[idx], 1);
    ngx_atomic_fetch_add(&gs->ttft_count, 1);
    ngx_atomic_fetch_add(&gs->ttft_sum_ms, (ngx_atomic_uint_t)ttft_ms);
}

void omni_metrics_record_tpot(omni_global_state_t *gs, ngx_msec_t tpot_ms)
{
    if (gs == NULL || tpot_ms <= 0)
        return;
    ngx_uint_t idx = omni_hist_bucket_index_ms(tpot_ms, OMNI_TPOT_BUCKET_BOUNDS_MS,
                                               (ngx_uint_t)(OMNI_TPOT_BUCKETS_COUNT - 1));
    ngx_atomic_fetch_add(&gs->tpot_buckets[idx], 1);
    ngx_atomic_fetch_add(&gs->tpot_count, 1);
    ngx_atomic_fetch_add(&gs->tpot_sum_ms, (ngx_atomic_uint_t)tpot_ms);
}

void omni_metrics_record_e2e(omni_global_state_t *gs, ngx_msec_t e2e_ms)
{
    if (gs == NULL || e2e_ms <= 0)
        return;

    ngx_uint_t idx = omni_hist_bucket_index_ms(e2e_ms, OMNI_E2E_BUCKET_BOUNDS_MS,
                                               (ngx_uint_t)(OMNI_E2E_BUCKETS_COUNT - 1));
    ngx_atomic_fetch_add(&gs->e2e_buckets[idx], 1);
    ngx_atomic_fetch_add(&gs->e2e_count, 1);
    ngx_atomic_fetch_add(&gs->e2e_sum_ms, (ngx_atomic_uint_t)e2e_ms);
}

static inline const char *omni_metrics_get_model_name(omni_global_state_t *gs) {
    if (gs && gs->model_name_len > 0) {
        return gs->model_name;
    }
    return "unknown";
}

#define UPSTREAM_METRIC(endpoint, name, help, value)                                                          \
    metrics_registry[index] = (omni_metric_desc_t){                                                           \
        name,                                                                                                 \
        help,                                                                                                 \
        OMNI_METRIC_GAUGE,                                                                                    \
        {{0}},                                                                                                \
        2,                                                                                                    \
        {.int_values = &endpoint->value},                                                                     \
        OMNI_VALUE_INT32,                                                                                     \
        1};                                                                                                   \
    snprintf(&metrics_registry[index].label_names[0][0],                                                      \
             OMNI_METRICS_MAX_LABEL_LEN, "endpoint=\"%s:%d\"", endpoint->comm.address.ip, endpoint->comm.address.port); \
    snprintf(&metrics_registry[index].label_names[1][0],                                                      \
             OMNI_METRICS_MAX_LABEL_LEN, "role=\"prefill\"");                                                 \
    index++;

#define PHASE_METRIC(phase, name, help, value)                    \
    metrics_registry[index] = (omni_metric_desc_t){               \
        name,                                                     \
        help,                                                     \
        OMNI_METRIC_GAUGE,                                        \
        {{0}},                                                    \
        1,                                                        \
        {.int_values = &global_state->groups[phase].value},       \
        OMNI_VALUE_INT32,                                         \
        1};                                                       \
    snprintf(&metrics_registry[index].label_names[0][0],          \
             OMNI_METRICS_MAX_LABEL_LEN, "phase=\"%s\"", #phase); \
    index++;

#define UPSTREAM_METRIC_MSEC(endpoint, name, help, value)                                                     \
    metrics_registry[index] = (omni_metric_desc_t){                                                           \
        name,                                                                                                 \
        help,                                                                                                 \
        OMNI_METRIC_GAUGE,                                                                                    \
        {{0}},                                                                                                \
        2,                                                                                                    \
        {.int64_values = (int64_t *)&endpoint->value},                                                        \
        OMNI_VALUE_INT64,                                                                                     \
        1};                                                                                                   \
    snprintf(&metrics_registry[index].label_names[0][0],                                                      \
             OMNI_METRICS_MAX_LABEL_LEN, "endpoint=\"%s:%d\"", endpoint->comm.address.ip, endpoint->comm.address.port); \
    snprintf(&metrics_registry[index].label_names[1][0],                                                      \
             OMNI_METRICS_MAX_LABEL_LEN, "role=\"prefill\"");                                                 \
    index++;

// Get or create the metrics registry
const omni_metric_desc_t *omni_metrics_get_registry(omni_global_state_t *global_state, size_t *count)
{
    if (metrics_registry != NULL)
    {
        if (count)
            *count = metrics_count;
        return metrics_registry;
    }

    if (!global_state)
    {
        return NULL;
    }

    // Calculate number of metrics needed
    size_t num_prefill = global_state->num_prefill_endpoints;
    size_t num_decode = global_state->num_decode_endpoints;

    // 9 metrics per endpoint type + 20 others, update the last value when add more
    metrics_count = (num_prefill * 4) + (num_decode * 5) + 20;

    // Allocate registry
    metrics_registry = ngx_alloc(metrics_count * sizeof(omni_metric_desc_t), ngx_cycle->log);
    if (!metrics_registry)
    {
        return NULL;
    }

    size_t index = 0;
    size_t cnt = 0;
    // Setup prefill metrics for each endpoint
    for (int i = 0; i < MAX_PREFILL_UPSTREAMS && cnt < num_prefill; i++)
    {
        omni_upstream_prefill_t *prefill = &global_state->prefill_states[i];
        if (prefill->comm.status != STATUS_ENABLE) {
            continue;
        }
        cnt++;
        // Prefill running requests
        UPSTREAM_METRIC(
            prefill,
            "omni_prefill_running_requests",
            "Number of running requests on prefill upstream",
            num_running);

        // Prefill number of tokens
        UPSTREAM_METRIC(
            prefill,
            "omni_prefill_num_tokens",
            "Number of tokens on prefill upstream",
            num_tokens);

        // Prefill last scheduled time
        UPSTREAM_METRIC_MSEC(
            prefill,
            "omni_prefill_last_scheduled_time",
            "Last scheduled time on prefill upstream",
            last_scheduled_time);

        // Prefill expected next time
        UPSTREAM_METRIC_MSEC(
            prefill,
            "omni_prefill_expected_next_time",
            "Expected next schedule time on prefill upstream",
            expected_next_schedule_time);
    }

    cnt = 0;
    // Setup decode metrics for each endpoint
    for (int i = 0; i < MAX_DECODE_UPSTREAMS && cnt < num_decode; i++)
    {
        omni_upstream_decode_t *decode = &global_state->decode_states[i];
        if (decode->comm.status != STATUS_ENABLE) {
            continue;
        }
        cnt++;
        UPSTREAM_METRIC(
            decode,
            "omni_decode_running_requests",
            "Number of running requests on decode upstream",
            num_running);

        UPSTREAM_METRIC(
            decode,
            "omni_decode_num_tokens",
            "Number of tokens on decode upstream",
            num_tokens);

        UPSTREAM_METRIC(
            decode,
            "omni_decode_generated_tokens_total",
            "Total tokens generated by decode upstream",
            generated_tokens);

        UPSTREAM_METRIC_MSEC(
            decode,
            "omni_decode_last_scheduled_time",
            "Last scheduled time on decode upstream",
            last_scheduled_time);

        UPSTREAM_METRIC_MSEC(
            decode,
            "omni_decode_expected_next_time",
            "Expected next schedule time on decode upstream",
            expected_next_schedule_time);
    }

    PHASE_METRIC(PHASE_TOKENIZING,
                 "omni_phase_waiting",
                 "Num of requests waiting at given phase", num_requests);
    PHASE_METRIC(PHASE_APC_MATCHING,
                 "omni_phase_waiting",
                 "Num of requests waiting at given phase", num_requests);
    PHASE_METRIC(PHASE_PREFILL_WAITING_SCHEDULE,
                 "omni_phase_waiting",
                 "Num of requests waiting at given phase", num_requests);
    PHASE_METRIC(PHASE_PREFILL_SCHEDULED,
                 "omni_phase_waiting",
                 "Num of requests waiting at given phase", num_requests);
    PHASE_METRIC(PHASE_PREFILLING,
                 "omni_phase_waiting",
                 "Num of requests waiting at given phase", num_requests);
    PHASE_METRIC(PHASE_DECODE_WAITING_SCHEDULE,
                 "omni_phase_waiting",
                 "Num of requests waiting at given phase", num_requests);
    PHASE_METRIC(PHASE_DECODE_SCHEDULED,
                 "omni_phase_waiting", "Num of requests waiting at given phase", num_requests);
    PHASE_METRIC(PHASE_DECODING,
                 "omni_phase_waiting",
                 "Num of requests waiting at given phase", num_requests);

    if (count)
    {
        metrics_count = index;
        *count = metrics_count;
    }

    return metrics_registry;
}

// Format a single metric value
static u_char *format_metric_value(u_char *p, u_char *end,
                                   const omni_metric_desc_t *desc)
{
    double value = 0.0;

    // Get the value based on type
    if (desc->value_type == OMNI_VALUE_INT32)
    {
        value = (double)(*desc->value.int_values);
    }
    else
    {
        value = *desc->value.double_values;
    }

    // Output metric name
    p = ngx_snprintf(p, end - p, "%s", desc->name);

    // Output labels if present
    if (desc->label_count > 0)
    {
        p = ngx_snprintf(p, end - p, "{");
        for (size_t i = 0; i < desc->label_count; i++)
        {
            p = ngx_snprintf(p, end - p, "%s", &desc->label_names[i]);
            if (i < desc->label_count - 1)
            {
                p = ngx_snprintf(p, end - p, ",");
            }
        }
        p = ngx_snprintf(p, end - p, "}");
    }

    // Output value
    p = ngx_snprintf(p, end - p, " %f\n", value);

    return p;
}

static u_char *export_histogram_series(u_char *p, u_char *end,
                                       const char *metric_name,
                                       const char *help,
                                       const ngx_atomic_t *buckets, ngx_uint_t nbuckets,
                                       const ngx_uint_t *bounds_ms,
                                       ngx_atomic_t sum_ms, ngx_atomic_t count,const char *model)
{
    // HELP/TYPE
    p = ngx_snprintf(p, end - p, "# HELP %s %s\n", metric_name, help);
    p = ngx_snprintf(p, end - p, "# TYPE %s histogram\n", metric_name);
    // accumulate
    ngx_uint_t cumulative = 0;
    for (ngx_uint_t i = 0; i < nbuckets - 1; i++)
    {
        ngx_atomic_t v = buckets ? buckets[i] : 0;
        cumulative += (ngx_uint_t)v;
        double le = ((double)bounds_ms[i]) / 1000.0;
        p = ngx_snprintf(p, end - p, "%s_bucket{engine=\"0\",model_name=\"%s\",le=\"%.3f\"} %ui\n", metric_name, model, le, cumulative);
    }
    // +Inf bucket
    p = ngx_snprintf(p, end - p, "%s_bucket{engine=\"0\",model_name=\"%s\",le=\"+Inf\"} %uA\n", metric_name, model, count);

    // sum / count（sec）
    p = ngx_snprintf(p, end - p, "%s_sum %.6f\n", metric_name, ((double)sum_ms) / 1000.0);
    p = ngx_snprintf(p, end - p, "%s_count %uA\n", metric_name, count);

    return p;
}
// Export all metrics in Prometheus format
ngx_str_t omni_metrics_export(omni_global_state_t *global_state)
{
    static u_char buffer[65536];
    u_char *p = buffer;
    u_char *end = buffer + sizeof(buffer);

    if (!global_state)
    {
        p = ngx_snprintf(p, end - p, "# ERROR: Global state not available\n");
        goto done;
    }

    // Get or create metrics registry
    size_t count = 0;
    const omni_metric_desc_t *registry = omni_metrics_get_registry(global_state, &count);
    if (!registry)
    {
        p = ngx_snprintf(p, end - p, "# ERROR: Failed to get metrics registry\n");
        goto done;
    }

    // Track which metrics we've already output HELP/TYPE for
    int help_output[256] = {0}; // Simple deduplication array

    // Export all metrics
    for (size_t i = 0; i < count; i++)
    {
        const omni_metric_desc_t *desc = &registry[i];

        // Output HELP and TYPE only once per metric name
        if (!help_output[i])
        {
            p = ngx_snprintf(p, end - p, "# HELP %s %s\n", desc->name, desc->help);

            const char *type_str = "";
            switch (desc->type)
            {
            case OMNI_METRIC_GAUGE:
                type_str = "gauge";
                break;
            case OMNI_METRIC_COUNTER:
                type_str = "counter";
                break;
            case OMNI_METRIC_HISTOGRAM:
                type_str = "histogram";
                break;
            }
            p = ngx_snprintf(p, end - p, "# TYPE %s %s\n", desc->name, type_str);

            help_output[i] = 1;

            // Mark other instances of same metric name as already documented
            for (size_t j = i + 1; j < count; j++)
            {
                if (ngx_strcmp(registry[j].name, desc->name) == 0)
                {
                    help_output[j] = 1;
                }
            }
        }

        // Format the metric value
        p = format_metric_value(p, end, desc);
    }
    const char *model = omni_metrics_get_model_name(global_state);

    p = export_histogram_series(
        p, end,
        "vllm:time_to_first_token_seconds",
        "Time to first token histogram (seconds)",
        global_state->ttft_buckets, OMNI_TTFT_BUCKETS_COUNT,
        OMNI_TTFT_BUCKET_BOUNDS_MS,
        global_state->ttft_sum_ms, global_state->ttft_count, model);

    p = export_histogram_series(
        p, end,
        "vllm:time_per_output_token_seconds",
        "Time per output token histogram (seconds)",
        global_state->tpot_buckets, OMNI_TPOT_BUCKETS_COUNT,
        OMNI_TPOT_BUCKET_BOUNDS_MS,
        global_state->tpot_sum_ms, global_state->tpot_count, model);

    p = export_histogram_series(
        p, end,
        "vllm:e2e_request_latency_seconds",
        "End-to-end request latency histogram (seconds)",
        global_state->e2e_buckets, OMNI_E2E_BUCKETS_COUNT,
        OMNI_E2E_BUCKET_BOUNDS_MS,
        global_state->e2e_sum_ms, global_state->e2e_count, model);
        
    p = ngx_snprintf(p, end - p,
                     "# HELP vllm:requests_success_total Total number of successful global proxy requests\n");
    p = ngx_snprintf(p, end - p,
                     "# TYPE vllm:requests_success_total counter\n");
    p = ngx_snprintf(p, end - p,
                     "vllm:requests_success_total %uA\n",
                     global_state->success_count);

    p = ngx_snprintf(p, end - p,
                     "# HELP vllm:requests_failure_total Total number of failed global proxy requests\n");
    p = ngx_snprintf(p, end - p,
                     "# TYPE vllm:requests_failure_total counter\n");
    p = ngx_snprintf(p, end - p,
                     "vllm:requests_failure_total %uA\n",
                     global_state->failure_count);
done:
    {
        ngx_str_t result;
        result.data = buffer;
        result.len = p - buffer;

        return result;
    }
}

ngx_str_t omni_health_status_export_json(omni_global_state_t *gs, ngx_pool_t *pool) {

    ngx_uint_t i;

    size_t required_size = 1024;

    uint16_t cnt = 0;
    for (i = 0; i < MAX_PREFILL_UPSTREAMS && cnt < gs->num_prefill_endpoints; i++) {
        if (gs->prefill_states[i].comm.status != STATUS_ENABLE) {
            continue;
        }
        cnt++;
        required_size += 300;
        required_size += gs->prefill_states[i].comm.address.name_str.len;
    }

    for (i = 0, cnt = 0; i < MAX_DECODE_UPSTREAMS && cnt < gs->num_decode_endpoints; i++) {
        if (gs->decode_states[i].comm.status != STATUS_ENABLE) {
            continue;
        }
        cnt++;
        required_size += 300;
        required_size += gs->decode_states[i].comm.address.name_str.len;
    }

    u_char *buf = ngx_palloc(pool, required_size);
    if (buf == NULL) {
        return (ngx_str_t){0, NULL};
    }

    u_char *p = buf;
    u_char *end = buf + required_size;
    ngx_uint_t is_first_node = 1;

    time_t now_utc = ngx_time();
    time_t now_beijing_t = now_utc + 8 * 3600;
    struct tm *now_tm = gmtime(&now_beijing_t);
    u_char time_buf[64];
    strftime((char *)time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", now_tm);

    ngx_uint_t healthy_prefill_count = 0;
    for (i = 0, cnt = 0; i < MAX_PREFILL_UPSTREAMS && cnt < gs->num_prefill_endpoints; i++) {
        if (gs->prefill_states[i].comm.status != STATUS_ENABLE) {
            continue;
        }
        cnt++;
        if (gs->prefill_states[i].healthy) {
            healthy_prefill_count++;
        }
    }

    ngx_uint_t healthy_decode_count = 0;
    for (i = 0, cnt = 0; i < MAX_DECODE_UPSTREAMS && cnt < gs->num_decode_endpoints; i++) {
        if (gs->decode_states[i].comm.status != STATUS_ENABLE) {
            continue;
        }
        cnt++;
        if (gs->decode_states[i].healthy) {
            healthy_decode_count++;
        }
    }

    ngx_uint_t x = healthy_prefill_count;
    ngx_uint_t m = 1;
    ngx_uint_t y = healthy_decode_count / gs->num_decode_endpoints;
    ngx_uint_t n = gs->num_decode_endpoints / DECODE_INSTANCE_GROUP_SIZE;


    ngx_uint_t code;
    ngx_str_t status;

    ngx_uint_t total_upstreams = gs->num_prefill_endpoints + gs->num_decode_endpoints;
    ngx_uint_t total_healthy = healthy_prefill_count + healthy_decode_count;

    if (x == 0 || y == 0) {
        code = 503;
        ngx_str_set(&status, "service failed");
    } else if (total_healthy == total_upstreams) {
        code = 200;
        ngx_str_set(&status, "OK");
    } else {
        code = 206;
        ngx_str_set(&status, "warn, unhealthy servers exist");
    }

    p = ngx_snprintf(p, end - p, "{\n");
    p = ngx_snprintf(p, end - p, "    \"code\": %ui,\n", code);
    p = ngx_snprintf(p, end - p, "    \"status\": \"%V\",\n", &status);
    p = ngx_snprintf(p, end - p, "    \"timestamp\": \"%s\",\n", time_buf);
    p = ngx_snprintf(p, end - p, "    \"summary\": \"%uiP%ui-%uiD%ui\",\n", x, m, y, n);
    p = ngx_snprintf(p, end - p, "    \"total_prefill_servers\": %d,\n", gs->num_prefill_endpoints);
    p = ngx_snprintf(p, end - p, "    \"health_prefill_servers\": %d,\n", healthy_prefill_count);
    p = ngx_snprintf(p, end - p, "    \"total_decode_servers\": %d,\n", gs->num_decode_endpoints);
    p = ngx_snprintf(p, end - p, "    \"health_decode_servers\": %d,\n", healthy_decode_count);
    p = ngx_snprintf(p, end - p, "    \"prefill\": [\n");

    // 1. for Prefill nodes
    for (i = 0, cnt = 0; i < MAX_PREFILL_UPSTREAMS && cnt < gs->num_prefill_endpoints; i++) {
        if (gs->prefill_states[i].comm.status != STATUS_ENABLE) {
            continue;
        }
        cnt++;
        p = ngx_snprintf(p, end - p, "%s        {\n", is_first_node ? "" : ",\n");
        if (gs->prefill_states[i].healthy == 1){
            p = ngx_snprintf(p, end - p, "            \"status\": \"running\",\n");
        } else {
            p = ngx_snprintf(p, end - p, "            \"status\": \"failed\",\n");
        }
        p = ngx_snprintf(p, end - p, "            \"pod_name\": \"\",\n");
        p = ngx_snprintf(p, end - p, "            \"url\": \"%V\",\n", &gs->prefill_states[i].comm.address.name_str);
        p = ngx_snprintf(p, end - p, "            \"role\": \"prefill\",\n");
        p = ngx_snprintf(p, end - p, "            \"index\": %d,\n", i);
        p = ngx_snprintf(p, end - p, "            \"port\": %ui,\n", gs->prefill_states[i].comm.address.port);
        if (gs->prefill_states[i].healthy == 1){
            p = ngx_snprintf(p, end - p, "            \"fault_message\": \"healthy\",\n");
        } else {
            p = ngx_snprintf(p, end - p, "            \"fault_message\": \"unhealthy\",\n");
        }
        p = ngx_snprintf(p, end - p, "            \"alert\": \"\"\n");
        p = ngx_snprintf(p, end - p, "        }");
        is_first_node = 0;
    }
    p = ngx_snprintf(p, end - p, "\n    ],\n");
    p = ngx_snprintf(p, end - p, "    \"decode\": [\n");

    // 2. for Decode nodes
    for (i = 0, cnt = 0; i < MAX_DECODE_UPSTREAMS && cnt < gs->num_decode_endpoints; i++) {
        if (gs->decode_states[i].comm.status != STATUS_ENABLE) {
            continue;
        }
        cnt++;
        ngx_uint_t index = i / DECODE_INSTANCE_GROUP_SIZE;
        if (i != 0){
            p = ngx_snprintf(p, end - p, "%s        {\n", is_first_node ? "" : ",\n");
        } else {
            p = ngx_snprintf(p, end - p, "        {\n");
        }
        if (gs->decode_states[i].healthy == 1){
            p = ngx_snprintf(p, end - p, "            \"status\": \"running\",\n");
        } else {
            p = ngx_snprintf(p, end - p, "            \"status\": \"failed\",\n");
        }
        p = ngx_snprintf(p, end - p, "            \"pod_name\": \"\",\n");
        p = ngx_snprintf(p, end - p, "            \"url\": \"%V\",\n", &gs->decode_states[i].comm.address.name_str);
        p = ngx_snprintf(p, end - p, "            \"role\": \"decode\",\n");
        p = ngx_snprintf(p, end - p, "            \"index\": %d,\n", index);
        p = ngx_snprintf(p, end - p, "            \"port\": %ui,\n", gs->decode_states[i].comm.address.port);
        if (gs->prefill_states[i].healthy == 1){
            p = ngx_snprintf(p, end - p, "            \"fault_message\": \"healthy\",\n");
        } else {
            p = ngx_snprintf(p, end - p, "            \"fault_message\": \"unhealthy\",\n");
        }
        p = ngx_snprintf(p, end - p, "            \"alert\": \"\"\n");
        p = ngx_snprintf(p, end - p, "        }");
        is_first_node = 0;
    }

    p = ngx_snprintf(p, end - p, "\n    ]\n");
    p = ngx_snprintf(p, end - p, "}\n");

    ngx_str_t result;
    result.data = buf;
    result.len = p - buf;

    FILE *f = fopen("/tmp/omni_proxy_health.json", "w");
    if (f) {
        fwrite(result.data, 1, result.len, f);
        fclose(f);
    }

    return result;
}
