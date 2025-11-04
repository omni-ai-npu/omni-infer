// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <omni_proxy.h>

void omni_proxy_schedule_decode(omni_global_state_t *gs, ngx_http_omni_loc_conf_t *olcf);
void omni_proxy_schedule_prefill(omni_global_state_t *gs, ngx_http_omni_loc_conf_t *olcf);
void omni_scheduler_record_prefill_batch_stat(omni_global_state_t *gs, uint32_t batch_size, ngx_msec_t duration);
