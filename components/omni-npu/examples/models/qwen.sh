#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

### QwQ-32B, tp8, pd-mixed, graph mode
bash "$(dirname $0)/../serve-single-instance.sh" --model /bigdata/models/QwQ-32B --tp-size 8 --bsz 32 --no-ep

### Qwen3-30B, prefill tp4, decode tp4+graph
# bash "$(dirname $0)/../serve-pd-disaggregate.sh" --model /bigdata/models/Qwen3-30B-A3B-Instruct-2507/ \
#     --prefill-tp-size 4 \
#     --decode-tp-size 4 \
#     --bsz 32
