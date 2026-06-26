#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

### DeepSeek-V3/V3.1-T/R1, tp16, pd-mixed, graph mode, mtp
bash "$(dirname $0)/../serve-single-instance.sh" --model /bigdata/models/DeepSeek-V3.1-MTP --tp-size 16 --mtp --bsz 32


### DeepSeek-V2-Lite, prefill tp8, decode dp8+graph
# bash "$(dirname $0)/../serve-pd-disaggregate.sh" --model /bigdata/models/DeepSeek-V2-Lite-Chat/ \
#     --prefill-tp-size 8 \
#     --decode-dp-size 8 \
#     --bsz 32
