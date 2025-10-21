# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

curl -X POST http://127.0.0.1:8089/v1/completions  \
     -H "Content-Type: application/json" \
     -d '{
     	 "model": "deepseek",
     	 "prompt": ["Hi, my name is", "The capital of France is"],
     	 "max_tokens": 35,
     	 "temperature": 0.0,
      	 "top_p": 1,
     	 "top_k": -1
 	 }' \
	 -m 99999