# Copyright (c) HuaWei Technologies Co., Ltd. 2025-2025. All rights reserved

import os


class EmsEnv:
    llm_engine = os.environ.get("LLM_ENGINE", "vllm")
    model_id = os.environ.get("MODEL_ID", "cc_kvstore@_@ds_default_ns_001")
    access_id = os.environ.get("ACCELERATE_ID", "cc_kvstore@_@ds_default_ns_001")
    access_key = os.environ.get("ACCELERATE_KEY", "")
    ems_timeout: int = int(os.environ.get("EMS_TIMEOUT", "5000"))
    ems_enable_write_rcache: bool = os.environ.get("EMS_ENABLE_WRITE_RCACHE", "1") == "1"
    ems_enable_read_local_only: bool = os.environ.get("EMS_ENABLE_READ_LOCAL_ONLY", "0") == "1"
    ems_num_min_reuse_tokens: int = int(os.environ.get("EMS_NUM_MIN_REUSE_TOKENS", "3072"))
    ems_num_min_load_blocks: int = int(os.environ.get("EMS_NUM_MIN_LOAD_BLOCKS", "1"))