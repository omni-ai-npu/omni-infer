#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_DIR="$(cd "$WORK_DIR/../../../" && pwd)"
VLLM_PATCH_FILE="${WORK_DIR}/vllm_cpu_kv_events.patch"
echo "TOP_DIR is ${TOP_DIR}"

#download model json file
cd ${WORK_DIR}
mkdir -p mock_model/
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json -O ./mock_model/config.json
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer.json -O ./mock_model/tokenizer.json
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/tokenizer_config.json -O ./mock_model/tokenizer_config.json
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/vocab.json -O ./mock_model/vocab.json
wget https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/merges.txt -O ./mock_model/merges.txt

#installing nginx
pkill nginx
cd ${WORK_DIR}/../
bash build.sh --skip-extras -c
pip3 install gcovr

#installing vllm version running on cpu
cd ${WORK_DIR}
git clone https://github.com/vllm-project/vllm
cd ${WORK_DIR}/vllm/
git checkout v0.14.0
git apply "${VLLM_PATCH_FILE}"
python -m pip install -U "cmake>=3.26" wheel packaging ninja "setuptools-scm>=8" "numpy<=2.3" msgpack "numba>=0.62.0" ijson
yum install -y numactl-devel
VLLM_TARGET_DEVICE=cpu python setup.py install
