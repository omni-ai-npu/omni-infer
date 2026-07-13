#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.


current_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
infer_engines_path="$current_dir/../../infer_engines"

if [ ! -d "$infer_engines_path" ]; then
    echo "Error: infer_engines directory not found at $infer_engines_path"
    exit 1
fi

cd "$infer_engines_path" || exit 1

if [ ! -d "vllm" ]; then
    echo "Error: vllm directory not found in $infer_engines_path"
    exit 1
fi

git config --global --add safe.directory "$(realpath vllm)"

cd vllm || return
git checkout -f
cd ..
bash bash_install_code.sh

pip uninstall vllm -y
pip uninstall omni_infer -y

cd vllm || return
SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e . --no-deps --no-build-isolation
cd ../../

pwd

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e . --no-deps --no-build-isolation

pip uninstall numpy -y
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.26 --no-deps --no-build-isolation