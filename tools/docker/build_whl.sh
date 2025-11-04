#!/usr/bin/env bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
set -exo pipefail

BASE_DIR=$(
    cd "$(dirname "$0")"
    pwd
)

# 代码下载需要网络代理
export http_proxy=${HTTP_PROXY}
export https_proxy=${HTTP_PROXY}
branch="release_v${OMNI_VERSION_NUM}"

version_ge() {
    local ver1="$1"
    local ver2="$2"
    local -a arr1 arr2
    
    # 1. 拆分版本号为3段，空值补0（适配 0.5、1.0 等不完整格式）
    IFS='.' read -r -a arr1 <<< "${ver1:-0.0.0}"  # 若ver1为空，默认0.0.0
    IFS='.' read -r -a arr2 <<< "${ver2:-0.0.0}"  # 若ver2为空，默认0.0.0
    
    # 2. 逐段比较（主→次→修订），不足3段的补0
    for i in 0 1 2; do
        local num1=${arr1[$i]:-0}
        local num2=${arr2[$i]:-0}
        
        num1=$((num1))
        num2=$((num2))
        
        if [ "$num1" -gt "$num2" ]; then
            return 0
        elif [ "$num1" -lt "$num2" ]; then
            return 1
        fi
    done
    
    return 0
}

# 需要预先准备 omniinfer 代码，可根据CI流程优化，在容器中下载会导致构建时间过长
git config --global http.sslVerify false
git clone -b "${branch}" https://gitee.com/omniai/omniinfer.git
cd omniinfer/infer_engines
git clone https://github.com/vllm-project/vllm.git

# 构建 whl 包
cd ${BASE_DIR}/omniinfer
chmod +x build/build.sh
chmod +x infer_engines/bash_install_code.sh

if version_ge "${OMNI_VERSION_NUM}" "0.5.0"; then
    bash -xe build/build.sh --cache 1
    cd ${BASE_DIR}/omniinfer/tools/quant/python
    python setup.py bdist_wheel
else
    bash -xe build/build.sh
fi

pip cache purge