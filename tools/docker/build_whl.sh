#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
set -exo pipefail

BASE_DIR=$(
    cd "$(dirname "$0")"
    pwd
)

# 代码下载需要网络代理
export http_proxy=${HTTP_PROXY}
export https_proxy=${HTTP_PROXY}
# The incoming value is a branch name (e.g. 'release_v0.6.0' or 'master').
# Use it directly if provided, otherwise default to 'master'.
branch="${BRANCH:-master}"
install_modules="${INSTALL_MODULES:-omni-npu,omni-proxy}"
# vllm_version="v0.12.0"
vllm_version="${VLLM_VERSION:-v0.12.0}"

cd /opt/

if [ -d "$/opt/vllm" ]; then
    echo "vllm already exists in infer_engines. Skipping clone."
else
    # Check if vllm exists in codes directory, copy if exists, otherwise clone
    if [ -d "${BASE_DIR}/dist/codes/vllm" ]; then
        echo "Directory ${BASE_DIR}/dist/codes/vllm already exists. Copying to infer_engines..."
        cp -r "${BASE_DIR}/dist/codes/vllm" "/opt/"
        cd /opt/vllm 
        git checkout ${vllm_version} 
    else
        echo "Cloning vllm repo from remote..."
        if ! git clone --depth 10 -b ${vllm_version} https://github.com/vllm-project/vllm.git; then
            echo "ERROR: Failed to clone vllm from remote repository."
            echo "Please manually download vllm to ./codes/vllm and run this script again."
            exit 1
        fi
    fi
fi
cd /opt/vllm 
echo "Successfully switched to commit ${vllm_version}."
sed -i 's/^gpt-oss >= 0\.0\.7$/#&/' /opt/vllm/requirements/common.txt

pip3 uninstall -y omni_infer vllm || true
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pip3 install --upgrade pip 
TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_TARGET_DEVICE=empty pip3 install --no-cache-dir -e /opt/vllm --no-build-isolation

cd ${BASE_DIR}

git config --global credential.helper store
git config --global http.sslverify false
git config --global https.sslverify false
# echo "https://gitee:password@gitee.com" > ~/.git-credentials
# chmod 600 ~/.git-credentials

# Check if omniinfer exists in dist/codes directory, copy if exists, otherwise clone
if [ -d "${BASE_DIR}/dist/codes/omniinfer" ]; then
    echo "Directory ${BASE_DIR}/dist/codes/omniinfer already exists. Copying to current path..."
    cp -r "${BASE_DIR}/dist/codes/omniinfer" "${BASE_DIR}/"
    cd ${BASE_DIR}/omniinfer
    git checkout "${branch}" || echo "Warning: failed to checkout branch ${branch} in omniinfer."
else
    echo "Cloning omniinfer repo (branch: ${branch})..."
    if ! git clone --depth 10 -b "${branch}" https://gitee.com/omniai/omniinfer.git; then
        echo "ERROR: Failed to clone omniinfer from remote repository."
        echo "Please manually download omniinfer to ./codes/omniinfer and run this script again."
        exit 1
    fi
fi


cd ${BASE_DIR}/omniinfer && bash build/build.sh -m "${install_modules}"

# rm -rf ~/.git-credentials

