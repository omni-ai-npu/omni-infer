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
# 华为内源需要直连，避免 cargo 访问 mirrors.tools.huawei.com 时走 HIS Proxy。
export no_proxy="${no_proxy:+${no_proxy},}mirrors.tools.huawei.com,rust.inhuawei.com"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}mirrors.tools.huawei.com,rust.inhuawei.com"
# The incoming value is a branch name (e.g. 'release_v0.6.0' or 'master').
# Use it directly if provided, otherwise default to 'master'.
branch="${BRANCH:-master}"
install_modules="${INSTALL_MODULES:-omni-proxy}"
skip_pull="${SKIP_PULL:-false}"
vllm_version="${VLLM_VERSION:-v0.14.0}"

# Load shared retry helper.
. /usr/local/bin/retry.sh

cd /opt/

if [ -d "/opt/vllm" ]; then
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

#git config --global credential.helper store
git config --global http.sslverify false
git config --global https.sslverify false

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


build_args=(-m "${install_modules}")
if [[ "${skip_pull}" == "True" || "${skip_pull}" == "true" || "${skip_pull}" == "1" ]]; then
    build_args+=(--skip-pull)
fi
cd ${BASE_DIR}/omniinfer && retry bash build/build.sh  --editable "${build_args[@]}"

if [[ ",${install_modules}," == *",omni-cache,"* ]]; then
    ox_binary="${BASE_DIR}/omniinfer/components/omni-cache/omni_cache/connector/backends/ox/ox"
    omni_cache_build_args=(-m omni-cache)
    if [[ "${skip_pull}" == "True" || "${skip_pull}" == "true" || "${skip_pull}" == "1" ]]; then
        omni_cache_build_args+=(--skip-pull)
    fi

    # build.sh may return success even when ox fails to compile; verify the
    # binary after each attempt and rebuild up to 10 times if missing.
    max_ox_retries=10
    ox_attempt=0
    while [ ! -f "${ox_binary}" ]; do
        ox_attempt=$((ox_attempt + 1))
        if [ "${ox_attempt}" -gt "${max_ox_retries}" ]; then
            echo "ERROR: omni-cache ox binary still missing at ${ox_binary} after ${max_ox_retries} rebuild attempts."
            exit 1
        fi
        echo "omni-cache ox binary not found at ${ox_binary}, rebuild attempt ${ox_attempt}/${max_ox_retries}..."
        cd ${BASE_DIR}/omniinfer && bash build/build.sh "${omni_cache_build_args[@]}" || true
    done
    echo "omni-cache ox binary found at ${ox_binary}"
fi


