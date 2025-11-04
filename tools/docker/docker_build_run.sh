#!/usr/bin/env bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
set -exo pipefail

ARCH="aarch64"
PROXY="http://username:passward@hostIP:port/"
HUGGING_FACE_PROXY="http://username:passward@hostIP:port/"
PIP_INDEX_URL="https://mirrors.huaweicloud.com/repository/pypi/simple"
PIP_TRUSTED_HOST="mirrors.huaweicloud.com"
MODEL_NAME="Qwen/Qwen2.5-0.5B"
BASE_IMAGE=test-infer-base:0.1
DEV_IMAGE=test-infer-dev:0.1
USER_IMAGE=test-infer-apiserver:0.1

## BASE_IMAGE: build base image witch CANN pytorch torch_npu
docker build -f Dockerfile.base \
    --build-arg ARCHITECTURE="${ARCH}" \
    --build-arg HTTP_PROXY="${PROXY}" \
    --build-arg PIP_INDEX_URL="${PIP_INDEX_URL}" \
    --build-arg PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST}" \
    --build-context driver=/usr/local/Ascend \
    --build-context etc=/etc \
    -t ${BASE_IMAGE} .

## BASE_IMAGE: build develop image
# docker build -f Dockerfile.omniinfer --target develop_image -t ${DEV_IMAGE} .
## step into dev container:
# docker run --rm -it -u root ${DEV_IMAGE}

## USER_IMAGE：build user image with apiserver
docker build -f Dockerfile.omniinfer \
    --build-arg HTTP_PROXY="${PROXY}" \
    --build-arg PIP_INDEX_URL="${PIP_INDEX_URL}" \
    --build-arg PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST}" \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    --target omininfer_openai \
    -t ${USER_IMAGE} .

## start apiserver and download model
docker run --rm -it --shm-size=500g \
    --net=host --privileged=true \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -e PORT=8301 \
    -e ASCEND_RT_VISIBLE_DEVICES=1 \
    -e HTTP_PROXY="${HUGGING_FACE_PROXY}" \
    -e MODEL_NAME="${MODEL_NAME}" \
    ${USER_IMAGE} \
    --model "${MODEL_NAME}"

# curl -X POST http://127.0.0.1:8301/v1/completions -H "Content-Type:application/json" -d '{"temperature":0,"max_tokens":50,"prompt": "how are you?"}'

## get dist whl,rpm files
TEMP_CONTAINER=dist_$(date +"%Y-%m-%d_%H%M%S")
docker run --name ${TEMP_CONTAINER} -d ${USER_IMAGE} 
docker cp ${TEMP_CONTAINER}:/dist ${TEMP_CONTAINER}/
docker rm -f ${TEMP_CONTAINER}