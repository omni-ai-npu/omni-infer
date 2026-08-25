#!/usr/bin/env bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
set -exo pipefail

# Default parameters (can be overridden via command-line --long-options)
ARCH="aarch64"
PROXY="http://username:passward@hostIP:port/"
HUGGING_FACE_PROXY="http://username:passward@hostIP:port/"
PIP_INDEX_URL="https://pypi.org/simple"
PIP_TRUSTED_HOST="pypi.org"
MODEL_NAME="Qwen/Qwen2.5-0.5B"
## CANN install mode: whole or split
CANN_INSTALL_MODE="whole"
## Custom operator packages to add (comma-separated list or bash script); default empty
CUSTOM_OPS=""
## NPU platform information (910B or 910C); default 910C
NPU_PLATFORM="910C"
## Whether to start the apiserver at the end (True/False). Default True
START_SERVER="True"
## Python version to use during image build (e.g. 3.10, 3.11)
PYTHON_VERSION="3.11.12"
# base image tags
BASE_IMAGE=test-infer-base:0.1
# L1 image tags
L1_IMAGE=test-infer-meddle:0.1
L2_IMAGE=test-infer-omniinfer:0.1
BRANCH=master
## BUILD_TARGET: which Dockerfiles to build: L1, L2, or both
# - L1: only build Dockerfile.base
# - L2: only build Dockerfile.omniinfer
# - both: build Dockerfile.base first, then Dockerfile.omniinfer (default)
BUILD_TARGET="both"
INSTALL_MODULES="omni-npu,omni-proxy"
VLLM_VERSION="v0.12.0"
BUILD_FOR_ROMA="false"
ROMA_IMAGE="test-infer-ROMA:0.1"


# Print usage/help
print_help() {
    cat <<EOF
Usage: $0 [options]

Options:
    -h, --help                         Show this help message and exit
    --arch <arch>                      Target architecture (default: ${ARCH})
    --proxy <proxy>                    HTTP proxy to use when building (default: ${PROXY})
    --hugging-face-proxy <proxy>      Proxy passed to the runtime container for model downloads (default: ${HUGGING_FACE_PROXY})
    --pip-index-url <url>             PIP index URL used during build (default: ${PIP_INDEX_URL})
    --pip-trusted-host <host>         PIP trusted host (default: ${PIP_TRUSTED_HOST})
    --model-name <name>               Model name used at runtime (default: ${MODEL_NAME})
    --cann-install-mode <mode>        CANN install mode for Dockerfile.base: whole or split (default: ${CANN_INSTALL_MODE})
    --npu-platform <platform>         NPU platform (910B or 910C) (default: ${NPU_PLATFORM})
    --base-image <image>              Tag for the base image build (default: ${BASE_IMAGE})
    --L1-image <image>               Tag for the dev image build (default: ${L1_IMAGE})
    --L2-image <image>              Tag for the apiserver/user image build (default: ${L2_IMAGE})
    --custom-ops <ops>                Custom operator packages to include (default: ${CUSTOM_OPS})
    --branch <tag>                    Omni code version/branch to include (default: ${BRANCH})
    --python-version <version>        Python version to use during build (default: ${PYTHON_VERSION})
    --build-target <L1|L2|both|skip> Select which builds to run (default: ${BUILD_TARGET})
    --start-server <True|False>     Whether to start the apiserver after build (default: ${START_SERVER})
    --vllm-version <version>        vLLM version to install (default: ${VLLM_VERSION})
    --build-for-roma <True|False>  Whether to build the Roma image (default: ${BUILD_FOR_ROMA})
    --roma-image <image>            Tag for the Roma image build (default: ${ROMA_IMAGE})
    --install-modules <modules>      Comma-separated list of omniinfer modules to install (default: ${INSTALL_MODULES})

Examples:
    $0 --arch aarch64 --L2-image my-image:latest --model-name "Qwen/Qwen2.5-0.5B"
    $0 --build-target L1 --cann-install-mode split
    # Only build the base image
    $0 --build-target L1
    # Only build the omniinfer/apiserver image (skip Dockerfile.base)
    $0 --build-target L2 --L2-image my-image:latest
    # Build both (default)
    $0 --build-target both
EOF
}

# Parse a single long option
parse_long_option() {
    case "$1" in
        --arch)
            ARCH="$2"
            ;;
        --proxy)
            PROXY="$2"
            ;;
        --hugging-face-proxy)
            HUGGING_FACE_PROXY="$2"
            ;;
        --pip-index-url)
            PIP_INDEX_URL="$2"
            ;;
        --pip-trusted-host)
            PIP_TRUSTED_HOST="$2"
            ;;
        --custom-ops)
            CUSTOM_OPS="$2"
            ;;
        --npu-platform)
            NPU_PLATFORM="$2"
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            ;;
        --model-name)
            MODEL_NAME="$2"
            ;;
        --cann-install-mode)
            CANN_INSTALL_MODE="$2"
            ;;
        --base-image)
            BASE_IMAGE="$2"
            ;;
        --L1-image)
            L1_IMAGE="$2"
            ;;
        --L2-image)
            L2_IMAGE="$2"
            ;;
        --branch)
            BRANCH="$2"
            ;;
        --start-server)
            START_SERVER="$2"
            ;;
        --install-modules)
            INSTALL_MODULES="$2"
            ;;
        --vllm-version)
            VLLM_VERSION="$2"
            ;;
        --build-for-roma)
            BUILD_FOR_ROMA="$2"
            ;;
        --roma-image)
            ROMA_IMAGE="$2"
            ;;
        --build-target)
            BUILD_TARGET="$2"
            ;;
        --help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_help
            ;;
    esac
    return 0
}

# ---------------------------------------------------------------------------
# Command-line parsing: treat --long-options
# Example: ./docker_build_run.sh --arch aarch64 --user-image myimg:latest
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            exit 0
            ;;
        --*)
            parse_long_option "$1" "$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_help
            exit 1
            ;;
    esac
done


## Print current configuration
echo "==== Current Configuration ===="
echo "ARCH: ${ARCH}"
echo "HTTP_PROXY: ${PROXY}"
echo "HUGGING_FACE_PROXY: ${HUGGING_FACE_PROXY}"
echo "PIP_INDEX_URL: ${PIP_INDEX_URL}"
echo "PIP_TRUSTED_HOST: ${PIP_TRUSTED_HOST}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "CANN_INSTALL_MODE: ${CANN_INSTALL_MODE}"
echo "NPU_PLATFORM: ${NPU_PLATFORM}"
echo "BASE_IMAGE: ${BASE_IMAGE}"
echo "L1_IMAGE: ${L1_IMAGE}"
echo "L2_IMAGE: ${L2_IMAGE}"
echo "BRANCH: ${BRANCH}"
echo "CUSTOM_OPS: ${CUSTOM_OPS}"
echo "BUILD_TARGET: ${BUILD_TARGET}"
echo "START_SERVER: ${START_SERVER}"
echo "PYTHON_VERSION: ${PYTHON_VERSION}"
echo "INSTALL_MODULES: ${INSTALL_MODULES}"
echo "VLLM_VERSION: ${VLLM_VERSION}"
echo "BUILD_FOR_ROMA: ${BUILD_FOR_ROMA}"
echo "ROMA_IMAGE: ${ROMA_IMAGE}"
echo "=================="

# Validate BUILD_TARGET
if [[ ! "${BUILD_TARGET}" =~ ^(L1|L2|both|skip)$ ]]; then
    echo "Unknown build target: ${BUILD_TARGET}. Use one of: L1, L2, both, skip." >&2
    exit 2
fi

# validate cann install mode
if [[ ! "${CANN_INSTALL_MODE}" =~ ^(whole|split)$ ]]; then
    echo "Unknown CANN_INSTALL_MODE: ${CANN_INSTALL_MODE}. Use one of: whole, split." >&2
    exit 2
fi

# validate npu platform
if [[ ! "${NPU_PLATFORM}" =~ ^(910B|910C)$ ]]; then
    echo "Unknown NPU_PLATFORM: ${NPU_PLATFORM}. Use one of: 910B, 910C." >&2
    exit 2
fi

## BASE_IMAGE: build base image with CANN pytorch torch_npu
if [[ "${BUILD_TARGET}" == "L1" || "${BUILD_TARGET}" == "both" ]]; then
    echo "Building base image (Dockerfile.base) -> ${L1_IMAGE}"
    docker build --progress=plain --no-cache -f Dockerfile.base \
        --build-arg ARCHITECTURE="${ARCH}" \
        --build-arg HTTP_PROXY="${PROXY}" \
        --build-arg CANN_INSTALL_MODE="${CANN_INSTALL_MODE}" \
        --build-arg PIP_INDEX_URL="${PIP_INDEX_URL}" \
        --build-arg PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST}" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        --build-arg BASE_IMAGE=${BASE_IMAGE} \
        -t ${L1_IMAGE} .
    # If the user only wanted to build the base image, stop here
    if [[ "${BUILD_TARGET}" == "L1" ]]; then
        echo "BUILD_TARGET=L1 selected — finished building L1 image. Exiting."
        exit 0
    fi
else
    echo "Skipping Dockerfile.base build (BUILD_TARGET=${BUILD_TARGET})"
fi

## BASE_IMAGE: build develop image
# docker build -f Dockerfile.omniinfer --target develop_image -t ${DEV_IMAGE} .
## step into dev container:
# docker run --rm -it -u root ${DEV_IMAGE}

## L2_IMAGE user image with apiserver
if [[ "${BUILD_TARGET}" == "L2" || "${BUILD_TARGET}" == "both" ]]; then
    echo "Building omniinfer_vllm image (Dockerfile.omniinfer) -> ${L2_IMAGE}"
    docker build --progress=plain --no-cache -f Dockerfile.omniinfer \
        --build-arg HTTP_PROXY="${PROXY}" \
        --build-arg PIP_INDEX_URL="${PIP_INDEX_URL}" \
        --build-arg PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST}" \
        --build-arg BRANCH="${BRANCH}" \
        --build-arg NPU_PLATFORM="${NPU_PLATFORM}" \
        --build-arg BASE_IMAGE=${L1_IMAGE} \
        --build-arg CUSTOM_OPS="${CUSTOM_OPS}" \
        --build-arg INSTALL_MODULES="${INSTALL_MODULES}" \
        --build-arg VLLM_VERSION="${VLLM_VERSION}" \
        --target omininfer_openai \
        -t ${L2_IMAGE} .
else
    echo "Skipping L2 image build (BUILD_TARGET=${BUILD_TARGET})"
fi

if [[ ("${BUILD_FOR_ROMA}" == "True" || "${BUILD_FOR_ROMA}" == "true" || "${BUILD_FOR_ROMA}" == "1")  && "${BUILD_TARGET}" != "L1" ]]; then
    echo "Building Roma image (Dockerfile.roma) -> ${ROMA_IMAGE}"
    docker build --progress=plain --no-cache -f Dockerfile.roma \
        --build-arg BASE_IMAGE=${L2_IMAGE} \
        -t ${ROMA_IMAGE} .
else
    echo "Skipping Roma image build (BUILD_FOR_ROMA=${BUILD_FOR_ROMA}, BUILD_TARGET=${BUILD_TARGET})"
fi

## get dist whl,rpm files
# Create a temp container name that includes a sanitized L2 image identifier
# so it's easy to distinguish artifacts when building different images.
# Sanitize image tag to allowed Docker name characters (alphanumeric, . _ -)


## start apiserver and download model (controlled by START_SERVER)
if [[ "${START_SERVER}" == "True" || "${START_SERVER}" == "true" || "${START_SERVER}" == "1" ]]; then
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
else
    echo "Skipping starting apiserver (START_SERVER=${START_SERVER})"
fi

# curl -X POST http://127.0.0.1:8301/v1/completions -H "Content-Type:application/json" -d '{"temperature":0,"max_tokens":50,"prompt": "how are you?"}'
