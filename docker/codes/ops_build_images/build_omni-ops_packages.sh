#!/bin/bash
set -euo pipefail

script_dir=$(dirname "$(readlink -f "$0")")

DEFAULT_OMNI_OPS_PATH="/workspace/dist/codes/code/omni-ops"
DEFAULT_PROJECT_VERSION="poc-0.8.0rc1"
DEFAULT_CANN_VERSION="8.5RC1"
DEFAULT_PTA_VERSION="2.6.0"
PIP_INDEX_URL="http://mirrors.tools.huawei.com/pypi/simple/"
PIP_TRUSTED_HOST="mirrors.tools.huawei.com"

print_help() {
    cat <<EOF
Usage:
  Dockerfile mode:
    $0 --npu-platform <910B|910C> [options]

  Host wrapper mode (backward compatible):
    $0 <container_name> <image_name> <omni_ops_path> <build_script_file>
       [compile_unit] [project_version] [cann_version]
       <output_suffix> <mounted_work_path> [pta_version]

Dockerfile mode options:
  --omni-ops-path <path>       omni-ops source path
  --npu-platform <platform>    910B or 910C
  --build-type <type>          inference, training, or both (default: both)
  --project-version <version>  release-* or poc-* package version
  --cann-version <version>     CANN version used in package names
  --pta-version <version>      PTA version used in wheel versions
  -h, --help                   Show this help
EOF
}

map_compile_unit() {
    case "$1" in
        910B|ascend910b)
            echo "ascend910b"
            ;;
        910C|ascend910_93)
            echo "ascend910_93"
            ;;
        *)
            echo "Unsupported NPU platform or compile unit: $1" >&2
            return 1
            ;;
    esac
}

run_release_build() {
    local omni_ops_path=$1
    local build_type=$2
    local compile_unit=$3
    local project_version=$4
    local cann_version=$5
    local pta_version=$6

    if [[ ! -d "$omni_ops_path" ]]; then
        echo "Error: omni-ops source path does not exist: $omni_ops_path" >&2
        return 1
    fi

    case "$project_version" in
        release-*|poc-*) ;;
        *)
            echo "Error: project version must start with release- or poc-: $project_version" >&2
            return 1
            ;;
    esac

    case "$build_type" in
        inference|both)
            bash "$script_dir/ops_scripts/build_omni-ops_inference_release.sh" \
                "$omni_ops_path" "$compile_unit" "$project_version" \
                "$cann_version" inference "$pta_version"
            ;;
    esac

    case "$build_type" in
        training|both)
            bash "$script_dir/ops_scripts/build_omni-ops_training_release.sh" \
                "$omni_ops_path" "$compile_unit" "$project_version" \
                "$cann_version" training "$pta_version"
            ;;
    esac
}

run_in_container() {
    local omni_ops_path="$DEFAULT_OMNI_OPS_PATH"
    local npu_platform="910C"
    local build_type="both"
    local project_version="$DEFAULT_PROJECT_VERSION"
    local cann_version="$DEFAULT_CANN_VERSION"
    local pta_version="$DEFAULT_PTA_VERSION"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --omni-ops-path)
                omni_ops_path=$2
                shift 2
                ;;
            --npu-platform)
                npu_platform=$2
                shift 2
                ;;
            --build-type)
                build_type=$2
                shift 2
                ;;
            --project-version)
                project_version=$2
                shift 2
                ;;
            --cann-version)
                cann_version=$2
                shift 2
                ;;
            --pta-version)
                pta_version=$2
                shift 2
                ;;
            -h|--help)
                print_help
                return 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                print_help >&2
                return 1
                ;;
        esac
    done

    if [[ ! "$build_type" =~ ^(inference|training|both)$ ]]; then
        echo "Error: build type must be inference, training, or both: $build_type" >&2
        return 1
    fi

    local compile_unit
    compile_unit=$(map_compile_unit "$npu_platform")

    run_release_build "$omni_ops_path" "$build_type" "$compile_unit" \
        "$project_version" "$cann_version" "$pta_version"

    local package_root
    package_root=$(readlink -f "$omni_ops_path/../../ops-packages")
    bash "$script_dir/../install_ops_by_whl.sh" --ops-path "$package_root"
}

run_host_wrapper() {
    if [[ $# -lt 9 ]]; then
        print_help >&2
        return 1
    fi

    local container_name=$1
    local image_name=$2
    local omni_ops_path=$3
    local build_script_file=$4
    local compile_unit=${5:-"ascend910_93"}
    local project_version=${6:-"$DEFAULT_PROJECT_VERSION"}
    local cann_version=${7:-"$DEFAULT_CANN_VERSION"}
    local output_suffix=$8
    local mounted_work_path=$9
    local pta_version=${10:-"$DEFAULT_PTA_VERSION"}

    echo "container_name: $container_name"
    echo "COMPILE_UNIT: $compile_unit"

    docker rm -f "$container_name" >/dev/null 2>&1 || true
    docker run -u root --rm --name "$container_name" \
        --ulimit nproc=65535:65535 --ipc=host \
        -v "$mounted_work_path:$mounted_work_path" \
        --shm-size=128g \
        --privileged \
        --entrypoint=bash \
        "$image_name" -c "
            source ~/.bashrc && \
            source /etc/profile && \
            pip config set global.index-url '$PIP_INDEX_URL' && \
            pip config set global.trusted-host '$PIP_TRUSTED_HOST' && \
            pip install --timeout 300 --retries 3 \
                google protobuf expecttest hypothesis attrs scipy \
                PyYAML decorator numpy psutil build && \
            bash '$script_dir/ops_scripts/$build_script_file' \
                '$omni_ops_path' '$compile_unit' '$project_version' \
                '$cann_version' '$output_suffix' '$pta_version'
        "
}

if [[ ${1:-} == -* ]]; then
    run_in_container "$@"
else
    run_host_wrapper "$@"
fi
