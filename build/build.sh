#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
readonly PYTHON_BIN="${OMNIINFER_PYTHON_BIN:-python3}"

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    readonly RED='\033[0;31m'
    readonly GREEN='\033[0;32m'
    readonly YELLOW='\033[1;33m'
    readonly NC='\033[0m'
else
    readonly RED=''
    readonly GREEN=''
    readonly YELLOW=''
    readonly NC=''
fi

ALL_MODULES=()
MODULES_TO_BUILD=()
COMMIT_MODULES=()
ROOT_PIP_ARGS=()

declare -A COMMIT_OF_MODULE=()

SKIP_PULL=0
SKIP_INSTALL=0
SKIP_COMPONENTS=0
SKIP_ROOT=0
USE_BUILD_ISOLATION=0
ROOT_BUILD_MODE="wheel"
ROOT_BUILD_MODE_SET=0

log_info() {
    printf '%b[INFO]%b %s\n' "${GREEN}" "${NC}" "$*"
}

log_warn() {
    printf '%b[WARN]%b %s\n' "${YELLOW}" "${NC}" "$*"
}

log_error() {
    printf '%b[ERROR]%b %s\n' "${RED}" "${NC}" "$*" >&2
}

die() {
    log_error "$*"
    exit 1
}

join_by() {
    local delimiter="$1"
    shift
    local first=1
    local value
    for value in "$@"; do
        if ((first)); then
            printf '%s' "${value}"
            first=0
        else
            printf '%s%s' "${delimiter}" "${value}"
        fi
    done
}

check_repository() {
    command -v git >/dev/null 2>&1 || die "git 未安装，请先安装 git"
    [[ -f "${PROJECT_ROOT}/.gitmodules" ]] || die ".gitmodules 文件不存在: ${PROJECT_ROOT}/.gitmodules"
    git -C "${PROJECT_ROOT}" rev-parse --show-toplevel >/dev/null 2>&1 \
        || die "项目根目录不是有效的 git 仓库: ${PROJECT_ROOT}"
}

read_all_modules() {
    local key
    local module_path
    local module

    ALL_MODULES=()
    while read -r key module_path; do
        [[ -n "${key}" && "${module_path}" == components/* ]] || continue
        module="${module_path#components/}"
        [[ -n "${module}" && "${module}" != */* ]] || continue
        ALL_MODULES+=("${module}")
    done < <(
        git config -f "${PROJECT_ROOT}/.gitmodules" \
            --get-regexp '^submodule\..*\.path$' 2>/dev/null || true
    )

    for module in "${ALL_MODULES[@]}"; do
        if [[ "${module}" == "omni-npu" ]]; then
            die "omni-npu 已集成到根项目，请先从 .gitmodules 中移除 components/omni-npu"
        fi
    done
}

show_help() {
    local available_modules="无"
    if ((${#ALL_MODULES[@]} > 0)); then
        available_modules="$(join_by ',' "${ALL_MODULES[@]}")"
    fi

    cat <<EOF
用法: $0 [选项] [-- <pip 参数>]

默认行为:
  更新并构建全部 components 子模块，最后将根项目构建为 wheel，
  输出到 build/dist。

组件选项:
  -m, --modules <列表>       指定 components 模块，多个模块以逗号分隔
  -s, --set <模块=commit>   将指定模块检出到固定 commit；可以重复指定
  -sp, --skip-pull          不更新子模块，使用本地已有源码
  --skip-components         不更新也不构建 components，只构建根项目

根项目选项:
  --wheel                   构建 wheel（默认），输出到 build/dist
  --editable                以 editable 模式安装根项目
  --build-isolation         启用 pip 构建隔离；默认使用当前 NPU Python 环境
  --skip-root               跳过根项目构建

通用选项:
  -si, --skip-install       跳过所有构建/安装，仅按需更新子模块
  -h, --help                显示帮助
  -- <pip 参数>             将剩余参数传递给根项目的 pip 命令

可用组件:
  ${available_modules}

环境变量:
  OMNIINFER_PYTHON_BIN      指定 Python 可执行文件，默认 python3

示例:
  $0
  $0 --editable
  $0 -m omni-cache,omni-eplb --wheel
  $0 --skip-pull --editable -- -v
  $0 --skip-components --wheel
  $0 --skip-root -m omni-cache
EOF
}

module_is_known() {
    local target="$1"
    local module
    for module in "${ALL_MODULES[@]}"; do
        [[ "${target}" == "${module}" ]] && return 0
    done
    return 1
}

module_is_selected() {
    local target="$1"
    local module
    for module in "${MODULES_TO_BUILD[@]}"; do
        [[ "${target}" == "${module}" ]] && return 0
    done
    return 1
}

set_submodule_commit() {
    local module_commit="$1"
    local module="${module_commit%%=*}"
    local commit="${module_commit#*=}"

    [[ -n "${module}" && -n "${commit}" && "${module_commit}" == *=* ]] \
        || die "--set 参数必须使用非空的 模块=commit 格式"
    [[ "${module}" =~ ^[[:alnum:]_.-]+$ ]] \
        || die "--set 中包含非法模块名: ${module}"

    if [[ -z "${COMMIT_OF_MODULE[$module]+configured}" ]]; then
        COMMIT_MODULES+=("${module}")
    fi
    COMMIT_OF_MODULE["${module}"]="${commit}"
}

set_root_build_mode() {
    local requested_mode="$1"
    if ((ROOT_BUILD_MODE_SET)) && [[ "${ROOT_BUILD_MODE}" != "${requested_mode}" ]]; then
        die "--wheel 与 --editable 不能同时使用"
    fi
    ROOT_BUILD_MODE="${requested_mode}"
    ROOT_BUILD_MODE_SET=1
}

validate_selection() {
    local module

    for module in "${MODULES_TO_BUILD[@]}"; do
        [[ -n "${module}" ]] || die "模块列表中不能包含空模块名"
        module_is_known "${module}" || die "未知模块: ${module}"
    done

    for module in "${COMMIT_MODULES[@]}"; do
        module_is_known "${module}" || die "--set 指定了未知模块: ${module}"
        module_is_selected "${module}" \
            || die "--set 指定的模块不在本次构建列表中: ${module}"
    done
}

parse_args() {
    local modules_str=""

    while (($# > 0)); do
        case "$1" in
            -m | --modules)
                (($# >= 2)) || die "选项 '$1' 需要模块列表参数"
                [[ -n "$2" && "$2" != -* ]] || die "选项 '$1' 需要非空模块列表参数"
                [[ "$2" != ,* && "$2" != *, && "$2" != *,,* ]] \
                    || die "模块列表不能以逗号开头、结尾或包含连续逗号"
                modules_str="$2"
                shift 2
                ;;
            -s | --set)
                (($# >= 2)) || die "选项 '$1' 需要 模块=commit 参数"
                set_submodule_commit "$2"
                shift 2
                ;;
            -sp | --skip-pull)
                SKIP_PULL=1
                shift
                ;;
            -si | --skip-install)
                SKIP_INSTALL=1
                shift
                ;;
            --skip-components)
                SKIP_COMPONENTS=1
                shift
                ;;
            --skip-root)
                SKIP_ROOT=1
                shift
                ;;
            --wheel)
                set_root_build_mode "wheel"
                shift
                ;;
            --editable)
                set_root_build_mode "editable"
                shift
                ;;
            --build-isolation)
                USE_BUILD_ISOLATION=1
                shift
                ;;
            -h | --help)
                show_help
                exit 0
                ;;
            --)
                shift
                ROOT_PIP_ARGS=("$@")
                break
                ;;
            *)
                die "未知选项: $1（使用 --help 查看帮助）"
                ;;
        esac
    done

    if [[ -n "${modules_str}" ]]; then
        IFS=',' read -r -a MODULES_TO_BUILD <<<"${modules_str}"
    else
        MODULES_TO_BUILD=("${ALL_MODULES[@]}")
    fi

    validate_selection
}

check_root_build_requirements() {
    [[ -f "${PROJECT_ROOT}/pyproject.toml" ]] \
        || die "根项目缺少 pyproject.toml"
    [[ -f "${PROJECT_ROOT}/setup.py" ]] \
        || die "根项目缺少 setup.py"
    [[ -f "${PROJECT_ROOT}/omni/__init__.py" ]] \
        || die "未找到 omni/__init__.py，请先将 omni/src/omni_npu 中的内容迁移到 omni"
    [[ -n "${ASCEND_TOOLKIT_HOME:-}" ]] \
        || die "构建根项目需要设置 ASCEND_TOOLKIT_HOME"

    command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
        || die "找不到 Python 可执行文件: ${PYTHON_BIN}"
    "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1 \
        || die "${PYTHON_BIN} 环境中未安装 pip"
    "${PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' \
        || die "omni_infer 需要 Python 3.11 或更高版本"

    if ((USE_BUILD_ISOLATION == 0)); then
        "${PYTHON_BIN}" -c 'import pybind11, torch' >/dev/null 2>&1 \
            || die "非隔离构建要求当前 Python 环境已安装 torch 和 pybind11"
    fi
}

update_components() {
    local module
    local module_dir
    local -a module_paths=()

    ((${#MODULES_TO_BUILD[@]} > 0)) || {
        log_warn "没有需要更新的 components 模块"
        return
    }

    for module in "${MODULES_TO_BUILD[@]}"; do
        module_paths+=("components/${module}")
    done

    log_info "同步子模块配置: $(join_by ',' "${MODULES_TO_BUILD[@]}")"
    git -C "${PROJECT_ROOT}" submodule sync --recursive -- "${module_paths[@]}"

    log_info "初始化并更新子模块"
    git -C "${PROJECT_ROOT}" submodule update \
        --init --recursive --remote -- "${module_paths[@]}"

    for module in "${COMMIT_MODULES[@]}"; do
        module_dir="${PROJECT_ROOT}/components/${module}"
        log_info "将 ${module} 检出到 commit ${COMMIT_OF_MODULE[$module]}"
        git -C "${module_dir}" checkout --detach "${COMMIT_OF_MODULE[$module]}"
    done
}

build_components() {
    local module
    local module_dir
    local build_script

    ((${#MODULES_TO_BUILD[@]} > 0)) || {
        log_warn "没有需要构建的 components 模块"
        return
    }

    for module in "${MODULES_TO_BUILD[@]}"; do
        module_dir="${PROJECT_ROOT}/components/${module}"
        build_script="${module_dir}/build/build.sh"

        [[ -d "${module_dir}" ]] || die "子模块目录不存在: ${module_dir}"
        [[ -f "${build_script}" ]] || die "未找到子模块构建脚本: ${build_script}"

        log_info "开始构建组件: ${module}"
        (
            cd -- "${module_dir}"
            bash "${build_script}"
        )
        log_info "组件构建成功: ${module}"
    done
}

build_root_project() {
    local -a isolation_args=()
    local -a command=()

    if ((USE_BUILD_ISOLATION == 0)); then
        isolation_args+=("--no-build-isolation")
    fi

    case "${ROOT_BUILD_MODE}" in
        wheel)
            local dist_dir="${PROJECT_ROOT}/build/dist"
            mkdir -p -- "${dist_dir}"
            command=(
                "${PYTHON_BIN}" -m pip wheel
                --no-deps
                --wheel-dir "${dist_dir}"
                "${isolation_args[@]}"
                "${ROOT_PIP_ARGS[@]}"
                "${PROJECT_ROOT}"
            )
            log_info "构建 omni_infer wheel，输出目录: ${dist_dir}"
            "${command[@]}"
            log_info "omni_infer wheel 构建完成"
            ;;
        editable)
            command=(
                "${PYTHON_BIN}" -m pip install
                "${isolation_args[@]}"
                "${ROOT_PIP_ARGS[@]}"
                --editable "${PROJECT_ROOT}"
            )
            log_info "以 editable 模式安装根项目 omni_infer"
            "${command[@]}"
            log_info "omni_infer editable 安装完成"
            ;;
        *)
            die "不支持的根项目构建模式: ${ROOT_BUILD_MODE}"
            ;;
    esac
}

main() {
    check_repository
    read_all_modules
    parse_args "$@"

    if ((SKIP_INSTALL == 0 && SKIP_ROOT == 0)); then
        check_root_build_requirements
    fi

    if ((SKIP_COMPONENTS)); then
        log_info "已跳过 components 更新与构建"
        if ((${#COMMIT_MODULES[@]} > 0)); then
            log_warn "--skip-components 已启用，--set 参数不会生效"
        fi
    else
        if ((${#MODULES_TO_BUILD[@]} > 0)); then
            log_info "本次处理组件: $(join_by ',' "${MODULES_TO_BUILD[@]}")"
        else
            log_warn ".gitmodules 中没有可处理的 components 模块"
        fi

        if ((SKIP_PULL)); then
            log_info "已跳过 components 更新"
            if ((${#COMMIT_MODULES[@]} > 0)); then
                log_warn "--skip-pull 已启用，--set 参数不会生效"
            fi
        else
            update_components
        fi
    fi

    if ((SKIP_INSTALL)); then
        log_info "已跳过所有组件构建和根项目构建"
        return
    fi

    if ((SKIP_COMPONENTS == 0)); then
        build_components
    fi

    if ((SKIP_ROOT)); then
        log_info "已跳过根项目构建"
    else
        build_root_project
    fi

    log_info "构建流程完成"
}

main "$@"

