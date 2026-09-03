#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 所有子模块的 pip 命令均使用本地环境：
export PIP_NO_INDEX=1
export PIP_NO_BUILD_ISOLATION=0
export PIP_DISABLE_PIP_VERSION_CHECK=1

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ALL_MODULES=()
MODULES_TO_BUILD=()
SKIP_INSTALL=0

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 从本地 components 目录获取全部模块
read_all_modules() {
    local components_dir="$PROJECT_ROOT/components"
    local module_dir

    if [[ ! -d "$components_dir" ]]; then
        log_error "components 目录不存在: $components_dir"
        exit 1
    fi

    ALL_MODULES=()
    for module_dir in "$components_dir"/*; do
        [[ -d "$module_dir" ]] || continue
        ALL_MODULES+=("$(basename "$module_dir")")
    done

    if [[ ${#ALL_MODULES[@]} -eq 0 ]]; then
        log_error "components 目录下没有找到任何本地模块: $components_dir"
        exit 1
    fi

    log_info "本地所有模块: ${ALL_MODULES[*]}"
}

show_help() {
    cat <<EOF
用法: $0 [-m|--modules <module1,module2,...>] [-si|--skip-install]

选项:
  -m, --modules <列表>   指定要编译的本地模块，以逗号分隔。
                        若不指定，则编译所有模块。
                        可用模块: ${ALL_MODULES[*]}
  -si, --skip-install   跳过编译安装，仅检查本地模块。
  -h, --help            显示帮助信息。

示例:
  $0
  $0 --modules omni-npu,omni-models
  $0 --skip-install
EOF
}

validate_modules() {
    local mod
    local avail
    local found
    local valid=1

    for mod in "${MODULES_TO_BUILD[@]}"; do
        found=0
        for avail in "${ALL_MODULES[@]}"; do
            if [[ "$mod" == "$avail" ]]; then
                found=1
                break
            fi
        done

        if [[ $found -eq 0 ]]; then
            log_error "未知模块: $mod" >&2
            valid=0
        fi
    done

    if [[ $valid -eq 0 ]]; then
        exit 1
    fi
}

parse_args() {
    local modules_str=""

    # 先读取本地模块，确保帮助信息和参数校验使用的是实际目录。
    read_all_modules

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -m|--modules)
                if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
                    log_error "选项 '$1' 需要一个非空参数，多个模块名用逗号分隔" >&2
                    exit 1
                fi
                modules_str="$2"
                shift 2
                ;;
            -si|--skip-install)
                SKIP_INSTALL=1
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知选项: $1" >&2
                show_help >&2
                exit 1
                ;;
        esac
    done

    if [[ -n "$modules_str" ]]; then
        IFS=',' read -r -a MODULES_TO_BUILD <<< "$modules_str"
        log_info "将编译以下本地模块: ${MODULES_TO_BUILD[*]}"
        validate_modules
    else
        MODULES_TO_BUILD=("${ALL_MODULES[@]}")
        log_warn "未指定模块，将编译所有本地模块: ${MODULES_TO_BUILD[*]}"
    fi
}

# 检查选定模块及其构建脚本是否存在。
check_local_modules() {
    local mod
    local module_dir
    local build_script

    for mod in "${MODULES_TO_BUILD[@]}"; do
        module_dir="$PROJECT_ROOT/components/$mod"
        build_script="$module_dir/build/build.sh"

        if [[ ! -d "$module_dir" ]]; then
            log_error "本地模块目录不存在: $module_dir"
            exit 1
        fi

        if [[ ! -f "$build_script" ]]; then
            log_error "模块构建脚本不存在: $build_script"
            exit 1
        fi
    done
}

# 执行单个模块的 build/build.sh。
check_and_build() {
    local module_dir="$1"
    local build_script="$module_dir/build/build.sh"

    log_info "找到 build.sh，开始编译安装: $module_dir"

    cd "$module_dir"

    if [[ ! -x "$build_script" ]]; then
        chmod +x "$build_script"
    fi

    if bash "$build_script"; then
        log_info "编译安装成功: $module_dir"
    else
        log_error "编译安装失败: $module_dir"
        exit 1
    fi

    cd "$PROJECT_ROOT"
}

# 遍历本地模块并执行测试、编译。
traverse_modules() {
    local mod
    local module_dir

    log_info "---Step 1：检查本地模块...---"
    for mod in "${MODULES_TO_BUILD[@]}"; do
        module_dir="$PROJECT_ROOT/components/$mod"
        log_info "使用本地模块: $mod ($module_dir)"
    done

    log_info "---Step 2：遍历本地模块执行 UT 测试...---"
    for mod in "${MODULES_TO_BUILD[@]}"; do
        module_dir="$PROJECT_ROOT/components/$mod"
        log_info "本地模块: $mod"

        # 当前没有启用 UT。如需启用，可在这里调用对应测试脚本。
        # check_and_test "$module_dir"
    done

    log_info "---Step 3：遍历本地模块执行编译安装...---"
    for mod in "${MODULES_TO_BUILD[@]}"; do
        module_dir="$PROJECT_ROOT/components/$mod"
        log_info "本地模块: $mod"
        check_and_build "$module_dir"
    done
}

main() {
    parse_args "$@"

    log_info "开始使用本地代码处理 OmniInfer 模块..."

    check_local_modules

    log_info "项目根目录: $PROJECT_ROOT"

    if [[ "$SKIP_INSTALL" == "0" ]]; then
        traverse_modules
    fi

    log_info "${MODULES_TO_BUILD[*]} 模块处理完成！"
}

main "$@"
