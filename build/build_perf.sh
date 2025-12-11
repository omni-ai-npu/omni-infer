#!/bin/bash
set -e

# 颜色输出函数
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
ALL_MODULES=("omni-npu" "omni-cache" "omni-eplb" "omni-proxy" "omni-models")

# 定义各模块所用的git仓库及分支
declare -A GIT_PATH_OF_MODULE
GIT_PATH_OF_MODULE["omni-npu"]="-b master https://gitee.com/omniai/omni-npu.git"
GIT_PATH_OF_MODULE["omni-cache"]="-b master https://gitee.com/omniai/omni-cache.git"
GIT_PATH_OF_MODULE["omni-eplb"]="-b master https://gitee.com/omniai/omni-eplb.git"
GIT_PATH_OF_MODULE["omni-proxy"]="-b master https://gitee.com/omniai/omni-proxy.git"
GIT_PATH_OF_MODULE["omni-models"]="-b master https://gitee.com/omniai/omni-models.git"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 git 是否安装
check_git() {
    if ! command -v git &> /dev/null; then
        log_error "git 未安装，请先安装 git"
        exit 1
    fi
}

# 初始化并更新子模块
init_submodules() {
    log_info "---Step1：初始化子模块...---"
    for mod in "${MODULES_TO_BUILD[@]}"; do
        if [ -e "components/$mod" ]; then
            log_warn "路径 components/$mod 存在"
            rm -rf components/$mod
        fi
        git submodule add --force ${GIT_PATH_OF_MODULE["$mod"]} components/$mod
    done
    git submodule init
    
    log_info "更新子模块..."
    git submodule update --recursive --remote
    
    if [ $? -ne 0 ]; then
        log_warn "子模块更新失败，尝试仅同步当前提交..."
        git submodule update --recursive
    fi
}

# 递归遍历子模块
traverse_submodules() {
    local base_dir="$1"
    # 获取所有子模块
    local submodules="${MODULES_TO_BUILD[@]}"
    
    # 遍历每个子仓UT测试
    log_info "---Step2：遍历子模块执行UT测试...---"
    for submodule in $submodules; do
        local submodule_path="$base_dir/components/$submodule"
        
        if [ -d "$submodule_path" ]; then
            log_info "开始子模块: $submodule"
            
            # 检查并执行UT测试 run_test.sh
            check_and_test "$submodule_path"
            
        else
            log_warn "子模块路径不存在: $submodule_path"
        fi
    done

    # 遍历每个子仓编译
    log_info "---Step3：遍历子模块执行编译...---"
    for submodule in $submodules; do
        local submodule_path="$base_dir/components/$submodule"
        
        if [ -d "$submodule_path" ]; then 
            log_info "子模块: $submodule"
            
            # 检查并执行 build.sh
            check_and_build "$submodule_path"
            
        else
            log_warn "子模块路径不存在: $submodule_path"
        fi
    done    
}

# 执行UT测试
check_and_test() {
    # TODO:加入UT
    echo "$@"
}

# 检查并执行 build.sh
check_and_build() {
    local module_dir="$1"
    local build_script="$module_dir/build/build.sh"
    
    # 检查 build.sh 是否存在
    if [ -f "$build_script" ]; then
        log_info "找到 build.sh，开始编译安装: $module_dir"
        
        # 进入目录
        cd "$module_dir"
        
        # 给 build.sh 添加执行权限（如果需要）
        if [ ! -x "$build_script" ]; then
            chmod +x "$build_script"
        fi
        
        # 执行编译安装
        if bash "$build_script"; then
            log_info "编译安装成功: $module_dir"
        else
            log_error "编译安装失败: $module_dir"
            # 失败1个即退出
            exit 1
        fi
        
        # 返回原目录
        cd - > /dev/null
    else
        log_error "未找到 build.sh 在: $module_dir/build/"
        exit 1
    fi
}

show_help() {
    cat <<EOF
用法: $0 [-m|--modules <module1,module2,...>]

选项:
  -m, --modules <列表>   指定要编译的模块（以逗号分隔）
                        若不指定，则编译所有模块。
                        模块列表"${ALL_MODULES[*]}"

示例:
  $0
  $0 -m module1,module3
  $0 --modules module2
EOF
}

validate_modules() {
    local valid=1
    for mod in "${MODULES_TO_BUILD[@]}"; do
        local found=0
        for avail in "${ALL_MODULES[@]}"; do
            if [[ "$mod" == "$avail" ]]; then
                found=1
                break
            fi
        done
        if [[ $found -eq 0 ]]; then
            log_error "错误: 未知模块 '$mod'" >&2
            valid=0
        fi
    done
    if [[ $valid -eq 0 ]]; then
        exit 1
    fi
}

parse_args() {
    local modules_str=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -m|--modules)
                if [[ -z "$2" || "$2" == -* ]]; then
                    log_error "错误: 选项 '$1' 需要一个非空参数（模块名，多个用逗号分隔）" >&2
                    exit 1
                fi
                modules_str="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "未知选项: $1" >&2
                show_help >&2
                exit 1
                ;;
        esac
    done

    # 如果指定了模块，按逗号分割；否则返回空数组
    if [[ -n "$modules_str" ]]; then
        IFS=',' read -r -a MODULES_TO_BUILD <<< "$modules_str"
    else
        MODULES_TO_BUILD=()
    fi

    if [[ ${#MODULES_TO_BUILD[@]} -eq 0 ]]; then
        log_warn "未指定模块，将编译所有模块。${ALL_MODULES}"
        MODULES_TO_BUILD=("${ALL_MODULES[@]}")
    else
        log_info "将编译以下模块: ${MODULES_TO_BUILD[*]}"
        validate_modules
    fi
}

# 主函数
main() {

    parse_args "$@"

    log_info "开始处理 Omniinfer 子模块UT测试编译安装..."
    
    # 检查 git
    check_git
    
    # 初始化子模块
    init_submodules
    
    # 获取项目根目录
    local project_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
    
    log_info "项目根目录: $project_root"
    
    # 遍历并处理所有子模块
    traverse_submodules "$project_root"
    
    log_info "所有子模块处理完成！"
}

# 运行主函数
main "$@"


