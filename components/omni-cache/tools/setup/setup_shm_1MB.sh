#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# 安全写入 /dev/shm 的脚本
# 在执行 dd 前检查内存是否足够，防止系统崩溃
set -euo pipefail

#############################
# 配置参数
# 第一个参数为 MAP_SIZE_BYTES（字节数），与 setup_hugetlbfs_2MB.sh 统一
# 如果未提供，则使用 SIZE_GB 环境变量或默认值
MAP_SIZE_BYTES="${1:-${MAP_SIZE_BYTES:-}}"
if [[ -n "$MAP_SIZE_BYTES" ]]; then
  SIZE_GB=$((MAP_SIZE_BYTES / 1024 / 1024 / 1024))
else
  SIZE_GB="${SIZE_GB:-200}"  # 默认 200GB
fi
TARGET_FILE="${TARGET_FILE:-/dev/shm/omni_cache}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.85}"  # 安全阈值 85%
FORCE_WRITE="${FORCE_WRITE:-0}"  # 强制写入（危险）

#############################
# 颜色辅助
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

#############################
# 检查系统内存是否足够
check_memory_available() {
  local required_bytes="$1"
  local required_gb=$((required_bytes / 1024 / 1024 / 1024))
  local total_mem_kb avail_mem_kb total_mem_gb avail_mem_gb

  # 从 /proc/meminfo 获取内存信息
  if [[ -f /proc/meminfo ]]; then
    total_mem_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
    avail_mem_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    # 如果没有 MemAvailable（旧内核），用 Free + Buffers + Cache 估算
    if [[ -z "$avail_mem_kb" ]]; then
      local free_kb buffers_kb cache_kb
      free_kb=$(awk '/^MemFree:/ {print $2}' /proc/meminfo)
      buffers_kb=$(awk '/^Buffers:/ {print $2}' /proc/meminfo)
      cache_kb=$(awk '/^Cached:/ {print $2}' /proc/meminfo)
      avail_mem_kb=$((free_kb + buffers_kb + cache_kb))
    fi
  else
    log_warn "无法读取 /proc/meminfo，跳过内存检查"
    return 0
  fi

  total_mem_gb=$((total_mem_kb / 1024 / 1024))
  avail_mem_gb=$((avail_mem_kb / 1024 / 1024))

  echo ""
  log_info "========== 内存状态 =========="
  log_info "系统总内存:   ${total_mem_gb} GB"
  log_info "可用内存:     ${avail_mem_gb} GB"
  log_info "请求写入:     ${required_gb} GB"
  log_info "安全阈值:     $(echo "scale=0; $SAFETY_MARGIN * 100" | bc)%"
  echo ""

  # 检查是否超过总内存（致命错误）
  if (( required_bytes > total_mem_kb * 1024 )); then
    log_error "=========================================="
    log_error "危险: 请求大小 ${required_gb} GB 超过系统总内存 ${total_mem_gb} GB！"
    log_error "=========================================="
    log_error "这将导致系统崩溃，操作已中止"
    return 1
  fi

  # 检查是否超过可用内存的安全阈值
  local safe_limit_bytes
  safe_limit_bytes=$(echo "$avail_mem_kb * 1024 * $SAFETY_MARGIN" | bc | cut -d. -f1)

  if (( required_bytes > safe_limit_bytes )); then
    local safe_gb=$((safe_limit_bytes / 1024 / 1024 / 1024))
    log_error "=========================================="
    log_error "警告: 请求大小 ${required_gb} GB 超过安全阈值 ${safe_gb} GB"
    log_error "=========================================="
    log_error ""
    log_error "建议:"
    log_error "  1. 释放部分内存后重试"
    log_error "  2. 使用更小的 SIZE_GB 值"
    log_error "  3. 强制执行 (风险自负): FORCE_WRITE=1 $0"
    log_error ""
    return 1
  fi

  log_info "内存检查通过"
  return 0
}

#############################
# 检查 /dev/shm 可用空间
check_shm_space() {
  local required_bytes="$1"
  local required_gb=$((required_bytes / 1024 / 1024 / 1024))
  local shm_dir
  shm_dir=$(dirname "$TARGET_FILE")
  local avail_bytes avail_gb

  avail_bytes=$(df -B1 "$shm_dir" 2>/dev/null | awk 'NR==2 {print $4}')

  if [[ -z "$avail_bytes" ]]; then
    log_warn "无法检测 $shm_dir 可用空间"
    return 0
  fi

  avail_gb=$((avail_bytes / 1024 / 1024 / 1024))

  log_info "========== /dev/shm 空间 =========="
  log_info "可用空间: ${avail_gb} GB"
  log_info "请求大小: ${required_gb} GB"
  echo ""

  if (( required_bytes > avail_bytes )); then
    log_error "/dev/shm 空间不足"
    log_error "提示: sudo mount -o remount,size=${required_gb}G /dev/shm"
    return 1
  fi

  log_info "/dev/shm 空间检查通过"
  return 0
}

#############################
# 执行 dd 写入
do_dd_write() {
  local target_file="$1"
  local size_bytes="$2"
  local size_mb=$((size_bytes / 1024 / 1024))

  log_info "开始写入: $target_file (${size_mb} MB)"

  local start_time
  start_time=$(date +%s)

  # 执行 dd，显示进度
  dd if=/dev/zero of="$target_file" bs=1M count="$size_mb" status=progress

  local end_time duration speed
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  speed=$((size_mb / (duration > 0 ? duration : 1)))

  echo ""
  log_info "写入完成: 耗时 ${duration} 秒, 速度 ~${speed} MB/s"

  # 验证文件
  local actual_size
  actual_size=$(stat -c%s "$target_file" 2>/dev/null || stat -f%z "$target_file" 2>/dev/null)
  log_info "文件大小: $((actual_size / 1024 / 1024 / 1024)) GB"
}

#############################
# 主流程
main() {
  local size_bytes=$((SIZE_GB * 1024 * 1024 * 1024))
  local size_mb=$((SIZE_GB * 1024 * 1024))

  echo ""
  log_info "=========================================="
  log_info "安全 dd 写入脚本"
  log_info "=========================================="
  log_info "目标文件: $TARGET_FILE"
  log_info "写入大小: ${SIZE_GB} GB (${size_mb} MB)"
  echo ""

  # 检查文件是否已存在，存在则删除
  if [[ -e "$TARGET_FILE" ]]; then
    log_warn "文件已存在: $TARGET_FILE，正在删除..."
    rm -f "$TARGET_FILE"
  fi

  # 强制模式跳过检查
  if [[ "$FORCE_WRITE" == "1" ]]; then
    log_warn "=========================================="
    log_warn "强制写入模式 - 已跳过安全检查"
    log_warn "=========================================="
  else
    # 内存检查
    if ! check_memory_available "$size_bytes"; then
      exit 1
    fi

    # /dev/shm 空间检查
    if ! check_shm_space "$size_bytes"; then
      exit 1
    fi
  fi

  # 执行写入
  do_dd_write "$TARGET_FILE" "$size_bytes"

  echo ""
  log_info "=========================================="
  log_info "完成!"
  log_info "=========================================="
  log_info "释放内存: rm $TARGET_FILE"
}

main "$@"