#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Reserve (or shrink) hugepages to MAP_SIZE_BYTES and ensure /dev/hugepages is
# a hugetlbfs mount. Page size is OMNI_KV_OFFLOAD_HUGEPAGE_SIZE bytes
# (default 2 MiB). Does not create the mmap file — NPUSharedOffloadRegion
# creates vllm_offload_*.mmap itself.
#
# Usage:
#   MAP_SIZE_BYTES=<bytes> bash tools/deploy/start_server/setup_hugetlbfs.sh
# pd_run.sh also invokes this when OMNI_KV_OFFLOAD_HUGEPAGE is enabled.
set -euo pipefail

MNT="${MNT:-/dev/hugepages}"
MAX_RETRY=10
MAP_SIZE_BYTES="${MAP_SIZE_BYTES:?MAP_SIZE_BYTES is required}"
page_bytes="${OMNI_KV_OFFLOAD_HUGEPAGE_SIZE:-2097152}"

log_info() { echo "[INFO] $*" >&2; }
log_error() { echo "[ERROR] $*" >&2; }

if ! [[ "$MAP_SIZE_BYTES" =~ ^[1-9][0-9]*$ ]]; then
  log_error "MAP_SIZE_BYTES must be a positive integer (bytes), got ${MAP_SIZE_BYTES}"
  exit 1
fi
if ! [[ "$page_bytes" =~ ^[1-9][0-9]*$ ]]; then
  log_error "OMNI_KV_OFFLOAD_HUGEPAGE_SIZE must be a positive integer (bytes), got ${page_bytes}"
  exit 1
fi

HUGEPGSZ_KB=$((page_bytes / 1024))
# mount -o pagesize= still needs 2M / 1G; derived from the numeric byte size.
if (( page_bytes % (1 << 30) == 0 )); then
  PAGE_OPT="$((page_bytes / (1 << 30)))G"
elif (( page_bytes % (1 << 20) == 0 )); then
  PAGE_OPT="$((page_bytes / (1 << 20)))M"
else
  log_error "OMNI_KV_OFFLOAD_HUGEPAGE_SIZE must be a multiple of 1MiB or 1GiB, got ${page_bytes}"
  exit 1
fi
if [[ "$PAGE_OPT" != "2M" && "$PAGE_OPT" != "1G" ]]; then
  log_error "hugetlbfs pagesize=${PAGE_OPT} is not 2M or 1G; kernel mount may reject it"
  exit 1
fi

nr_path="/sys/kernel/mm/hugepages/hugepages-${HUGEPGSZ_KB}kB/nr_hugepages"
free_path="/sys/kernel/mm/hugepages/hugepages-${HUGEPGSZ_KB}kB/free_hugepages"
needed=$(( (MAP_SIZE_BYTES + page_bytes - 1) / page_bytes ))

if [[ ! -e "$nr_path" ]]; then
  log_error "kernel does not expose ${nr_path}"
  exit 1
fi

is_hugetlbfs_mount() {
  # Match npu_shared_offload_region._hugetlbfs_mounted: /proc/mounts path + fstype.
  awk -v p="$MNT" '$2 == p && $3 == "hugetlbfs" { found = 1 } END { exit !found }' /proc/mounts
}

ensure_hugetlbfs_mount() {
  mkdir -p "$MNT"
  if is_hugetlbfs_mount; then
    log_info "hugetlbfs already mounted at $MNT"
    return 0
  fi
  if mountpoint -q "$MNT" 2>/dev/null; then
    local fstype
    fstype=$(findmnt -n -o FSTYPE "$MNT" 2>/dev/null || echo unknown)
    log_info "unmounting non-hugetlbfs at $MNT (fstype=${fstype})"
    if ! umount "$MNT" 2>/dev/null; then
      log_info "umount failed; mounting hugetlbfs on top of $MNT"
    fi
  fi
  mount -t hugetlbfs -o pagesize="${PAGE_OPT}",mode=0770 none "$MNT"
  log_info "hugetlbfs mounted at $MNT (pagesize=${PAGE_OPT})"
  if ! is_hugetlbfs_mount; then
    log_error "failed to put a hugetlbfs mount at $MNT"
    awk '$2 == "/dev/hugepages" || /hugetlb/' /proc/mounts >&2 || true
    exit 1
  fi
}

remove_stale_offload_mmaps() {
  local files=()
  shopt -s nullglob
  files=("$MNT"/vllm_offload_*.mmap)
  shopt -u nullglob
  if (( ${#files[@]} == 0 )); then
    log_info "no leftover vllm_offload_*.mmap under $MNT"
    return 0
  fi
  log_info "removing leftover mmap files so hugepage size can change: ${files[*]}"
  rm -f -- "${files[@]}"
}

_last_pct=-1
_last_ts=0

progress_line() {
  local cur=$1 total=$2 attempt=$3
  if (( total <= 0 )); then total=1; fi
  if (( cur > total )); then cur=$total; fi
  local pct=$(( cur * 100 / total ))
  local gb=$(( cur * page_bytes / 1024 / 1024 / 1024 ))
  local total_gb=$(( total * page_bytes / 1024 / 1024 / 1024 ))
  local barw=40
  local filled=$(( barw * cur / total ))
  local bar
  bar="$(printf '%*s' "$filled" '' | tr ' ' '#')"
  bar+="$(printf '%*s' $((barw - filled)) '' | tr ' ' '-')"
  local msg
  msg=$(printf "adjusting [%s] %3d%% (%d/%d pages, %d/%d GB) attempt %d/%d" \
    "$bar" "$pct" "$cur" "$total" "$gb" "$total_gb" "$attempt" "$MAX_RETRY")
  if [[ -t 2 ]]; then
    printf "\r[INFO] %s" "$msg" >&2
    return
  fi
  local now
  now=$(date +%s)
  if (( pct != _last_pct || now - _last_ts >= 5 )); then
    log_info "$msg"
    _last_pct=$pct
    _last_ts=$now
  fi
}

read_nr() { cat "$nr_path" 2>/dev/null || echo 0; }

adjust_hugepages() {
  local current actual write_pid write_rc i
  current=$(read_nr)
  log_info "adjusting hugepages: page=${PAGE_OPT} current=${current} needed=${needed} (~$((needed * page_bytes / 1024 / 1024 / 1024)) GB)"
  if (( current == needed )); then
    log_info "hugepage pool already at ${needed} pages"
    return 0
  fi
  if (( current > needed )); then
    log_info "shrinking hugepage pool ${current} -> ${needed}"
  else
    log_info "growing hugepage pool ${current} -> ${needed}"
  fi

  local ok=0
  for i in $(seq 1 "$MAX_RETRY"); do
    _last_pct=-1
    _last_ts=0
    remove_stale_offload_mmaps
    log_info "adjust attempt ${i}/${MAX_RETRY}: writing ${needed} to ${nr_path}"
    set +e
    echo "$needed" > "$nr_path" &
    write_pid=$!
    while kill -0 "$write_pid" 2>/dev/null; do
      actual=$(read_nr)
      progress_line "$actual" "$needed" "$i"
      sleep 1
    done
    wait "$write_pid"
    write_rc=$?
    set -e
    if [[ -t 2 ]]; then
      echo >&2
    fi
    actual=$(read_nr)
    progress_line "$actual" "$needed" "$i"
    if [[ -t 2 ]]; then
      echo >&2
    fi
    if (( actual == needed )); then
      log_info "HugePages adjusted: ${actual} pages (~$((actual * page_bytes / 1024 / 1024 / 1024)) GB)"
      ok=1
      break
    fi
    log_info "attempt ${i}/${MAX_RETRY} incomplete: rc=${write_rc} actual=${actual} needed=${needed} free=$(cat "$free_path" 2>/dev/null || echo N/A)"
    sleep 1
  done

  actual=$(read_nr)
  if (( actual < needed )); then
    log_error "failed to reserve ${needed} hugepages (current=${actual})"
    exit 1
  fi
  if (( ok != 1 )); then
    log_info "WARNING: requested ${needed} pages but pool is ${actual}; leftover users under $MNT:"
    ls -l "$MNT" >&2 || true
    grep -E 'HugePages_|Hugepagesize|Hugetlb' /proc/meminfo >&2 || true
  fi
}

# Mount first so leftover hugetlbfs files can be removed before shrinking.
ensure_hugetlbfs_mount
remove_stale_offload_mmaps
adjust_hugepages
if ! is_hugetlbfs_mount; then
  log_error "$MNT is not a hugetlbfs mount after setup"
  exit 1
fi
log_info "hugetlbfs ready at $MNT nr_hugepages=$(read_nr) free=$(cat "$free_path" 2>/dev/null || echo N/A)"
