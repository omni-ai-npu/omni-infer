#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Setup 2-MB HugePages and mount hugetlbfs, create a file and optionally zero-fill all pages.
set -euo pipefail

#############################
# Tunable parameters
PAGES=${1:-}                         # if empty, auto-calc from MAP_SIZE_BYTES
MNT="${MNT:-/dev/hugepages}"         # mount point
HUGEPGSZ_KB=2048                     # 2 MB
MAX_RETRY=10
SLEEP=2
OMNI_FILE="${OMNI_FILE:-omni_cache}" # filename under hugetlbfs mount
MAP_SIZE_BYTES="${MAP_SIZE_BYTES:-1099511627776}"  # default 1 TiB
ZERO_FILL="${ZERO_FILL:-1}"          # 1 = write zero to every page; 0 = skip

#############################
# Color helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

#############################
# Progress bar / spinner
_spinner_frames='|/-\'
_spinner_idx=0
_progress_active=0

progress_bar() {
  local cur=${1:-0} total=${2:-1} prefix="${3:-}"
  if (( total <= 0 )); then total=1; fi
  if (( cur > total )); then cur=$total; fi

  local cols barw percent filled empty
  cols=$(tput cols 2>/dev/null || echo 80)
  barw=$(( cols - 40 ))
  (( barw < 10 )) && barw=10

  percent=$(( cur * 100 / total ))
  filled=$(( barw * cur / total ))
  empty=$(( barw - filled ))

  local bar_fill bar_empty
  bar_fill=$(printf '%*s' "$filled" '' | tr ' ' '#')
  bar_empty=$(printf '%*s' "$empty" '' | tr ' ' ' ')

  local spin_char=${_spinner_frames:_spinner_idx:1}
  _spinner_idx=$(( (_spinner_idx + 1) % ${#_spinner_frames} ))

  printf "\r%s [%s%s] %3d%% (%d/%d) %s" \
    "$prefix" "$bar_fill" "$bar_empty" "$percent" "$cur" "$total" "$spin_char"
  _progress_active=1
}

progress_done() {
  if (( _progress_active == 1 )); then
    printf "\n"
    _progress_active=0
  fi
}

#############################
# Utility helpers
huge_sys_dir="/sys/kernel/mm/hugepages/hugepages-${HUGEPGSZ_KB}kB"
nr_path="${huge_sys_dir}/nr_hugepages"
free_path="${huge_sys_dir}/free_hugepages"
resv_path="${huge_sys_dir}/resv_hugepages"

need_pages_from_size() {
  local size_bytes="$1"
  local sz_per_page=$((HUGEPGSZ_KB * 1024))
  echo $(( (size_bytes + sz_per_page - 1) / sz_per_page ))
}

#############################
# 1. Reserve hugepages (retry + progress bar)
reserve_pages() {
  local wanted=$1
  if [[ ! -e "$nr_path" ]]; then
    log_error "This kernel does not expose ${nr_path}. Check hugepage support and page size."
    return 1
  fi

  local current
  current=$(cat "$nr_path" 2>/dev/null || echo 0)
  local target=$(( current > wanted ? current : wanted ))

  for i in $(seq 1 $MAX_RETRY); do
    echo "$target" > "$nr_path"
    local wait_left=$SLEEP
    while (( wait_left >= 0 )); do
      actual=$(cat "$nr_path" 2>/dev/null || echo 0)
      progress_bar "$actual" "$target" "Reserving 2MB HugePages (attempt $i/$MAX_RETRY)"
      if [[ "$actual" -ge "$target" ]]; then
        progress_done
        log_info "HugePages(2MB) reserved: $actual"
        return 0
      fi
      sleep 0.3
      wait_left=$((wait_left - 1))
    done
  done
  progress_done
  actual=$(cat "$nr_path" 2>/dev/null || echo 0)
  log_error "Failed to reserve $target hugepages after $MAX_RETRY attempts (current=$actual)"
  return 1
}

#############################
# 2. Mount hugetlbfs
mount_hugetlbfs() {
  if mountpoint -q "$MNT"; then
    if findmnt -n -o FSTYPE "$MNT" | grep -q "^hugetlbfs$"; then
      log_info "hugetlbfs already mounted at $MNT (keeping it)"
      return 0
    else
      log_info "Unmounting non-hugetlbfs at $MNT"
      umount "$MNT" || { log_error "umount failed"; return 1; }
    fi
  fi

  mkdir -p "$MNT"
  mount -t hugetlbfs -o pagesize=2M,mode=0770 none "$MNT"
  log_info "hugetlbfs mounted at $MNT (2 MB pages)"
}

#############################
# 3. Create mmap file and optionally zero-fill
create_mmap_file() {
  local file="$MNT/$OMNI_FILE"
  local size="$MAP_SIZE_BYTES"

  if [[ -e "$file" ]]; then
    log_info "Removing existing file: $file"
    rm -f -- "$file"
  fi

  log_info "Creating hugetlbfs file: $file size=${size} bytes (~$((size/1024/1024/1024)) GB)"
  truncate -s "$size" "$file"
  chmod 660 "$file"

  # Zero-fill: touch every page to force hugepage allocation and clear content
  if [[ "${ZERO_FILL:-1}" == "1" ]]; then
    log_info "Zero-filling entire file (forcing all hugepages to be allocated and set to 0)..."
    if ! command -v python3 > /dev/null; then
      log_error "python3 not found; cannot mmap-zero-fill. Set ZERO_FILL=0 to skip."
      return 1
    fi
    python3 - "$file" "$size" <<'PY_EOF'
import mmap, os, sys, time
path, size_str = sys.argv[1], sys.argv[2]
size = int(size_str)
PAGE = 2 * 1024 * 1024   # 2 MB
num_pages = (size + PAGE - 1) // PAGE

fd = os.open(path, os.O_RDWR)
try:
    mm = mmap.mmap(fd, size, flags=mmap.MAP_SHARED,
                   prot=mmap.PROT_READ | mmap.PROT_WRITE)
    try:
        for i in range(num_pages):
            off = i * PAGE
            mm[off] = 0
            if (i + 1) % max(1, num_pages // 100) == 0 or i == num_pages - 1:
                percent = (i + 1) * 100 // num_pages
                sys.stderr.write(f"\rZero-fill progress: {percent}% ({i+1}/{num_pages} pages)")
                sys.stderr.flush()
        sys.stderr.write("\n")
    finally:
        mm.close()
finally:
    os.close(fd)
PY_EOF
    sync
    log_info "Zero-fill completed."
  else
    log_info "ZERO_FILL=0, skipping zero-fill (file will remain sparse, pages allocated on first access)"
  fi

  log_info "Created hugetlbfs file: $file size=$(stat -c%s "$file") bytes"
}

#############################
# 4. Main
main() {
  if [[ $EUID -ne 0 ]]; then
    log_error "Please run as root or with sudo"
    exit 1
  fi

  local k_hps_kb
  k_hps_kb=$(awk '/Hugepagesize/ {print $2}' /proc/meminfo || echo 0)
  if [[ "$k_hps_kb" -ne "$HUGEPGSZ_KB" ]]; then
    log_info "Kernel Hugepagesize is ${k_hps_kb} kB; targeting 2MB pool via ${huge_sys_dir}"
  fi

  local needed_pages
  if [[ -n "${PAGES:-}" ]]; then
    needed_pages="$PAGES"
    log_info "Using user-specified pages: $needed_pages (2MB each)"
  else
    needed_pages=$(need_pages_from_size "$MAP_SIZE_BYTES")
    log_info "Auto pages for ${MAP_SIZE_BYTES} bytes (~$((MAP_SIZE_BYTES/1024/1024/1024)) GB): $needed_pages (2MB each)"
  fi

  reserve_pages "$needed_pages"

  if [[ -r "$free_path" ]]; then
    log_info "free_hugepages=$(cat "$free_path"), resv_hugepages=$(cat "$resv_path" 2>/dev/null || echo N/A)"
  fi

  mount_hugetlbfs
  create_mmap_file

  log_info "HugePages setup completed successfully!"
  log_info "Verify:"
  log_info "  cat $nr_path (reserved pages)"
  log_info "  mount | grep hugetlbfs"
  log_info "  ls -lh $MNT/$OMNI_FILE"
}

trap 'progress_done' EXIT
main "$@"
