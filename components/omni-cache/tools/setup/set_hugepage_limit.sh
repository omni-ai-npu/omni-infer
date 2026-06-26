#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# set_hugepage_limit.sh
# Purpose: Permanently adjust the 2MB hugepage limit (auto-calc or manual).
# Usage:
#   sudo ./set_hugepage_limit.sh                         # Auto-calc (reserve 10% for system)
#   sudo ./set_hugepage_limit.sh --target-pages 1048576  # Manual page count

set -euo pipefail

#############################
# Default parameters
HUGEPGSZ_KB=2048              # 2MB
TARGET_PAGES=""               # Empty means auto-calc
MAX_RETRY=5

# Parse arguments
for arg in "$@"; do
    case $arg in
        --target-pages=*)
            TARGET_PAGES="${arg#*=}"
            ;;
        --target-pages)
            # Supports --target-pages 1048576 format
            TARGET_PAGES="$2"
            shift
            ;;
        --help)
            echo "Usage: $0 [--target-pages=N]"
            echo "  --target-pages=N  : Manually specify number of hugepages (2MB each), otherwise auto-calculated."
            exit 0
            ;;
    esac
done

#############################
# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

#############################
# 1. Check root
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)."
    exit 1
fi

#############################
# 2. Calculate target page count (if not manually specified)
total_mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
total_mem_gb=$(( total_mem_kb / 1024 / 1024 ))
if [[ -z "$TARGET_PAGES" ]]; then
    # Reserve 10% of memory for the system (at least 20GB, at most 200GB)
    reserve_gb=$(( total_mem_gb / 10 ))
    (( reserve_gb < 20 )) && reserve_gb=20
    (( reserve_gb > 200 )) && reserve_gb=200
    max_huge_gb=1228  # ~1.2 TiB cap
    huge_gb=$(( total_mem_gb - reserve_gb ))
    (( huge_gb > max_huge_gb )) && huge_gb=$max_huge_gb
    if (( huge_gb <= 0 )); then
        log_error "Not enough memory to reserve hugepages (total ${total_mem_gb}GB, need at least ${reserve_gb}GB for system)."
        exit 1
    fi
    huge_mb=$(( huge_gb * 1024 ))
    TARGET_PAGES=$(( huge_mb / (HUGEPGSZ_KB / 1024) ))
    log_info "Physical memory: ${total_mem_gb}GB, reserving ${reserve_gb}GB for system → target hugepages: ${TARGET_PAGES} pages (~${huge_gb}GB)"
else
    log_info "User-specified target pages: $TARGET_PAGES ( ~$(( TARGET_PAGES * HUGEPGSZ_KB / 1024 / 1024 )) GB )"
fi

#############################
# 3. Kill processes that are using 2MB hugepages (otherwise nr_hugepages may not be adjustable)
log_warn "Searching for processes using 2MB hugepages..."
pids=$(grep -l "KernelPageSize:  2048 kB" /proc/*/smaps 2>/dev/null | awk -F/ '{print $3}' | sort -u || true)
if [[ -n "$pids" ]]; then
    log_warn "The following processes are using 2MB hugepages and will be killed:"
    for pid in $pids; do
        cmd=$(ps -p "$pid" -o cmd= 2>/dev/null || echo "unknown")
        echo "  PID $pid: $cmd"
    done
    read -p "Continue to kill these processes? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        log_error "User aborted."
        exit 1
    fi
    for pid in $pids; do
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 2
    log_info "Killed all processes using hugepages."
else
    log_info "No processes currently using 2MB hugepages."
fi

#############################
# 4. Persist configuration via sysctl
sysctl_conf="/etc/sysctl.conf"
if grep -q "^vm.nr_hugepages" "$sysctl_conf"; then
    sed -i "s/^vm.nr_hugepages=.*/vm.nr_hugepages=${TARGET_PAGES}/" "$sysctl_conf"
else
    echo "vm.nr_hugepages=${TARGET_PAGES}" >> "$sysctl_conf"
fi
log_info "Updated $sysctl_conf with vm.nr_hugepages=${TARGET_PAGES}"

#############################
# 5. Apply immediately (write 0 to release, then write target)
nr_path="/sys/kernel/mm/hugepages/hugepages-${HUGEPGSZ_KB}kB/nr_hugepages"
if [[ ! -f "$nr_path" ]]; then
    log_error "Kernel does not support 2MB hugepages."
    exit 1
fi

log_info "Resetting hugepage pool (writing 0 to $nr_path)..."
echo 0 > "$nr_path"
sleep 3

log_info "Setting hugepage count to $TARGET_PAGES ..."
echo "$TARGET_PAGES" > "$nr_path" || {
    log_error "Failed to set hugepages. Check available memory or fragmentation."
    exit 1
}

# Wait and verify actual allocation
for i in $(seq 1 $MAX_RETRY); do
    sleep 2
    actual=$(cat "$nr_path")
    if (( actual >= TARGET_PAGES )); then
        log_info "Successfully allocated $actual hugepages (≥ $TARGET_PAGES)."
        break
    fi
    if (( i == MAX_RETRY )); then
        log_error "After $MAX_RETRY attempts, only $actual hugepages allocated (need $TARGET_PAGES)."
        log_error "System may have memory fragmentation. Try rebooting and re-run."
        exit 1
    fi
done

#############################
# 6. Final verification
log_info "Hugepage limit adjustment completed permanently!"
echo "=========================================="
grep HugePages_ /proc/meminfo
echo "=========================================="
log_info "Configuration is permanent (reboot will keep vm.nr_hugepages=${TARGET_PAGES})."