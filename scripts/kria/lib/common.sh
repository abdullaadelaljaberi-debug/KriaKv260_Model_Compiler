#!/usr/bin/env bash
# scripts/kria/lib/common.sh
# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers for Kria-side scripts. Sourced by every kria/*.sh.
# Mirrors scripts/host/lib/common.sh but with Kria-specific helpers.
# ─────────────────────────────────────────────────────────────────────────────

set -o pipefail

# ─── verbosity ──────────────────────────────────────────────────────────────
# Controlled by VERBOSE env var or --verbose / --quiet flags handled by
# each script. Scripts set $VERBOSITY before sourcing this file (default 1).
VERBOSITY="${VERBOSITY:-1}"   # 0=quiet  1=normal  2=verbose

# ─── colored output ─────────────────────────────────────────────────────────
if [[ -t 1 ]] && command -v tput &>/dev/null && (( $(tput colors 2>/dev/null || echo 0) > 0 )); then
    _C_RED=$(tput setaf 1);   _C_GREEN=$(tput setaf 2)
    _C_YELLOW=$(tput setaf 3); _C_BLUE=$(tput setaf 4)
    _C_BOLD=$(tput bold);     _C_RESET=$(tput sgr0)
else
    _C_RED=""; _C_GREEN=""; _C_YELLOW=""; _C_BLUE=""; _C_BOLD=""; _C_RESET=""
fi

log_info()  { (( VERBOSITY >= 1 )) && printf '%s[*]%s %s\n' "$_C_BLUE"   "$_C_RESET" "$*"; }
log_ok()    { (( VERBOSITY >= 1 )) && printf '%s[OK]%s %s\n' "$_C_GREEN"  "$_C_RESET" "$*"; }
log_warn()  { printf '%s[!]%s %s\n'  "$_C_YELLOW" "$_C_RESET" "$*"; }
log_err()   { printf '%s[FAIL]%s %s\n' "$_C_RED"  "$_C_RESET" "$*" >&2; }
log_step()  { printf '\n%s%s━━ %s%s\n' "$_C_BOLD" "$_C_BLUE" "$*" "$_C_RESET"; }
log_debug() { (( VERBOSITY >= 2 )) && printf '%s[..]%s %s\n' "$_C_BLUE" "$_C_RESET" "$*"; }
die()       {
    log_err "$*"
    # Record failure in summary if one is active
    if [[ -n "${SUMMARY_CURRENT_KEY:-}" ]]; then
        # Truncate detail to one line for the table
        local short
        short=$(echo "$*" | head -1 | cut -c1-50)
        summary_stage_failed "$short"
    fi
    exit 1
}

# Run a command, logging it at debug level, and showing output only when verbose.
run_cmd() {
    log_debug "+ $*"
    if (( VERBOSITY >= 2 )); then
        "$@"
    else
        local out
        out=$("$@" 2>&1) || { rc=$?; echo "$out" >&2; return $rc; }
    fi
}

# ─── repo-root detection ────────────────────────────────────────────────────
find_repo_root() {
    local d="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
    d="$(cd "$(dirname "$d")" && pwd)"
    while [[ "$d" != "/" ]]; do
        [[ -f "$d/pyproject.toml" ]] && { echo "$d"; return; }
        d="$(dirname "$d")"
    done
    die "could not locate repo root (no pyproject.toml found walking up)"
}

REPO_ROOT="$(find_repo_root)"

# ─── basic prereq detection ─────────────────────────────────────────────────
have_cmd() { command -v "$1" &>/dev/null; }

is_root() { [[ $EUID -eq 0 ]]; }

# Run a command via sudo unless we're already root. Errors if sudo missing.
need_sudo() {
    if is_root; then
        "$@"
    else
        have_cmd sudo || die "this step requires root or sudo, but neither is available"
        sudo "$@"
    fi
}

# Returns 0 if the user is in the docker / dialout / video / etc. group.
in_group() {
    id -nG "$USER" 2>/dev/null | tr ' ' '\n' | grep -qx "$1"
}

# ─── Kria-specific detection ────────────────────────────────────────────────
# Returns 0 if running on a Kria SOM. Best heuristic: /proc/device-tree/compatible
# on Kria contains "xlnx,zynqmp" plus a board-specific string.
is_kria() {
    [[ -r /proc/device-tree/compatible ]] || return 1
    grep -qa "xlnx,zynqmp" /proc/device-tree/compatible 2>/dev/null
}

# Returns the SOM model: kv260, kr260, kd240, or "unknown".
kria_model() {
    if [[ -r /proc/device-tree/compatible ]]; then
        local compat
        compat=$(tr -d '\0' < /proc/device-tree/compatible 2>/dev/null | tr ',' '\n')
        if grep -qi "kv260" <<<"$compat"; then echo "kv260"; return; fi
        if grep -qi "kr260" <<<"$compat"; then echo "kr260"; return; fi
        if grep -qi "kd240" <<<"$compat"; then echo "kd240"; return; fi
    fi
    echo "unknown"
}

# Returns Ubuntu version string (e.g., "22.04"), or empty if not Ubuntu.
ubuntu_version() {
    [[ -r /etc/os-release ]] || { echo ""; return; }
    local id ver
    id=$(awk -F= '/^ID=/ {print $2}' /etc/os-release | tr -d '"')
    [[ "$id" == "ubuntu" ]] || { echo ""; return; }
    awk -F= '/^VERSION_ID=/ {print $2}' /etc/os-release | tr -d '"'
}

# Returns "3.5", "2.5", or "" depending on what (if any) VAI version is installed.
# Detection strategy:
#   1. Primary: check libvart deb package version (this is the canonical VAI
#      runtime package; the name 'libvart' is used in VAI 2.5 and 3.5).
#   2. Fallback: check libxir version (always co-installed with libvart).
#   3. Last-ditch: check if /usr/bin/xir exists and parse from libvart.so.* symlinks.
vai_installed_version() {
    # Primary: libvart package (NOT libvart-runtime — that's a different name)
    local pkg_ver
    pkg_ver=$(dpkg -s libvart 2>/dev/null | awk -F': *' '/^Version:/ {print $2}' | head -1)
    if [[ -n "$pkg_ver" ]]; then
        if [[ "$pkg_ver" == 3.5* ]]; then echo "3.5"; return; fi
        if [[ "$pkg_ver" == 2.5* ]]; then echo "2.5"; return; fi
        # Some other version
        echo "$pkg_ver"
        return
    fi

    # Fallback: libxir package
    pkg_ver=$(dpkg -s libxir 2>/dev/null | awk -F': *' '/^Version:/ {print $2}' | head -1)
    if [[ -n "$pkg_ver" ]]; then
        if [[ "$pkg_ver" == 3.5* ]]; then echo "3.5"; return; fi
        if [[ "$pkg_ver" == 2.5* ]]; then echo "2.5"; return; fi
        echo "$pkg_ver"
        return
    fi

    # Last-ditch: try to read version from the actual .so file name
    # /usr/lib/libvart-runner.so.3.5.0 → "3.5"
    local so_ver
    so_ver=$(ls /usr/lib/libvart-runner.so.*.*.* 2>/dev/null \
             | head -1 | sed -E 's|.*\.so\.([0-9]+\.[0-9]+)\..*|\1|')
    if [[ -n "$so_ver" ]]; then
        echo "$so_ver"
        return
    fi

    # Nothing found
    echo ""
}

# Returns "yes" if Kria-PYNQ is installed (pynq_venv exists), "no" otherwise.
kria_pynq_installed() {
    [[ -d /usr/local/share/pynq-venv ]] && echo "yes" || echo "no"
}

# ─── disk space ─────────────────────────────────────────────────────────────
disk_free_gb() {
    local path="$1" min_gb="$2"
    local avail_gb
    avail_gb=$(df -BG --output=avail "$path" 2>/dev/null | tail -1 | tr -dc '0-9')
    [[ -n "$avail_gb" ]] && (( avail_gb >= min_gb ))
}

# ─── confirm prompt ─────────────────────────────────────────────────────────
confirm() {
    local prompt="${1:-Proceed?}" reply
    if [[ -n "${YES:-}" ]]; then
        # Allow tests / CI / "I know what I'm doing" runs
        echo "$prompt [auto-yes]"
        return 0
    fi
    read -r -p "$prompt [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]]
}

# ─── argument parsing helper ────────────────────────────────────────────────
# Pop --verbose / --quiet / --yes flags from the script's args. Each script
# calls this once at the top.
parse_common_flags() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --verbose) VERBOSITY=2; shift ;;
            --quiet)   VERBOSITY=0; shift ;;
            --yes|-y)  YES=1; shift ;;
            --help|-h) return 99 ;;
            --) shift; break ;;
            *) break ;;
        esac
    done
    # Return remaining args via global ARGS_REMAINING
    ARGS_REMAINING=( "$@" )
    return 0
}

# ─── log to file ────────────────────────────────────────────────────────────
# Tee output to a log file so install runs are debuggable post-hoc.
# Usage: log_to_file <path>
log_to_file() {
    local logfile="$1"
    mkdir -p "$(dirname "$logfile")"
    exec > >(tee -a "$logfile") 2>&1
    log_debug "Logging to $logfile"
}

COMMON_LIB_VERSION="0.1.0"

# ─────────────────────────────────────────────────────────────────────────────
# Stage status tracking + end-of-run summary
# ─────────────────────────────────────────────────────────────────────────────
# Usage in scripts:
#   summary_init                              # call once at start
#   summary_stage_start "1/5" "Description"   # before stage runs
#   summary_stage_done    [detail]            # green ✓
#   summary_stage_skipped [detail]            # gray ↷
#   summary_stage_failed  [detail]            # red ✗
#   summary_print                             # call once at end (also auto on trap)
#   summary_set_action "what to do next"      # printed if any stage failed
#
# Each script should:
#   summary_init
#   trap 'summary_print' EXIT
#   ... do work, calling summary_stage_* at boundaries ...

declare -ga SUMMARY_STAGE_KEYS=()
declare -gA SUMMARY_STAGE_TITLES=()
declare -gA SUMMARY_STAGE_STATUS=()
declare -gA SUMMARY_STAGE_DETAIL=()
declare -gA SUMMARY_STAGE_DURATION=()
SUMMARY_SCRIPT_START=0
SUMMARY_STAGE_START=0
SUMMARY_CURRENT_KEY=""
SUMMARY_ACTION_HINT=""

summary_init() {
    SUMMARY_STAGE_KEYS=()
    SUMMARY_STAGE_TITLES=()
    SUMMARY_STAGE_STATUS=()
    SUMMARY_STAGE_DETAIL=()
    SUMMARY_STAGE_DURATION=()
    SUMMARY_SCRIPT_START=$SECONDS
    SUMMARY_CURRENT_KEY=""
    SUMMARY_ACTION_HINT=""
}

summary_stage_start() {
    local key="$1" title="$2"
    SUMMARY_STAGE_KEYS+=( "$key" )
    SUMMARY_STAGE_TITLES["$key"]="$title"
    SUMMARY_STAGE_STATUS["$key"]="RUNNING"
    SUMMARY_STAGE_DETAIL["$key"]=""
    SUMMARY_STAGE_START=$SECONDS
    SUMMARY_CURRENT_KEY="$key"
}

summary_stage_done() {
    local detail="${1:-}"
    [[ -z "$SUMMARY_CURRENT_KEY" ]] && return
    SUMMARY_STAGE_STATUS["$SUMMARY_CURRENT_KEY"]="DONE"
    SUMMARY_STAGE_DETAIL["$SUMMARY_CURRENT_KEY"]="$detail"
    SUMMARY_STAGE_DURATION["$SUMMARY_CURRENT_KEY"]=$(( SECONDS - SUMMARY_STAGE_START ))
    SUMMARY_CURRENT_KEY=""
}

summary_stage_skipped() {
    local detail="${1:-already done}"
    [[ -z "$SUMMARY_CURRENT_KEY" ]] && return
    SUMMARY_STAGE_STATUS["$SUMMARY_CURRENT_KEY"]="SKIPPED"
    SUMMARY_STAGE_DETAIL["$SUMMARY_CURRENT_KEY"]="$detail"
    SUMMARY_STAGE_DURATION["$SUMMARY_CURRENT_KEY"]=$(( SECONDS - SUMMARY_STAGE_START ))
    SUMMARY_CURRENT_KEY=""
}

summary_stage_failed() {
    local detail="${1:-error}"
    [[ -z "$SUMMARY_CURRENT_KEY" ]] && return
    SUMMARY_STAGE_STATUS["$SUMMARY_CURRENT_KEY"]="FAILED"
    SUMMARY_STAGE_DETAIL["$SUMMARY_CURRENT_KEY"]="$detail"
    SUMMARY_STAGE_DURATION["$SUMMARY_CURRENT_KEY"]=$(( SECONDS - SUMMARY_STAGE_START ))
    SUMMARY_CURRENT_KEY=""
}

summary_set_action() {
    SUMMARY_ACTION_HINT="$1"
}

# Pretty-print the table. Called manually or via EXIT trap.
summary_print() {
    # If summary was never initialized, skip silently
    [[ -z "$SUMMARY_SCRIPT_START" || "${#SUMMARY_STAGE_KEYS[@]}" -eq 0 ]] && return 0

    # If a stage is still running (script exited mid-stage), mark it as failed
    if [[ -n "$SUMMARY_CURRENT_KEY" ]]; then
        SUMMARY_STAGE_STATUS["$SUMMARY_CURRENT_KEY"]="FAILED"
        SUMMARY_STAGE_DETAIL["$SUMMARY_CURRENT_KEY"]="interrupted"
        SUMMARY_STAGE_DURATION["$SUMMARY_CURRENT_KEY"]=$(( SECONDS - SUMMARY_STAGE_START ))
    fi

    local total_elapsed=$(( SECONDS - SUMMARY_SCRIPT_START ))
    local total_str
    if (( total_elapsed >= 60 )); then
        total_str="$((total_elapsed / 60))m $((total_elapsed % 60))s"
    else
        total_str="${total_elapsed}s"
    fi

    echo
    printf '%s━━ Install summary %s%s\n' "$_C_BOLD" "$(printf '━%.0s' {1..58})" "$_C_RESET"
    echo

    local any_failed=0
    local key title status detail dur dur_str status_str
    for key in "${SUMMARY_STAGE_KEYS[@]}"; do
        title="${SUMMARY_STAGE_TITLES[$key]}"
        status="${SUMMARY_STAGE_STATUS[$key]}"
        detail="${SUMMARY_STAGE_DETAIL[$key]:-}"
        dur="${SUMMARY_STAGE_DURATION[$key]:-0}"

        if (( dur >= 60 )); then
            dur_str="$((dur / 60))m$((dur % 60))s"
        else
            dur_str="${dur}s"
        fi

        case "$status" in
            DONE)    status_str="${_C_GREEN}[✓] DONE${_C_RESET}    " ;;
            SKIPPED) status_str="${_C_BLUE}[↷] SKIPPED${_C_RESET} " ;;
            FAILED)  status_str="${_C_RED}[✗] FAILED${_C_RESET}  "; any_failed=1 ;;
            *)       status_str="${_C_YELLOW}[?] $status${_C_RESET}  " ;;
        esac

        # Format: "  [1/5] First-boot config              [✓] DONE    1s  hostname=kria-lpr"
        printf "  [%s] %-40s %s %5s  %s\n" "$key" "$title" "$status_str" "$dur_str" "$detail"
    done

    echo
    printf '%s━━ Time %s%s\n' "$_C_BOLD" "$(printf '━%.0s' {1..68})" "$_C_RESET"
    echo
    printf "  Total elapsed: %s\n" "$total_str"
    if [[ -n "${LOG_FILE:-}" ]]; then
        printf "  Logs: %s\n" "$LOG_FILE"
    fi
    echo

    if (( any_failed )); then
        printf '%s━━ Action needed %s%s\n' "$_C_BOLD" "$(printf '━%.0s' {1..59})" "$_C_RESET"
        echo
        if [[ -n "$SUMMARY_ACTION_HINT" ]]; then
            printf "  %s\n" "$SUMMARY_ACTION_HINT"
        else
            printf "  One or more stages failed. Inspect %s for details.\n" "${LOG_FILE:-output}"
            printf "  After fixing, re-run the script — completed stages will be skipped.\n"
        fi
        echo
    fi
}
