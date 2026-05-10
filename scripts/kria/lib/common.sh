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
die()       { log_err "$*"; exit 1; }

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
# Detection is based on installed deb packages.
vai_installed_version() {
    if dpkg -s libvart-runtime 2>/dev/null | grep -qE '^Version:\s*3\.5'; then
        echo "3.5"
    elif dpkg -s libvart-runtime 2>/dev/null | grep -qE '^Version:\s*2\.5'; then
        echo "2.5"
    elif dpkg -s libvart-runtime 2>/dev/null | grep -q '^Version:'; then
        # Some other version
        dpkg -s libvart-runtime 2>/dev/null | awk -F: '/^Version:/ {print $2}' | xargs
    else
        echo ""
    fi
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
