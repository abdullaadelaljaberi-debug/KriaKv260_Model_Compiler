#!/usr/bin/env bash
# scripts/host/lib/common.sh
# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers for KriaKv260_Model_Compiler host-side scripts.
# Sourced by every script under scripts/host/, never executed directly.
# ─────────────────────────────────────────────────────────────────────────────

set -o pipefail   # propagate failures through pipes

# ─── colored output ─────────────────────────────────────────────────────────
if [[ -t 1 ]] && command -v tput &>/dev/null; then
    _C_RED=$(tput setaf 1);   _C_GREEN=$(tput setaf 2)
    _C_YELLOW=$(tput setaf 3); _C_BLUE=$(tput setaf 4)
    _C_BOLD=$(tput bold);     _C_RESET=$(tput sgr0)
else
    _C_RED=""; _C_GREEN=""; _C_YELLOW=""; _C_BLUE=""; _C_BOLD=""; _C_RESET=""
fi

log_info()  { printf '%s[*]%s %s\n'  "$_C_BLUE"   "$_C_RESET" "$*"; }
log_ok()    { printf '%s[✓]%s %s\n'  "$_C_GREEN"  "$_C_RESET" "$*"; }
log_warn()  { printf '%s[!]%s %s\n'  "$_C_YELLOW" "$_C_RESET" "$*"; }
log_err()   { printf '%s[✗]%s %s\n'  "$_C_RED"    "$_C_RESET" "$*" >&2; }
log_step()  { printf '\n%s%s━━ %s%s\n' "$_C_BOLD" "$_C_BLUE" "$*" "$_C_RESET"; }
die()       { log_err "$*"; exit 1; }

# ─── repo-root detection ─────────────────────────────────────────────────────
# Walks up from the script's location until it finds pyproject.toml.
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

# ─── prereq detection ────────────────────────────────────────────────────────
# Returns 0 if the command exists, 1 otherwise. Does not print anything.
have_cmd() { command -v "$1" &>/dev/null; }

# Returns 0 if the docker daemon is reachable. May fail with permission denied
# if the user is not in the docker group.
docker_ok() {
    have_cmd docker || return 1
    docker info &>/dev/null
}

# Returns 0 if NVIDIA-Container-Toolkit is wired up to docker.
nvidia_docker_ok() {
    have_cmd docker || return 1
    # The toolkit registers a runtime named "nvidia"; check both the
    # legacy default-runtime and the modern --gpus=all path.
    docker info 2>/dev/null | grep -qE 'Runtimes:.*nvidia' \
        || docker run --rm --gpus all hello-world &>/dev/null
}

# Returns 0 if an NVIDIA driver is loaded. Doesn't touch docker.
nvidia_driver_ok() {
    have_cmd nvidia-smi || return 1
    nvidia-smi &>/dev/null
}

# Returns 0 if there's at least N gigabytes free at the given path.
disk_free_gb() {
    local path="$1" min_gb="$2"
    local avail_gb
    avail_gb=$(df -BG --output=avail "$path" 2>/dev/null | tail -1 | tr -dc '0-9')
    [[ -n "$avail_gb" ]] && (( avail_gb >= min_gb ))
}

# Returns 0 if the host can reach the public internet (HEAD request to a CDN).
internet_ok() {
    have_cmd curl || return 1
    curl --silent --head --max-time 5 https://www.google.com &>/dev/null
}

# ─── safe sudo ───────────────────────────────────────────────────────────────
# Wraps sudo so we abort cleanly if the user doesn't have it; useful for
# scripts that need to install packages but should fail nicely otherwise.
need_sudo() {
    if [[ $EUID -eq 0 ]]; then
        # Already root, no sudo needed
        "$@"
    else
        have_cmd sudo || die "this step requires root or sudo, but neither is available"
        sudo "$@"
    fi
}

# ─── confirm ─────────────────────────────────────────────────────────────────
# Yes/no prompt, default no. Returns 0 on yes.
confirm() {
    local prompt="${1:-Proceed?}" reply
    read -r -p "$prompt [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]]
}

# ─── trap helper ─────────────────────────────────────────────────────────────
# Set an error message that prints if the script exits with a non-zero status.
# Usage: trap_err_msg "something specific went wrong"
trap_err_msg() {
    local msg="$*"
    trap 'rc=$?; if [[ $rc -ne 0 ]]; then log_err "'"$msg"' (exit $rc)"; fi' EXIT
}

# ─── version of this library ─────────────────────────────────────────────────
COMMON_LIB_VERSION="0.1.0"
