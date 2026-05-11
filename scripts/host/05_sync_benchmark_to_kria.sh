#!/usr/bin/env bash
# scripts/host/05_sync_benchmark_to_kria.sh
# ─────────────────────────────────────────────────────────────────────────────
# Sync staged benchmark data (models + datasets) from this host to the Kria.
# Uses rsync (not scp) — handles the directory tree, resumable, only copies
# changed/missing files.
#
# Mirrors 03_sync_to_kria.sh for SSH-key-auth setup and connection multiplexing,
# but extends with:
#   - rsync (recursive, with progress)
#   - post-sync verification: file count + total size
#   - Optional --dry-run mode to preview what would change
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

STAGE_ROOT="$REPO_ROOT/build/benchmark_stage"
REMOTE_BASE="/home/ubuntu/KriaKv260_Model_Compiler/notebooks"

usage() {
    cat <<EOF
Usage: $(basename "$0") <user@host> [--dry-run]

Sync the staged benchmark data tree to the Kria.

Arguments:
  user@host    SSH target, typically: ubuntu@<board-ip>
  --dry-run    Show what would be transferred without actually doing it

Source:      $STAGE_ROOT/{Models_VAI35,Datasets}/
Destination: <user@host>:$REMOTE_BASE/{Models_VAI35,Datasets}/

Prereqs:
  - You must have run scripts/host/04_stage_benchmark.sh first
  - The Kria must be reachable via SSH
  - Free disk on the Kria: ≥ same as the stage size

This uses rsync with --partial-dir support, so an interrupted sync resumes
cleanly. On first run, sets up SSH key auth (one password prompt).

Example:
  bash $(basename "$0") ubuntu@10.42.0.27
  bash $(basename "$0") ubuntu@10.42.0.27 --dry-run
EOF
    exit 2
}

[[ $# -lt 1 ]] && usage
KRIA="$1"
DRY_RUN=0
if [[ "${2:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

log_step "Sync benchmark data to $KRIA"
log_info "stage : $STAGE_ROOT"
log_info "remote: $KRIA:$REMOTE_BASE/"
(( DRY_RUN == 1 )) && log_warn "DRY RUN — no files will be transferred"

# ─── prereqs ───────────────────────────────────────────────────────────────
if [[ ! -d "$STAGE_ROOT" ]]; then
    die "stage root not found: $STAGE_ROOT
  Run scripts/host/04_stage_benchmark.sh first."
fi
if [[ ! -d "$STAGE_ROOT/Models_VAI35" ]] && [[ ! -d "$STAGE_ROOT/Datasets" ]]; then
    die "Stage root has neither Models_VAI35/ nor Datasets/ — staging incomplete."
fi
if ! have_cmd rsync; then
    die "rsync not installed:  sudo apt install rsync"
fi

# Report local stage size
local_size=$(du -sh "$STAGE_ROOT" | cut -f1)
local_files=$(find "$STAGE_ROOT" -type f -o -type l | wc -l)
log_info "stage size: $local_size  ($local_files files/symlinks)"

# ─── SSH key auth (same pattern as 03_sync_to_kria.sh) ─────────────────────
ssh_keyauth_works() {
    ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
        "$KRIA" "echo ok" &>/dev/null
}

setup_ssh_key_auth() {
    log_step "First-time SSH setup for $KRIA"
    log_info "  Setting up SSH key auth — you'll be prompted once for the Kria password."

    if [[ ! -f "$HOME/.ssh/id_ed25519" && ! -f "$HOME/.ssh/id_rsa" ]]; then
        log_info "Generating ~/.ssh/id_ed25519..."
        mkdir -p "$HOME/.ssh"; chmod 700 "$HOME/.ssh"
        ssh-keygen -t ed25519 -f "$HOME/.ssh/id_ed25519" -N "" \
            -C "auto-generated for $KRIA" -q || die "ssh-keygen failed"
        log_ok "  generated ~/.ssh/id_ed25519"
    fi

    have_cmd ssh-copy-id || die "ssh-copy-id missing. Install:  sudo apt install openssh-client"
    log_info "Copying public key to $KRIA..."
    ssh-copy-id "$KRIA" || die "ssh-copy-id failed."

    ssh_keyauth_works || die "key copied but passwordless auth still fails."
    log_ok "Passwordless SSH configured."
}

if ssh_keyauth_works; then
    log_ok "passwordless SSH works"
else
    setup_ssh_key_auth
fi

# Connection multiplexing for the rest of this script
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
SSH_CTL="$HOME/.ssh/cm-bench-${KRIA//[^a-zA-Z0-9]/_}-$$"
SSH_OPTS=(
    -o BatchMode=yes
    -o ControlMaster=auto
    -o ControlPath="$SSH_CTL"
    -o ControlPersist=60s
)
cleanup_ssh_ctl() {
    ssh "${SSH_OPTS[@]}" -O exit "$KRIA" &>/dev/null || true
    rm -f "$SSH_CTL" 2>/dev/null || true
}
trap cleanup_ssh_ctl EXIT

# ─── remote pre-flight ──────────────────────────────────────────────────────
log_info "Checking remote disk space..."
remote_free_kb=$(ssh "${SSH_OPTS[@]}" "$KRIA" \
    "df --output=avail '$REMOTE_BASE' 2>/dev/null | tail -1 | tr -dc '0-9'" || echo "0")
if [[ -z "$remote_free_kb" || "$remote_free_kb" == "0" ]]; then
    log_warn "Could not query remote disk space — proceeding anyway."
else
    remote_free_gb=$(( remote_free_kb / 1024 / 1024 ))
    local_size_kb=$(du -sk "$STAGE_ROOT" | cut -f1)
    local_size_gb=$(( local_size_kb / 1024 / 1024 ))
    log_info "  remote free: ${remote_free_gb} GB"
    log_info "  local stage: ${local_size_gb} GB"
    if (( remote_free_gb < local_size_gb + 2 )); then
        die "Insufficient remote disk: need ~$((local_size_gb + 2)) GB, have ${remote_free_gb} GB.
  Free space on the Kria (rm -rf old Models_VAI35/ Datasets/ if any), then re-run."
    fi
fi

# Ensure remote base dir exists
ssh "${SSH_OPTS[@]}" "$KRIA" "mkdir -p '$REMOTE_BASE'" \
    || die "could not create $REMOTE_BASE on $KRIA"

# ─── rsync ─────────────────────────────────────────────────────────────────
# --copy-links is CRITICAL: the host-side staging uses symlinks under
# Datasets/imagenet_sample/images/ that point into the source extraction
# dir (imagenetv2-matched-frequency-format-val/). Without --copy-links,
# rsync would preserve the symlinks AS symlinks — and since their targets
# (laptop's /home/aaljaberi/.../build/...) don't exist on the Kria, every
# ImageNet image becomes a broken pointer. --copy-links sends the real
# file contents in place of each symlink, so the Kria gets actual JPEGs.
RSYNC_FLAGS=(
    --archive              # -rlptgoD (preserve perms, times, symlinks, etc.)
    --copy-links           # dereference symlinks during transfer (CRITICAL — see comment above)
    --human-readable
    --partial              # keep partial files for resume
    --info=progress2       # nice progress display
    --exclude='_downloads/'        # don't sync cache dirs
    --exclude='.stage_state.json'  # local state, not for Kria
    --exclude='imagenetv2-matched-frequency-format-val/'  # source for symlinks; --copy-links makes this redundant on Kria
)
(( DRY_RUN == 1 )) && RSYNC_FLAGS+=( --dry-run )

# rsync each top-level dir separately so progress is clearer
for sub in Models_VAI35 Datasets; do
    src="$STAGE_ROOT/$sub/"
    [[ -d "$src" ]] || { log_warn "  $sub/ not staged locally — skipping"; continue; }

    log_step "rsync $sub"
    # Note: trailing slash on src means "contents of"; without it, the directory itself goes inside.
    rsync "${RSYNC_FLAGS[@]}" \
        -e "ssh ${SSH_OPTS[*]}" \
        "$src" \
        "$KRIA:$REMOTE_BASE/$sub/" \
        || die "rsync of $sub failed."
    log_ok "  $sub synced"
done

(( DRY_RUN == 1 )) && { log_warn "DRY RUN: no files actually transferred."; exit 0; }

# ─── verification ──────────────────────────────────────────────────────────
log_step "Verification"

log_info "Counting remote files..."
remote_files=$(ssh "${SSH_OPTS[@]}" "$KRIA" \
    "find $REMOTE_BASE/Models_VAI35 $REMOTE_BASE/Datasets -type f 2>/dev/null | wc -l" || echo "0")
log_info "  remote files:  $remote_files"
# Show each top-level dir's actual size separately. The previous version
# tried to sum the `du -sh` output with awk, which broke because du's output
# mixes units (947M + 3.3G = "950M (approx)" due to suffix-stripping).
log_info "  remote sizes:"
ssh "${SSH_OPTS[@]}" "$KRIA" \
    "du -sh $REMOTE_BASE/Models_VAI35 $REMOTE_BASE/Datasets 2>/dev/null" \
    | sed 's|^|    |' || log_warn "  could not query remote sizes"

# Spot-check: one xmodel + the imagenet labels.txt
log_info "Spot-checking key files..."
ok_xmodel=$(ssh "${SSH_OPTS[@]}" "$KRIA" \
    "find $REMOTE_BASE/Models_VAI35 -name '*.xmodel' 2>/dev/null | head -1" || echo "")
ok_labels=$(ssh "${SSH_OPTS[@]}" "$KRIA" \
    "ls $REMOTE_BASE/Datasets/imagenet_sample/labels.txt 2>/dev/null" || echo "")
ok_coco=$(ssh "${SSH_OPTS[@]}" "$KRIA" \
    "ls $REMOTE_BASE/Datasets/coco_val2017/annotations/instances_val2017.json 2>/dev/null" || echo "")

[[ -n "$ok_xmodel" ]] && log_ok "  xmodel:        $ok_xmodel" || log_warn "  no .xmodel found on remote"
[[ -n "$ok_labels" ]] && log_ok "  imagenet:      labels.txt present" || log_warn "  imagenet labels.txt missing"
[[ -n "$ok_coco"   ]] && log_ok "  coco anns:     present" || log_warn "  COCO annotations missing"

echo
log_ok "Benchmark data is on the Kria."
log_info "Next: on the Kria,"
log_info "  cd ~/KriaKv260_Model_Compiler && git pull"
log_info "  sudo bash scripts/kria/run_live.sh yolov5n     # to launch Jupyter"
log_info "  # then open notebooks/04_vai35_benchmark.ipynb in the browser"
