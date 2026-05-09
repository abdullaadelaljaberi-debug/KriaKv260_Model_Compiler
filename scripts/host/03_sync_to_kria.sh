#!/usr/bin/env bash
# scripts/host/03_sync_to_kria.sh
# ─────────────────────────────────────────────────────────────────────────────
# Sync a compiled xmodel to the Kria board via scp. Verifies the file via
# size + sha256 before reporting success.
#
# Usage:
#   bash scripts/host/03_sync_to_kria.sh <user@host> <variant>
#
# Example:
#   bash scripts/host/03_sync_to_kria.sh ubuntu@10.42.0.27 yolov5n
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

# Where xmodels land on the Kria. Documented in MODELS.md and used by all
# deploy notebooks/scripts. Hardcoded by design for reproducibility.
KRIA_XMODEL_DIR="/home/ubuntu/xmodels_vai35"

usage() {
    cat <<EOF
Usage: $(basename "$0") <user@host> <variant>

Arguments:
  user@host    SSH target. Typically: ubuntu@<board-ip>
  variant      The variant name compiled earlier (e.g. yolov5n)

The script reads the xmodel from out/<variant>/<variant>_kv260.xmodel
and copies it to <user@host>:$KRIA_XMODEL_DIR/<variant>/<variant>_kv260.xmodel

Example:
  bash $(basename "$0") ubuntu@10.42.0.27 yolov5n
EOF
    exit 2
}

[[ $# -lt 2 ]] && usage
KRIA="$1"
VARIANT="$2"

LOCAL_XMODEL="$REPO_ROOT/out/$VARIANT/${VARIANT}_kv260.xmodel"
[[ -f "$LOCAL_XMODEL" ]] || die "xmodel not found: $LOCAL_XMODEL
  Did you run scripts/host/02_compile.sh first?"

REMOTE_DIR="$KRIA_XMODEL_DIR/$VARIANT"
REMOTE_PATH="$REMOTE_DIR/${VARIANT}_kv260.xmodel"

log_step "Sync $VARIANT to $KRIA"
log_info "local : $LOCAL_XMODEL"
log_info "remote: $KRIA:$REMOTE_PATH"

# ─── ssh sanity ─────────────────────────────────────────────────────────────
log_info "Testing SSH..."
if ! ssh -o ConnectTimeout=5 -o BatchMode=no "$KRIA" "echo 'ssh-ok'" 2>/dev/null | grep -q ssh-ok; then
    die "SSH to $KRIA failed.
  - Is the board powered on and reachable?
  - Did you set up SSH keys?  ssh-copy-id $KRIA
  - Try manually:  ssh $KRIA"
fi

# ─── ensure remote dir exists ───────────────────────────────────────────────
log_info "Ensuring remote dir exists..."
ssh "$KRIA" "mkdir -p '$REMOTE_DIR'" || die "could not create $REMOTE_DIR on $KRIA"

# ─── compute local sha256 ───────────────────────────────────────────────────
log_info "Hashing local xmodel..."
LOCAL_SHA=$(sha256sum "$LOCAL_XMODEL" | awk '{print $1}')
LOCAL_SZ=$(stat -c%s "$LOCAL_XMODEL")
log_ok "local: ${LOCAL_SZ} bytes  sha256:${LOCAL_SHA:0:16}..."

# ─── scp ────────────────────────────────────────────────────────────────────
log_info "Copying via scp..."
scp -p "$LOCAL_XMODEL" "$KRIA:$REMOTE_PATH"

# ─── verify on remote ───────────────────────────────────────────────────────
log_info "Verifying remote checksum..."
REMOTE_SHA=$(ssh "$KRIA" "sha256sum '$REMOTE_PATH'" | awk '{print $1}')
if [[ "$REMOTE_SHA" != "$LOCAL_SHA" ]]; then
    die "checksum mismatch!
  local : $LOCAL_SHA
  remote: $REMOTE_SHA
  The file may have been corrupted in transit. Re-run this script."
fi

log_ok "verified: sha256 matches"
log_ok "Remote file: $KRIA:$REMOTE_PATH"
echo
log_info "Next: SSH to the board and run the live demo:"
log_info "  ssh $KRIA"
log_info "  cd KriaKv260_Model_Compiler"
log_info "  bash scripts/kria/run_live.sh $VARIANT       (Pass 5)"
