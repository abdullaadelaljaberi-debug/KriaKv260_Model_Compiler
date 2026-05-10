#!/usr/bin/env bash
# scripts/host/03_sync_to_kria.sh
# ─────────────────────────────────────────────────────────────────────────────
# Sync a compiled xmodel to the Kria board via scp. Verifies via sha256
# checksum on the remote side.
#
# On first run, sets up passwordless SSH key authentication automatically:
# generates a key if needed, copies it to the Kria (one password prompt),
# then verifies it works. Subsequent runs use the key silently.
#
# Subsequent SSH/scp invocations within the same script run are coalesced
# via SSH connection multiplexing.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

KRIA_XMODEL_DIR="/home/ubuntu/xmodels_vai35"

usage() {
    cat <<EOF
Usage: $(basename "$0") <user@host> <variant>

Arguments:
  user@host    SSH target. Typically: ubuntu@<board-ip>
  variant      The variant name compiled earlier (e.g. yolov5n)

Reads the xmodel from out/<variant>/<variant>_kv260.xmodel
and copies it to <user@host>:$KRIA_XMODEL_DIR/<variant>/<variant>_kv260.xmodel

On first run, sets up passwordless SSH (one-time, requires the password ONCE).
Subsequent runs are silent.

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

# ─── SSH key auth setup ────────────────────────────────────────────────────
# Probe whether passwordless SSH already works. BatchMode=yes means SSH
# will fail immediately if it would need to prompt — exactly the test we want.
ssh_keyauth_works() {
    ssh -o BatchMode=yes \
        -o ConnectTimeout=5 \
        -o StrictHostKeyChecking=accept-new \
        "$KRIA" "echo ok" &>/dev/null
}

setup_ssh_key_auth() {
    log_step "First-time SSH setup for $KRIA"
    log_info "  Without key auth, this script needs ~4 password prompts per run."
    log_info "  Setting up SSH key auth — you'll be prompted ONCE for the Kria password."
    echo

    # Generate a key if the user doesn't have one
    if [[ ! -f "$HOME/.ssh/id_ed25519" && ! -f "$HOME/.ssh/id_rsa" ]]; then
        log_info "Generating ~/.ssh/id_ed25519 (no passphrase)..."
        mkdir -p "$HOME/.ssh"
        chmod 700 "$HOME/.ssh"
        if ! ssh-keygen -t ed25519 -f "$HOME/.ssh/id_ed25519" -N "" \
                -C "auto-generated for $KRIA" -q; then
            die "ssh-keygen failed"
        fi
        log_ok "  generated ~/.ssh/id_ed25519"
    else
        log_info "  using existing SSH key in ~/.ssh/"
    fi

    # Copy the public key to the Kria — the one and only password prompt
    if ! have_cmd ssh-copy-id; then
        die "ssh-copy-id not available. Install openssh-client:
  sudo apt install openssh-client"
    fi

    log_info "Copying public key to $KRIA (enter Kria password when prompted):"
    if ! ssh-copy-id "$KRIA"; then
        die "ssh-copy-id failed. Check connectivity and password, then re-run."
    fi

    # Verify it works now
    if ssh_keyauth_works; then
        log_ok "Passwordless SSH configured. Future runs will be silent."
    else
        die "Key copied but passwordless auth still fails.
  Investigate ~/.ssh/authorized_keys on the Kria manually."
    fi
}

log_info "Testing SSH..."
if ssh_keyauth_works; then
    log_ok "passwordless SSH works"
else
    setup_ssh_key_auth
fi

# ─── enable SSH connection multiplexing for the rest of this script run ────
# All subsequent ssh/scp invocations within ~60 seconds reuse a single
# authenticated connection. Saves time and prevents stale prompts.
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
SSH_CTL="$HOME/.ssh/cm-${KRIA//[^a-zA-Z0-9]/_}-$$"
SSH_OPTS=(
    -o BatchMode=yes
    -o ControlMaster=auto
    -o ControlPath="$SSH_CTL"
    -o ControlPersist=60s
)
# Make sure the control socket gets torn down when this script exits
cleanup_ssh_ctl() {
    ssh "${SSH_OPTS[@]}" -O exit "$KRIA" &>/dev/null || true
    rm -f "$SSH_CTL" 2>/dev/null || true
}
trap cleanup_ssh_ctl EXIT

# ─── ensure remote dir exists ───────────────────────────────────────────────
log_info "Ensuring remote dir exists..."
ssh "${SSH_OPTS[@]}" "$KRIA" "mkdir -p '$REMOTE_DIR'" \
    || die "could not create $REMOTE_DIR on $KRIA"

# ─── compute local sha256 ───────────────────────────────────────────────────
log_info "Hashing local xmodel..."
LOCAL_SHA=$(sha256sum "$LOCAL_XMODEL" | awk '{print $1}')
LOCAL_SZ=$(stat -c%s "$LOCAL_XMODEL")
log_ok "local: ${LOCAL_SZ} bytes  sha256:${LOCAL_SHA:0:16}..."

# ─── scp ────────────────────────────────────────────────────────────────────
log_info "Copying via scp..."
scp "${SSH_OPTS[@]}" -p "$LOCAL_XMODEL" "$KRIA:$REMOTE_PATH"

# ─── verify on remote ───────────────────────────────────────────────────────
log_info "Verifying remote checksum..."
REMOTE_SHA=$(ssh "${SSH_OPTS[@]}" "$KRIA" "sha256sum '$REMOTE_PATH'" | awk '{print $1}')
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
