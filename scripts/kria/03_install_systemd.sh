#!/usr/bin/env bash
# scripts/kria/03_install_systemd.sh
# ─────────────────────────────────────────────────────────────────────────────
# Installs a systemd unit that re-runs 02_apply_tuning.sh on every boot.
# Without this, tuning is lost across reboots.
#
# Idempotent: re-running updates the unit file if the script's path has
# changed, otherwise just reloads systemd.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

parse_common_flags "$@"

UNIT_NAME="kriakv260-tuning.service"
UNIT_PATH="/etc/systemd/system/$UNIT_NAME"
TUNING_SCRIPT="$SCRIPT_DIR/02_apply_tuning.sh"

log_step "Install systemd unit: $UNIT_NAME"

# ─── prereq: 02_apply_tuning.sh must exist ──────────────────────────────────
if [[ ! -x "$TUNING_SCRIPT" ]]; then
    die "$TUNING_SCRIPT not found or not executable.
  Make sure scripts/kria/02_apply_tuning.sh exists and chmod +x'd."
fi

# ─── write the unit file ────────────────────────────────────────────────────
# Type=oneshot: runs once on boot, exits.
# RemainAfterExit=yes: systemd reports the unit as "active" after the script
#   finishes, so systemctl status shows green. Otherwise it shows as "dead".
# After=multi-user.target: run late in boot, after USB has settled.
# StandardOutput=journal: collected by journalctl -u <unit>.

UNIT_CONTENT=$(cat <<EOF
[Unit]
Description=KriaKv260_Model_Compiler runtime tuning (USB + CPU + camera)
After=multi-user.target
Wants=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash $TUNING_SCRIPT --quiet
StandardOutput=journal
StandardError=journal
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
EOF
)

# Check whether the file would change. Idempotency: don't bother writing /
# restarting if the content is the same.
if [[ -f "$UNIT_PATH" ]] && diff -q <(echo "$UNIT_CONTENT") "$UNIT_PATH" &>/dev/null; then
    log_ok "  $UNIT_PATH already up to date"
else
    log_info "  Writing $UNIT_PATH"
    echo "$UNIT_CONTENT" | need_sudo tee "$UNIT_PATH" >/dev/null
    log_ok "  unit file written"
fi

# ─── reload + enable ───────────────────────────────────────────────────────
log_info "Reloading systemd..."
need_sudo systemctl daemon-reload

log_info "Enabling $UNIT_NAME..."
need_sudo systemctl enable "$UNIT_NAME" 2>&1 | tail -2

# Start the unit now (which runs the tuning script once)
log_info "Starting $UNIT_NAME (will run 02_apply_tuning.sh)..."
if need_sudo systemctl start "$UNIT_NAME"; then
    sleep 1
    log_ok "  service started"
else
    log_warn "  systemctl start failed. Inspect:"
    log_warn "    sudo journalctl -u $UNIT_NAME -n 50"
fi

# Report status
echo
log_info "Service status:"
need_sudo systemctl status "$UNIT_NAME" --no-pager -l 2>&1 | head -15

echo
log_ok "Persistence configured."
log_info "  - 02_apply_tuning.sh will run on every boot"
log_info "  - Disable later with:  sudo systemctl disable $UNIT_NAME"
log_info "  - Manual re-run:       sudo systemctl restart $UNIT_NAME"
log_info "  - Inspect logs:        sudo journalctl -u $UNIT_NAME"
