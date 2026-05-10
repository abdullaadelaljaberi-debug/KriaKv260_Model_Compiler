#!/usr/bin/env bash
# scripts/kria/run_live.sh
# ─────────────────────────────────────────────────────────────────────────────
# Launch a Jupyter server with the live demo notebook ready to run.
#
# Usage:
#   bash scripts/kria/run_live.sh <variant>
#
# Example:
#   bash scripts/kria/run_live.sh yolov5n
#
# After launch, follow the printed URL on your laptop's browser.
#
# Recommended laptop-side access pattern (more secure than network exposure):
#   ssh -L 8888:localhost:8888 ubuntu@<kria-ip>
#   then point your laptop's browser at http://localhost:8888
# See docs/USAGE.md for the SSH-tunneled workflow.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

parse_common_flags "$@"
set -- "${ARGS_REMAINING[@]}"

usage() {
    cat <<EOF
Usage: $(basename "$0") <variant>

Launch a Jupyter server with the live demo notebook (notebooks/02_deploy_live.ipynb)
configured for the given variant.

Arguments:
  variant      One of the variants compiled and synced: yolov5n, yolov5s,
               yolox_tiny, yolox_nano.

By default Jupyter binds to localhost:8888. Access via SSH tunnel from your
laptop:

  ssh -L 8888:localhost:8888 $USER@<this-board's-ip>
  # then on your laptop, browse http://localhost:8888

Environment overrides:
  JUPYTER_PORT  Port to bind to (default: 8888)
  JUPYTER_HOST  Bind address (default: 127.0.0.1; set to 0.0.0.0 for network access)
EOF
    exit 2
}

[[ $# -lt 1 ]] && usage
VARIANT="$1"

JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_HOST="${JUPYTER_HOST:-127.0.0.1}"

# ─── prereqs ───────────────────────────────────────────────────────────────
log_step "Live demo: $VARIANT"

# 1. xmodel exists?
XMODEL="/home/ubuntu/xmodels_vai35/$VARIANT/${VARIANT}_kv260.xmodel"
if [[ ! -f "$XMODEL" ]]; then
    log_err "xmodel not found: $XMODEL"
    log_err "  From your laptop, sync it:"
    log_err "    bash scripts/host/03_sync_to_kria.sh ubuntu@<this-board-ip> $VARIANT"
    exit 1
fi
log_ok "xmodel found: $XMODEL"

# 2. Notebook exists?
NOTEBOOK="$REPO_ROOT/notebooks/02_deploy_live.ipynb"
if [[ ! -f "$NOTEBOOK" ]]; then
    log_warn "Live notebook not found at $NOTEBOOK"
    log_warn "  This notebook is delivered in Pass 6. For now, falling back to your"
    log_warn "  existing notebook 09v2 if you have it staged."
    # Look for the user's existing live notebook
    fallback=$(find "$HOME" -maxdepth 3 -name "09*v2*.ipynb" 2>/dev/null | head -1)
    if [[ -z "$fallback" ]]; then
        die "No live notebook found. Stage notebook 09v2 in your home dir."
    fi
    NOTEBOOK="$fallback"
    log_info "Using fallback: $NOTEBOOK"
fi

# 3. Camera available?
if [[ ! -e /dev/video0 ]]; then
    log_warn "  No /dev/video0 — plug in the camera before running the notebook."
fi

# 4. pynq-venv exists?
PYNQ_VENV="/usr/local/share/pynq-venv"
if [[ ! -d "$PYNQ_VENV" ]]; then
    die "pynq-venv not found at $PYNQ_VENV.
  Did you run scripts/kria/01_install_vai35.sh?"
fi

# ─── apply tuning if not already in performance mode ───────────────────────
gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "")
if [[ "$gov" != "performance" ]]; then
    log_warn "  CPU governor is '$gov', not 'performance' — applying tuning"
    bash "$SCRIPT_DIR/02_apply_tuning.sh" --quiet || \
        log_warn "  tuning script reported issues but continuing"
fi

# ─── set environment for the notebook ──────────────────────────────────────
# The notebook reads $LPR_VARIANT and $LPR_XMODEL to know which model to load
# without hardcoding paths inside the .ipynb file.
export LPR_VARIANT="$VARIANT"
export LPR_XMODEL="$XMODEL"

# ─── launch jupyter ────────────────────────────────────────────────────────
log_step "Launching JupyterLab"
log_info "  port    : $JUPYTER_PORT"
log_info "  bind    : $JUPYTER_HOST"
log_info "  variant : $VARIANT"
log_info "  notebook: $NOTEBOOK"

cat <<EOF

  ┌─ Access this notebook from your laptop ──────────────────
  │
  │   1. SSH-tunnel on your laptop (replace IP):
  │        ssh -L $JUPYTER_PORT:localhost:$JUPYTER_PORT $USER@<kria-ip>
  │
  │   2. Open the URL Jupyter prints below (with the token)
  │      in your laptop's browser.
  │
  │   3. The notebook's first cell reads LPR_VARIANT='$VARIANT'
  │      and LPR_XMODEL from the environment — just Run All.
  │
  │   Ctrl-C twice to stop Jupyter.
  │
  └──────────────────────────────────────────────────────────

EOF

# Activate the pynq-venv and launch JupyterLab
source "$PYNQ_VENV/bin/activate"
exec jupyter lab \
    --no-browser \
    --ip="$JUPYTER_HOST" \
    --port="$JUPYTER_PORT" \
    --notebook-dir="$REPO_ROOT/notebooks"
