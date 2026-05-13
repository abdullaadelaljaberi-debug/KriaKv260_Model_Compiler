#!/usr/bin/env bash
# scripts/kria/run_benchmark.sh
# ─────────────────────────────────────────────────────────────────────────────
# Launch JupyterLab on the Kria with the VAI 3.5 benchmark notebook ready.
#
# Companion to run_live.sh but tailored for the benchmark workflow:
#   - Loads the pynq-venv (so the DPU runtime libs are visible)
#   - Sources XRT environment (if present)
#   - Unloads k26-starter-kits so the DPU overlay can program
#   - Prints a copy-paste URL with the Kria's LAN IP filled in
#
# Usage:
#   sudo bash scripts/kria/run_benchmark.sh
#
# After it starts, open the URL it prints in your laptop's browser.
# Ctrl-C twice in this terminal to stop JupyterLab.
#
# Environment overrides:
#   JUPYTER_PORT     Port to bind (default: 8888)
#   JUPYTER_HOST     Bind address (default: 0.0.0.0; set 127.0.0.1 for
#                    localhost-only access, requires SSH tunnel from laptop)
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

JUPYTER_PORT="${JUPYTER_PORT:-8888}"
JUPYTER_HOST="${JUPYTER_HOST:-0.0.0.0}"
NOTEBOOK_DIR="$REPO_ROOT/notebooks"
NOTEBOOK_FILE="$NOTEBOOK_DIR/04_vai35_benchmark.ipynb"

# ─── pre-flight ────────────────────────────────────────────────────────────
log_step "VAI 3.5 benchmark — launch JupyterLab"

# 0. Must be root: PYNQ-DPU mmaps the FPGA configuration registers.
if [[ $EUID -ne 0 ]]; then
    log_err "This script must be run as root: PYNQ-DPU needs root for FPGA mmap."
    log_err ""
    log_err "Re-run as:"
    log_err "  sudo bash scripts/kria/run_benchmark.sh"
    exit 1
fi
log_ok "running as root"

# 1. Notebook present?
if [[ ! -f "$NOTEBOOK_FILE" ]]; then
    log_err "Notebook not found: $NOTEBOOK_FILE"
    log_err ""
    log_err "On the Kria:"
    log_err "    cd ~/KriaKv260_Model_Compiler && git pull"
    exit 1
fi
log_ok "notebook  : $NOTEBOOK_FILE"

# 2. pynq-venv?
PYNQ_VENV="/usr/local/share/pynq-venv"
if [[ ! -d "$PYNQ_VENV" ]]; then
    die "pynq-venv not found at $PYNQ_VENV.
  Did you run scripts/kria/01_install_vai35.sh?"
fi

# 3. Staged data exists?
MODELS_DIR="$NOTEBOOK_DIR/Models_VAI35"
DATASETS_DIR="$NOTEBOOK_DIR/Datasets"
if [[ ! -d "$MODELS_DIR" ]] || [[ ! -d "$DATASETS_DIR" ]]; then
    log_warn "  Models_VAI35/ or Datasets/ missing under $NOTEBOOK_DIR"
    log_warn "  The benchmark will fail without them. Run:"
    log_warn "    bash scripts/host/04_stage_benchmark.sh         # on laptop"
    log_warn "    bash scripts/host/05_sync_benchmark_to_kria.sh ubuntu@<this-ip>"
fi

# ─── fix any root-owned files from previous Jupyter-as-root sessions ───────
# When this script runs as root and the user later runs cells that write
# files (CSVs, the report), those land owned by root. Next time the user
# scp's from the laptop, they'll hit "Permission denied" on the existing
# files. Fix proactively.
if [[ -d "$REPO_ROOT" ]]; then
    log_info "  ensuring repo owned by ubuntu (so scp from laptop works)"
    chown -R ubuntu:ubuntu "$REPO_ROOT" 2>/dev/null || \
        log_warn "  could not chown — non-fatal, ignore"
fi

# ─── prep the FPGA: unload starter-kit so DPU overlay can program ──────────
if have_cmd xmutil; then
    if xmutil listapps 2>/dev/null | grep -q "k26-starter-kits.*XRT_FLAT.*0,"; then
        log_info "  unloading k26-starter-kits..."
        xmutil unloadapp >/dev/null 2>&1 || \
            log_warn "  xmutil unloadapp returned non-zero (may still work)"
        log_ok "  starter-kit unloaded; PL ready for DPU bitstream"
    else
        log_debug "  no starter-kit loaded — PL is ready"
    fi
else
    log_warn "  xmutil not found — skipping starter-kit unload"
fi

# ─── load environment for the kernel ───────────────────────────────────────
[[ -f /etc/profile.d/pynq_venv.sh ]] && source /etc/profile.d/pynq_venv.sh
[[ -f /opt/xilinx/xrt/setup.sh   ]] && source /opt/xilinx/xrt/setup.sh

# ─── print friendly URL ────────────────────────────────────────────────────
log_step "Launching JupyterLab"
log_info "  port    : $JUPYTER_PORT"
log_info "  bind    : $JUPYTER_HOST"
log_info "  dir     : $NOTEBOOK_DIR"

KRIA_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ -z "$KRIA_IP" ]] && KRIA_IP="<kria-ip>"

if [[ "$JUPYTER_HOST" == "0.0.0.0" ]]; then
    cat <<EOF

  ┌─ Open this in your laptop's browser ─────────────────────
  │
  │   Once Jupyter prints its URL below (with the token),
  │   replace 'localhost' or '127.0.0.1' with the Kria's IP:
  │
  │      http://$KRIA_IP:$JUPYTER_PORT/lab?token=<copy-from-below>
  │
  │   Open notebooks/04_vai35_benchmark.ipynb after login.
  │
  │   Ctrl-C twice in this terminal to stop Jupyter.
  │
  └──────────────────────────────────────────────────────────

EOF
else
    cat <<EOF

  ┌─ Access via SSH tunnel ──────────────────────────────────
  │
  │   (JUPYTER_HOST=$JUPYTER_HOST → localhost-only, tunnel required)
  │
  │   1. On your laptop, tunnel:
  │        ssh -L $JUPYTER_PORT:localhost:$JUPYTER_PORT ubuntu@$KRIA_IP
  │
  │   2. Open URL Jupyter prints below in your laptop browser.
  │
  │   Ctrl-C twice in this terminal to stop Jupyter.
  │
  └──────────────────────────────────────────────────────────

EOF
fi

# ─── activate venv and launch ──────────────────────────────────────────────
source "$PYNQ_VENV/bin/activate"
exec jupyter lab \
    --no-browser \
    --ip="$JUPYTER_HOST" \
    --port="$JUPYTER_PORT" \
    --allow-root \
    --notebook-dir="$NOTEBOOK_DIR"
