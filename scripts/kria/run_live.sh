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

# 0. Must be root: PYNQ-DPU needs root to mmap the FPGA configuration registers
# (pynq.ps._ClocksUltrascale raises "Root permissions required" otherwise).
if [[ $EUID -ne 0 ]]; then
    log_err "This script must be run as root: the PYNQ-DPU stack mmaps the FPGA"
    log_err "configuration registers, which requires root permissions."
    log_err ""
    log_err "Re-run as:"
    log_err "  sudo bash scripts/kria/run_live.sh $VARIANT"
    exit 1
fi
log_ok "running as root (needed for FPGA mmap)"

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

    # Search candidates: current $HOME, plus the invoking user's home if running under sudo.
    # Under `sudo`, $HOME == /root (not the original user's home), so we'd miss notebooks
    # staged at /home/ubuntu/. SUDO_USER is set by sudo to the invoking username.
    declare -a search_homes=( "$HOME" )
    if [[ -n "${SUDO_USER:-}" ]] && [[ "$SUDO_USER" != "root" ]]; then
        sudo_home=$(getent passwd "$SUDO_USER" | cut -d: -f6)
        if [[ -n "$sudo_home" ]] && [[ "$sudo_home" != "$HOME" ]]; then
            search_homes+=( "$sudo_home" )
        fi
    fi

    fallback=""
    for h in "${search_homes[@]}"; do
        fallback=$(find "$h" -maxdepth 3 -name "09*v2*.ipynb" 2>/dev/null | head -1)
        [[ -n "$fallback" ]] && break
    done

    if [[ -z "$fallback" ]]; then
        die "No live notebook found. Stage notebook 09v2 in one of:
  $(printf '    %s\n' "${search_homes[@]}")
Or wait for Pass 6's notebooks/02_deploy_live.ipynb."
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
# without hardcoding paths inside the .ipynb file. $REPO_ROOT helps the
# notebook find the lpr_pipeline package.
export LPR_VARIANT="$VARIANT"
export LPR_XMODEL="$XMODEL"
export REPO_ROOT="$REPO_ROOT"

# ─── prepare the FPGA: unload starter-kit so PYNQ can program the DPU ──────
# KV260 boots with `k26-starter-kits` loaded in XRT_FLAT mode. That mode
# doesn't expose an xclbin-compatible PL, so PYNQ's device enumeration finds
# nothing and DpuOverlay() raises "No Devices Found".
#
# The fix is the standard Kria-PYNQ procedure: unload the starter-kit, then
# let DpuOverlay program the DPU bitstream into the now-empty PL.
#
# If no app is loaded (someone already did this), `xmutil unloadapp` is a
# harmless no-op.
if have_cmd xmutil; then
    current_app=$(sudo xmutil listapps 2>/dev/null \
                    | awk 'NR>1 && $NF != "" && $NF != "0,"  {print $1; exit}')
    # Detect specifically the k26-starter-kits in active slot
    if sudo xmutil listapps 2>/dev/null | grep -q "k26-starter-kits.*XRT_FLAT.*0,"; then
        log_info "  Unloading k26-starter-kits so PYNQ can program the DPU..."
        sudo xmutil unloadapp >/dev/null 2>&1 || \
            log_warn "  xmutil unloadapp returned non-zero (may still work)"
        log_ok "  starter-kit unloaded; PL ready for DPU bitstream"
    else
        log_debug "  no starter-kit loaded — PL is ready"
    fi
else
    log_warn "  xmutil not found — skipping starter-kit unload"
    log_warn "  If DpuOverlay raises 'No Devices Found', install xrt-tools"
fi

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

# Activate the pynq-venv and launch JupyterLab.
# --allow-root is needed when run via sudo (Jupyter refuses to start as root
# by default for safety; the PYNQ-DPU stack requires root for FPGA mmap).
#
# Source pynq_venv.sh and xrt setup.sh (if present) so the Jupyter kernels
# inherit the LD_LIBRARY_PATH=/usr/lib patch from Pass 5 Stage 5d, plus any
# XRT environment vars. AMD's reference workflow uses `sudo su` to get a
# login shell that loads /etc/profile.d/*.sh automatically; we replicate
# that by sourcing them explicitly here.
[[ -f /etc/profile.d/pynq_venv.sh ]] && source /etc/profile.d/pynq_venv.sh
[[ -f /opt/xilinx/xrt/setup.sh   ]] && source /opt/xilinx/xrt/setup.sh

source "$PYNQ_VENV/bin/activate"
jupyter_args=(
    --no-browser
    --ip="$JUPYTER_HOST"
    --port="$JUPYTER_PORT"
    --notebook-dir="$REPO_ROOT/notebooks"
)
if [[ $EUID -eq 0 ]]; then
    jupyter_args+=( --allow-root )
fi
exec jupyter lab "${jupyter_args[@]}"
