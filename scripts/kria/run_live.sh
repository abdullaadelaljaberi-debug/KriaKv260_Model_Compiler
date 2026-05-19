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
#   bash scripts/kria/run_live.sh yolov11n
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
Usage: $(basename "$0") <variant> [mode]

Launch a Jupyter server with one of the live-demo notebooks configured for
the given variant.

Arguments:
  variant      One of the variants compiled and synced:
               - yolov5n, yolov5s (LPR demo notebooks)
               - yolov11n         (egg detection demo notebook)
               - yolox_tiny, yolox_nano (deferred — needs GraphRunner notebook)
  mode         For yolov5* variants: 'text' (default) or 'visual'.
                 - text:   02_deploy_text.ipynb   — max-throughput, HTML status
                                                    only, no per-frame rendering.
                                                    Use for benchmarks.
                 - visual: 03_deploy_visual.ipynb — live video preview + bounding
                                                    boxes + interactive sliders.
                                                    Use for demos and tuning.
               For yolov11n: mode is ignored; always launches
                 eggs/05_deploy_visual.ipynb (interactive notebook with
                 USB camera / video file / image folder selector).

By default Jupyter binds to 0.0.0.0:8888 — reachable from any device on
your LAN. Open the URL Jupyter prints (it includes a single-use token) in
your laptop's browser, replacing 'localhost' with the Kria's IP.

To restrict access to the Kria itself (requires an SSH tunnel from your
laptop to reach it), set:

  JUPYTER_HOST=127.0.0.1 sudo -E bash $(basename "$0") <variant>

Environment overrides:
  JUPYTER_PORT  Port to bind to (default: 8888)
  JUPYTER_HOST  Bind address (default: 0.0.0.0; set to 127.0.0.1 for
                localhost-only access requiring an SSH tunnel)
EOF
    exit 2
}

[[ $# -lt 1 ]] && usage
VARIANT="$1"
MODE="${2:-text}"

# ─── Dispatch variant → notebook ───────────────────────────────────────────
# Different variants use different notebooks. yolov5* uses the LPR notebooks
# (text/visual); yolov11n uses the eggs notebook (interactive input selector
# so the user picks camera/video/folder in-notebook).
case "$VARIANT" in
    yolov5n|yolov5s)
        case "$MODE" in
            text)   NOTEBOOK_FILENAME="02_deploy_text.ipynb"   ;;
            visual) NOTEBOOK_FILENAME="03_deploy_visual.ipynb" ;;
            *)
                log_err "Unknown mode for $VARIANT: $MODE (expected 'text' or 'visual')"
                usage
                ;;
        esac
        ;;
    yolov11n|yolov11s)
        NOTEBOOK_FILENAME="eggs/05_deploy_visual.ipynb"
        if [[ "$MODE" != "text" ]]; then
            # MODE was explicitly passed but we ignore it for yolov11n.
            log_info "  (note: mode '$MODE' ignored for yolov11n; eggs notebook handles its own input selection)"
        fi
        ;;
    yolox_tiny|yolox_nano)
        log_err "$VARIANT requires a multi-DPU-subgraph notebook (GraphRunner-based),"
        log_err "which is not yet implemented. Use the benchmark notebook for these"
        log_err "variants until the live-demo path is added."
        exit 1
        ;;
    *)
        log_err "Unknown variant: $VARIANT"
        log_err "Supported variants:"
        log_err "  yolov5n, yolov5s  — LPR demo (text/visual)"
        log_err "  yolov11n          — egg detection demo"
        log_err "  yolox_*           — not yet implemented for live demo"
        exit 1
        ;;
esac

JUPYTER_PORT="${JUPYTER_PORT:-8888}"
# Default to 0.0.0.0 (LAN-accessible) since this is the thesis workflow:
# you reach Jupyter from your laptop's browser directly, no SSH tunnel.
# Set JUPYTER_HOST=127.0.0.1 to revert to localhost-only (requires a tunnel).
JUPYTER_HOST="${JUPYTER_HOST:-0.0.0.0}"

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
NOTEBOOK="$REPO_ROOT/notebooks/$NOTEBOOK_FILENAME"
if [[ ! -f "$NOTEBOOK" ]]; then
    log_err "Notebook not found: $NOTEBOOK"
    log_err ""
    log_err "Make sure you've pulled the latest repo."
    log_err ""
    log_err "Available notebooks ship in v0.7 / v0.8 / v0.9:"
    log_err "    notebooks/02_deploy_text.ipynb        (yolov5* max throughput)"
    log_err "    notebooks/03_deploy_visual.ipynb      (yolov5* visual)"
    log_err "    notebooks/eggs/05_deploy_visual.ipynb (yolov11n eggs demo)"
    log_err ""
    log_err "From your laptop:"
    log_err "    git pull origin main"
    log_err "On the Kria:"
    log_err "    cd ~/KriaKv260_Model_Compiler && git pull"
    exit 1
fi
log_ok "notebook  : $NOTEBOOK"

# 3. Camera available? Only warn for yolov5* (LPR notebook assumes camera);
# the eggs notebook lets the user pick non-camera input modes interactively.
if [[ "$VARIANT" == yolov5* ]] && [[ ! -e /dev/video0 ]]; then
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

# Discover the Kria's primary LAN IP so we can print a copy-pasteable URL.
# `hostname -I` returns space-separated IPv4 addresses across all interfaces;
# the first is typically the LAN address. Falls back to '<kria-ip>' as a
# literal placeholder if discovery fails.
KRIA_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ -z "$KRIA_IP" ]] && KRIA_IP="<kria-ip>"

if [[ "$JUPYTER_HOST" == "0.0.0.0" ]]; then
    # LAN-bound: browser hits the Kria directly.
    cat <<EOF

  ┌─ Open this in your laptop's browser ─────────────────────
  │
  │   Once Jupyter prints its URL below (with the token),
  │   replace 'localhost' or '127.0.0.1' with the Kria's IP:
  │
  │      http://$KRIA_IP:$JUPYTER_PORT/lab?token=<copy-from-below>
  │
  │   The notebook's first cell reads LPR_VARIANT='$VARIANT'
  │   and LPR_XMODEL from the environment — just Run All.
  │
  │   Ctrl-C twice to stop Jupyter.
  │
  └──────────────────────────────────────────────────────────

EOF
else
    # Localhost-only: user needs an SSH tunnel.
    cat <<EOF

  ┌─ Access this notebook from your laptop ──────────────────
  │
  │   (JUPYTER_HOST=$JUPYTER_HOST → localhost-only, tunnel required)
  │
  │   1. SSH-tunnel on your laptop (replace IP):
  │        ssh -L $JUPYTER_PORT:localhost:$JUPYTER_PORT $USER@$KRIA_IP
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
fi

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
