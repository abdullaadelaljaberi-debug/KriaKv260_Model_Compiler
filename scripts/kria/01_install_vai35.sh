#!/usr/bin/env bash
# scripts/kria/01_install_vai35.sh
# ─────────────────────────────────────────────────────────────────────────────
# Installs the Kria-PYNQ stack and upgrades the VAI runtime to 3.5 on the
# board. Idempotent: re-running on an already-correctly-installed system
# detects the state and skips heavy steps.
#
# This script is the result of reverse-engineering from session notes plus
# AMD's official documentation. Sources:
#   - https://github.com/Xilinx/Kria-PYNQ
#   - https://github.com/Xilinx/Vitis-AI/blob/v3.5/board_setup/mpsoc/
#   - https://github.com/Xilinx/DPU-PYNQ/tree/design_contest_3.5
#
# IMPORTANT: this is a reconstruction. Test against an already-working board
# first (which should produce mostly "already installed, skipping" output)
# before attempting a fresh install. If anything fails, the script logs to
# /var/log/kriakv260_install.log for debugging.
#
# Stages:
#   1. Optional: A3-partial first-boot config (hostname, password prompt)
#   2. xlnx-config.sysinit (Xilinx-recommended system tuning)
#   3. Kria-PYNQ install (if not already present)
#   4. VAI 2.5 → 3.5 upgrade (if VAI 3.5 not already present)
#       a. Download vai3.5_kr260.zip
#       b. dpkg-install the debs in dependency order
#       c. Copy lack_lib helper libs (Kria-PYNQ-specific workaround)
#       d. Apply glog 0.5.0 workaround
#   5. DPU bitstream from DPU-PYNQ design_contest_3.5 branch
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

# Pop --verbose / --quiet flags
parse_common_flags "$@"

# Config — exposed as env vars so users can override
# Default URL for vai3.5_kr260.zip. Hosted on AMD's openDownload portal
# If AMD moves it, override with:
#   VAI35_KR260_ZIP_URL=... bash 01_install_vai35.sh
VAI35_KR260_ZIP_URL="${VAI35_KR260_ZIP_URL:-https://www.xilinx.com/bin/public/openDownload?filename=vai3.5_kr260.zip}"

# Kria-PYNQ git repo branch. Pinning to v3.0.1 — current stable as of pipeline writing.
KRIA_PYNQ_REPO="${KRIA_PYNQ_REPO:-https://github.com/Xilinx/Kria-PYNQ.git}"
KRIA_PYNQ_BRANCH="${KRIA_PYNQ_BRANCH:-v3.0}"

# DPU-PYNQ design contest branch — has the VAI 3.5 DPU bitstream
DPU_PYNQ_REPO="${DPU_PYNQ_REPO:-https://github.com/Xilinx/DPU-PYNQ.git}"
DPU_PYNQ_BRANCH="${DPU_PYNQ_BRANCH:-design_contest_3.5}"

# Where to stage downloads. Cleaned up after successful install.
STAGE_DIR="${STAGE_DIR:-$HOME/.cache/kriakv260_install}"

# Log file — appended, not overwritten, so you can re-run and see history.
LOG_FILE="${LOG_FILE:-$HOME/kriakv260_install.log}"

log_to_file "$LOG_FILE"
log_info "Logging to $LOG_FILE"

# ─── Initialize summary tracker ─────────────────────────────────────────────
summary_init
trap 'summary_print' EXIT

# ─── prereq sanity: 00_check_prereqs.sh must have passed ────────────────────
if ! bash "$SCRIPT_DIR/00_check_prereqs.sh" --quiet; then
    die "00_check_prereqs.sh failed. Run it directly to see what's wrong:
  bash scripts/kria/00_check_prereqs.sh"
fi

log_step "Kria install — script will skip already-done steps"

# ─── STAGE 1: First-boot config (A3 partial automation) ────────────────────
log_step "[1/5] First-boot config"
summary_stage_start "1/5" "First-boot config"

current_hostname=$(hostname)
if [[ "$current_hostname" == "kria" ]] || [[ "$current_hostname" == "ubuntu" ]]; then
    log_info "Current hostname '$current_hostname' looks default. Set a custom one?"
    if confirm "  Set hostname to 'kria-lpr'? (or N to skip)"; then
        need_sudo hostnamectl set-hostname kria-lpr
        log_ok "hostname set to kria-lpr (effective after reboot)"
        summary_stage_done "hostname=kria-lpr"
    else
        log_info "  keeping current hostname '$current_hostname'"
        summary_stage_done "hostname=$current_hostname (kept)"
    fi
else
    log_ok "hostname '$current_hostname' (custom, leaving alone)"
    summary_stage_done "hostname=$current_hostname"
fi

# Password change is NOT automated (would require expect, fragile).
# We just check whether the user is still on the default ubuntu password.
# We can't actually verify the password, so we just nudge.
if id -u ubuntu &>/dev/null && [[ "$USER" == "ubuntu" ]]; then
    log_warn "  Reminder: if you haven't changed the default 'ubuntu' password,"
    log_warn "  run 'passwd' after this script finishes. Default passwords are"
    log_warn "  a security risk, especially on a board with a public IP."
fi

# Network config: refuse to touch. User had to SSH in to run this script,
# so their network already works.
log_info "  Skipping network config (already working — you SSH'd in)"

# ─── STAGE 2: xlnx-config.sysinit ───────────────────────────────────────────
log_step "[2/5] xlnx-config.sysinit (Xilinx-recommended system tuning)"
summary_stage_start "2/5" "xlnx-config.sysinit"

if ! have_cmd xlnx-config; then
    log_info "Installing xlnx-config snap..."
    need_sudo snap install xlnx-config --classic \
        || die "snap install xlnx-config failed.
  Check: snap list, snap refresh, sudo journalctl -u snapd"
    log_ok "xlnx-config installed"
fi

# xlnx-config.sysinit is interactive (asks to install various components).
# We pipe 'yes' to accept defaults, BUT only once per machine. The .done
# stamp prevents re-running and asking confirmations on every script run.
SYSINIT_STAMP="/var/local/kriakv260_sysinit.done"
if [[ -f "$SYSINIT_STAMP" ]]; then
    log_ok "xlnx-config.sysinit already run (stamp: $SYSINIT_STAMP)"
    summary_stage_skipped "stamp present"
else
    log_info "Running xlnx-config.sysinit — installs base components"
    log_warn "  This step is interactive on a fresh system. Accept the defaults."
    log_warn "  It can take 5-10 min on first run."
    if confirm "  Run xlnx-config.sysinit now?"; then
        need_sudo xlnx-config.sysinit || die "xlnx-config.sysinit failed"
        need_sudo touch "$SYSINIT_STAMP"
        log_ok "xlnx-config.sysinit complete"
        summary_stage_done "ran sysinit, accepted defaults"
    else
        log_warn "  Skipped. Re-run this script to do it later."
        summary_stage_skipped "user declined"
    fi
fi

# ─── STAGE 3: Kria-PYNQ ─────────────────────────────────────────────────────
log_step "[3/5] Kria-PYNQ stack"
summary_stage_start "3/5" "Kria-PYNQ stack"

if [[ "$(kria_pynq_installed)" == "yes" ]]; then
    log_ok "Kria-PYNQ already installed at /usr/local/share/pynq-venv"
    KRIA_PYNQ_FRESH_INSTALL=0
else
    KRIA_PYNQ_FRESH_INSTALL=1
    log_info "Cloning Kria-PYNQ ($KRIA_PYNQ_BRANCH branch)..."
    mkdir -p "$STAGE_DIR"
    cd "$STAGE_DIR"
    if [[ -d Kria-PYNQ ]]; then
        cd Kria-PYNQ && git fetch && git checkout "$KRIA_PYNQ_BRANCH"
    else
        git clone --depth 1 -b "$KRIA_PYNQ_BRANCH" "$KRIA_PYNQ_REPO"
        cd Kria-PYNQ
    fi

    log_info "Running Kria-PYNQ install.sh (board: KV260) — takes 20-30 min"
    log_warn "  Lots of output. We're logging it to $LOG_FILE."
    if ! need_sudo bash install.sh -b KV260; then
        die "Kria-PYNQ install.sh failed.
  Check $LOG_FILE for the error.
  Common causes:
    - Network drop mid-install (re-run; install.sh is mostly idempotent)
    - Disk full (df -h to check)
    - Conflicting Python venv (sudo rm -rf /usr/local/share/pynq-venv and re-run)"
    fi
    log_ok "Kria-PYNQ installed"
fi

# ─── Stage 3 post-fixes — apply even on re-runs (idempotent) ───────────────
# Kria-PYNQ's install.sh sometimes leaves the venv with numpy 2.x while cv2
# was built against numpy 1.x, breaking imports. We pin numpy<2 unconditionally
# (pip is a no-op if already at the right version).
STAGE3_DETAIL=""
if [[ -d /usr/local/share/pynq-venv ]]; then
    log_info "  Pinning numpy<2 in pynq-venv (cv2 ABI compatibility)"
    current_numpy=$(/usr/local/share/pynq-venv/bin/python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "?")
    log_debug "    current numpy in venv: $current_numpy"
    if [[ "$current_numpy" == 2.* ]]; then
        log_warn "    numpy $current_numpy is 2.x — downgrading to 1.x for cv2 compat"
        need_sudo /usr/local/share/pynq-venv/bin/pip install --quiet 'numpy<2' \
            || log_warn "    numpy downgrade failed; continuing"

        # Note: we previously tried `pynq-get-notebooks pynq_composable` here,
        # but it always fails with 'No device found in the system' on a system
        # without a loaded DPU overlay. We don't use pynq_composable's notebooks
        # anyway — the package's Python is what matters and `import pynq_composable`
        # works once numpy is fixed.
        STAGE3_DETAIL="numpy downgraded $current_numpy → <2"
    else
        log_ok "    numpy $current_numpy (compatible with bundled cv2)"
        STAGE3_DETAIL="numpy=$current_numpy, healthy"
    fi

    # Sanity check the full venv
    if /usr/local/share/pynq-venv/bin/python -c "import numpy, cv2, pynq, pynq_dpu" 2>/dev/null; then
        log_ok "  pynq-venv healthy: numpy + cv2 + pynq + pynq_dpu all importable"
        if (( KRIA_PYNQ_FRESH_INSTALL == 1 )); then
            summary_stage_done "fresh install; $STAGE3_DETAIL"
        else
            summary_stage_skipped "$STAGE3_DETAIL"
        fi
    else
        log_warn "  pynq-venv health check failed — see /home/ubuntu/kriakv260_install.log"
        /usr/local/share/pynq-venv/bin/python -c "import numpy, cv2, pynq, pynq_dpu" 2>&1 | tail -5
        summary_stage_failed "venv health check failed"
        # Don't die: maybe user can recover. Let other stages run for diagnostics.
    fi
else
    summary_stage_failed "pynq-venv directory not found"
fi

# ─── STAGE 4: VAI 3.5 upgrade ──────────────────────────────────────────────
# Source: https://github.com/amd/Kria-RoboticsAI
#         files/scripts/install_update_kr260_to_vitisai35.sh
# Adapted to: KV260 (vs KR260 in the original); idempotent stamps; quiet mode;
# better error handling.
log_step "[4/5] VAI 3.5 runtime upgrade"
summary_stage_start "4/5" "VAI 3.5 runtime upgrade"

vai_v=$(vai_installed_version)
VAI35_STAMP="/var/local/kriakv260_vai35.done"

if [[ "$vai_v" == "3.5" ]]; then
    # VAI 3.5 runtime libs are already installed. If the stamp file is
    # missing (e.g., VAI 3.5 was installed via a different path before this
    # script existed, or a prior run completed the install but failed before
    # stamping), the stamp must be written here so subsequent runs skip
    # cleanly without re-attempting the download. Without this, the script
    # would fall into the else branch and try to "upgrade VAI 3.5 → VAI 3.5",
    # which is wasted work and can fail spuriously on SSL/network issues.
    if [[ ! -f "$VAI35_STAMP" ]]; then
        log_info "VAI 3.5 runtime detected (libvart $vai_v) but stamp missing."
        log_info "  Verified install is at target version — marking stage done."
        need_sudo touch "$VAI35_STAMP"
    fi
    log_ok "VAI 3.5 already installed and patched — skipping upgrade"
    summary_stage_skipped "VAI 3.5 + stamp present"
else
    if [[ -n "$vai_v" ]]; then
        log_info "Current VAI version: $vai_v. Upgrading to 3.5."
    else
        log_info "No VAI runtime detected. Installing 3.5."
    fi

    # ── 4a. Download vai3.5_kr260.zip from xilinx.com ─────────────────────
    mkdir -p "$STAGE_DIR"
    ZIP_PATH="$STAGE_DIR/vai3.5_kr260.zip"
    if [[ -f "$ZIP_PATH" ]] && unzip -tq "$ZIP_PATH" &>/dev/null; then
        log_ok "  $ZIP_PATH already downloaded and valid"
    else
        log_info "  Downloading from $VAI35_KR260_ZIP_URL"
        # --ca-directory=/etc/ssl/certs ensures wget uses the system CA
        # bundle (xilinx.com's Let's Encrypt cert chain isn't picked up
        # automatically by wget on some Kria images, despite the
        # ca-certificates package being correct).
        if ! wget --ca-directory=/etc/ssl/certs --show-progress \
                  -O "$ZIP_PATH" "$VAI35_KR260_ZIP_URL"; then
            die "Download failed. URL may have changed. Override with:
  VAI35_KR260_ZIP_URL=<new_url> bash scripts/kria/01_install_vai35.sh"
        fi
        if ! unzip -tq "$ZIP_PATH"; then
            die "Downloaded file is corrupt: $ZIP_PATH"
        fi
        log_ok "  Downloaded ($(stat -c%s "$ZIP_PATH" | numfmt --to=iec))"
    fi

    # Unzip into a subdirectory of stage
    EXTRACT_DIR="$STAGE_DIR/vai3.5_kr260"
    if [[ ! -d "$EXTRACT_DIR/target/runtime_deb" ]]; then
        rm -rf "$EXTRACT_DIR"
        unzip -q "$ZIP_PATH" -d "$STAGE_DIR"
        # The zip extracts to a vai3.5_kr260/ dir directly inside STAGE_DIR
        if [[ ! -d "$EXTRACT_DIR" ]]; then
            die "After unzipping, $EXTRACT_DIR not found.
  Inspect: ls -la $STAGE_DIR"
        fi
        log_debug "  Extracted to $EXTRACT_DIR"
    else
        log_ok "  Already extracted: $EXTRACT_DIR"
    fi

    # ── 4b. Run AMD's setup.sh on the runtime_deb dir ──────────────────────
    # This installs all the .deb files in the correct order. Don't try to
    # parse / reorder them; the bundled setup.sh has been tested by AMD.
    RUNTIME_DEB_DIR="$EXTRACT_DIR/target/runtime_deb"
    if [[ ! -f "$RUNTIME_DEB_DIR/setup.sh" ]]; then
        die "Expected $RUNTIME_DEB_DIR/setup.sh not found.
  The zip's layout may have changed. Inspect: ls -R $EXTRACT_DIR"
    fi

    log_info "  Running AMD's setup.sh (installs VAI 3.5 debs)"
    (
        cd "$RUNTIME_DEB_DIR"
        need_sudo bash setup.sh
    ) || die "setup.sh failed. Check $LOG_FILE for the error."
    log_ok "  VAI 3.5 debs installed via setup.sh"

    # ── 4c. Copy lack_lib helper libs ──────────────────────────────────────
    # lack_lib.tar.gz lives in vai3.5_kr260/target/ (one level up from runtime_deb)
    LACK_LIB_TGZ="$EXTRACT_DIR/target/lack_lib.tar.gz"
    if [[ -f "$LACK_LIB_TGZ" ]]; then
        log_info "  Extracting and installing lack_lib helper libs"
        (
            cd "$EXTRACT_DIR/target"
            tar -xzf lack_lib.tar.gz
            need_sudo cp -r lack_lib/* /usr/lib/
        ) || die "lack_lib install failed"
        need_sudo ldconfig
        log_ok "  helper libs installed"
    else
        log_warn "  No $LACK_LIB_TGZ found — skipping (layout may have changed)"
    fi

    # ── 4d. Copy xbutil2 to /usr/bin/unwrapped/ ────────────────────────────
    XBUTIL2="$EXTRACT_DIR/xbutil_tool/xbutil2"
    if [[ -f "$XBUTIL2" ]]; then
        log_info "  Installing xbutil2 to /usr/bin/unwrapped/"
        need_sudo mkdir -p /usr/bin/unwrapped
        need_sudo cp "$XBUTIL2" /usr/bin/unwrapped/
        need_sudo chmod +x /usr/bin/unwrapped/xbutil2
        log_ok "  xbutil2 installed"
    else
        log_warn "  $XBUTIL2 not found — skipping"
    fi

    # Verify VAI runtime is now 3.5
    vai_v=$(vai_installed_version)
    if [[ "$vai_v" == "3.5" ]]; then
        log_ok "  VAI 3.5 runtime verified"
        summary_stage_done "installed via AMD setup.sh"
    else
        die "VAI install completed but vai_installed_version returned '$vai_v', expected '3.5'.
  Inspect installed VAI debs with:
    dpkg -l | grep -iE 'vart|xir|unilog|target-factory'
  And libvart specifically:
    dpkg -s libvart | head -5"
    fi
fi


# ─── STAGE 5: DPU-PYNQ for VAI 3.5 + post-install patches ──────────────────
log_step "[5/5] DPU-PYNQ for VAI 3.5 + post-install patches"
summary_stage_start "5/5" "DPU-PYNQ for VAI 3.5"

# Per AMD's script:
#   git clone -b design_contest_3.5 DPU-PYNQ
#   source pynq_venv.sh
#   pip install . --no-build-isolation
#   purge old pynq-dpu notebooks, get new ones
#   patch pynq_venv.sh to export LD_LIBRARY_PATH=/usr/lib
#   strip /usr/bin/ prefix from xdputil script
DPU_PYNQ_DIR="$STAGE_DIR/DPU-PYNQ"

if [[ -f "$VAI35_STAMP" ]]; then
    log_ok "post-install patches already applied (stamp: $VAI35_STAMP)"
    summary_stage_skipped "stamp present"
else
    # 5a. Clone DPU-PYNQ design_contest_3.5
    if [[ -d "$DPU_PYNQ_DIR/.git" ]]; then
        log_ok "  DPU-PYNQ already cloned at $DPU_PYNQ_DIR"
    else
        log_info "  Cloning DPU-PYNQ ($DPU_PYNQ_BRANCH branch)..."
        git clone --depth 1 -b "$DPU_PYNQ_BRANCH" "$DPU_PYNQ_REPO" "$DPU_PYNQ_DIR" \
            || die "DPU-PYNQ clone failed. Check network access to github.com."
        log_ok "  DPU-PYNQ cloned"
    fi

    # 5b. pip install DPU-PYNQ into pynq-venv.
    #
    # Two problems we work around here:
    #
    # (1) pynqutils download_overlays.py has a real bug: when no FPGA device
    #     is currently loaded (true during a fresh install), it does
    #         if len(devices) == 0 and type(Device.devices[0]) == EmbeddedDevice:
    #     where both `devices` and `Device.devices` are empty lists, so the
    #     `[0]` raises IndexError. We patch the file to fall back to the
    #     "embedded board" assumption when Device.devices is empty.
    #
    # (2) Running pip via `sudo /path/to/python` strips the environment,
    #     losing what `source pynq_venv.sh` did. AMD's reference script
    #     avoids this by `sudo su`-ing first and running everything as root.
    #     We replicate that by using `sudo bash -c "..."` so the source +
    #     pip-install all happen in a single root shell.

    # 5b-i. Patch pynqutils download_overlays.py (idempotent)
    PYNQUTILS_DL="/usr/local/share/pynq-venv/lib/python3.10/site-packages/pynqutils/setup_utils/download_overlays.py"
    if [[ -f "$PYNQUTILS_DL" ]]; then
        if grep -q "len(devices) == 0 and type(Device.devices\[0\])" "$PYNQUTILS_DL" \
                && ! grep -q "not Device.devices or type(Device.devices\[0\])" "$PYNQUTILS_DL"; then
            log_info "  Patching pynqutils download_overlays.py (IndexError on empty device list)"
            need_sudo sed -i \
                's|len(devices) == 0 and type(Device.devices\[0\]) == EmbeddedDevice|len(devices) == 0 and (not Device.devices or type(Device.devices[0]) == EmbeddedDevice)|' \
                "$PYNQUTILS_DL"
            log_ok "  pynqutils patched"
        else
            log_debug "  pynqutils download_overlays.py already patched (or different version)"
        fi
    else
        log_warn "  $PYNQUTILS_DL not found — patch skipped"
    fi

    # 5b-ii. Install via a single sudo bash so env survives
    log_info "  Installing DPU-PYNQ into pynq-venv (no-build-isolation)"
    if ! need_sudo bash -c "
        set -e
        [[ -f /etc/profile.d/pynq_venv.sh ]] && source /etc/profile.d/pynq_venv.sh
        [[ -f /opt/xilinx/xrt/setup.sh ]] && source /opt/xilinx/xrt/setup.sh
        export BOARD=KV260
        cd '$DPU_PYNQ_DIR'
        /usr/local/share/pynq-venv/bin/python3 -m pip install . --no-build-isolation
    "; then
        die "DPU-PYNQ pip install failed. Check $LOG_FILE for the error.
  Common causes:
    - pynqutils version differs from what we patched (inspect $PYNQUTILS_DL)
    - Missing build-essential (verify: which gcc make)
    - Network drop mid-build (re-run; pip will resume)
    - XRT/zocl kernel module not loaded (verify: lsmod | grep zocl)"
    fi
    log_ok "  DPU-PYNQ installed in pynq-venv"

    # 5c. Refresh pynq-dpu notebooks (same sudo bash -c pattern as 5b)
    log_info "  Refreshing pynq-dpu notebooks at /home/root/jupyter_notebooks/pynq-dpu"
    if [[ -d /home/root/jupyter_notebooks ]]; then
        need_sudo rm -rf /home/root/jupyter_notebooks/pynq-dpu
        need_sudo bash -c "
            set +u
            [[ -f /etc/profile.d/pynq_venv.sh ]] && source /etc/profile.d/pynq_venv.sh
            cd /home/root/jupyter_notebooks
            /usr/local/share/pynq-venv/bin/pynq get-notebooks pynq-dpu -p . --force
        " || log_warn "  pynq get-notebooks failed (non-fatal: 'No device found' is OK here)"
        log_ok "  notebooks refreshed"
    else
        log_warn "  /home/root/jupyter_notebooks doesn't exist — skipping notebook refresh"
    fi

    # 5d. Patch /etc/profile.d/pynq_venv.sh to export LD_LIBRARY_PATH=/usr/lib
    # Required so VART can find libunilog.so etc.
    PYNQ_VENV_SH="/etc/profile.d/pynq_venv.sh"
    if [[ -f "$PYNQ_VENV_SH" ]]; then
        if grep -q "export LD_LIBRARY_PATH=/usr/lib" "$PYNQ_VENV_SH"; then
            log_ok "  pynq_venv.sh already exports LD_LIBRARY_PATH=/usr/lib"
        else
            log_info "  Appending LD_LIBRARY_PATH=/usr/lib to $PYNQ_VENV_SH"
            echo "export LD_LIBRARY_PATH=/usr/lib" | need_sudo tee -a "$PYNQ_VENV_SH" >/dev/null
            log_ok "  LD_LIBRARY_PATH patch applied"
        fi
    else
        log_warn "  $PYNQ_VENV_SH not found — LD_LIBRARY_PATH patch skipped"
    fi

    # 5e. Patch /usr/bin/xdputil to remove the '/usr/bin/' prefix on the python3 invocation
    # Otherwise xdputil ignores pynq-venv's python3.
    if [[ -f /usr/bin/xdputil ]]; then
        if grep -q "^#!/usr/bin/python3" /usr/bin/xdputil; then
            log_info "  Patching /usr/bin/xdputil to use pynq-venv's python3"
            need_sudo sed -i "s/\/usr\/bin\///g" /usr/bin/xdputil
            log_ok "  xdputil patched"
        else
            log_ok "  /usr/bin/xdputil already patched (or different layout)"
        fi
    else
        log_warn "  /usr/bin/xdputil not found — patch skipped"
    fi

    # 5f. Drop the stamp file so this whole stage skips on re-runs
    need_sudo touch "$VAI35_STAMP"
    log_ok "  marked complete: $VAI35_STAMP"
    summary_stage_done "DPU-PYNQ installed; patches applied"
fi

# Set action hint in case any earlier stage failed
summary_set_action "If a stage above failed, inspect:
    grep -B2 -A20 'FAIL' $LOG_FILE | tail -50
  Then re-run this script — completed stages will be skipped automatically."

# ─── Done ───────────────────────────────────────────────────────────────────
echo
log_ok "Install complete."
log_warn "  ⚠ AMD recommends rebooting the board after this install."
log_warn "    Run: sudo reboot"
log_warn "    After reboot, validate by running a small pynq_dpu example."
echo
log_ok "Next steps:"
log_info "  1. (After reboot) Apply camera + system tuning:"
log_info "       bash scripts/kria/02_apply_tuning.sh"
log_info "  2. Persist tuning across reboots:"
log_info "       bash scripts/kria/03_install_systemd.sh"
log_info "  3. Run the live demo:"
log_info "       bash scripts/kria/run_live.sh yolov5n"
echo
log_info "Cleanup (optional): rm -rf $STAGE_DIR"
log_info "Logs: $LOG_FILE"

exit 0
