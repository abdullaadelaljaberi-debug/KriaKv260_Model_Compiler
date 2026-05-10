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
# Default URL for vai3.5_kr260.zip. The package is hosted on AMD's GitHub
# Vitis-AI v3.5 release. If AMD moves it, override with:
#   VAI35_KR260_ZIP_URL=... bash 01_install_vai35.sh
VAI35_KR260_ZIP_URL="${VAI35_KR260_ZIP_URL:-https://github.com/Xilinx/Vitis-AI/releases/download/v3.5/vai3.5_kr260.zip}"

# Kria-PYNQ git repo branch. Pinning to v3.0.1 — current stable as of pipeline writing.
KRIA_PYNQ_REPO="${KRIA_PYNQ_REPO:-https://github.com/Xilinx/Kria-PYNQ.git}"
KRIA_PYNQ_BRANCH="${KRIA_PYNQ_BRANCH:-v3.0.1}"

# DPU-PYNQ design contest branch — has the VAI 3.5 DPU bitstream
DPU_PYNQ_REPO="${DPU_PYNQ_REPO:-https://github.com/Xilinx/DPU-PYNQ.git}"
DPU_PYNQ_BRANCH="${DPU_PYNQ_BRANCH:-design_contest_3.5}"

# Where to stage downloads. Cleaned up after successful install.
STAGE_DIR="${STAGE_DIR:-$HOME/.cache/kriakv260_install}"

# Log file — appended, not overwritten, so you can re-run and see history.
LOG_FILE="${LOG_FILE:-$HOME/kriakv260_install.log}"

log_to_file "$LOG_FILE"
log_info "Logging to $LOG_FILE"

# ─── prereq sanity: 00_check_prereqs.sh must have passed ────────────────────
if ! bash "$SCRIPT_DIR/00_check_prereqs.sh" --quiet; then
    die "00_check_prereqs.sh failed. Run it directly to see what's wrong:
  bash scripts/kria/00_check_prereqs.sh"
fi

log_step "Kria install — script will skip already-done steps"

# ─── STAGE 1: First-boot config (A3 partial automation) ────────────────────
log_step "[1/5] First-boot config"

current_hostname=$(hostname)
if [[ "$current_hostname" == "kria" ]] || [[ "$current_hostname" == "ubuntu" ]]; then
    log_info "Current hostname '$current_hostname' looks default. Set a custom one?"
    if confirm "  Set hostname to 'kria-lpr'? (or N to skip)"; then
        need_sudo hostnamectl set-hostname kria-lpr
        log_ok "hostname set to kria-lpr (effective after reboot)"
    else
        log_info "  keeping current hostname '$current_hostname'"
    fi
else
    log_ok "hostname '$current_hostname' (custom, leaving alone)"
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
else
    log_info "Running xlnx-config.sysinit — installs base components"
    log_warn "  This step is interactive on a fresh system. Accept the defaults."
    log_warn "  It can take 5-10 min on first run."
    if confirm "  Run xlnx-config.sysinit now?"; then
        need_sudo xlnx-config.sysinit || die "xlnx-config.sysinit failed"
        need_sudo touch "$SYSINIT_STAMP"
        log_ok "xlnx-config.sysinit complete"
    else
        log_warn "  Skipped. Re-run this script to do it later."
    fi
fi

# ─── STAGE 3: Kria-PYNQ ─────────────────────────────────────────────────────
log_step "[3/5] Kria-PYNQ stack"

if [[ "$(kria_pynq_installed)" == "yes" ]]; then
    log_ok "Kria-PYNQ already installed at /usr/local/share/pynq-venv"
else
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

# ─── STAGE 4: VAI 3.5 upgrade ──────────────────────────────────────────────
log_step "[4/5] VAI 3.5 runtime upgrade"

vai_v=$(vai_installed_version)
if [[ "$vai_v" == "3.5" ]]; then
    log_ok "VAI 3.5 already installed — skipping upgrade"
else
    if [[ -n "$vai_v" ]]; then
        log_info "Current VAI version: $vai_v. Upgrading to 3.5."
    else
        log_info "No VAI runtime detected. Installing 3.5."
    fi

    # ── 4a. Download vai3.5_kr260.zip ──────────────────────────────────────
    mkdir -p "$STAGE_DIR"
    ZIP_PATH="$STAGE_DIR/vai3.5_kr260.zip"
    if [[ -f "$ZIP_PATH" ]] && unzip -tq "$ZIP_PATH" &>/dev/null; then
        log_ok "  $ZIP_PATH already downloaded and valid"
    else
        log_info "  Downloading $VAI35_KR260_ZIP_URL"
        if ! wget --show-progress -O "$ZIP_PATH" "$VAI35_KR260_ZIP_URL"; then
            die "Download failed. Verify URL:
  $VAI35_KR260_ZIP_URL
  Set VAI35_KR260_ZIP_URL env var to override."
        fi
        if ! unzip -tq "$ZIP_PATH"; then
            die "Downloaded file is corrupt: $ZIP_PATH"
        fi
        log_ok "  Downloaded ($(stat -c%s "$ZIP_PATH" | numfmt --to=iec))"
    fi

    # Unzip into a subdirectory of stage so we can inspect contents
    EXTRACT_DIR="$STAGE_DIR/vai3.5_kr260"
    rm -rf "$EXTRACT_DIR"
    mkdir -p "$EXTRACT_DIR"
    unzip -q "$ZIP_PATH" -d "$EXTRACT_DIR"
    log_debug "  Extracted to $EXTRACT_DIR"

    # ── 4b. Install debs in dependency order ───────────────────────────────
    # Order matters because each package depends on the previous one(s).
    # Standard VAI dependency chain:
    #   unilog → xir → target_factory → vart-runtime → vitis-ai-library
    log_info "  Installing VAI 3.5 deb packages in dependency order"

    # Find the debs (the zip's internal layout may vary; we search recursively)
    declare -a DEBS_FOUND=()
    while IFS= read -r -d '' deb; do
        DEBS_FOUND+=("$deb")
    done < <(find "$EXTRACT_DIR" -name "*.deb" -print0)

    if (( ${#DEBS_FOUND[@]} == 0 )); then
        die "No .deb files found in $EXTRACT_DIR.
  Layout of the zip may have changed. Inspect:
    ls -R $EXTRACT_DIR"
    fi

    # Install in the canonical dependency order. We grep for package name
    # substrings; this is more robust to filename variations than hardcoding.
    install_deb_by_pattern() {
        local pattern="$1"
        local deb=""
        for d in "${DEBS_FOUND[@]}"; do
            if [[ "$(basename "$d")" =~ $pattern ]]; then
                deb="$d"
                break
            fi
        done
        if [[ -z "$deb" ]]; then
            log_warn "  no deb matching '$pattern' — skipping"
            return 0
        fi
        log_info "  dpkg -i $(basename "$deb")"
        need_sudo dpkg -i "$deb" || {
            # Try to resolve any missing deps with apt
            log_warn "  dpkg failed; attempting 'apt --fix-broken install'"
            need_sudo apt-get install -f -y || \
                die "Could not install $(basename "$deb"). Check $LOG_FILE."
        }
    }

    install_deb_by_pattern "^unilog"
    install_deb_by_pattern "^xir"
    install_deb_by_pattern "^target.factory|^target-factory"
    install_deb_by_pattern "^libvart|^vart"
    install_deb_by_pattern "^libvitis.ai.library|^vitis.ai.library"

    log_ok "  VAI 3.5 debs installed"

    # ── 4c. Copy lack_lib helper libs (workaround) ─────────────────────────
    # On some Kria-PYNQ + VAI 3.5 combos, certain symbols aren't in the system
    # /usr/lib so we drop in additional .so files shipped in the zip's
    # lack_lib/ subdirectory.
    LACK_LIB_DIR=$(find "$EXTRACT_DIR" -maxdepth 3 -type d -name "lack_lib" | head -1)
    if [[ -n "$LACK_LIB_DIR" ]] && [[ -d "$LACK_LIB_DIR" ]]; then
        log_info "  Copying helper libs from lack_lib/ → /usr/lib/"
        need_sudo cp -v "$LACK_LIB_DIR"/* /usr/lib/ 2>&1 | tail -5
        need_sudo ldconfig
        log_ok "  helper libs installed"
    else
        log_warn "  No lack_lib/ directory in the zip — may not be needed"
        log_warn "  for this VAI 3.5 release. If glog/symbol errors occur"
        log_warn "  at runtime, check the zip's actual contents."
    fi

    # ── 4d. glog 0.5.0 workaround ─────────────────────────────────────────
    # Older Ubuntu may ship glog 0.4.x; VAI 3.5 needs 0.5+. The pynq-venv
    # may have an older bundled glog that we need to override.
    log_info "  glog version check"
    glog_v=$(dpkg -s libgoogle-glog0v5 2>/dev/null | awk -F: '/^Version/ {print $2}' | xargs)
    if [[ -n "$glog_v" ]]; then
        log_ok "    libgoogle-glog0v5 $glog_v installed"
    else
        log_warn "    libgoogle-glog0v5 not found. Installing from apt..."
        need_sudo apt-get update
        need_sudo apt-get install -y libgoogle-glog0v5 \
            || log_warn "    apt install libgoogle-glog0v5 failed — may need a PPA"
    fi

    # Verify installed version is 3.5
    vai_v=$(vai_installed_version)
    if [[ "$vai_v" == "3.5" ]]; then
        log_ok "  VAI 3.5 successfully installed"
    else
        die "VAI install completed but version is '$vai_v', expected '3.5'.
  Check $LOG_FILE for clues, or inspect:
    dpkg -s libvart-runtime"
    fi
fi

# ─── STAGE 5: DPU bitstream ────────────────────────────────────────────────
log_step "[5/5] DPU bitstream (VAI 3.5 / design_contest_3.5 branch)"

# DPU-PYNQ has a separate repo with the bitstream files (.bit, .hwh, .xclbin).
# We clone the design_contest_3.5 branch which targets VAI 3.5.
BITSTREAM_DIR="/home/$USER/dpu_pynq_vai35"
if [[ -d "$BITSTREAM_DIR/boards/kv260" ]]; then
    log_ok "DPU-PYNQ already cloned at $BITSTREAM_DIR"
else
    log_info "Cloning DPU-PYNQ ($DPU_PYNQ_BRANCH branch) to $BITSTREAM_DIR..."
    git clone --depth 1 -b "$DPU_PYNQ_BRANCH" "$DPU_PYNQ_REPO" "$BITSTREAM_DIR" \
        || die "DPU-PYNQ clone failed. Check network access to github.com."
fi

# Verify the KV260 bitstream files are present
KV260_BS_DIR="$BITSTREAM_DIR/boards/kv260"
if [[ -d "$KV260_BS_DIR" ]]; then
    bs_files=$(find "$KV260_BS_DIR" -maxdepth 2 -name "*.bit" -o -name "*.xclbin" -o -name "*.hwh" 2>/dev/null | wc -l)
    if (( bs_files >= 3 )); then
        log_ok "DPU bitstream files present ($bs_files files in $KV260_BS_DIR)"
    else
        log_warn "Only $bs_files bitstream files found. Expected ≥3 (.bit, .hwh, .xclbin)"
        log_warn "Check: ls $KV260_BS_DIR"
    fi
else
    log_warn "No $KV260_BS_DIR — branch layout may have changed."
    log_warn "Check the DPU-PYNQ repo manually."
fi

# ─── Done ───────────────────────────────────────────────────────────────────
echo
log_ok "Install complete. Next steps:"
log_info "  1. Apply camera + system tuning:"
log_info "       bash scripts/kria/02_apply_tuning.sh"
log_info "  2. Persist tuning across reboots:"
log_info "       bash scripts/kria/03_install_systemd.sh"
log_info "  3. Run the live demo:"
log_info "       bash scripts/kria/run_live.sh yolov5n"
echo
log_info "Cleanup (optional): rm -rf $STAGE_DIR"
log_info "Logs: $LOG_FILE"
