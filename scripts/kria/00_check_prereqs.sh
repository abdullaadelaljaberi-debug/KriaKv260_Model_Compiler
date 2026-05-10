#!/usr/bin/env bash
# scripts/kria/00_check_prereqs.sh
# ─────────────────────────────────────────────────────────────────────────────
# Verifies the Kria board is in a sensible state before the install scripts
# touch anything. Fails fast with copy-paste fix commands.
#
# Refuses to proceed on:
#   - Non-Kria hardware (different SOM)
#   - Wrong SOM model (KR260, KD240 — pipeline targets KV260 only)
#   - Wrong Ubuntu version (must be 22.04; 20.04 needs reflash)
#   - Wrong architecture (must be aarch64)
#
# Reports but doesn't fail on:
#   - Already-installed VAI 3.5 (will be skipped by 01_install_vai35.sh)
#   - Kria-PYNQ present
#   - Internet present
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

# Pop --verbose / --quiet flags
parse_common_flags "$@"

FAIL=0
fail() { FAIL=$((FAIL + 1)); }

log_step "Kria board prerequisite check"

# ─── 1. Hardware: is this a Kria? ───────────────────────────────────────────
log_info "Hardware platform"
if is_kria; then
    log_ok "Kria SOM detected"
else
    log_err "  Not a Kria SOM (no xlnx,zynqmp in device-tree/compatible)"
    log_err "  This pipeline targets the AMD Kria KV260. If you're running"
    log_err "  on x86/other hardware, you cannot use these Kria-side scripts."
    fail
fi

# ─── 2. SOM model: must be KV260 ────────────────────────────────────────────
log_info "SOM model"
model=$(kria_model)
case "$model" in
    kv260)
        log_ok "KV260 Vision AI Starter Kit"
        ;;
    kr260)
        log_err "  KR260 detected — this pipeline targets KV260 only"
        log_err "  The KR260 has a different DPU configuration. The compiled"
        log_err "  xmodels' fingerprint (0x101000056010407) won't match."
        log_err "  See docs/MODELS.md → DPU compatibility note."
        fail
        ;;
    kd240)
        log_err "  KD240 detected — this pipeline targets KV260 only"
        log_err "  Same fingerprint issue as KR260 above."
        fail
        ;;
    *)
        log_warn "  unknown SOM ($model) — pipeline tested only on KV260"
        log_warn "  Proceed at your own risk. The xmodel fingerprint check at"
        log_warn "  deploy time will catch true incompatibilities."
        ;;
esac

# ─── 3. Operating system: must be Ubuntu 22.04 ──────────────────────────────
log_info "Operating system"
uv=$(ubuntu_version)
if [[ -z "$uv" ]]; then
    log_err "  Not running Ubuntu (cannot determine version from /etc/os-release)"
    log_err "  This pipeline targets Canonical's Ubuntu 22.04 LTS for Kria SOMs."
    fail
elif [[ "$uv" == "22.04" ]]; then
    log_ok "Ubuntu 22.04 LTS"
elif [[ "$uv" == "20.04" ]]; then
    log_err "  Ubuntu 20.04 detected — pipeline requires 22.04 LTS for Kria SOMs"
    log_err ""
    log_err "  REASON: the VAI 3.5 deb packages in this pipeline are built for"
    log_err "  Ubuntu 22.04. Installing them on 20.04 will fail or produce a"
    log_err "  broken setup. The 20.04 → 22.04 in-place upgrade is high-risk"
    log_err "  on Xilinx kernels and not supported by this script."
    log_err ""
    log_err "  FIX: re-flash the SD card with Ubuntu 22.04 LTS for Kria SOMs"
    log_err "  following the official Canonical tutorial:"
    log_err "    https://canonical-kria.readthedocs-hosted.com/en/latest/"
    log_err "  Then run this script again."
    fail
else
    log_warn "  Ubuntu $uv detected (tested on 22.04 only). May or may not work."
    log_warn "  If installation fails, please reflash with 22.04 LTS."
fi

# ─── 4. Architecture: must be aarch64 ───────────────────────────────────────
log_info "CPU architecture"
arch="$(uname -m)"
if [[ "$arch" == "aarch64" ]]; then
    log_ok "aarch64 (ARM 64-bit)"
else
    log_err "  Architecture is '$arch' but Kria SOMs are aarch64"
    log_err "  Something is very wrong — are you sure you're on the Kria?"
    fail
fi

# ─── 5. Sudo access ─────────────────────────────────────────────────────────
log_info "sudo access"
if is_root; then
    log_warn "  Running as root. The install scripts use sudo where needed,"
    log_warn "  so running as plain user is recommended. Continuing anyway."
elif have_cmd sudo && sudo -n true 2>/dev/null; then
    log_ok "passwordless sudo"
elif have_cmd sudo; then
    log_ok "sudo present (will prompt for password during install)"
else
    log_err "  sudo not installed and not root"
    fail
fi

# ─── 6. Disk space ──────────────────────────────────────────────────────────
log_info "Disk space"
if disk_free_gb / 5; then
    avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
    log_ok "${avail_gb} GB free at /  (need ≥5 GB for the VAI install)"
else
    avail_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9' || echo "?")
    log_err "  only ${avail_gb} GB free at / (need ≥5 GB)"
    log_err "  Free up space:  sudo apt clean && sudo apt autoremove"
    fail
fi

# ─── 7. Internet (informational — pip needs it during VAI install) ─────────
log_info "Internet connectivity"
if curl --silent --head --max-time 5 https://github.com &>/dev/null; then
    log_ok "outbound HTTPS works"
else
    log_warn "  cannot reach github.com — VAI install needs to download packages"
    log_warn "  Check network. If you're on a managed network, you may need a proxy."
fi

# ─── 8. xlnx-config (informational) ─────────────────────────────────────────
log_info "xlnx-config tooling"
if have_cmd xlnx-config; then
    log_ok "xlnx-config present"
else
    log_warn "  xlnx-config not on PATH — required to load DPU bitstream"
    log_warn "  Install via:  sudo snap install xlnx-config --classic"
    log_warn "  (the next install script will offer to do this automatically)"
fi

# ─── 9. Kria-PYNQ (informational) ───────────────────────────────────────────
log_info "Kria-PYNQ"
if [[ "$(kria_pynq_installed)" == "yes" ]]; then
    log_ok "Kria-PYNQ already installed at /usr/local/share/pynq-venv"
else
    log_warn "  Kria-PYNQ not yet installed"
    log_warn "  Will be installed by:  bash scripts/kria/01_install_vai35.sh"
fi

# ─── 10. VAI version (informational) ────────────────────────────────────────
log_info "Vitis-AI runtime"
vai_v=$(vai_installed_version)
if [[ -z "$vai_v" ]]; then
    log_warn "  No VAI runtime detected (libvart-runtime not installed)"
elif [[ "$vai_v" == "3.5" ]]; then
    log_ok "VAI 3.5 already installed — 01_install_vai35.sh will skip the upgrade"
elif [[ "$vai_v" == "2.5" ]]; then
    log_warn "  VAI 2.5 installed — 01_install_vai35.sh will upgrade to 3.5"
else
    log_warn "  Unrecognized VAI version: $vai_v"
fi

# ─── 11. Repo layout sanity ─────────────────────────────────────────────────
log_info "Repo layout"
for d in scripts/kria scripts/host lpr_pipeline data/weights; do
    if [[ -d "$REPO_ROOT/$d" ]]; then
        log_ok "  $d/"
    else
        log_err "  $d/ missing"
        fail
    fi
done

# ─── summary ────────────────────────────────────────────────────────────────
echo
if (( FAIL == 0 )); then
    log_ok "All required checks passed."
    log_info "Next:  bash scripts/kria/01_install_vai35.sh"
    exit 0
else
    log_err "$FAIL check(s) failed. Address the items above before continuing."
    exit 1
fi
