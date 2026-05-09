#!/usr/bin/env bash
# scripts/host/00_check_prereqs.sh
# ─────────────────────────────────────────────────────────────────────────────
# Verifies the host PC has everything needed to run the compile pipeline.
# Reports each check pass/fail and prints a copy-paste fix command when
# something's missing.
#
# Exit codes:
#   0  all checks passed
#   1  at least one check failed (compile pipeline won't work yet)
# ─────────────────────────────────────────────────────────────────────────────
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

# Min requirements (matches docs/HOST_SETUP.md)
MIN_DISK_GB=50            # for VAI image + intermediate quantization data
MIN_DOCKER_VERSION=20.10
VAI_IMAGE="xilinx/vitis-ai-pytorch-gpu:3.5.0.001"

FAIL=0
fail() { FAIL=$((FAIL + 1)); }

log_step "Host PC prerequisite check"

# ─── 1. Operating system ────────────────────────────────────────────────────
log_info "Operating system"
if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    if [[ "$ID" == "ubuntu" ]] && [[ "${VERSION_ID%%.*}" -ge 22 ]]; then
        log_ok "Ubuntu ${VERSION_ID} (${VERSION_CODENAME})"
    else
        log_warn "Detected: $PRETTY_NAME"
        log_warn "  Pipeline tested on Ubuntu 22.04 / 24.04 LTS. Other distros may work but are unsupported."
    fi
else
    log_warn "Cannot determine distro (no /etc/os-release)"
fi

# ─── 2. Architecture ────────────────────────────────────────────────────────
log_info "CPU architecture"
arch="$(uname -m)"
if [[ "$arch" == "x86_64" ]]; then
    log_ok "x86_64"
else
    log_err "Architecture is '$arch' — pipeline requires x86_64 (the Vitis-AI Docker image is x86_64-only)"
    fail
fi

# ─── 3. Disk space ──────────────────────────────────────────────────────────
log_info "Disk space"
if disk_free_gb "$REPO_ROOT" "$MIN_DISK_GB"; then
    avail_gb=$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9')
    log_ok "${avail_gb} GB free at $REPO_ROOT (need ≥${MIN_DISK_GB} GB)"
else
    avail_gb=$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9' || echo "?")
    log_err "Only ${avail_gb} GB free at $REPO_ROOT (need ≥${MIN_DISK_GB} GB for the VAI image + intermediates)"
    log_err "  Free up space, or run from a partition with more headroom"
    fail
fi

# ─── 4. Docker installed ────────────────────────────────────────────────────
log_info "Docker"
if have_cmd docker; then
    docker_v=$(docker --version 2>/dev/null | awk '{print $3}' | tr -d ',')
    log_ok "docker $docker_v"
    if ! docker_ok; then
        log_err "  Docker is installed but the daemon is not reachable"
        log_err "  Most common cause: your user is not in the 'docker' group"
        log_err "  Fix:  sudo usermod -aG docker \$USER  &&  newgrp docker"
        log_err "        (or log out and back in)"
        fail
    fi
else
    log_err "docker not installed"
    log_err "  Fix:  follow the official Docker Engine install for Ubuntu:"
    log_err "        https://docs.docker.com/engine/install/ubuntu/"
    log_err "  Or quick install:"
    log_err "        curl -fsSL https://get.docker.com | sudo sh"
    log_err "        sudo usermod -aG docker \$USER && newgrp docker"
    fail
fi

# ─── 5. NVIDIA driver ───────────────────────────────────────────────────────
log_info "NVIDIA driver"
if nvidia_driver_ok; then
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    log_ok "driver $drv on $gpu"
    # Vitis-AI 3.5 ships CUDA 11.8 in the image; needs driver ≥520
    drv_major=$(echo "$drv" | cut -d. -f1)
    if (( drv_major < 520 )); then
        log_warn "Driver < 520; Vitis-AI 3.5 PyTorch GPU image needs CUDA 11.8 → driver ≥520"
        log_warn "  Update via:  sudo apt install nvidia-driver-535"
    fi
else
    log_err "nvidia-smi not present or driver not loaded"
    log_err "  Fix:  sudo apt install nvidia-driver-535  (then reboot)"
    log_err "  Or use Ubuntu's 'Software & Updates' → 'Additional Drivers' tab"
    fail
fi

# ─── 6. NVIDIA Container Toolkit ────────────────────────────────────────────
log_info "NVIDIA Container Toolkit"
if nvidia_docker_ok; then
    log_ok "GPU is accessible from docker containers"
else
    if have_cmd docker && nvidia_driver_ok; then
        log_err "Docker can't see the GPU (Container Toolkit not installed/configured)"
        log_err "  Fix:"
        log_err "    distribution=\$(. /etc/os-release; echo \$ID\$VERSION_ID)"
        log_err "    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \\"
        log_err "      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        log_err "    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\"
        log_err "      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\"
        log_err "      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        log_err "    sudo apt update && sudo apt install -y nvidia-container-toolkit"
        log_err "    sudo nvidia-ctk runtime configure --runtime=docker"
        log_err "    sudo systemctl restart docker"
        fail
    else
        log_warn "Skipped (docker or NVIDIA driver missing — fix those first)"
    fi
fi

# ─── 7. Internet (for pulling VAI image) ────────────────────────────────────
log_info "Internet connectivity"
if internet_ok; then
    log_ok "outbound HTTPS works"
else
    log_err "Cannot reach the public internet"
    log_err "  The VAI 3.5 docker image is ~10 GB and must be pulled from Docker Hub"
    log_err "  Check your network / proxy configuration before continuing"
    fail
fi

# ─── 8. Vitis-AI image already pulled? (informational, not a failure) ──────
log_info "Vitis-AI 3.5 image"
if docker_ok && docker image inspect "$VAI_IMAGE" &>/dev/null; then
    sz=$(docker image inspect --format='{{.Size}}' "$VAI_IMAGE" | numfmt --to=iec)
    log_ok "$VAI_IMAGE  ($sz)"
else
    log_warn "$VAI_IMAGE not yet pulled"
    log_warn "  Run  bash scripts/host/01_install_vai.sh  to pull it (~10 GB download)"
fi

# ─── 9. Python ───────────────────────────────────────────────────────────────
log_info "Python (host-side helpers)"
if have_cmd python3; then
    pyv=$(python3 --version | awk '{print $2}')
    log_ok "python3 $pyv"
else
    log_warn "python3 not on PATH (only needed for some host-side helpers)"
fi

# ─── 10. Repo layout sanity check ───────────────────────────────────────────
log_info "Repo layout"
expected_dirs=(scripts/host scripts/kria lpr_pipeline data/weights data/calib data/eval)
for d in "${expected_dirs[@]}"; do
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
    log_ok "All checks passed. Next:  bash scripts/host/01_install_vai.sh"
    exit 0
else
    log_err "$FAIL check(s) failed. Address the items above before continuing."
    exit 1
fi
