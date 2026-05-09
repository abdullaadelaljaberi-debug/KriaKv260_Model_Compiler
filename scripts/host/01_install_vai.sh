#!/usr/bin/env bash
# scripts/host/01_install_vai.sh
# ─────────────────────────────────────────────────────────────────────────────
# Set up the Vitis-AI 3.5 docker image. Three paths:
#
#   1. If a local image is already present (CPU or GPU), report and exit OK.
#   2. If $VAI_IMAGE env var is set, use exactly that.
#   3. If neither, print build-from-source instructions and exit non-zero.
#
# AMD does NOT publicly distribute the VAI 3.5 GPU image on Docker Hub.
# Either pull the CPU image (latest tag), or build from source.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

log_step "Vitis-AI 3.5 image setup"

# ─── prereq: docker reachable ───────────────────────────────────────────────
if ! docker_ok; then
    die "docker daemon not reachable. Run scripts/host/00_check_prereqs.sh."
fi

# ─── auto-detect existing image ─────────────────────────────────────────────
# Prefer GPU > CPU; prefer pinned tag > latest. We don't enforce a specific
# version because (a) AMD's tag scheme has changed, (b) "latest" is fine for
# our purposes, (c) the inner conda env path is what actually matters.
detect_local_image() {
    # GPU first
    docker images --format '{{.Repository}}:{{.Tag}}' \
        | grep -E '^xilinx/vitis-ai-pytorch-gpu:' \
        | head -1
}

detect_local_cpu_image() {
    docker images --format '{{.Repository}}:{{.Tag}}' \
        | grep -E '^xilinx/vitis-ai-pytorch-cpu:' \
        | head -1
}

# Honor explicit override
if [[ -n "${VAI_IMAGE:-}" ]]; then
    if docker image inspect "$VAI_IMAGE" &>/dev/null; then
        sz=$(docker image inspect --format='{{.Size}}' "$VAI_IMAGE" | numfmt --to=iec)
        log_ok "Using \$VAI_IMAGE override: $VAI_IMAGE ($sz)"
        exit 0
    else
        die "\$VAI_IMAGE='$VAI_IMAGE' is not a local image. Pull or build it first."
    fi
fi

# Auto-detect
gpu_img=$(detect_local_image)
cpu_img=$(detect_local_cpu_image)

if [[ -n "$gpu_img" ]]; then
    sz=$(docker image inspect --format='{{.Size}}' "$gpu_img" | numfmt --to=iec)
    log_ok "Found local GPU image: $gpu_img ($sz)"
    log_info "The compile script will use this image with --gpus all"
    log_info "Override with:  export VAI_IMAGE='$gpu_img'  (already auto-detected)"
    exit 0
fi

if [[ -n "$cpu_img" ]]; then
    sz=$(docker image inspect --format='{{.Size}}' "$cpu_img" | numfmt --to=iec)
    log_ok "Found local CPU image: $cpu_img ($sz)"
    log_warn "  CPU image — quantization will run ~5× slower than GPU"
    log_warn "  yolov5n calib (200 imgs): ~10-15 min on CPU vs ~2-5 min on GPU"
    log_info "  For most thesis work this is fine. To speed up later, see:"
    log_info "    bottom of this script (build GPU from source)"
    exit 0
fi

# ─── neither found: instructions to obtain ─────────────────────────────────
log_warn "No Vitis-AI PyTorch image present locally."
echo
log_info "Vitis-AI 3.5 GPU images are NOT distributed on Docker Hub."
log_info "You have two options:"
echo
log_info "  ───── Option A: pull the CPU image (~12 GB, ready in 20-40 min) ─────"
log_info "  docker pull xilinx/vitis-ai-pytorch-cpu:latest"
log_info "  # Slower quantization (~10-15 min/model on CPU vs ~2-5 min on GPU)"
echo
log_info "  ───── Option B: build the GPU image from source (~25 GB, 30-60 min build) ─────"
log_info "  cd /tmp"
log_info "  git clone https://github.com/Xilinx/Vitis-AI.git -b v3.5"
log_info "  cd Vitis-AI/docker"
log_info "  ./docker_build.sh -t gpu -f pytorch"
log_info "  # Builds image as xilinx/vitis-ai-pytorch-gpu:<commit-hash>"
echo
log_info "After either option:"
log_info "  bash scripts/host/01_install_vai.sh    # re-run, should now detect the image"
exit 1
