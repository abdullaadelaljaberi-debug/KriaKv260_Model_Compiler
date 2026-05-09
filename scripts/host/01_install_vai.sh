#!/usr/bin/env bash
# scripts/host/01_install_vai.sh
# ─────────────────────────────────────────────────────────────────────────────
# Pulls the Vitis-AI 3.5 PyTorch GPU docker image. Idempotent — if the image
# is already present, exits successfully without re-pulling.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

# Docker tag pinned in one place. Update here if AMD releases a newer 3.5.x.
VAI_IMAGE="${VAI_IMAGE:-xilinx/vitis-ai-pytorch-gpu:3.5.0.001}"

log_step "Vitis-AI 3.5 docker image: $VAI_IMAGE"

# ─── prereq: docker reachable ───────────────────────────────────────────────
if ! docker_ok; then
    die "docker daemon not reachable. Run scripts/host/00_check_prereqs.sh and follow its hints."
fi

# ─── already pulled? ────────────────────────────────────────────────────────
if docker image inspect "$VAI_IMAGE" &>/dev/null; then
    sz=$(docker image inspect --format='{{.Size}}' "$VAI_IMAGE" | numfmt --to=iec)
    log_ok "Image already present ($sz). Nothing to do."
    log_info "To force re-pull:  docker pull $VAI_IMAGE"
    exit 0
fi

# ─── disk space sanity ──────────────────────────────────────────────────────
# The compressed download is ~10 GB but unpacks to ~25 GB on disk.
docker_root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo "/var/lib/docker")
if ! disk_free_gb "$docker_root" 30; then
    avail_gb=$(df -BG --output=avail "$docker_root" 2>/dev/null | tail -1 | tr -dc '0-9' || echo "?")
    log_err "Only ${avail_gb} GB free at $docker_root (need ≥30 GB for unpacked image)"
    log_err "  Free up docker space:  docker system prune -a"
    log_err "  Or relocate the docker root via /etc/docker/daemon.json"
    exit 1
fi

# ─── pull ───────────────────────────────────────────────────────────────────
log_info "Pulling — this is a ~10 GB compressed download. Be patient."
if docker pull "$VAI_IMAGE"; then
    log_ok "Pulled $VAI_IMAGE"
else
    log_err "docker pull failed."
    log_err "  Common causes:"
    log_err "    - Authentication: docker hub may require login for some images"
    log_err "    - Network: try again on a stable connection"
    log_err "    - Disk space: docker may have run out mid-extract"
    exit 1
fi

# ─── post-install GPU sanity ────────────────────────────────────────────────
log_info "Verifying GPU is reachable from inside the image..."
if docker run --rm --gpus all "$VAI_IMAGE" \
        bash -c 'nvidia-smi --query-gpu=name --format=csv,noheader | head -1' \
        2>/dev/null | grep -q '.'; then
    log_ok "GPU visible from container"
else
    log_warn "Could not run nvidia-smi inside the container."
    log_warn "  This may be fine for some setups, but the compile step requires GPU access."
    log_warn "  Re-run scripts/host/00_check_prereqs.sh and verify the NVIDIA Container Toolkit step."
fi

echo
log_ok "Vitis-AI 3.5 ready."
log_info "Next:  drop your trained .pt weights into data/weights/ and calibration"
log_info "       images into data/calib/, then:"
log_info "         bash scripts/host/02_compile.sh yolov5 yolov5n data/weights/<name>.pt data/calib/"
