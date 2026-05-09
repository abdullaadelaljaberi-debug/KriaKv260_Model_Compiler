#!/usr/bin/env bash
# scripts/host/02_compile.sh
# ─────────────────────────────────────────────────────────────────────────────
# Main compile entrypoint. Runs the lpr_pipeline.compile module inside the
# Vitis-AI 3.5 Docker container, with the user's weights and calibration
# images mounted in.
#
# Auto-detects whether to use a GPU or CPU image, picking the GPU image
# preferentially. Override via $VAI_IMAGE env var.
#
# Usage:
#   bash scripts/host/02_compile.sh <family> <variant> <weights.pt> <calib_dir> [output_path]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

N_CALIB="${N_CALIB:-200}"
NUM_CLASSES="${NUM_CLASSES:-1}"

usage() {
    cat <<EOF
Usage: $(basename "$0") <family> <variant> <weights> <calib_dir> [output]

Arguments:
  family       One of: yolov5, yolox, yolov7, yolov4_csp, ssd_mobilenetv2
               (yolov7/yolov4_csp/ssd_mobilenetv2 are stubs and will fail)
  variant      One of the variant names registered in lpr_pipeline.shared.models
               Examples: yolov5n, yolov5s, yolox_tiny, yolox_nano
  weights      Path to your trained .pt checkpoint (relative to repo root)
  calib_dir    Directory of calibration images (≥100 representative JPGs/PNGs)
  output       Optional. Where to write the final .xmodel.
               Default: out/<variant>/<variant>_kv260.xmodel

Environment overrides:
  VAI_IMAGE    Docker image tag. Default: auto-detect (GPU preferred over CPU)
  N_CALIB      Number of calibration images to use (default: $N_CALIB)
  NUM_CLASSES  Number of object classes the model was trained on (default: 1)

Example:
  bash $(basename "$0") yolov5 yolov5n \\
       data/weights/yolov5n_lpr.pt \\
       data/calib/ \\
       out/yolov5n_kv260.xmodel
EOF
    exit 2
}

[[ $# -lt 4 ]] && usage

FAMILY="$1"
VARIANT="$2"
WEIGHTS_REL="$3"
CALIB_REL="$4"
OUT_REL="${5:-out/$VARIANT/${VARIANT}_kv260.xmodel}"

# ─── resolve paths to absolute, relative to repo root ──────────────────────
abspath_in_repo() {
    local p="$1"
    p="$(realpath -m --relative-base="$REPO_ROOT" "$REPO_ROOT/$p")"
    if [[ "$p" == /* ]] || [[ "$p" == ../* ]]; then
        die "path must be inside the repo: $1 (resolved: $p)"
    fi
    echo "$REPO_ROOT/$p"
}

WEIGHTS=$(abspath_in_repo "$WEIGHTS_REL")
CALIB=$(abspath_in_repo "$CALIB_REL")
OUT=$(abspath_in_repo "$OUT_REL")

log_step "Compile: $FAMILY → $VARIANT"
log_info "weights : $WEIGHTS_REL"
log_info "calib   : $CALIB_REL"
log_info "output  : $OUT_REL"
log_info "nc      : $NUM_CLASSES"

# ─── prereq checks ──────────────────────────────────────────────────────────
if ! docker_ok; then
    die "docker daemon not reachable. Run scripts/host/00_check_prereqs.sh."
fi

# ─── pick the docker image ─────────────────────────────────────────────────
# Auto-detect unless caller set $VAI_IMAGE explicitly.
USE_GPU=0
if [[ -n "${VAI_IMAGE:-}" ]]; then
    if ! docker image inspect "$VAI_IMAGE" &>/dev/null; then
        die "\$VAI_IMAGE='$VAI_IMAGE' not present locally. Pull or build it first."
    fi
    [[ "$VAI_IMAGE" == *"-gpu:"* ]] && USE_GPU=1
else
    # Prefer GPU
    img=$(docker images --format '{{.Repository}}:{{.Tag}}' \
            | grep -E '^xilinx/vitis-ai-pytorch-gpu:' | head -1 || true)
    if [[ -n "$img" ]]; then
        VAI_IMAGE="$img"
        USE_GPU=1
    else
        img=$(docker images --format '{{.Repository}}:{{.Tag}}' \
                | grep -E '^xilinx/vitis-ai-pytorch-cpu:' | head -1 || true)
        if [[ -n "$img" ]]; then
            VAI_IMAGE="$img"
        else
            die "No Vitis-AI image found. Run scripts/host/01_install_vai.sh."
        fi
    fi
fi

if (( USE_GPU == 1 )); then
    log_info "image   : $VAI_IMAGE  [GPU acceleration]"
else
    log_info "image   : $VAI_IMAGE  [CPU — quantization will be slower]"
fi

[[ -f "$WEIGHTS" ]] || die "weights file not found: $WEIGHTS_REL
  Place your trained checkpoint at this path before compiling.
  See docs/USAGE.md → 'Where to get weights' for guidance."

[[ -d "$CALIB" ]] || die "calibration directory not found: $CALIB_REL
  Place ≥100 representative images (JPG/PNG/BMP) in this directory."

mkdir -p "$(dirname "$OUT")"

# ─── set up working directory ───────────────────────────────────────────────
WORK_DIR="$REPO_ROOT/build/$VARIANT"
mkdir -p "$WORK_DIR"
log_info "work dir: $WORK_DIR"

# ─── launch the container ───────────────────────────────────────────────────
log_step "Launching Vitis-AI 3.5 container"

docker_args=(
    --rm
    --user "$(id -u):$(id -g)"
    --workdir /workspace
    -v "$REPO_ROOT:/workspace:rw"
    -v "$HOME/.cache:/home/$(whoami)/.cache:rw"
    -e PYTHONPATH=/workspace
    -e PYTHONDONTWRITEBYTECODE=1
)

# Only add --gpus all if using a GPU image (otherwise causes errors on CPU image)
if (( USE_GPU == 1 )); then
    docker_args+=( --gpus all )
fi

# Only attach a TTY if running interactively (lets you Ctrl-C cleanly)
if [[ -t 0 ]]; then
    docker_args+=( -it )
fi

# Build the python command — exec the compile entrypoint inside the container
PY_CMD=$(cat <<EOF
import sys, os, traceback
from pathlib import Path
sys.path.insert(0, '/workspace')

from lpr_pipeline.shared.models import get_spec
from lpr_pipeline.compile import get_compiler, NotImplementedFamilyError
from lpr_pipeline.compile.base import CompileInputs, CompileError

try:
    spec = get_spec("$VARIANT")
    if spec.family != "$FAMILY":
        print(f"  ✗ Variant {spec.name} belongs to family {spec.family!r}, "
              f"not {'$FAMILY'!r}.")
        sys.exit(2)

    inputs = CompileInputs(
        spec       = spec,
        weights    = Path("/workspace/${WEIGHTS#$REPO_ROOT/}"),
        calib_dir  = Path("/workspace/${CALIB#$REPO_ROOT/}"),
        work_dir   = Path("/workspace/build/$VARIANT"),
        out_xmodel = Path("/workspace/${OUT#$REPO_ROOT/}"),
        nc         = $NUM_CLASSES,
        n_calib    = $N_CALIB,
    )

    compiler = get_compiler("$FAMILY")
    out      = compiler.run(inputs)
    print(f"\n✓ Wrote {out}")
    sys.exit(0)

except NotImplementedFamilyError as e:
    print(f"\n✗ {e}")
    sys.exit(3)
except CompileError as e:
    print(f"\n✗ Compile failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Unexpected: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
EOF
)

# Run with conda env activated
docker run "${docker_args[@]}" "$VAI_IMAGE" bash -lc "
    set -e
    source /opt/vitis_ai/conda/etc/profile.d/conda.sh
    conda activate vitis-ai-pytorch
    python -c '$PY_CMD'
"

# ─── result ─────────────────────────────────────────────────────────────────
if [[ -f "$OUT" ]]; then
    sz=$(stat -c%s "$OUT" | numfmt --to=iec)
    log_ok "Final xmodel: $OUT_REL  ($sz)"
    echo
    log_info "Next:  bash scripts/host/03_sync_to_kria.sh ubuntu@<board-ip> $VARIANT"
else
    die "Compile reported success but $OUT_REL is missing"
fi
