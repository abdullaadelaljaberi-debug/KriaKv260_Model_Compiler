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
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

N_CALIB="${N_CALIB:-200}"
NUM_CLASSES="${NUM_CLASSES:-1}"
SWAP_ACTIVATIONS="${SWAP_ACTIVATIONS:-true}"

# Convert SWAP_ACTIVATIONS env var to Python boolean literal
case "${SWAP_ACTIVATIONS,,}" in
    true|yes|1|on)  SWAP_PY="True" ;;
    false|no|0|off) SWAP_PY="False" ;;
    *) die "SWAP_ACTIVATIONS must be true/false, got: $SWAP_ACTIVATIONS" ;;
esac

usage() {
    cat <<EOF
Usage: $(basename "$0") <family> <variant> <weights> <calib_dir> [output]

Arguments:
  family       One of: yolov5, yolov11, yolox,
                       ssdlite, retinanet, classification
                       (stubs: yolov7, yolov4_csp, ssd_mobilenetv2)
  variant      e.g. yolov5n, yolov11n, yolox_tiny,
                       ssdlite_bstld, retinanet_vineset, resnet50_gtsrb
  weights      Path to your trained .pt checkpoint (relative to repo root)
  calib_dir    Directory of calibration images (≥100 JPGs/PNGs)
  output       Optional. Where to write the final .xmodel.
               Default: out/<variant>/<variant>_kv260.xmodel

Environment overrides:
  VAI_IMAGE    Docker image tag. Default: auto-detect (GPU preferred over CPU)
  N_CALIB      Number of calibration images to use (default: $N_CALIB)
  NUM_CLASSES  Number of object classes the model was trained on (default: 1)
  SWAP_ACTIVATIONS  true (default) — auto-swap SiLU → LeakyReLU(0.1015625) for
               DPU compatibility. Set to false to preserve the original
               activations (xmodel will fragment into multi-DPU subgraphs;
               must be deployed via vitis_ai_library.GraphRunner instead of
               pynq_dpu.overlay.load_model). See docs/MODELS.md.
EOF
    exit 2
}

[[ $# -lt 4 ]] && usage

FAMILY="$1"
VARIANT="$2"
WEIGHTS_REL="$3"
CALIB_REL="$4"
OUT_REL="${5:-out/$VARIANT/${VARIANT}_kv260.xmodel}"

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

log_step "Compile: $FAMILY -> $VARIANT"
log_info "weights : $WEIGHTS_REL"
log_info "calib   : $CALIB_REL"
log_info "output  : $OUT_REL"
log_info "nc      : $NUM_CLASSES"
log_info "swap    : $SWAP_ACTIVATIONS  (SiLU → LeakyReLU)"

if ! docker_ok; then
    die "docker daemon not reachable. Run scripts/host/00_check_prereqs.sh."
fi

# ─── pick the docker image ─────────────────────────────────────────────────
USE_GPU=0
if [[ -n "${VAI_IMAGE:-}" ]]; then
    if ! docker image inspect "$VAI_IMAGE" &>/dev/null; then
        die "\$VAI_IMAGE='$VAI_IMAGE' not present locally."
    fi
    [[ "$VAI_IMAGE" == *"-gpu:"* ]] && USE_GPU=1
else
    img=$(docker images --format '{{.Repository}}:{{.Tag}}' \
            | grep -E '^xilinx/vitis-ai-pytorch-gpu:' | head -1)
    if [[ -n "$img" ]]; then
        VAI_IMAGE="$img"
        USE_GPU=1
    else
        img=$(docker images --format '{{.Repository}}:{{.Tag}}' \
                | grep -E '^xilinx/vitis-ai-pytorch-cpu:' | head -1)
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
    log_info "image   : $VAI_IMAGE  [CPU - quantization will be slower]"
fi

[[ -f "$WEIGHTS" ]] || die "weights file not found: $WEIGHTS_REL"
[[ -d "$CALIB" ]] || die "calibration directory not found: $CALIB_REL"

mkdir -p "$(dirname "$OUT")"

WORK_DIR="$REPO_ROOT/build/$VARIANT"
mkdir -p "$WORK_DIR"
log_info "work dir: $WORK_DIR"

# ─── write the compile driver to a tempfile ────────────────────────────────
# We write the Python entrypoint to /workspace/build/.compile_<pid>.py inside
# the repo so it's mounted into the container, then run it. This avoids the
# nested-quote nightmare of trying to pass multi-line Python through bash -c.
mkdir -p "$REPO_ROOT/build"
DRIVER_FILE="$REPO_ROOT/build/.compile_$$.py"
trap 'rm -f "$DRIVER_FILE"' EXIT

cat > "$DRIVER_FILE" <<PYEOF
import sys, os, traceback
from pathlib import Path
sys.path.insert(0, '/workspace')

from lpr_pipeline.shared.models import get_spec
from lpr_pipeline.compile import get_compiler, NotImplementedFamilyError
from lpr_pipeline.compile.base import CompileInputs, CompileError

VARIANT = "$VARIANT"
FAMILY  = "$FAMILY"
WEIGHTS = "/workspace/${WEIGHTS#$REPO_ROOT/}"
CALIB   = "/workspace/${CALIB#$REPO_ROOT/}"
WORKDIR = "/workspace/build/$VARIANT"
OUT     = "/workspace/${OUT#$REPO_ROOT/}"
NC      = $NUM_CLASSES
N_CALIB = $N_CALIB
SWAP_ACT = $SWAP_PY

try:
    spec = get_spec(VARIANT)
    if spec.family != FAMILY:
        print(f"  [FAIL] Variant {spec.name} belongs to family {spec.family!r}, "
              f"not {FAMILY!r}.")
        sys.exit(2)

    inputs = CompileInputs(
        spec       = spec,
        weights    = Path(WEIGHTS),
        calib_dir  = Path(CALIB),
        work_dir   = Path(WORKDIR),
        out_xmodel = Path(OUT),
        nc         = NC,
        n_calib    = N_CALIB,
        swap_activations = SWAP_ACT,
    )

    compiler = get_compiler(FAMILY)
    out      = compiler.run(inputs)
    print(f"\n[OK] Wrote {out}")
    sys.exit(0)

except NotImplementedFamilyError as e:
    print(f"\n[STUB] {e}")
    sys.exit(3)
except CompileError as e:
    print(f"\n[ERROR] Compile failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Unexpected: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
PYEOF

# ─── launch the container ───────────────────────────────────────────────────
log_step "Launching Vitis-AI 3.5 container"

docker_args=( --rm
    --user "$(id -u):$(id -g)"
    --workdir /workspace
    -v "$REPO_ROOT:/workspace:rw"
    -v "$HOME/.cache:/home/$(whoami)/.cache:rw"
    -e PYTHONPATH=/workspace
    -e PYTHONDONTWRITEBYTECODE=1
    -e PIP_CACHE_DIR=/workspace/build/.pip_cache
)

(( USE_GPU == 1 )) && docker_args+=( --gpus all )

if [[ -t 0 ]]; then
    docker_args+=( -it )
fi

DRIVER_REL=".compile_$$.py"

docker run "${docker_args[@]}" "$VAI_IMAGE" bash -lc "
    source /opt/vitis_ai/conda/etc/profile.d/conda.sh
    conda activate vitis-ai-pytorch
    # Install ultralytics so torch.load can unpickle modern YOLOv5/v8 checkpoints.
    # See HOST_SETUP.md for notes on the scipy/numpy version conflict warnings.
    pip install --quiet --disable-pip-version-check ultralytics 2>&1 | tail -3 || true
    python /workspace/build/$DRIVER_REL
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
