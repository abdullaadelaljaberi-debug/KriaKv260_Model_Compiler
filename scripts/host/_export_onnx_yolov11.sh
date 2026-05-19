#!/usr/bin/env bash
# scripts/host/_export_onnx_yolov11.sh
# Runs _export_onnx_yolov11.py inside the vitis-ai-onnx-cpu container.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

WEIGHTS="${1:-data/weights/yolo11n_eggs_dpu.pt}"
OUTPUT="${2:-build/yolov11n_onnx/yolo11n_eggs_dpu.onnx}"
IMGSZ="${IMGSZ:-640}"
NC="${NC:-1}"

# Resolve to absolute paths under REPO_ROOT
WEIGHTS_ABS="$(realpath "$REPO_ROOT/$WEIGHTS")"
OUTPUT_ABS="$REPO_ROOT/$OUTPUT"
mkdir -p "$(dirname "$OUTPUT_ABS")"

# Container-side paths (under /workspace mount)
WEIGHTS_CONTAINER="/workspace/${WEIGHTS_ABS#$REPO_ROOT/}"
OUTPUT_CONTAINER="/workspace/${OUTPUT_ABS#$REPO_ROOT/}"

log_step "ONNX export: $WEIGHTS → $OUTPUT"
log_info "  imgsz: $IMGSZ"
log_info "  nc:    $NC"

if ! docker image inspect vitis-ai-onnx-cpu:eggs &>/dev/null; then
    die "vitis-ai-onnx-cpu:eggs not found. Build it or pull it first."
fi

docker run --rm \
    --user "$(id -u):$(id -g)" \
    --workdir /workspace \
    -v "$REPO_ROOT:/workspace:rw" \
    -e PYTHONPATH=/workspace \
    -e PYTHONDONTWRITEBYTECODE=1 \
    vitis-ai-onnx-cpu:eggs bash -lc "
        source /opt/vitis_ai/conda/etc/profile.d/conda.sh
        conda activate vitis-ai-pytorch
        python /workspace/scripts/host/_export_onnx_yolov11.py \
            --weights '$WEIGHTS_CONTAINER' \
            --output  '$OUTPUT_CONTAINER' \
            --imgsz   $IMGSZ \
            --nc      $NC
    "

rtn=$?
if [[ $rtn -ne 0 ]]; then
    die "ONNX export failed (exit $rtn)"
fi

if [[ -f "$OUTPUT_ABS" ]]; then
    sz=$(stat -c%s "$OUTPUT_ABS" | numfmt --to=iec)
    log_ok "ONNX file: $OUTPUT ($sz)"
else
    die "Export reported success but $OUTPUT is missing"
fi
