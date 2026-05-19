#!/usr/bin/env bash
# scripts/host/_quantize_onnx_yolov11.sh
# Runs _quantize_onnx_yolov11.py inside the vitis-ai-onnx-cpu:eggs container.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

INPUT="${1:-build/yolov11n_onnx/yolo11n_eggs_dpu.onnx}"
CALIB="${2:-data/calib_v2_hardneg}"
OUTPUT="${3:-out/yolov11n_onnx/yolov11n_eggs_kv260.xmodel}"
IMGSZ="${IMGSZ:-640}"
N_CALIB="${N_CALIB:-200}"
PER_CHANNEL="${PER_CHANNEL:-true}"   # set to "false" for ablation

# Resolve to absolute paths under REPO_ROOT
INPUT_ABS="$REPO_ROOT/$INPUT"
CALIB_ABS="$REPO_ROOT/$CALIB"
OUTPUT_ABS="$REPO_ROOT/$OUTPUT"
mkdir -p "$(dirname "$OUTPUT_ABS")"

# Container-side paths
INPUT_CONTAINER="/workspace/${INPUT_ABS#$REPO_ROOT/}"
CALIB_CONTAINER="/workspace/${CALIB_ABS#$REPO_ROOT/}"
OUTPUT_CONTAINER="/workspace/${OUTPUT_ABS#$REPO_ROOT/}"

log_step "ONNX PTQ + compile: $INPUT → $OUTPUT"
log_info "  calib:        $CALIB ($N_CALIB images)"
log_info "  imgsz:        $IMGSZ"
log_info "  per_channel:  $PER_CHANNEL"

if [[ ! -f "$INPUT_ABS" ]]; then
    die "Input ONNX not found: $INPUT (run _export_onnx_yolov11.sh first)"
fi
if [[ ! -d "$CALIB_ABS" ]]; then
    die "Calib dir not found: $CALIB"
fi

if ! docker image inspect vitis-ai-onnx-cpu:eggs &>/dev/null; then
    die "vitis-ai-onnx-cpu:eggs not found. Run the docker commit step from the conversation to build it."
fi

# Build the per-channel flag
if [[ "$PER_CHANNEL" == "true" ]] || [[ "$PER_CHANNEL" == "1" ]]; then
    PC_FLAG="--per-channel"
else
    PC_FLAG="--no-per-channel"
fi

docker run --rm \
    --user "$(id -u):$(id -g)" \
    --workdir /workspace \
    -v "$REPO_ROOT:/workspace:rw" \
    -e PYTHONPATH=/workspace \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e HOME=/tmp \
    vitis-ai-onnx-cpu:eggs bash -lc "
        source /opt/vitis_ai/conda/etc/profile.d/conda.sh
        conda activate vitis-ai-pytorch
        python /workspace/scripts/host/_quantize_onnx_yolov11.py \
            --input    '$INPUT_CONTAINER' \
            --calib    '$CALIB_CONTAINER' \
            --output   '$OUTPUT_CONTAINER' \
            --imgsz    $IMGSZ \
            --n-calib  $N_CALIB \
            $PC_FLAG
    "

rtn=$?
if [[ $rtn -ne 0 ]]; then
    die "ONNX PTQ + compile failed (exit $rtn)"
fi

if [[ -f "$OUTPUT_ABS" ]]; then
    sz=$(stat -c%s "$OUTPUT_ABS" | numfmt --to=iec)
    log_ok "Final xmodel: $OUTPUT ($sz)"
    log_info ""
    log_info "Next — sync to Kria with a unique name so it doesn't clobber the NNDCT xmodel:"
    log_info "  scp $OUTPUT_ABS ubuntu@10.42.0.189:/home/ubuntu/xmodels_vai35/yolov11n/yolov11n_kv260_onnx.xmodel"
else
    die "Compile reported success but $OUTPUT is missing"
fi
