#!/usr/bin/env bash
# Master orchestrator: runs all 24 trainings sequentially.
#
# - Skips any training whose output weight file already exists (resumable).
# - Logs each training to data/weights/{detection,classification}/<name>_<dataset>.log
# - Continues past failures so a single bad config doesn't waste 15 hours of compute.
# - Prints a summary table at the end.
#
# Usage:
#     bash scripts/host/train_all.sh                  # run everything
#     bash scripts/host/train_all.sh --detection      # only detection trainings
#     bash scripts/host/train_all.sh --classification # only classification
#     bash scripts/host/train_all.sh --resume         # skip what's already done

set -u   # exit on undefined variable but NOT on error -- we want to continue past failures

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Defaults
RUN_DETECTION=1
RUN_CLASSIFICATION=1
RESUME=1   # always resume by default

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --detection)     RUN_CLASSIFICATION=0; shift ;;
        --classification) RUN_DETECTION=0; shift ;;
        --no-resume)     RESUME=0; shift ;;
        --resume)        RESUME=1; shift ;;
        *) echo "unknown flag: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# The 24 trainings
# ---------------------------------------------------------------------------

# Detection: 6 models x 3 datasets = 18 trainings
# Format: "model dataset epochs batch [imgsz]"
DETECTION_RUNS=(
    "yolov5n bstld 50 32"
    "yolov5n license_plates 50 32"
    "yolov5n vineset 50 32"
    "yolov5s bstld 50 24"
    "yolov5s license_plates 50 24"
    "yolov5s vineset 50 24"
    "yolov11n bstld 50 32"
    "yolov11n license_plates 50 32"
    "yolov11n vineset 50 32"
    "yolov11s bstld 50 16"
    "yolov11s license_plates 50 16"
    "yolov11s vineset 50 16"
    "ssdlite bstld 60 32"
    "ssdlite license_plates 60 32"
    "ssdlite vineset 60 32"
    "retinanet bstld 30 8"          # smaller batch — retinanet is heavy
    "retinanet license_plates 30 8"
    "retinanet vineset 30 8"
)

# Classification: 3 models x 2 datasets = 6 trainings
# Format: "model dataset epochs batch"
CLASSIFICATION_RUNS=(
    "resnet50 gtsrb 30 64"
    "resnet50 oxford_pets 40 32"
    "mobilenetv2 gtsrb 30 64"
    "mobilenetv2 oxford_pets 40 32"
    "inceptionv3 gtsrb 30 32"
    "inceptionv3 oxford_pets 40 16"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log_step() {
    echo ""
    echo "============================================================================"
    echo ">>> $1"
    echo "============================================================================"
}

log_info() { echo "    $1"; }

# Track outcomes
declare -a SUCCESSES=()
declare -a FAILURES=()
declare -a SKIPPED=()

run_detection() {
    local spec="$1"
    read -r model dataset epochs batch imgsz <<< "$spec"
    imgsz="${imgsz:-}"

    local output="data/weights/detection/${model}_${dataset}"
    # YOLO models save .pt; torchvision models save .pth
    case "$model" in
        yolov5*|yolov11*) output_file="${output}.pt" ;;
        *)                output_file="${output}.pth" ;;
    esac

    if [[ "$RESUME" -eq 1 && -f "$output_file" ]]; then
        log_info "SKIP: ${model}_${dataset} (output already exists)"
        SKIPPED+=("${model}_${dataset}")
        return 0
    fi

    log_step "Detection: ${model} on ${dataset} (epochs=$epochs, batch=$batch)"

    # Capture full stdout to a per-run log file so we don't lose progress
    # if the terminal closes mid-run.
    local log_file="data/weights/detection/${model}_${dataset}.run.log"
    mkdir -p "$(dirname "$log_file")"

    local cmd=(python3 scripts/host/train_detection.py
               --model "$model" --dataset "$dataset"
               --epochs "$epochs" --batch "$batch")
    [[ -n "$imgsz" ]] && cmd+=(--imgsz "$imgsz")

    # 'tee' shows output live AND writes it to the log file.
    # ${PIPESTATUS[0]} captures python's exit code (not tee's).
    if "${cmd[@]}" 2>&1 | tee "$log_file"; then
        true  # placeholder; real check below
    fi
    local rc=${PIPESTATUS[0]}
    if [[ $rc -eq 0 ]]; then
        SUCCESSES+=("${model}_${dataset}")
    else
        FAILURES+=("${model}_${dataset}")
        log_info "FAILED: ${model}_${dataset} (exit=$rc, log=$log_file)"
    fi
}

run_classification() {
    local spec="$1"
    read -r model dataset epochs batch <<< "$spec"

    local output_file="data/weights/classification/${model}_${dataset}.pth"

    if [[ "$RESUME" -eq 1 && -f "$output_file" ]]; then
        log_info "SKIP: ${model}_${dataset} (output already exists)"
        SKIPPED+=("${model}_${dataset}")
        return 0
    fi

    log_step "Classification: ${model} on ${dataset} (epochs=$epochs, batch=$batch)"

    local log_file="data/weights/classification/${model}_${dataset}.run.log"
    mkdir -p "$(dirname "$log_file")"

    if python3 scripts/host/train_classification.py \
            --model "$model" --dataset "$dataset" \
            --epochs "$epochs" --batch "$batch" 2>&1 | tee "$log_file"; then
        true
    fi
    local rc=${PIPESTATUS[0]}
    if [[ $rc -eq 0 ]]; then
        SUCCESSES+=("${model}_${dataset}")
    else
        FAILURES+=("${model}_${dataset}")
        log_info "FAILED: ${model}_${dataset} (exit=$rc, log=$log_file)"
    fi
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

START_TIME=$(date +%s)

log_step "Training pipeline start"
log_info "Detection runs:       $RUN_DETECTION (${#DETECTION_RUNS[@]} trainings)"
log_info "Classification runs:  $RUN_CLASSIFICATION (${#CLASSIFICATION_RUNS[@]} trainings)"
log_info "Resume mode:          $RESUME"

if [[ "$RUN_DETECTION" -eq 1 ]]; then
    log_step "Stage 2a: Detection"
    for spec in "${DETECTION_RUNS[@]}"; do
        run_detection "$spec"
    done
fi

if [[ "$RUN_CLASSIFICATION" -eq 1 ]]; then
    log_step "Stage 2b: Classification"
    for spec in "${CLASSIFICATION_RUNS[@]}"; do
        run_classification "$spec"
    done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

log_step "Training pipeline summary"

echo ""
echo "  Wall clock:       ${HOURS}h ${MINUTES}m"
echo ""
echo "  Successes (${#SUCCESSES[@]}):"
for s in "${SUCCESSES[@]}"; do echo "    [OK]   $s"; done
echo ""
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo "  Skipped (already done) (${#SKIPPED[@]}):"
    for s in "${SKIPPED[@]}"; do echo "    [SKIP] $s"; done
    echo ""
fi
if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "  Failures (${#FAILURES[@]}):"
    for f in "${FAILURES[@]}"; do echo "    [FAIL] $f"; done
    echo ""
    echo "  ⚠ Re-run with --resume to retry failed trainings only."
    exit 1
fi

echo "  All trainings completed successfully."
echo ""
echo "  Next steps:"
echo "    1. Run scripts/host/02_compile.sh for each variant to produce xmodels"
echo "    2. Run scripts/host/03_sync_to_kria.sh ubuntu@<kria-ip> <variant> for each"
echo "    3. Open notebooks/06_full_matrix_benchmark.ipynb on the Kria"
echo ""
