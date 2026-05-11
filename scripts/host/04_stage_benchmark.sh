#!/usr/bin/env bash
# scripts/host/04_stage_benchmark.sh
# ─────────────────────────────────────────────────────────────────────────────
# Stage VAI 3.5 benchmark data on the HOST PC (not the Kria).
#
# Why this lives on the host:
#   The earlier in-notebook auto-download on the Kria corrupted a 256 GB SD
#   card under sustained writes. Downloading on a laptop's SSD instead is
#   much safer, faster, and resumable across days.
#
# After this completes, push to the Kria with:
#   bash scripts/host/05_sync_benchmark_to_kria.sh ubuntu@<board-ip>
#
# Then on the Kria, the benchmark notebook (notebooks/04_vai35_benchmark.ipynb)
# expects everything already in place — it does NO downloads.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

STAGE_ROOT="$REPO_ROOT/build/benchmark_stage"
PY_HELPER="$SCRIPT_DIR/_stage_benchmark.py"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Stage the VAI 3.5 benchmark data on this host PC. Downloads ~12 GB:
  - VAI 3.0 pre-compiled KV260 xmodels (~9 GB, 34 models)
  - ImageNetV2 matched-frequency with labels (~1.3 GB, 10k images)
  - COCO val2017 images + annotations (~1.0 GB)
  - VOC2007 test set (~430 MB)

Output goes to:  $STAGE_ROOT

Options:
  --skip-models       Only download datasets
  --skip-datasets     Only download models
  --only <name>       Download a single model by name (for catalogue iteration)
  --min-free-gb <N>   Override the pre-flight disk check (default: 15)
  -h, --help          This help

After staging completes:
  bash scripts/host/05_sync_benchmark_to_kria.sh ubuntu@<board-ip>

Re-runnable: completed downloads are skipped, partial files resume.
EOF
    exit 2
}

# Parse arguments — pass through to the Python helper, but catch -h locally
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        *) EXTRA_ARGS+=( "$1" ); shift ;;
    esac
done

log_step "VAI 3.5 benchmark staging"
log_info "Stage root: $STAGE_ROOT"
log_info "Helper:     $PY_HELPER"

# Pre-flight: python3 + Python helper present
if ! have_cmd python3; then
    die "python3 not found. Install with:  sudo apt install python3"
fi
if [[ ! -f "$PY_HELPER" ]]; then
    die "Helper script not found: $PY_HELPER"
fi

# certifi makes SSL more robust on older systems (Kria's CA bundle was
# incomplete; laptops are usually fine, but the check is cheap).
if ! python3 -c "import certifi" 2>/dev/null; then
    log_warn "certifi not installed — SSL fallback will be used if needed"
    log_warn "  Install with:  pip install certifi"
fi

# Pre-flight: internet
if ! internet_ok; then
    log_warn "Cannot reach the public internet"
    log_warn "  If you're behind a proxy, set http_proxy / https_proxy before re-running"
fi

mkdir -p "$STAGE_ROOT"

log_info "Launching Python staging tool..."
echo

# Pass through to the Python helper
python3 "$PY_HELPER" --stage-root "$STAGE_ROOT" "${EXTRA_ARGS[@]}"
rc=$?

echo
if (( rc == 0 )); then
    log_ok "Staging complete."
    log_info "  Stage size: $(du -sh "$STAGE_ROOT" | cut -f1)"
    log_info "  Free disk:  $(df -h "$STAGE_ROOT" | tail -1 | awk '{print $4}') remaining"
    echo
    log_info "Next: push to the Kria"
    log_info "  bash scripts/host/05_sync_benchmark_to_kria.sh ubuntu@<board-ip>"
elif (( rc == 2 )); then
    log_err "Pre-flight check failed (insufficient disk). Free space and re-run."
else
    log_err "Staging encountered errors (exit $rc). Re-run to retry — completed items will be skipped."
fi
exit $rc
