#!/usr/bin/env bash
# scripts/host/check_compile_inputs.sh
# ─────────────────────────────────────────────────────────────────────────────
# Pre-compile diagnostic. Verifies all layers of the compile path WITHOUT
# actually running quantization or vai_c_xir. Catches the common failure
# modes before you spend 15 minutes finding out.
#
# Usage:
#   bash scripts/host/check_compile_inputs.sh <variant> <weights> <calib_dir>
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") <variant> <weights> <calib_dir>

Verifies inputs for a compile WITHOUT actually quantizing.
Reports each check pass/fail with a clear next step.

Arguments:
  variant      e.g. yolov5n, yolov5s, yolox_tiny, yolox_nano
  weights      path to trained .pt checkpoint (relative to repo root)
  calib_dir    directory of calibration images
EOF
    exit 2
}

[[ $# -lt 3 ]] && usage

VARIANT="$1"
WEIGHTS_REL="$2"
CALIB_REL="$3"

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

FAIL=0
fail() { FAIL=$((FAIL + 1)); }

log_step "Pre-compile diagnostic: $VARIANT"

# ─── 1. Variant is in the registry ──────────────────────────────────────────
log_info "Variant lookup"
spec_check=$(cd "$REPO_ROOT" && python3 - "$VARIANT" <<'PY'
import sys
sys.path.insert(0, '.')
try:
    from lpr_pipeline.shared.models import get_spec
    s = get_spec(sys.argv[1])
    print(f"OK|{s.family}|{s.imgsz}|{s.nc}|{s.reg_max}|{s.status}")
except KeyError as e:
    print(f"BAD|{e}")
    sys.exit(1)
except Exception as e:
    print(f"BAD|{type(e).__name__}: {e}")
    sys.exit(1)
PY
)

if [[ "$spec_check" == OK* ]]; then
    IFS='|' read -r _ family imgsz nc reg_max status <<<"$spec_check"
    log_ok "  family=$family  imgsz=$imgsz  nc=$nc  reg_max=$reg_max  status=$status"
    if [[ "$status" == "stub" ]]; then
        log_warn "  ⚠  This variant's compile path is a stub. Compile will raise NotImplementedFamilyError."
        log_warn "      Variants with full compile support: yolov5n, yolov5s, yolox_tiny, yolox_nano"
        fail
    fi
else
    log_err "  ${spec_check#BAD|}"
    fail
    family="?"; imgsz="?"
fi

# ─── 2. Weights file ────────────────────────────────────────────────────────
log_info "Weights file"
if [[ ! -f "$WEIGHTS" ]]; then
    log_err "  not found: $WEIGHTS_REL"
    log_err "  Place your trained .pt at this path and re-run."
    fail
else
    sz=$(stat -c%s "$WEIGHTS" | numfmt --to=iec)
    log_ok "  $WEIGHTS_REL  ($sz)"

    # Check magic bytes — accept both 'PK\x03\x04' (zip) and other torch formats.
    # PyTorch's standard .save() format is a zip archive starting with 'PK'.
    first_byte=$(head -c 2 "$WEIGHTS" | od -An -tx1 | tr -d ' ')
    if [[ "$first_byte" == "504b" ]]; then
        log_ok "  valid torch checkpoint (PK zip header)"
    else
        log_warn "  unusual file header (0x$first_byte). May still load fine; will verify in dry-run."
    fi
fi

# ─── 3. Calibration directory ───────────────────────────────────────────────
log_info "Calibration directory"
if [[ ! -d "$CALIB" ]]; then
    log_err "  not found: $CALIB_REL"
    fail
else
    n_jpg=$(find "$CALIB" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \
              -o -iname '*.png' -o -iname '*.bmp' \) 2>/dev/null | wc -l)
    if (( n_jpg < 1 )); then
        log_err "  $CALIB_REL has 0 images"
        fail
    elif (( n_jpg < 50 )); then
        log_warn "  $CALIB_REL has $n_jpg images — quantization accuracy will suffer"
    else
        log_ok "  $CALIB_REL  ($n_jpg images)"
    fi
fi

# ─── 4. Pre-existing build artifacts (informational) ────────────────────────
log_info "Build/output state"
WORK_DIR="$REPO_ROOT/build/$VARIANT"
OUT_DIR="$REPO_ROOT/out/$VARIANT"
if [[ -d "$WORK_DIR" ]]; then
    n=$(find "$WORK_DIR" -type f 2>/dev/null | wc -l)
    if (( n > 0 )); then
        log_warn "  Existing work dir: $WORK_DIR ($n files) — will be overwritten"
    fi
fi
if [[ -f "$OUT_DIR/${VARIANT}_kv260.xmodel" ]]; then
    sz=$(stat -c%s "$OUT_DIR/${VARIANT}_kv260.xmodel" | numfmt --to=iec)
    log_warn "  Existing xmodel will be overwritten: $OUT_DIR/${VARIANT}_kv260.xmodel ($sz)"
fi

# ─── 5. Docker + image ──────────────────────────────────────────────────────
log_info "Docker + Vitis-AI image"
USE_GPU=0
VAI_IMAGE=""
if ! docker_ok; then
    log_err "  docker daemon not reachable"
    fail
else
    img=$(docker images --format '{{.Repository}}:{{.Tag}}' \
            | grep -E '^xilinx/vitis-ai-pytorch-gpu:' | head -1)
    if [[ -n "$img" ]]; then
        log_ok "  GPU image will be used: $img"
        VAI_IMAGE="$img"
        USE_GPU=1
    else
        img=$(docker images --format '{{.Repository}}:{{.Tag}}' \
                | grep -E '^xilinx/vitis-ai-pytorch-cpu:' | head -1)
        if [[ -n "$img" ]]; then
            log_ok "  CPU image will be used: $img  (slower but works)"
            VAI_IMAGE="$img"
        else
            log_err "  No Vitis-AI image found locally"
            fail
        fi
    fi
fi

# ─── 6. Inside-container sanity (THE BIG ONE) ──────────────────────────────
# Writes the test script to a tempfile inside the repo's build/ dir, then runs
# it inside the container. Avoids all the quoting nightmares of trying to pass
# multi-line Python through `bash -c "python -c \"...\""`.
if [[ -n "$VAI_IMAGE" ]] && [[ -f "$WEIGHTS" ]]; then
    log_info "Container dry-run (loads checkpoint, finds Detect head, no quantize)"

    mkdir -p "$REPO_ROOT/build"
    DRY_RUN_FILE="$REPO_ROOT/build/.dry_run_$$.py"
    # Clean up the tempfile when the script exits
    trap 'rm -f "$DRY_RUN_FILE"' EXIT

    cat > "$DRY_RUN_FILE" <<'PYEOF'
import sys, os
sys.path.insert(0, '/workspace')
ok = True

def chk(label, fn):
    global ok
    try:
        fn()
        print(f"  [PASS] {label}")
    except Exception as e:
        print(f"  [FAIL] {label}: {type(e).__name__}: {e}")
        ok = False

# 1. lpr_pipeline imports
def t1():
    from lpr_pipeline.shared.models import get_spec
    from lpr_pipeline.compile import get_compiler
    from lpr_pipeline.compile.base import CompileInputs
chk("lpr_pipeline imports", t1)

# 2. PyTorch + vai_q_pytorch
def t2():
    import torch, pytorch_nndct
    print(f"      torch={torch.__version__}  pytorch_nndct={getattr(pytorch_nndct, '__version__', '?')}")
chk("torch + vai_q_pytorch", t2)

# 3. vai_c_xir on PATH
def t3():
    import shutil
    if not shutil.which("vai_c_xir"):
        raise RuntimeError("vai_c_xir not found on PATH")
chk("vai_c_xir on PATH", t3)

# 4. Architecture file present
def t4():
    from pathlib import Path
    arch = Path("/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json")
    if not arch.is_file():
        raise FileNotFoundError(arch)
chk("KV260 arch.json present", t4)

# 5. Weights load
WEIGHTS_PATH = os.environ["WEIGHTS_IN_CONTAINER"]
def t5():
    import torch
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        m = ckpt["model"]
    else:
        m = ckpt
    m = m.float().eval()
    globals()["_loaded_model"] = m
    print(f"      model class: {type(m).__name__}")
chk("Load weights with torch.load", t5)

# 6. Find Detect head
def t6():
    m = globals()["_loaded_model"]
    detect = None
    for sub in m.modules():
        cls = type(sub).__name__
        if cls in ("Detect", "DetectAux", "v8Detect", "v6Detect"):
            detect = sub
            print(f"      detect head class: {cls}")
            break
    if detect is None:
        kids = [type(c).__name__ for c in m.children()]
        raise RuntimeError(f"no Detect head found. children={kids}")
chk("Detect head found", t6)

# 7. Calib loader builds (one image)
def t7():
    from PIL import Image
    import numpy as np, torch
    from pathlib import Path
    cdir = Path(os.environ["CALIB_IN_CONTAINER"])
    imgs = [p for p in cdir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    if not imgs:
        raise RuntimeError("no images")
    img = Image.open(imgs[0]).convert("RGB")
    arr = np.asarray(img.resize((320, 320)))
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    if t.shape != (3, 320, 320):
        raise ValueError(f"unexpected shape: {t.shape}")
chk("Read first calib image (PIL to tensor)", t7)

if not ok:
    sys.exit(2)
print("\n  All in-container checks passed.")
PYEOF

    docker_args=( --rm
        --user "$(id -u):$(id -g)"
        --workdir /workspace
        -v "$REPO_ROOT:/workspace:rw"
        -e PYTHONPATH=/workspace
        -e PYTHONDONTWRITEBYTECODE=1
        -e WEIGHTS_IN_CONTAINER="/workspace/${WEIGHTS#$REPO_ROOT/}"
        -e CALIB_IN_CONTAINER="/workspace/${CALIB#$REPO_ROOT/}"
    )
    (( USE_GPU == 1 )) && docker_args+=( --gpus all )

    # The script lives in /workspace/build/.dry_run_<pid>.py; just run it.
    DRY_RUN_REL=".dry_run_$$.py"
    echo "  (running container check; ~60 sec — first run installs ultralytics)"
    if docker run "${docker_args[@]}" "$VAI_IMAGE" bash -lc "
        source /opt/vitis_ai/conda/etc/profile.d/conda.sh
        conda activate vitis-ai-pytorch
        # Install ultralytics so torch.load can unpickle modern YOLOv5/v8 checkpoints.
        # The scipy/numpy version warnings can be safely ignored — quantization
        # has not been observed to fail with the upgraded versions.
        pip install --quiet --disable-pip-version-check ultralytics 2>&1 | tail -3 || true
        python /workspace/build/$DRY_RUN_REL
    "; then
        :
    else
        log_err "  Container dry-run failed (see above)"
        fail
    fi
fi

# ─── summary ────────────────────────────────────────────────────────────────
echo
if (( FAIL == 0 )); then
    log_ok "All pre-compile checks passed. Ready to compile:"
    log_info ""
    log_info "  bash scripts/host/02_compile.sh $family $VARIANT $WEIGHTS_REL $CALIB_REL"
    exit 0
else
    log_err "$FAIL check(s) failed. Address the items above before running 02_compile.sh."
    exit 1
fi
