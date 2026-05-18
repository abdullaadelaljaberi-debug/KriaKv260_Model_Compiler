#!/usr/bin/env python3
"""
scripts/host/_train_yolov11.py — YOLOv11 training helper for DPU deployment.

Trains YOLOv11n with the DPU-friendly architecture (C2PSA → C2PSA_DPU and
DWConv → Conv via monkey-patches) on a user-supplied dataset. Produces a
.pt checkpoint ready for the standard compile pipeline:

    bash scripts/host/02_compile.sh yolov11 yolov11n <out.pt> <calib_dir>

Why this is a separate step
---------------------------

YOLOv11n's stock architecture contains operations (matmul, softmax, chunk,
split, depthwise convolutions in the Detect head) that either don't compile
to the KV260 DPU or fragment the resulting xmodel into many subgraphs. We
replace these blocks with DPU-friendly equivalents before training, then
fine-tune so the new operations learn useful behavior. The substitutions
are mathematically different from the originals (HardSigmoid replaces
softmax-based attention, plain Conv replaces DWConv+Conv pairs), so
retraining is required — you cannot apply the substitutions to an already-
trained model and expect good accuracy.

See docs/YOLOV11.md for the full discussion.

Usage
-----

::

    python3 scripts/host/_train_yolov11.py \\
        --weights stock_yolo11n.pt \\
        --data    my_dataset/data.yaml \\
        --output  data/weights/my_model_dpu.pt \\
        --epochs  50 \\
        --batch   16

If you don't have a stock YOLOv11n.pt, Ultralytics will download one
automatically on first ``YOLO('yolo11n.yaml')``. Or train from scratch
by passing ``--weights yolo11n.yaml`` instead of a .pt.

The script auto-detects an NVIDIA GPU if present. With an A2000 8GB:
training a single-class dataset like the eggs example takes ~12 minutes
for 50 epochs at batch 16.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path

import torch
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Make the lpr_pipeline package importable when this script is invoked
# directly as a file (e.g. `python3 scripts/host/_train_yolov11.py`).
#
# Python's default behavior is to add the *script's directory* to sys.path[0],
# not the current working directory. Since this script lives at
# <repo>/scripts/host/, the repo root isn't on sys.path by default and
# `import lpr_pipeline.c2psa_dpu` fails. We resolve this by walking up two
# directories from this file's location (scripts/host/ → scripts/ → <repo>)
# and prepending the repo root to sys.path.
#
# Idempotent: if the repo is already on sys.path (e.g. via PYTHONPATH or
# invocation as `python3 -m`), this is a no-op.
# ─────────────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Compute device detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_device() -> str:
    """Return 'cuda' if a NVIDIA GPU is usable, else 'cpu' (with a warning)."""
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU detected: {name} ({vram_gb:.1f} GB VRAM, "
              f"{n} device{'s' if n > 1 else ''})")
        return "cuda"
    print("  ⚠ No CUDA GPU detected. Training will run on CPU.")
    print("    For a small dataset this is feasible but slow (20-50× slower).")
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset config repair
# ─────────────────────────────────────────────────────────────────────────────

def resolve_dataset_paths(yaml_path: Path) -> Path:
    """Rewrite data.yaml with absolute paths.

    Roboflow exports use ``train: ../train/images`` relative paths that
    assume the yaml lives one level deep in a parent directory. This
    function tries three resolution conventions per split and uses the
    first one that exists. Saves a new yaml alongside the original and
    returns its path.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    parent = yaml_path.parent.resolve()
    out = parent / "data_abs.yaml"

    new_cfg = dict(cfg)
    for split in ("train", "val", "test"):
        if split not in cfg:
            continue
        rel = cfg[split]
        # Three conventions:
        #   (a) strip leading "../" components, look in yaml's own dir
        #   (b) literal join (yaml_parent + rel)
        #   (c) one level up + path
        rel_clean = rel.lstrip("./")
        rel_basename = rel
        while rel_basename.startswith("../"):
            rel_basename = rel_basename[3:]

        candidates = [
            (parent / rel_basename).resolve(),
            (parent / rel).resolve(),
            (parent.parent / rel_clean).resolve(),
        ]

        abs_path = next((c for c in candidates if c.exists() and c.is_dir()), None)

        if abs_path is not None:
            new_cfg[split] = str(abs_path)
            print(f"  {split:>5}: {abs_path}")
        else:
            print(f"  warn: {split} not found at any of:")
            for c in candidates:
                print(f"         {c}")
            new_cfg.pop(split, None)

    if "val" not in new_cfg:
        raise SystemExit("data.yaml has no usable val split — cannot train.")

    with open(out, "w") as f:
        yaml.safe_dump(new_cfg, f, sort_keys=False)
    print(f"  wrote: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch application
# ─────────────────────────────────────────────────────────────────────────────

def apply_dpu_monkey_patches() -> None:
    """Replace C2PSA → C2PSA_DPU and DWConv → Conv in Ultralytics namespaces.

    Must be called BEFORE any ``YOLO()`` or ``DetectionModel()`` construction.
    """
    import ultralytics.nn.tasks
    import ultralytics.nn.modules
    import ultralytics.nn.modules.block

    from lpr_pipeline.c2psa_dpu import C2PSA_DPU
    from lpr_pipeline.detect_dpu import apply_dwconv_monkey_patch

    # C2PSA in three places (different files import it differently).
    ultralytics.nn.tasks.C2PSA = C2PSA_DPU
    ultralytics.nn.modules.C2PSA = C2PSA_DPU
    ultralytics.nn.modules.block.C2PSA = C2PSA_DPU
    print("  patched: ultralytics.{tasks,modules,modules.block}.C2PSA → C2PSA_DPU")

    apply_dwconv_monkey_patch()
    print("  patched: ultralytics.nn.modules.head.DWConv → Conv")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, required=True,
                        help="Path to YOLOv11n .pt or .yaml to start from")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to dataset's data.yaml (Ultralytics format)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to copy the trained best.pt when done "
                             "(parent dir created if missing)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16, fits 8GB VRAM at imgsz=640)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--lr0", type=float, default=1e-3,
                        help="Initial learning rate (default: 1e-3, fine-tune-friendly)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early-stop patience in epochs (default: 20)")
    parser.add_argument("--workdir", type=Path, default=Path("./runs/train_dpu"),
                        help="Where Ultralytics writes intermediates "
                             "(default: ./runs/train_dpu)")
    parser.add_argument("--name", default="exp",
                        help="Experiment subfolder name (default: exp)")
    args = parser.parse_args()

    print("=" * 70)
    print("  YOLOv11n DPU-friendly training")
    print("=" * 70)
    print(f"  weights : {args.weights}")
    print(f"  data    : {args.data}")
    print(f"  output  : {args.output}")
    print(f"  epochs  : {args.epochs}")
    print(f"  batch   : {args.batch}")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  lr0     : {args.lr0}")
    print(f"  workdir : {args.workdir}/{args.name}")
    print()

    if not args.weights.exists() and not str(args.weights).endswith(".yaml"):
        raise SystemExit(f"weights file not found: {args.weights}\n"
                         f"  (note: .yaml is also accepted to train from scratch)")
    if not args.data.exists():
        raise SystemExit(f"data.yaml not found: {args.data}")

    # ─── monkey-patches FIRST ───────────────────────────────────────────
    print("─── Applying DPU-friendly monkey-patches ──────────────────────────")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        apply_dpu_monkey_patches()
    print()

    # ─── device ─────────────────────────────────────────────────────────
    print("─── Detecting compute device ──────────────────────────────────────")
    device = detect_device()
    print()

    # ─── dataset paths ──────────────────────────────────────────────────
    print("─── Preparing dataset config ──────────────────────────────────────")
    data_yaml = resolve_dataset_paths(args.data)
    print()

    # ─── load + patch verification ──────────────────────────────────────
    print("─── Loading Ultralytics ───────────────────────────────────────────")
    from ultralytics import YOLO
    m = YOLO(str(args.weights))
    n_params = sum(p.numel() for p in m.model.parameters())
    print(f"  loaded: {args.weights.name}  ({n_params:,} params)")
    print()

    # ─── train ──────────────────────────────────────────────────────────
    print("─── Launching trainer ─────────────────────────────────────────────")
    print(f"  Note: Ultralytics' trainer rebuilds the model from YAML during")
    print(f"  setup_model(). Our monkey-patches take effect there, so the")
    print(f"  trained model will have C2PSA_DPU at index 10 and plain Conv")
    print(f"  blocks (not DWConv) in the Detect head's cv3 branch.")
    print()

    m.train(
        data         = str(data_yaml),
        epochs       = args.epochs,
        batch        = args.batch,
        imgsz        = args.imgsz,
        lr0          = args.lr0,
        device       = device,
        project      = str(args.workdir),
        name         = args.name,
        patience     = args.patience,
        exist_ok     = True,
        plots        = True,
        save         = True,
        save_period  = 10,
        verbose      = True,
        warmup_epochs= 1.0,
        cos_lr       = True,
        mosaic       = 0.5,
        mixup        = 0.0,
        freeze       = None,
    )

    # ─── copy best.pt to the requested output location ──────────────────
    # Ultralytics' save_dir handling is unreliable — depending on the
    # version, the cwd, and the project name, best.pt can end up in any
    # of half a dozen places. Rather than guessing, glob for it across
    # the likely roots and pick the most recent match.
    print()
    print("─── Locating best.pt and copying to --output ──────────────────────")

    search_roots = [
        Path("runs"),                                # cwd-relative
        args.workdir,                                # explicit --workdir
        args.workdir.parent,                          # one above (Ultralytics quirk)
        Path.home() / ".pyenv" / "runs",             # pyenv-shimmed user homedir
        Path.cwd() / "runs",                          # explicit absolute
    ]

    found = []
    for root in search_roots:
        if root.exists():
            # Recursive glob; restrict to paths matching '*/<name>/weights/best.pt'
            # to avoid picking up unrelated files in the search roots.
            for hit in root.rglob(f"*/{args.name}/weights/best.pt"):
                found.append(hit)

    # Deduplicate by resolved path and pick the most recent.
    found = sorted({p.resolve(): p for p in found}.values(),
                   key=lambda p: p.stat().st_mtime, reverse=True)

    if not found:
        print(f"  ✗ best.pt not found. Searched recursively under:")
        for r in search_roots:
            marker = "(exists)" if r.exists() else "(missing)"
            print(f"         {r} {marker}")
        print(f"  Look manually under runs/ or ~/.pyenv/runs/ — "
              f"if you find it, copy it to {args.output} yourself.")
        sys.exit(1)

    best_pt = found[0]
    if len(found) > 1:
        print(f"  found {len(found)} candidates; picking most recent:")
        for p in found[:5]:
            print(f"    {p}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, args.output)
    print(f"  source: {best_pt}")
    print(f"  copied: {args.output}")
    print()

    # ─── final verification ─────────────────────────────────────────────
    print("─── Verifying saved model has DPU-friendly architecture ──────────")
    m2 = YOLO(str(args.output))
    n_params_final = sum(p.numel() for p in m2.model.parameters())
    layer10 = m2.model.model[10]
    detect = m2.model.model[23]
    n_dwconv = sum(1 for mod in detect.modules() if type(mod).__name__ == "DWConv")

    print(f"  saved model params: {n_params_final:,}")
    print(f"  layer 10 type:      {type(layer10).__name__}")
    print(f"  DWConv in Detect:   {n_dwconv}")

    if type(layer10).__name__ == "C2PSA_DPU" and n_dwconv == 0:
        print(f"  ✓ DPU-friendly architecture confirmed")
    else:
        print(f"  ✗ WARNING: monkey-patches may not have held")
        print(f"    Expected: layer 10 = C2PSA_DPU, DWConv count = 0")
        sys.exit(2)

    print()
    print("=" * 70)
    print("  Training complete")
    print("=" * 70)
    print(f"  Next step — compile:")
    print(f"    NUM_CLASSES=<your_nc> bash scripts/host/02_compile.sh \\")
    print(f"        yolov11 yolov11n {args.output} <calib_dir>")


if __name__ == "__main__":
    main()
