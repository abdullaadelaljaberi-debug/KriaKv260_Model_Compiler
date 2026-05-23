#!/usr/bin/env python3
"""
Calibration set generator for the Kria KV260 thesis pipeline.

NNDCT post-training quantization needs a small set of representative images
(typically 100-1000) to compute per-tensor activation scales. From our prior
work, IN-DOMAIN calibration (drawing samples from the same distribution as
the deployment target) consistently outperforms mixed or out-of-domain sets.

This script produces one calibration directory per dataset:
    data/calib/
    ├── detection/
    │   ├── bstld/             # 200 images from BSTLD training set
    │   ├── license_plates/    # 200 images from LPR training set
    │   └── vineset/           # 200 images from VineSet training set
    └── classification/
        ├── gtsrb/             # 200 images from GTSRB training set
        └── oxford_pets/       # 200 images from Oxford Pets training set

The same directory is used by every model targeting that dataset. So the
NNDCT compile step for resnet50_oxford_pets, mobilenetv2_oxford_pets,
and inceptionv3_oxford_pets all read from data/calib/classification/oxford_pets/.

Usage:
    python3 scripts/host/prep_calibration.py --all
    python3 scripts/host/prep_calibration.py --dataset bstld
    python3 scripts/host/prep_calibration.py --all --n 300       # override default 200
    python3 scripts/host/prep_calibration.py --all --seed 1337   # different seed
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data"

DETECTION_DATASETS = ["bstld", "license_plates", "vineset"]
CLASSIFICATION_DATASETS = ["gtsrb", "oxford_pets"]
ALL_DATASETS = DETECTION_DATASETS + CLASSIFICATION_DATASETS

DEFAULT_N = 200

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_step(msg: str) -> None:
    print(f"\n{'='*72}\n>>> {msg}\n{'='*72}")

def log_info(msg: str) -> None:
    print(f"    {msg}")

def log_err(msg: str) -> None:
    print(f"    ERROR: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".webp")


def find_detection_images(dataset: str) -> List[Path]:
    """Return list of image paths in the detection dataset's training split."""
    img_dir = DATA_ROOT / "datasets" / "detection" / dataset / "train" / "images"
    if not img_dir.exists():
        return []
    return [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]


def find_classification_images(dataset: str) -> List[Path]:
    """Return list of image paths across all classes in the classification dataset's training split."""
    train_dir = DATA_ROOT / "datasets" / "classification" / dataset / "train"
    if not train_dir.exists():
        return []
    images: List[Path] = []
    for class_dir in train_dir.iterdir():
        if not class_dir.is_dir():
            continue
        for p in class_dir.iterdir():
            if p.suffix.lower() in IMAGE_EXTS:
                images.append(p)
    return images


# ---------------------------------------------------------------------------
# Sampling and copy
# ---------------------------------------------------------------------------

def build_calibration_set(dataset: str, n: int, seed: int) -> bool:
    """Sample n images from the dataset and copy them to data/calib/<task>/<dataset>/.

    Returns True if at least 1 image was placed; False if the source dataset
    was missing or empty.
    """
    if dataset in DETECTION_DATASETS:
        source_images = find_detection_images(dataset)
        task = "detection"
    elif dataset in CLASSIFICATION_DATASETS:
        source_images = find_classification_images(dataset)
        task = "classification"
    else:
        log_err(f"unknown dataset: {dataset}")
        return False

    if not source_images:
        log_err(f"no training images found for {dataset} — has prep_datasets.py been run?")
        return False

    out_dir = DATA_ROOT / "calib" / task / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear any previously-built calibration set so we get a fresh sample.
    # This makes the script idempotent across different --seed values.
    for old in out_dir.iterdir():
        if old.is_file():
            old.unlink()

    # Deterministic sampling
    rng = random.Random(seed)
    n_sample = min(n, len(source_images))
    if n_sample < n:
        log_info(f"  WARNING: only {n_sample} images available (requested {n})")
    sampled = rng.sample(source_images, n_sample)

    for src in sampled:
        # Use a flat naming scheme so the NNDCT calibration loop can just
        # glob('*') over the directory.
        dest = out_dir / src.name
        # Handle name collisions (rare; classification datasets put images
        # in per-class subdirectories so two classes may have same filename)
        if dest.exists():
            dest = out_dir / f"{src.parent.name}_{src.name}"
        shutil.copy(src, dest)

    log_info(f"  {dataset}: {n_sample} images -> {out_dir}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                       help="build calibration sets for every dataset")
    group.add_argument("--dataset", choices=ALL_DATASETS,
                       help="build a calibration set for one dataset")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help=f"number of images per calibration set (default {DEFAULT_N})")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for deterministic sampling (default 42)")
    args = ap.parse_args()

    if args.all:
        selected = ALL_DATASETS
    else:
        selected = [args.dataset]

    log_step(f"Building calibration sets (n={args.n}, seed={args.seed})")
    success = 0
    failed = 0
    for dataset in selected:
        if build_calibration_set(dataset, args.n, args.seed):
            success += 1
        else:
            failed += 1

    log_step("Summary")
    log_info(f"  {success} calibration set(s) built successfully")
    if failed > 0:
        log_info(f"  {failed} failed (dataset not prepared?)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
