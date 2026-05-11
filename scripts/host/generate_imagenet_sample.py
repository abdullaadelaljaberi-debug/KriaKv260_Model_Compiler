#!/usr/bin/env python3
"""
generate_imagenet_sample.py — Reproducible ImageNet benchmark sample.

Generates a small, reproducible ImageNet-style image sample from a Kaggle
archive (or any folder-per-class ImageNet-derived zip). The sample is used
by 04_vai35_benchmark.ipynb's classification accuracy loop.

Source archive convention:
    A .zip with subfolders named after class names, each containing images
    of that class. Common compatible Kaggle datasets:
      - "ImageNet-Mini"  (https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000)
      - "ImageNet-1k Mini" and similar one-folder-per-class repackages

    The archive's per-class folder names should match (or normalise to)
    the human-readable class names in imagenet_class_index.json. Names
    are compared case-insensitively with spaces/hyphens/apostrophes
    treated as equivalent.

Output:
    <target>/images/img_NNNN_<classname>.<ext>     (sampled images)
    <target>/labels.txt                            (format C: "filename classname")

The output directory layout is exactly what 04_vai35_benchmark.ipynb's
load_imagenet_dataset() expects.

Reproducibility:
    Random sampling uses a fixed seed (default 42). The same archive +
    same N + same seed produces an identical sample every time.

Usage:
    python3 generate_imagenet_sample.py \\
        --archive ~/Downloads/archive.zip \\
        --output ./Datasets/imagenet_sample \\
        --n 500 \\
        --seed 42

After staging locally, sync to the Kria with rsync (do this from your laptop):
    rsync -avh --copy-links --delete --exclude='.ipynb_checkpoints' \\
        ./Datasets/imagenet_sample/ \\
        ubuntu@<kria-ip>:/home/ubuntu/KriaKv260_Model_Compiler/notebooks/Datasets/imagenet_sample/
"""
import argparse
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path


IMAGE_EXTS = ('.png', '.jpg', '.jpeg')


def process_archive(zip_path: Path, target_dir: Path,
                    sample_size: int = 500, seed: int = 42):
    """Extract archive, sample N images at random, write image+labels output."""
    extract_dir = target_dir.parent / "_temp_archive_extract"
    images_out_dir = target_dir / "images"
    labels_file = target_dir / "labels.txt"

    # --- 1. Extract archive ---
    if not zip_path.exists():
        raise SystemExit(f"Archive not found: {zip_path}")
    if not extract_dir.exists():
        print(f"[1/5] Extracting {zip_path.name} to {extract_dir} ...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
    else:
        print(f"[1/5] Reusing existing extraction at {extract_dir}")

    # --- 2. Walk for images ---
    print(f"[2/5] Scanning for images ...")
    all_images = []
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if fname.lower().endswith(IMAGE_EXTS):
                class_name = os.path.basename(root)
                # Skip the extract-dir level (no real class folder)
                if class_name in ('', os.path.basename(extract_dir)):
                    continue
                all_images.append({
                    'path': os.path.join(root, fname),
                    'class_name': class_name,
                })
    n_classes = len({x['class_name'] for x in all_images})
    print(f"      Found {len(all_images):,} images across {n_classes} class folders.")
    if not all_images:
        raise SystemExit("No images found. Check the archive structure.")

    if len(all_images) < sample_size:
        print(f"      WARNING: only {len(all_images)} images available; "
              f"reducing sample size from {sample_size}.")
        sample_size = len(all_images)

    # --- 3. Random sample with fixed seed ---
    print(f"[3/5] Sampling {sample_size} images (random.seed={seed}) ...")
    rng = random.Random(seed)
    sampled = rng.sample(all_images, sample_size)

    # --- 4. Copy + write labels.txt (format C) ---
    print(f"[4/5] Building dataset in {target_dir} ...")
    images_out_dir.mkdir(parents=True, exist_ok=True)
    n_seen_classes = set()
    with open(labels_file, 'w') as f:
        for i, img in enumerate(sampled):
            ext = os.path.splitext(img['path'])[1]
            new_name = f"img_{i:04d}_{img['class_name']}{ext}"
            shutil.copy2(img['path'], images_out_dir / new_name)
            f.write(f"{new_name} {img['class_name']}\n")
            n_seen_classes.add(img['class_name'])
    print(f"      Wrote {sample_size} files, {len(n_seen_classes)} unique classes")

    # --- 5. Cleanup ---
    print(f"[5/5] Removing temporary extraction directory ...")
    shutil.rmtree(extract_dir)

    print()
    print(f"Done. Dataset ready at: {target_dir}")
    print(f"  Images : {images_out_dir} ({sample_size} files)")
    print(f"  Labels : {labels_file} ({sample_size} lines, format C)")
    print(f"  Sample : {sample_size} of {len(all_images):,} ({100*sample_size/len(all_images):.2f}%)")
    print()
    print("Next: rsync this directory to the Kria — see the docstring at the top.")


def main():
    p = argparse.ArgumentParser(
        description="Generate a reproducible ImageNet sample for benchmark accuracy testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else None,
    )
    p.add_argument('--archive', required=True, type=Path,
                   help='Path to source archive.zip')
    p.add_argument('--output', required=True, type=Path,
                   help='Output dataset directory (e.g. ./Datasets/imagenet_sample)')
    p.add_argument('--n', type=int, default=500,
                   help='Number of images to sample (default 500)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility (default 42)')
    args = p.parse_args()

    process_archive(args.archive, args.output, args.n, args.seed)


if __name__ == '__main__':
    main()
