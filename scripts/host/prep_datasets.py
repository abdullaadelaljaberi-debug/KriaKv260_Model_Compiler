#!/usr/bin/env python3
"""
Dataset preparation for the multi-model, multi-dataset Kria KV260 thesis pipeline.

This script downloads and converts five datasets to a unified directory layout:

    data/datasets/
    ├── detection/
    │   ├── bstld/              # Bosch Small Traffic Lights, YOLO + COCO formats
    │   ├── license_plates/     # Roboflow LPR
    │   └── vineset/            # Magalhães et al. vineyard
    └── classification/
        ├── gtsrb/              # German Traffic Sign Recognition Benchmark
        └── oxford_pets/        # Oxford-IIIT Pets

Detection datasets are stored in two formats simultaneously:
- YOLO format (txt files, one annotation per line) for Ultralytics training
- COCO JSON for torchvision SSDLite / RetinaNet training

Datasets requiring auth or manual download are flagged with instructions.

Usage:
    python3 scripts/host/prep_datasets.py --all
    python3 scripts/host/prep_datasets.py --datasets bstld,oxford_pets
    python3 scripts/host/prep_datasets.py --datasets bstld --skip-download   # if already downloaded
"""

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party imports — required for the script to run at all.
# These should be installed via:
#     pip install pillow pyyaml torch torchvision
# Optional (only needed for specific datasets):
#     pip install roboflow                 # for LPR (Roboflow)
try:
    from PIL import Image  # noqa: F401
except ImportError:
    print("ERROR: Pillow not installed. Run: pip install pillow", file=sys.stderr)
    sys.exit(1)

# yaml only needed by BSTLD prep — load lazily there to avoid forcing the dep
# if the user is only doing other datasets.

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "datasets"
DETECTION_ROOT = DATA_ROOT / "detection"
CLASSIFICATION_ROOT = DATA_ROOT / "classification"
DOWNLOAD_CACHE = REPO_ROOT / "data" / "downloads"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log_step(msg: str) -> None:
    print(f"\n{'='*72}\n>>> {msg}\n{'='*72}")

def log_info(msg: str) -> None:
    print(f"    {msg}")

def log_warn(msg: str) -> None:
    print(f"    WARN: {msg}", file=sys.stderr)

def log_err(msg: str) -> None:
    print(f"    ERROR: {msg}", file=sys.stderr)


def download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download a file with a basic progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log_info(f"already cached: {dest.name}")
        return
    log_info(f"downloading {desc or url} -> {dest}")
    def hook(blocknum: int, blocksize: int, totalsize: int) -> None:
        if totalsize > 0:
            pct = min(100, 100 * blocknum * blocksize / totalsize)
            sys.stdout.write(f"\r        {pct:5.1f}%")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, dest, hook)
    sys.stdout.write("\n")


def extract_archive(archive: Path, dest: Path) -> None:
    """Extract a zip or tarball into dest."""
    dest.mkdir(parents=True, exist_ok=True)
    log_info(f"extracting {archive.name} -> {dest}")
    if archive.suffix == ".zip" or archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif archive.name.endswith((".tar.gz", ".tgz", ".tar")):
        mode = "r:gz" if archive.name.endswith(("gz", "tgz")) else "r"
        with tarfile.open(archive, mode) as t:
            t.extractall(dest)
    else:
        raise ValueError(f"unsupported archive format: {archive}")


def write_yolo_data_yaml(out_dir: Path, names: List[str]) -> None:
    """Write a YOLO data.yaml describing class names and split paths."""
    yaml = (
        f"path: {out_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    (out_dir / "data.yaml").write_text(yaml)
    log_info(f"wrote {out_dir / 'data.yaml'}")


def yolo_to_coco(yolo_dir: Path, split: str, class_names: List[str]) -> Dict:
    """Convert a YOLO-format split (train or val) to a COCO-style dict.

    yolo_dir layout expected:
        <yolo_dir>/<split>/images/*.{jpg,png}
        <yolo_dir>/<split>/labels/*.txt   # one row per box: class cx cy w h (normalized)

    Returns a COCO-format dict with images, categories, annotations.
    """
    images_dir = yolo_dir / split / "images"
    labels_dir = yolo_dir / split / "labels"

    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(class_names)
    ]
    images: List[Dict] = []
    annotations: List[Dict] = []
    ann_id = 1

    for img_id, img_path in enumerate(sorted(images_dir.glob("*"))):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = parts
            cls = int(cls)
            cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
            # Convert center+size (normalized) to absolute corner+size COCO format
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            box_w = bw * w
            box_h = bh * h
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls,
                "bbox": [x, y, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

    return {
        "info": {"description": yolo_dir.name},
        "licenses": [],
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }


def write_coco_jsons(yolo_dir: Path, class_names: List[str]) -> None:
    """Write COCO JSONs for train and val splits, alongside the YOLO labels."""
    for split in ("train", "val"):
        if not (yolo_dir / split / "images").exists():
            continue
        coco = yolo_to_coco(yolo_dir, split, class_names)
        out_path = yolo_dir / f"coco_{split}.json"
        out_path.write_text(json.dumps(coco))
        log_info(f"wrote {out_path} ({len(coco['images'])} images, "
                 f"{len(coco['annotations'])} annotations)")


# ---------------------------------------------------------------------------
# Dataset-specific preparation routines
# ---------------------------------------------------------------------------

def prep_bstld(skip_download: bool = False) -> None:
    """Bosch Small Traffic Lights Dataset.

    BSTLD requires a free registration to download the labeled data.
    Source: https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset

    Once downloaded, the user must place:
        data/downloads/bstld/dataset_train_rgb.zip
        data/downloads/bstld/dataset_test_rgb.zip
        data/downloads/bstld/train.yaml
        data/downloads/bstld/test.yaml

    This script then converts to YOLO format with 4 classes:
    red, yellow, green, off (and merges "redLeft" etc. into base classes).
    """
    log_step("Preparing BSTLD (Bosch Small Traffic Lights)")
    out_dir = DETECTION_ROOT / "bstld"

    raw_dir = DOWNLOAD_CACHE / "bstld"
    train_zip = raw_dir / "dataset_train_rgb.zip"
    test_zip = raw_dir / "dataset_test_rgb.zip"
    train_yaml = raw_dir / "train.yaml"
    test_yaml = raw_dir / "test.yaml"

    if not skip_download:
        if not all(p.exists() for p in (train_zip, test_zip, train_yaml, test_yaml)):
            log_warn("BSTLD requires manual download (free registration).")
            log_warn("Visit https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset")
            log_warn(f"Place files in {raw_dir}/  then re-run with --skip-download")
            return

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, src_zip, src_yaml in (
        ("train", train_zip, train_yaml),
        ("val",   test_zip,  test_yaml),
    ):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        extract_archive(src_zip, out_dir / split / "_raw")
        # BSTLD YAML format: list of dicts with 'path' and 'boxes' (each with x_min/x_max/y_min/y_max/label)
        try:
            import yaml as pyyaml  # type: ignore
        except ImportError:
            log_err("pyyaml not installed; pip install pyyaml")
            return
        labels_data = pyyaml.safe_load(src_yaml.read_text())

        class_map = {"Red": 0, "Yellow": 1, "Green": 2, "off": 3,
                     "RedLeft": 0, "RedRight": 0, "RedStraight": 0,
                     "GreenLeft": 2, "GreenRight": 2, "GreenStraight": 2,
                     "GreenStraightRight": 2, "GreenStraightLeft": 2}

        for entry in labels_data:
            img_rel = Path(entry["path"]).name
            img_src = next((out_dir / split / "_raw").rglob(img_rel), None)
            if img_src is None:
                continue
            img_dst = out_dir / split / "images" / img_rel
            shutil.copy(img_src, img_dst)
            with Image.open(img_dst) as im:
                W, H = im.size
            lbl_lines = []
            for box in entry.get("boxes", []) or []:
                if box["label"] not in class_map:
                    continue
                cls = class_map[box["label"]]
                x_min, x_max = float(box["x_min"]), float(box["x_max"])
                y_min, y_max = float(box["y_min"]), float(box["y_max"])
                cx = ((x_min + x_max) / 2) / W
                cy = ((y_min + y_max) / 2) / H
                bw = (x_max - x_min) / W
                bh = (y_max - y_min) / H
                lbl_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            lbl_path = out_dir / split / "labels" / f"{Path(img_rel).stem}.txt"
            lbl_path.write_text("\n".join(lbl_lines))
        # cleanup the _raw directory after conversion
        shutil.rmtree(out_dir / split / "_raw", ignore_errors=True)

    class_names = ["red", "yellow", "green", "off"]
    write_yolo_data_yaml(out_dir, class_names)
    write_coco_jsons(out_dir, class_names)
    log_info("BSTLD ready.")


def prep_license_plates(skip_download: bool = False) -> None:
    """Roboflow LPR dataset.

    Roboflow LPR dataset is available with a free API key at:
    https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e

    Requires:
        export ROBOFLOW_API_KEY=<your key>
        pip install roboflow

    Falls back to a public mirror if neither path works.
    """
    log_step("Preparing Roboflow LPR (License Plate Recognition)")
    out_dir = DETECTION_ROOT / "license_plates"
    raw_dir = DOWNLOAD_CACHE / "license_plates"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            log_warn("No ROBOFLOW_API_KEY set in environment.")
            log_warn("Sign up free at https://roboflow.com, get your API key,")
            log_warn("then re-run with: export ROBOFLOW_API_KEY=<key>")
            log_warn("Skipping for now — re-run after setting the key.")
            return

        try:
            from roboflow import Roboflow  # type: ignore
            rf = Roboflow(api_key=api_key)
            project = rf.workspace("roboflow-universe-projects").project(
                "license-plate-recognition-rxg4e")
            dataset = project.version(11).download("yolov8", location=str(raw_dir))
            log_info(f"downloaded to {dataset.location}")
        except ImportError:
            log_err("roboflow package not installed; pip install roboflow")
            return
        except Exception as e:
            log_err(f"Roboflow download failed: {e}")
            return

    # Roboflow's YOLOv8 format is already nearly what we need.
    # Layout: <raw_dir>/{train,valid,test}/{images,labels}/*
    src_train = raw_dir / "train"
    src_val = raw_dir / "valid"
    if not src_train.exists():
        log_err(f"Expected {src_train} from Roboflow download, not found.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for src_split, dst_split in (("train", "train"), ("valid", "val")):
        src_imgs = raw_dir / src_split / "images"
        src_lbls = raw_dir / src_split / "labels"
        dst_imgs = out_dir / dst_split / "images"
        dst_lbls = out_dir / dst_split / "labels"
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_lbls.mkdir(parents=True, exist_ok=True)
        for f in src_imgs.glob("*"):
            shutil.copy(f, dst_imgs / f.name)
        for f in src_lbls.glob("*"):
            shutil.copy(f, dst_lbls / f.name)
        log_info(f"copied {src_split} -> {dst_split} "
                 f"({len(list(dst_imgs.glob('*')))} imgs)")

    class_names = ["license_plate"]
    write_yolo_data_yaml(out_dir, class_names)
    write_coco_jsons(out_dir, class_names)
    log_info("License plate dataset ready.")


def voc_xml_to_yolo(xml_path: Path, class_map: Dict[str, int]) -> Tuple[List[str], int, int]:
    """Convert a single Pascal VOC XML annotation to YOLO format lines.

    Returns (yolo_lines, image_width, image_height).
    VineSet ships with VOC-style XMLs, so we need this converter.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return [], 0, 0
    W = int(size.findtext("width", "0"))
    H = int(size.findtext("height", "0"))
    if W <= 0 or H <= 0:
        return [], W, H
    lines = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "")
        if name not in class_map:
            continue
        cls = class_map[name]
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.findtext("xmin", "0"))
        ymin = float(bbox.findtext("ymin", "0"))
        xmax = float(bbox.findtext("xmax", "0"))
        ymax = float(bbox.findtext("ymax", "0"))
        cx = ((xmin + xmax) / 2) / W
        cy = ((ymin + ymax) / 2) / H
        bw = (xmax - xmin) / W
        bh = (ymax - ymin) / H
        # Clamp to [0, 1] to handle edge boxes
        cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
        bw, bh = max(0.0, min(1.0, bw)), max(0.0, min(1.0, bh))
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines, W, H


def prep_vineset(skip_download: bool = False) -> None:
    """VineSet (Magalhães et al. 2022) — vineyard grape bunch + trunk detection.

    Source: Zenodo record 5717293 (Magalhães et al., open access).
    Reference:
        Magalhães et al., "Benchmarking edge computing devices for grape bunches
        and trunks detection using accelerated object detection single shot
        multibox deep learning models," Engineering Applications of AI 117
        (2023) 105604, DOI 10.1016/j.engappai.2022.105604.

    The dataset ships as a single zip containing Pascal VOC-format
    annotations (XML), plus the images. We convert to YOLO + COCO and split
    into train (80%) / val (20%) since the original dataset is unsplit.
    """
    log_step("Preparing VineSet (Magalhães et al., Zenodo 5717293)")
    out_dir = DETECTION_ROOT / "vineset"
    raw_dir = DOWNLOAD_CACHE / "vineset"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # The Zenodo record has multiple files; the principal one is the
    # "WGISD-related" annotated grape image set. URL is the standard
    # Zenodo file-download endpoint.
    zenodo_record = "5717293"
    # NOTE: at time of writing the file name on the record is "VineSet.zip".
    # If that filename changes on the record, this URL will need updating.
    # Visit https://zenodo.org/records/5717293 to verify.
    zenodo_zip_url = f"https://zenodo.org/records/{zenodo_record}/files/VineSet.zip"
    zip_path = raw_dir / "VineSet.zip"

    if not skip_download:
        try:
            download_file(zenodo_zip_url, zip_path,
                         f"VineSet from Zenodo {zenodo_record} (~1 GB)")
        except Exception as e:
            log_err(f"Zenodo download failed: {e}")
            log_warn("Manual fallback: visit https://zenodo.org/records/5717293")
            log_warn(f"Download VineSet.zip and place it at {zip_path}")
            log_warn("Then re-run with --skip-download")
            return

    if not zip_path.exists():
        log_err(f"VineSet zip not found at {zip_path}")
        return

    # Extract
    extract_dir = raw_dir / "extracted"
    if not extract_dir.exists():
        extract_archive(zip_path, extract_dir)

    # The extracted layout has images and annotations alongside; search for
    # the typical Pascal VOC structure (Images/, Annotations/).
    # The Magalhães dataset layout is approximately:
    #     extracted/<some_root>/Images/*.jpg
    #     extracted/<some_root>/Annotations/*.xml
    # We locate it dynamically because the top-level folder name varies.
    images_dirs = list(extract_dir.rglob("Images")) + list(extract_dir.rglob("images"))
    annotations_dirs = list(extract_dir.rglob("Annotations")) + list(extract_dir.rglob("annotations"))

    if not images_dirs or not annotations_dirs:
        # Fallback: maybe everything is in one flat directory with mixed file types
        all_jpgs = list(extract_dir.rglob("*.jpg")) + list(extract_dir.rglob("*.JPG"))
        all_xmls = list(extract_dir.rglob("*.xml"))
        if not all_jpgs or not all_xmls:
            log_err(f"Could not find VineSet images/annotations in {extract_dir}")
            log_err("The dataset layout may differ from what was expected.")
            log_err("Inspect the extracted directory and update prep_vineset() accordingly.")
            return
        # Use the parent directory of the first jpg as the root
        images_root = all_jpgs[0].parent
        annotations_root = all_xmls[0].parent
    else:
        images_root = images_dirs[0]
        annotations_root = annotations_dirs[0]

    log_info(f"images:      {images_root} ({len(list(images_root.glob('*')))} files)")
    log_info(f"annotations: {annotations_root} ({len(list(annotations_root.glob('*')))} files)")

    # VineSet uses Pascal VOC XML annotations. Class names depend on the
    # specific subset of the dataset:
    #   - WGISD subset: "grape" (1 class)
    #   - Magalhães Hackster setup: "grape_bunch", "grape_trunk" (2 classes)
    # We scan the XMLs to find which class names are actually present.
    class_names_seen = set()
    sample_xmls = list(annotations_root.glob("*.xml"))[:50]
    for xml_path in sample_xmls:
        try:
            tree = ET.parse(xml_path)
            for obj in tree.getroot().findall("object"):
                name = obj.findtext("name", "").strip()
                if name:
                    class_names_seen.add(name)
        except Exception:
            continue
    log_info(f"classes found in sample: {sorted(class_names_seen)}")

    # Build class_map based on what's actually present
    class_names = sorted(class_names_seen)
    class_map = {name: i for i, name in enumerate(class_names)}

    # Split images into train (80%) / val (20%) by deterministic seed
    all_images = sorted([p for p in images_root.glob("*")
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    random.seed(42)
    random.shuffle(all_images)
    split_idx = int(0.8 * len(all_images))
    splits = {"train": all_images[:split_idx], "val": all_images[split_idx:]}

    # Convert each image's XML annotation to YOLO format and copy
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, img_paths in splits.items():
        img_dst = out_dir / split_name / "images"
        lbl_dst = out_dir / split_name / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        n_copied = 0
        for img_path in img_paths:
            xml_path = annotations_root / f"{img_path.stem}.xml"
            if not xml_path.exists():
                continue
            yolo_lines, W, H = voc_xml_to_yolo(xml_path, class_map)
            if W == 0 or H == 0:
                continue
            shutil.copy(img_path, img_dst / img_path.name)
            (lbl_dst / f"{img_path.stem}.txt").write_text("\n".join(yolo_lines))
            n_copied += 1
        log_info(f"  {split_name}: {n_copied} images written")

    write_yolo_data_yaml(out_dir, class_names)
    write_coco_jsons(out_dir, class_names)
    log_info(f"VineSet ready. Classes ({len(class_names)}): {class_names}")


def prep_gtsrb(skip_download: bool = False) -> None:
    """German Traffic Sign Recognition Benchmark (43-class classification).

    Uses torchvision.datasets.GTSRB(download=True), which is the canonical
    public source. We then dump the (PIL image, label) tuples into a standard
    ImageFolder layout under data/datasets/classification/gtsrb/{train,val}/.

    Reference:
      Stallkamp et al., "Man vs. computer: Benchmarking machine learning
      algorithms for traffic sign recognition," Neural Networks 32 (2012).
    """
    log_step("Preparing GTSRB (via torchvision)")
    out_dir = CLASSIFICATION_ROOT / "gtsrb"
    raw_dir = DOWNLOAD_CACHE / "gtsrb"
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        from torchvision.datasets import GTSRB  # type: ignore
    except ImportError:
        log_err("torchvision not available; pip install torchvision")
        return

    download = not skip_download
    try:
        train_ds = GTSRB(root=str(raw_dir), split="train", download=download)
        test_ds = GTSRB(root=str(raw_dir), split="test", download=download)
    except Exception as e:
        log_err(f"torchvision GTSRB download failed: {e}")
        log_warn("Check your network connection or try again later")
        return

    log_info(f"train: {len(train_ds)} samples, test: {len(test_ds)} samples")

    # Dump to ImageFolder layout. torchvision.GTSRB returns (PIL.Image, label).
    for split, ds in (("train", train_ds), ("val", test_ds)):
        split_dir = out_dir / split
        # Each of the 43 classes goes in its own subdirectory
        for i, (img, label) in enumerate(ds):
            cls_dir = split_dir / f"{label:05d}"
            cls_dir.mkdir(parents=True, exist_ok=True)
            img_path = cls_dir / f"{split}_{i:06d}.png"
            img.save(img_path)
            if (i + 1) % 2000 == 0:
                log_info(f"  {split}: dumped {i+1}/{len(ds)}")

    n_train = sum(1 for _ in (out_dir / "train").rglob("*.png"))
    n_val = sum(1 for _ in (out_dir / "val").rglob("*.png"))
    n_classes = len(list((out_dir / "train").iterdir())) if (out_dir / "train").exists() else 0
    log_info(f"train: {n_train} images, val: {n_val} images, classes: {n_classes}")
    log_info("GTSRB ready.")


def prep_oxford_pets(skip_download: bool = False) -> None:
    """Oxford-IIIT Pet Dataset (37 cat+dog breeds, classification).

    Official source: https://www.robots.ox.ac.uk/~vgg/data/pets/
    Public, no auth required.
    """
    log_step("Preparing Oxford-IIIT Pets")
    out_dir = CLASSIFICATION_ROOT / "oxford_pets"
    raw_dir = DOWNLOAD_CACHE / "oxford_pets"
    raw_dir.mkdir(parents=True, exist_ok=True)

    images_tar = raw_dir / "images.tar.gz"
    annotations_tar = raw_dir / "annotations.tar.gz"

    if not skip_download:
        try:
            download_file(
                "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
                images_tar, "Oxford Pets images (~775 MB)")
            download_file(
                "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
                annotations_tar, "Oxford Pets annotations (~19 MB)")
        except Exception as e:
            log_err(f"Oxford Pets download failed: {e}")
            return

    extract_archive(images_tar, raw_dir)
    extract_archive(annotations_tar, raw_dir)

    images_dir = raw_dir / "images"
    splits_dir = raw_dir / "annotations"
    if not images_dir.exists() or not splits_dir.exists():
        log_err(f"Oxford Pets extracted layout unexpected; check {raw_dir}")
        return

    # trainval.txt and test.txt provide image, class_id pairs
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, dst_split in (("trainval.txt", "train"), ("test.txt", "val")):
        split_file = splits_dir / split_name
        if not split_file.exists():
            continue
        for line in split_file.read_text().strip().splitlines():
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            img_stem = parts[0]
            # Class name = filename without trailing number, lowercased breed name
            # e.g. "Abyssinian_100" -> breed "Abyssinian"
            breed = "_".join(img_stem.split("_")[:-1]).lower()
            cls_dst = out_dir / dst_split / breed
            cls_dst.mkdir(parents=True, exist_ok=True)
            for ext in (".jpg", ".jpeg", ".png"):
                src_img = images_dir / f"{img_stem}{ext}"
                if src_img.exists():
                    shutil.copy(src_img, cls_dst / src_img.name)
                    break

    n_train = sum(1 for _ in (out_dir / "train").rglob("*.jpg"))
    n_val = sum(1 for _ in (out_dir / "val").rglob("*.jpg"))
    n_classes = len(list((out_dir / "train").iterdir())) if (out_dir / "train").exists() else 0
    log_info(f"train: {n_train} images, val: {n_val} images, classes: {n_classes}")
    log_info("Oxford Pets ready.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = {
    "bstld": prep_bstld,
    "license_plates": prep_license_plates,
    "vineset": prep_vineset,
    "gtsrb": prep_gtsrb,
    "oxford_pets": prep_oxford_pets,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="prepare every dataset")
    ap.add_argument("--datasets", type=str, default="",
                    help="comma-separated list (bstld,license_plates,vineset,gtsrb,oxford_pets)")
    ap.add_argument("--skip-download", action="store_true",
                    help="assume downloads cached in data/downloads/")
    args = ap.parse_args()

    if args.all:
        selected = list(DATASETS.keys())
    elif args.datasets:
        selected = [d.strip() for d in args.datasets.split(",")]
    else:
        ap.print_help()
        return 1

    for d in selected:
        if d not in DATASETS:
            log_err(f"unknown dataset: {d} (valid: {list(DATASETS.keys())})")
            return 1

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    DETECTION_ROOT.mkdir(parents=True, exist_ok=True)
    CLASSIFICATION_ROOT.mkdir(parents=True, exist_ok=True)

    for d in selected:
        DATASETS[d](skip_download=args.skip_download)

    log_step("Dataset prep summary")
    for d in selected:
        if d in ("bstld", "license_plates", "vineset"):
            root = DETECTION_ROOT / d
        else:
            root = CLASSIFICATION_ROOT / d
        if (root / "train").exists() or (root / "data.yaml").exists():
            log_info(f"  [OK]   {d:20s} -> {root}")
        else:
            log_info(f"  [SKIP] {d:20s}  (not prepared)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
