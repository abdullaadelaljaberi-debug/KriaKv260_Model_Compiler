#!/usr/bin/env python3
"""
Detection training driver for the Kria KV260 thesis pipeline.

Trains one of six detection variants on one of three datasets:
    YOLOv5n / YOLOv5s / YOLOv11n / YOLOv11s   -- via Ultralytics
    SSDLite-MobileNetV3                        -- via torchvision
    RetinaNet-ResNet50-FPN                     -- via torchvision

Each model trains on YOLO-format labels (Ultralytics) or COCO JSON (torchvision)
that prep_datasets.py has already produced under data/datasets/detection/<name>/.

Usage:
    python3 scripts/host/train_detection.py \\
        --model yolov11s --dataset bstld --epochs 50 --batch 16

    python3 scripts/host/train_detection.py \\
        --model ssdlite --dataset license_plates --epochs 60 --batch 32

Output:
    data/weights/detection/<model>_<dataset>.pt   (Ultralytics format)
    data/weights/detection/<model>_<dataset>.pth  (torchvision format)
    data/weights/detection/<model>_<dataset>.log  (training log)
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

# Third-party — required to run at all. Fail fast and clearly if missing.
try:
    import torch
    import torch.nn as nn
    import torch.utils.data
    import torchvision
    import torchvision.transforms.functional as TF
    from PIL import Image
except ImportError as e:
    print(f"ERROR: required dependency missing: {e}", file=sys.stderr)
    print("Install with: pip install torch torchvision pillow", file=sys.stderr)
    sys.exit(1)

# NOTE: ultralytics is NOT imported here. It must be imported AFTER the
# YOLOv11 monkey-patches are applied (see train_yolo() below). Importing it
# at module top would freeze the C2PSA and DWConv class references before
# we've had a chance to substitute them.

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data"
WEIGHTS_OUT = DATA_ROOT / "weights" / "detection"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YOLO_MODELS = {
    "yolov5n":  {"family": "yolov5",  "weights_url": "yolov5nu.pt", "imgsz": 640},
    "yolov5s":  {"family": "yolov5",  "weights_url": "yolov5su.pt", "imgsz": 640},
    "yolov11n": {"family": "yolov11", "weights_url": "yolo11n.pt",  "imgsz": 640},
    "yolov11s": {"family": "yolov11", "weights_url": "yolo11s.pt",  "imgsz": 640},
}

TORCHVISION_MODELS = {
    "ssdlite":   {"framework": "torchvision", "imgsz": 320},
    "retinanet": {"framework": "torchvision", "imgsz": 640},  # 640 not 800: matches YOLO inputs and fits 8GB GPU
}

ALL_MODELS = {**YOLO_MODELS, **TORCHVISION_MODELS}

DATASET_CONFIG = {
    "bstld": {
        "yaml": "data/datasets/detection/bstld/data.yaml",
        "coco_train": "data/datasets/detection/bstld/coco_train.json",
        "coco_val": "data/datasets/detection/bstld/coco_val.json",
        "img_root": "data/datasets/detection/bstld",
        "num_classes": 4,
    },
    "license_plates": {
        "yaml": "data/datasets/detection/license_plates/data.yaml",
        "coco_train": "data/datasets/detection/license_plates/coco_train.json",
        "coco_val": "data/datasets/detection/license_plates/coco_val.json",
        "img_root": "data/datasets/detection/license_plates",
        "num_classes": 1,
    },
    "vineset": {
        "yaml": "data/datasets/detection/vineset/data.yaml",
        "coco_train": "data/datasets/detection/vineset/coco_train.json",
        "coco_val": "data/datasets/detection/vineset/coco_val.json",
        "img_root": "data/datasets/detection/vineset",
        "num_classes": 2,
    },
}

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
# YOLO path (Ultralytics)
# ---------------------------------------------------------------------------

def train_yolo(model_name: str, dataset: str, epochs: int, batch: int,
               imgsz: int, seed: int, output: Path) -> int:
    """Train a YOLOv5 or YOLOv11 variant via Ultralytics."""
    family = YOLO_MODELS[model_name]["family"]
    pretrained = YOLO_MODELS[model_name]["weights_url"]

    log_step(f"Training {model_name} on {dataset} (Ultralytics, family={family})")
    log_info(f"epochs={epochs}, batch={batch}, imgsz={imgsz}, seed={seed}")

    # For YOLOv11 we MUST apply the C2PSA_DPU and DWConv->Conv monkey-patches
    # BEFORE the first YOLO() instantiation so that Ultralytics' DetectionTrainer
    # picks them up during model rebuild from YAML.
    if family == "yolov11":
        log_info("applying YOLOv11 DPU-friendly architectural surgery...")
        try:
            sys.path.insert(0, str(REPO_ROOT))
            # Apply C2PSA -> C2PSA_DPU substitution (HardSigmoid-gated conv)
            from lpr_pipeline.c2psa_dpu import C2PSA_DPU  # type: ignore
            import ultralytics.nn.modules.block as block_mod  # type: ignore
            block_mod.C2PSA = C2PSA_DPU
            log_info("  C2PSA -> C2PSA_DPU substitution applied")

            # Apply DWConv -> Conv replacement in Detect head
            from lpr_pipeline.detect_dpu import apply_dwconv_monkey_patch  # type: ignore
            apply_dwconv_monkey_patch()
            log_info("  Detect head DWConv -> Conv substitution applied")
        except ImportError as e:
            log_err(f"YOLOv11 surgery modules not found: {e}")
            log_err(f"Make sure lpr_pipeline/ is at {REPO_ROOT}/lpr_pipeline/")
            return 1

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        log_err("ultralytics not installed; pip install ultralytics")
        return 1

    cfg = DATASET_CONFIG[dataset]
    data_yaml = REPO_ROOT / cfg["yaml"]
    if not data_yaml.exists():
        log_err(f"data.yaml not found: {data_yaml}")
        log_err("run prep_datasets.py first")
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    log_file = output.with_suffix(".log")

    # Ultralytics auto-downloads pretrained weights on first instantiation
    model = YOLO(pretrained)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        seed=seed,
        project=str(output.parent),
        name=output.stem,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )

    # Ultralytics writes best.pt to <project>/<name>/weights/best.pt
    best_pt = output.parent / output.stem / "weights" / "best.pt"
    if best_pt.exists():
        log_info(f"copying {best_pt} -> {output}")
        shutil.copy(best_pt, output)
        log_info(f"final weights: {output}")
        return 0
    else:
        log_err(f"best.pt not found at {best_pt}")
        return 1


# ---------------------------------------------------------------------------
# Torchvision path (SSDLite, RetinaNet)
# ---------------------------------------------------------------------------

def train_torchvision(model_name: str, dataset: str, epochs: int, batch: int,
                       imgsz: int, seed: int, output: Path) -> int:
    """Train SSDLite-MobileNetV3-Large or RetinaNet-ResNet50-FPN via torchvision."""
    log_step(f"Training {model_name} on {dataset} (torchvision)")
    log_info(f"epochs={epochs}, batch={batch}, imgsz={imgsz}, seed={seed}")

    from torchvision.models.detection import (
        ssdlite320_mobilenet_v3_large,
        retinanet_resnet50_fpn,
        SSDLite320_MobileNet_V3_Large_Weights,
        RetinaNet_ResNet50_FPN_Weights,
    )
    from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead
    from functools import partial

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(f"device: {device}")
    if device.type == "cuda":
        log_info(f"GPU: {torch.cuda.get_device_name(0)}")

    cfg = DATASET_CONFIG[dataset]
    num_classes = cfg["num_classes"] + 1  # +1 for background
    coco_train = REPO_ROOT / cfg["coco_train"]
    coco_val = REPO_ROOT / cfg["coco_val"]
    if not coco_train.exists():
        log_err(f"COCO train JSON not found: {coco_train}")
        log_err("run prep_datasets.py first")
        return 1

    # Build the model with the right number of classes
    if model_name == "ssdlite":
        log_info(f"loading SSDLite-MobileNetV3-Large pretrained on COCO...")
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        model = ssdlite320_mobilenet_v3_large(weights=weights, weights_backbone=None)
        # Replace classification head for our num_classes
        in_channels = [m.in_channels for m in model.head.classification_head.module_list]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels, num_anchors, num_classes, norm_layer)
    elif model_name == "retinanet":
        log_info(f"loading RetinaNet-ResNet50-FPN pretrained on COCO...")
        weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1
        model = retinanet_resnet50_fpn(weights=weights, weights_backbone=None)
        # Replace classification head for our num_classes
        in_features = model.head.classification_head.cls_logits.in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_features, num_anchors, num_classes,
            norm_layer=partial(nn.GroupNorm, 32))
    else:
        log_err(f"unknown torchvision model: {model_name}")
        return 1
    model.to(device)

    # COCO-format dataset class
    img_root = REPO_ROOT / cfg["img_root"] / "train" / "images"
    val_img_root = REPO_ROOT / cfg["img_root"] / "val" / "images"
    train_ds = CocoDetection(str(img_root), str(coco_train), imgsz=imgsz)
    val_ds = CocoDetection(str(val_img_root), str(coco_val), imgsz=imgsz)

    log_info(f"train: {len(train_ds)} images, val: {len(val_ds)} images")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch, shuffle=True, num_workers=4,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch, shuffle=False, num_workers=4,
        collate_fn=collate_fn,
    )

    # Optimizer + LR schedule
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 0.7), int(epochs * 0.9)], gamma=0.1)

    # Training loop
    output.parent.mkdir(parents=True, exist_ok=True)
    log_file = output.with_suffix(".log")
    log_fp = open(log_file, "w")
    log_info(f"writing log to {log_file}")

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            n_batches += 1

        lr_scheduler.step()
        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = time.time() - t0
        msg = (f"epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  "
               f"lr={optimizer.param_groups[0]['lr']:.5f}  ({elapsed:.1f}s)")
        log_info(msg)
        log_fp.write(msg + "\n")
        log_fp.flush()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "dataset": dataset,
                "num_classes": num_classes,
                "epoch": epoch + 1,
                "loss": avg_loss,
            }, output)
            log_info(f"  saved checkpoint -> {output} (loss={avg_loss:.4f})")

    log_fp.close()
    log_info(f"training complete. Best loss: {best_loss:.4f}. Weights: {output}")
    return 0


# ---------------------------------------------------------------------------
# COCO-format Dataset for torchvision models
# ---------------------------------------------------------------------------

class CocoDetection:
    """A minimal COCO-style dataset, returning (image_tensor, target_dict).

    Used by torchvision detection models (SSDLite, RetinaNet) which expect
    targets in dict form with 'boxes' (x1,y1,x2,y2) and 'labels' (int64) keys.
    """

    def __init__(self, img_root: str, ann_file: str, imgsz: int = 640):
        self.img_root = Path(img_root)
        self.imgsz = imgsz
        with open(ann_file) as f:
            self.coco = json.load(f)
        self.img_id_to_anns: dict = {}
        for ann in self.coco["annotations"]:
            self.img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.coco["images"])

    def __getitem__(self, idx: int):
        img_info = self.coco["images"][idx]
        img_path = self.img_root / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize to imgsz x imgsz (simple, not aspect-preserving)
        img = img.resize((self.imgsz, self.imgsz))
        img_tensor = TF.to_tensor(img)

        scale_x = self.imgsz / orig_w
        scale_y = self.imgsz / orig_h

        anns = self.img_id_to_anns.get(img_info["id"], [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            # torchvision wants (x1, y1, x2, y2)
            boxes.append([x * scale_x, y * scale_y,
                          (x + w) * scale_x, (y + h) * scale_y])
            # COCO category_id starts at 0; torchvision wants 1-indexed (0=bg)
            labels.append(ann["category_id"] + 1)

        if not boxes:
            # Empty annotations: provide a dummy zero-area box (torchvision handles it)
            boxes = [[0.0, 0.0, 1.0, 1.0]]
            labels = [0]  # background

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_info["id"]]),
        }
        return img_tensor, target


def collate_fn(batch):
    """Variable-length collate for torchvision detection."""
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(ALL_MODELS.keys()))
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="override default image size for this model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=None,
                    help="override default output path")
    args = ap.parse_args()

    imgsz = args.imgsz or ALL_MODELS[args.model]["imgsz"]
    if args.output is None:
        ext = ".pt" if args.model in YOLO_MODELS else ".pth"
        args.output = WEIGHTS_OUT / f"{args.model}_{args.dataset}{ext}"

    log_step(f"Plan: train {args.model} on {args.dataset}")
    log_info(f"  epochs:  {args.epochs}")
    log_info(f"  batch:   {args.batch}")
    log_info(f"  imgsz:   {imgsz}")
    log_info(f"  seed:    {args.seed}")
    log_info(f"  output:  {args.output}")

    if args.model in YOLO_MODELS:
        return train_yolo(args.model, args.dataset, args.epochs, args.batch,
                          imgsz, args.seed, args.output)
    elif args.model in TORCHVISION_MODELS:
        return train_torchvision(args.model, args.dataset, args.epochs, args.batch,
                                  imgsz, args.seed, args.output)
    else:
        log_err(f"unknown model framework for {args.model}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
