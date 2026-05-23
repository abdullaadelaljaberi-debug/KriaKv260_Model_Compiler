#!/usr/bin/env python3
"""
Classification training driver for the Kria KV260 thesis pipeline.

Trains one of three classifiers on one of two datasets:
    ResNet50 / MobileNetV2 / InceptionV3   -- via torchvision

Datasets are in ImageFolder layout under data/datasets/classification/<name>/{train,val}/.

Usage:
    python3 scripts/host/train_classification.py \\
        --model resnet50 --dataset gtsrb --epochs 30 --batch 64

    python3 scripts/host/train_classification.py \\
        --model mobilenetv2 --dataset oxford_pets --epochs 40 --batch 32

Output:
    data/weights/classification/<model>_<dataset>.pth
    data/weights/classification/<model>_<dataset>.log

Notes:
- We FREEZE the backbone for the first few epochs, then unfreeze. This is a
  transfer-learning recipe known to work well for small target datasets.
- ImageNet-pretrained weights from torchvision are used as the starting point.
- InceptionV3 requires 299x299 input; the others use 224x224.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Third-party — required to run at all. Fail fast and clearly if missing.
try:
    import torch
    import torch.nn as nn
    import torch.utils.data
    import torchvision
    from torchvision import datasets, transforms
except ImportError as e:
    print(f"ERROR: required dependency missing: {e}", file=sys.stderr)
    print("Install with: pip install torch torchvision", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data"
WEIGHTS_OUT = DATA_ROOT / "weights" / "classification"


CLASSIFICATION_MODELS = {
    "resnet50":    {"imgsz": 224, "torch_fn": "resnet50",
                    "weights": "ResNet50_Weights.IMAGENET1K_V2"},
    "mobilenetv2": {"imgsz": 224, "torch_fn": "mobilenet_v2",
                    "weights": "MobileNet_V2_Weights.IMAGENET1K_V2"},
    "inceptionv3": {"imgsz": 299, "torch_fn": "inception_v3",
                    "weights": "Inception_V3_Weights.IMAGENET1K_V1"},
}

DATASET_CONFIG = {
    "gtsrb": {
        "root": "data/datasets/classification/gtsrb",
        "num_classes": 43,
    },
    "oxford_pets": {
        "root": "data/datasets/classification/oxford_pets",
        "num_classes": 37,
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
# Main training routine
# ---------------------------------------------------------------------------

def train_classification(model_name: str, dataset: str, epochs: int, batch: int,
                          imgsz: int, seed: int, output: Path,
                          freeze_epochs: int = 5) -> int:
    """Train a classification model with transfer learning from ImageNet."""

    log_step(f"Training {model_name} on {dataset}")
    log_info(f"epochs={epochs}, batch={batch}, imgsz={imgsz}, seed={seed}")
    log_info(f"backbone frozen for first {freeze_epochs} epochs")

    from torchvision.models import (
        resnet50, ResNet50_Weights,
        mobilenet_v2, MobileNet_V2_Weights,
        inception_v3, Inception_V3_Weights,
    )

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(f"device: {device}")
    if device.type == "cuda":
        log_info(f"GPU: {torch.cuda.get_device_name(0)}")

    cfg = DATASET_CONFIG[dataset]
    num_classes = cfg["num_classes"]
    data_root = REPO_ROOT / cfg["root"]
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.exists() or not val_dir.exists():
        log_err(f"dataset not prepared: {data_root}")
        log_err("run prep_datasets.py first")
        return 1

    # Transforms - ImageNet normalization (because we use ImageNet-pretrained backbones)
    train_tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)
    log_info(f"train: {len(train_ds)} images, val: {len(val_ds)} images")
    log_info(f"classes ({len(train_ds.classes)}): {train_ds.classes[:5]}{'...' if len(train_ds.classes) > 5 else ''}")
    assert len(train_ds.classes) == num_classes, (
        f"dataset has {len(train_ds.classes)} classes but config says {num_classes}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)

    # Build the model with ImageNet-pretrained weights, then replace final layer
    if model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenetv2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "inceptionv3":
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                             aux_logits=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        # InceptionV3 has an aux classifier that needs replacement too
        if model.AuxLogits is not None:
            aux_in = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(aux_in, num_classes)
    else:
        log_err(f"unknown model: {model_name}")
        return 1
    model.to(device)

    # Freeze backbone, unfreeze only the final layer
    if freeze_epochs > 0:
        for p in model.parameters():
            p.requires_grad = False
        if model_name == "resnet50":
            for p in model.fc.parameters():
                p.requires_grad = True
        elif model_name == "mobilenetv2":
            for p in model.classifier.parameters():
                p.requires_grad = True
        elif model_name == "inceptionv3":
            for p in model.fc.parameters():
                p.requires_grad = True
            if model.AuxLogits is not None:
                for p in model.AuxLogits.fc.parameters():
                    p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.01, momentum=0.9, weight_decay=1e-4,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)

    # Training loop
    output.parent.mkdir(parents=True, exist_ok=True)
    log_file = output.with_suffix(".log")
    log_fp = open(log_file, "w")
    log_info(f"writing log to {log_file}")

    best_acc = 0.0
    for epoch in range(epochs):
        # Unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs and freeze_epochs > 0:
            log_info(f"epoch {epoch+1}: unfreezing backbone (full fine-tune)")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4,
            )
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - freeze_epochs, eta_min=1e-6)

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        n_train = 0
        t0 = time.time()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if model_name == "inceptionv3":
                outputs, aux = model(images)
                loss = criterion(outputs, labels) + 0.4 * criterion(aux, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            n_train += images.size(0)

        train_loss /= n_train
        train_acc = train_correct / n_train
        lr_scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        n_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # InceptionV3 returns (main, aux)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                n_val += images.size(0)
        val_acc = val_correct / n_val

        elapsed = time.time() - t0
        msg = (f"epoch {epoch+1}/{epochs}  "
               f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
               f"val_acc={val_acc:.4f}  lr={optimizer.param_groups[0]['lr']:.5f}  "
               f"({elapsed:.1f}s)")
        log_info(msg)
        log_fp.write(msg + "\n")
        log_fp.flush()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "dataset": dataset,
                "num_classes": num_classes,
                "class_names": train_ds.classes,
                "imgsz": imgsz,
                "epoch": epoch + 1,
                "val_acc": val_acc,
            }, output)
            log_info(f"  saved checkpoint -> {output} (val_acc={val_acc:.4f})")

    log_fp.close()
    log_info(f"training complete. Best val acc: {best_acc:.4f}. Weights: {output}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(CLASSIFICATION_MODELS.keys()))
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="override default image size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze-epochs", type=int, default=5,
                    help="how many epochs to keep backbone frozen")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    imgsz = args.imgsz or CLASSIFICATION_MODELS[args.model]["imgsz"]
    if args.output is None:
        args.output = WEIGHTS_OUT / f"{args.model}_{args.dataset}.pth"

    log_step(f"Plan: train {args.model} on {args.dataset}")
    log_info(f"  epochs:        {args.epochs}")
    log_info(f"  batch:         {args.batch}")
    log_info(f"  imgsz:         {imgsz}")
    log_info(f"  seed:          {args.seed}")
    log_info(f"  freeze epochs: {args.freeze_epochs}")
    log_info(f"  output:        {args.output}")

    return train_classification(
        args.model, args.dataset, args.epochs, args.batch,
        imgsz, args.seed, args.output, args.freeze_epochs,
    )


if __name__ == "__main__":
    sys.exit(main())
