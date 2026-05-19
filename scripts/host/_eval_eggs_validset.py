"""Float-model evaluation on the eggs validation set.

Measures TPs / FPs / FNs / precision / recall / mAP@0.5 for the float PyTorch
models on the 32-image eggs valid set (1,717 labeled eggs total).

This is the COMPLEMENT to the industrial-image FP test: there, any detection
was an FP. Here, predictions must match ground-truth boxes by IoU >= 0.5 to
count as TPs; otherwise they're FPs.

Compares:
  - yolov11n_eggs_dpu.pt (3.6M params)
  - yolov11s_eggs_dpu.pt (13.5M params)
  - yolov5s_eggs_dpu.pt  (9.1M params)

Usage:
  cd ~/Documents/Girona_Masters/Thesis/KriaKv260_Model_Compiler
  python3 scripts/host/_eval_eggs_validset.py
"""
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cv2

EGGS_DIR = Path("/home/aaljaberi/Documents/Girona_Masters/Thesis/yolo11n_eggs/egg.v4-egg.yolov11")
VALID_IMAGES = EGGS_DIR / "valid" / "images"
VALID_LABELS = EGGS_DIR / "valid" / "labels"

# Models to compare
MODELS = {
    "yolov11n": "data/weights/yolo11n_eggs_dpu.pt",
    "yolov5s":  "data/weights/yolo5s_eggs_dpu.pt",
    "yolov11s": "data/weights/yolo11s_eggs_dpu.pt",
}

THRESHOLDS = [0.50, 0.70, 0.85]
IOU_MATCH = 0.50  # IoU threshold for matching prediction to ground truth


def load_gt(label_path, img_w, img_h):
    """Load YOLO-format labels and convert to xyxy in pixel coordinates."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = (float(x) for x in parts[:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def iou_matrix(pred_boxes, gt_boxes):
    """Vectorised IoU between each pair of (pred, gt) boxes. Returns NxM."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)

    p = pred_boxes[:, None, :]    # N,1,4
    g = gt_boxes[None, :, :]      # 1,M,4

    xa = np.maximum(p[..., 0], g[..., 0])
    ya = np.maximum(p[..., 1], g[..., 1])
    xb = np.minimum(p[..., 2], g[..., 2])
    yb = np.minimum(p[..., 3], g[..., 3])

    inter = np.clip(xb - xa, 0, None) * np.clip(yb - ya, 0, None)
    p_area = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
    g_area = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    union = p_area + g_area - inter
    return inter / np.maximum(union, 1e-6)


def match_predictions(pred_boxes, pred_confs, gt_boxes, iou_thresh=0.5):
    """Match predictions to ground truth greedily by confidence (descending).

    Returns (tp_mask, fp_mask, gt_matched_mask).
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    tp = np.zeros(n_pred, dtype=bool)
    fp = np.zeros(n_pred, dtype=bool)
    gt_matched = np.zeros(n_gt, dtype=bool)

    if n_pred == 0:
        return tp, fp, gt_matched

    # Sort predictions by confidence descending
    order = np.argsort(-pred_confs)

    iou = iou_matrix(pred_boxes, gt_boxes)

    for pred_idx in order:
        if n_gt == 0:
            fp[pred_idx] = True
            continue
        # Best matching ground truth not yet claimed
        ious_for_this = iou[pred_idx].copy()
        ious_for_this[gt_matched] = -1  # already-matched GTs ineligible
        best_gt = int(np.argmax(ious_for_this))
        best_iou = ious_for_this[best_gt]
        if best_iou >= iou_thresh:
            tp[pred_idx] = True
            gt_matched[best_gt] = True
        else:
            fp[pred_idx] = True

    return tp, fp, gt_matched


def evaluate_model(name, weight_path, conf_thresholds):
    """Evaluate one model across multiple confidence thresholds."""
    print(f"\n{'═'*70}")
    print(f"  {name}  ({weight_path})")
    print('═'*70)

    model = YOLO(weight_path)

    # Get all validation images
    images = sorted(VALID_IMAGES.glob("*.jpg"))

    # Pre-cache predictions at the LOWEST threshold (we can filter higher ones in post)
    min_conf = min(conf_thresholds)

    # Storage: per-image GT boxes + per-image (preds, confs)
    per_image = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  warn: could not read {img_path.name}")
            continue
        h, w = img.shape[:2]

        label_path = VALID_LABELS / (img_path.stem + ".txt")
        gt = load_gt(label_path, w, h)

        # Run inference once at the lowest threshold; filter higher thresholds later
        results = model(str(img_path), conf=min_conf, iou=0.45, verbose=False)
        if len(results[0].boxes) > 0:
            preds = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
        else:
            preds = np.zeros((0, 4), dtype=np.float32)
            confs = np.zeros((0,), dtype=np.float32)

        per_image.append((img_path.name, gt, preds, confs))

    # Now evaluate at each threshold
    for conf in conf_thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_gt = 0
        per_image_summary = []

        for name_img, gt, preds, confs in per_image:
            # Filter predictions at this threshold
            keep = confs >= conf
            preds_kept = preds[keep]
            confs_kept = confs[keep]

            tp, fp, gt_matched = match_predictions(
                preds_kept, confs_kept, gt, iou_thresh=IOU_MATCH
            )

            tp_count = int(tp.sum())
            fp_count = int(fp.sum())
            fn_count = int((~gt_matched).sum())

            total_tp += tp_count
            total_fp += fp_count
            total_fn += fn_count
            total_gt += len(gt)
            per_image_summary.append((name_img, tp_count, fp_count, fn_count, len(gt)))

        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        print(f"\n=== conf >= {conf} ===")
        print(f"  Aggregate over {len(per_image)} valid images ({total_gt} ground-truth eggs):")
        print(f"    TPs:        {total_tp}")
        print(f"    FPs:        {total_fp}  (predictions matching no GT)")
        print(f"    FNs:        {total_fn}  (GT eggs with no prediction)")
        print(f"    Precision:  {precision:.4f}")
        print(f"    Recall:     {recall:.4f}")
        print(f"    F1:         {f1:.4f}")

        # Worst per-image cases at the deployment threshold
        if conf == 0.85:
            worst_fp = sorted(per_image_summary, key=lambda x: -x[2])[:3]
            worst_fn = sorted(per_image_summary, key=lambda x: -x[3])[:3]
            if worst_fp[0][2] > 0:
                print(f"\n  Worst-FP images @ 0.85 (predictions with no matching egg):")
                for n, tp_, fp_, fn_, ng in worst_fp:
                    if fp_ > 0:
                        print(f"    {fp_:3d} FPs  /  TPs={tp_:3d}  FNs={fn_:3d}  GT={ng:3d}  -  {n}")
            if worst_fn[0][3] > 0:
                print(f"\n  Worst-FN images @ 0.85 (missed eggs):")
                for n, tp_, fp_, fn_, ng in worst_fn:
                    print(f"    {fn_:3d} FNs  /  TPs={tp_:3d}  FPs={fp_:3d}  GT={ng:3d}  -  {n}")


def main():
    print(f"\nFloat-model eggs validation evaluation")
    print(f"  Test set: {VALID_IMAGES} ({len(list(VALID_IMAGES.glob('*.jpg')))} images)")

    # Count total ground-truth eggs
    total_gt = sum(
        len([line for line in p.read_text().splitlines() if line.strip()])
        for p in VALID_LABELS.glob("*.txt")
    )
    print(f"  Total ground-truth eggs: {total_gt}")
    print(f"  IoU match threshold:     {IOU_MATCH}")
    print(f"  Confidence thresholds:   {THRESHOLDS}")

    for name, path in MODELS.items():
        evaluate_model(name, path, THRESHOLDS)

    print(f"\n{'═'*70}")
    print("  Summary key:")
    print('═'*70)
    print("  TP = prediction matched a ground-truth egg by IoU >= 0.5")
    print("  FP = prediction with no matching GT (false detection)")
    print("  FN = GT egg with no matching prediction (missed)")
    print("  Precision = TP / (TP+FP) — fraction of predictions that are real")
    print("  Recall    = TP / (TP+FN) — fraction of real eggs found")
    print()


if __name__ == "__main__":
    main()
