"""Output decoders for compiled xmodels.

The DPU outputs raw conv responses; the head's post-processing (sigmoid,
grid-cell projection, DFL distribution → distance, NMS) is too irregular for
the DPU and runs on CPU here.

Currently implements:

  `decode_yolov5u` — modern Ultralytics YOLOv5 (anchor-free DFL). Both
                    yolov5n and yolov5s use this format with the same
                    `reg_max=16, nc=1` layout in our LPR pipeline.

To add later:

  `decode_yolov5_legacy` — classic anchor-based YOLOv5 (the
                          .yaml-defined yolov5n/s/m/l/x from
                          ultralytics/yolov5)
  `decode_yolox`         — YOLOX anchor-free, different layout
  `decode_ssd`           — SSD-MobileNet, totally different output structure
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


# Box format used internally: (x1, y1, x2, y2, score, class_idx).
Detection = Tuple[float, float, float, float, float, int]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable enough for the range we see post-DPU."""
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Subtract max for numerical stability before exp."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def decode_yolov5u(
    outputs: List[np.ndarray],
    imgsz: int,
    nc: int,
    reg_max: int,
    conf_thresh: float,
    iou_thresh: float,
    max_detections: int = 300,
) -> List[Detection]:
    """Decode the three-scale DFL output of a YOLOv5u xmodel.

    Pipeline:

      1. **Sort outputs** by spatial size (largest first = stride 8, then 16, 32).
         The DPU's output order is determined by graph traversal and isn't
         guaranteed to match the stride order. Sorting makes us robust.

      2. **Threshold each scale**: keep cells where the class score (after
         sigmoid) exceeds `conf_thresh`. Skip scales with no hits — saves a
         softmax for the (often majority) empty scales.

      3. **DFL decode**: each retained cell has `4 * reg_max` channels that
         encode 4 box-side distance distributions over `reg_max` discrete bins.
         Softmax + weighted sum collapses each distribution to a scalar
         distance in *stride units*. Multiply by stride to get pixel distances.

      4. **Reconstruct boxes**: cell-center + (distances × stride) → corners.

      5. **Concatenate all scales, then NMS** via OpenCV's NMSBoxes
         (C++ implementation; faster than a pure-python loop).

      6. **Cap to `max_detections`**: extra safety. Most scenes have ≤5 plates,
         so 300 is plenty of headroom while preventing pathological outputs
         from runaway models.

    Parameters
    ----------
    outputs : List[np.ndarray]
        The DPU's output buffers. Each has shape `[1, H, W, 4*reg_max + nc]`
        in NHWC (the compile pipeline's `_make_inference_model` ensures this
        layout). With imgsz=320 and reg_max=16, nc=1, expect three scales:
        `[1, 40, 40, 65]`, `[1, 20, 20, 65]`, `[1, 10, 10, 65]`.
    imgsz : int
        Model input side in pixels.
    nc, reg_max : int
        Number of classes; DFL distribution width. Both come from the spec
        in `lpr_pipeline.shared.models`.
    conf_thresh, iou_thresh : float
        Confidence threshold (pre-NMS); IoU threshold for NMS.
    max_detections : int
        Hard cap on the number of returned detections after NMS.

    Returns
    -------
    list of `(x1, y1, x2, y2, score, class_idx)` tuples, in model-input
    pixel coords (use `unletterbox` to map back to camera-frame coords).
    """
    # Sort by descending H — largest map first (= smallest stride = highest resolution).
    outputs = sorted(outputs, key=lambda a: -a.shape[1])
    strides = [imgsz // a.shape[1] for a in outputs]
    proj    = np.arange(reg_max, dtype=np.float32)   # bin indices [0, 1, ..., reg_max-1]

    boxes_all:  List[np.ndarray] = []
    scores_all: List[np.ndarray] = []

    for out, stride in zip(outputs, strides):
        # Class scores: last `nc` channels, post-sigmoid. We extract early and
        # threshold to skip the expensive softmax on empty scales.
        cls = _sigmoid(out[..., 4*reg_max : 4*reg_max + nc])  # shape (1, H, W, nc)
        # For LPR (nc=1) we squeeze the class dim; for multi-class we'd argmax.
        cls = cls[0, :, :, 0] if nc == 1 else cls[0].max(axis=-1)

        keep = cls > conf_thresh
        if not keep.any():
            continue

        ys, xs = np.where(keep)                      # cell-grid indices

        # DFL: 4 × reg_max channels per cell, reshape to (N, 4, reg_max).
        reg  = out[0, ys, xs, :4*reg_max].reshape(-1, 4, reg_max)
        ltrb = (_softmax(reg, axis=-1) * proj).sum(axis=-1)   # (N, 4) in stride units

        # Cell centers (+ 0.5 to land in the middle of each grid cell).
        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride

        # ltrb = (left, top, right, bottom) distances from cell center.
        x1 = cx - ltrb[:, 0] * stride
        y1 = cy - ltrb[:, 1] * stride
        x2 = cx + ltrb[:, 2] * stride
        y2 = cy + ltrb[:, 3] * stride

        boxes_all.append(np.stack([x1, y1, x2, y2], axis=1))
        scores_all.append(cls[ys, xs])

    if not boxes_all:
        return []

    boxes  = np.concatenate(boxes_all,  axis=0)
    scores = np.concatenate(scores_all, axis=0)

    # OpenCV's NMSBoxes wants (x, y, w, h), not (x1, y1, x2, y2).
    boxes_xywh = np.stack(
        [boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]],
        axis=1,
    ).tolist()

    # NMSBoxes returns indices into the input arrays.
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_thresh, iou_thresh,
                            top_k=max_detections)
    if len(idxs) == 0:
        return []
    idxs = np.asarray(idxs).flatten()[:max_detections]

    return [
        (float(boxes[i, 0]), float(boxes[i, 1]),
         float(boxes[i, 2]), float(boxes[i, 3]),
         float(scores[i]), 0)
        for i in idxs
    ]
